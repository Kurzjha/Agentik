from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from config import HarnessConfig
from memory import SessionInsights, SessionMemory
from tools import ToolContext, ToolExecutionError, build_default_tools


@dataclass(frozen=True, slots=True)
class TaskProfile:
    kind: str
    should_act: bool
    requires_verification: bool
    preferred_validation_commands: tuple[str, ...]

    @classmethod
    def from_user_input(cls, user_input: str) -> "TaskProfile":
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in ("why", "explain", "what is", "how does")):
            return cls(
                kind="analysis",
                should_act=False,
                requires_verification=False,
                preferred_validation_commands=(),
            )
        if any(keyword in lowered for keyword in ("test", "bug", "fix", "implement", "build", "create")):
            return cls(
                kind="implementation",
                should_act=True,
                requires_verification=True,
                preferred_validation_commands=(
                    "python -m pytest",
                    "python main.py --self-check",
                    "python app/main.py --self-check",
                    "python main.py",
                ),
            )
        return cls(
            kind="general",
            should_act=True,
            requires_verification=False,
            preferred_validation_commands=("python main.py --self-check", "python app/main.py --self-check", "python main.py"),
        )


@dataclass(slots=True)
class AgentHarness:
    config: HarnessConfig
    session_id: str
    depth: int = 0
    tools: dict[str, Any] = field(default_factory=build_default_tools)
    memory: SessionMemory = field(init=False)
    wrote_files_in_run: bool = field(init=False, default=False)
    verification_requested: bool = field(init=False, default=False)
    verification_succeeded: bool = field(init=False, default=False)
    validation_in_progress: bool = field(init=False, default=False)
    validation_failed: bool = field(init=False, default=False)
    last_validation_error: str = field(init=False, default="")
    dependency_install_completed: bool = field(init=False, default=False)
    progress_step: int = field(init=False, default=0)
    user_input: str = field(init=False, default="")
    task_profile: TaskProfile = field(init=False)
    session_insights: SessionInsights = field(init=False)
    assistant_turns: int = field(init=False, default=0)
    tool_calls_executed: int = field(init=False, default=0)
    context_rounds_without_action: int = field(init=False, default=0)
    last_validation_command: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.memory = SessionMemory(self.config.session_root / f"{self.session_id}.jsonl")
        self.task_profile = TaskProfile(
            kind="general",
            should_act=True,
            requires_verification=False,
            preferred_validation_commands=(),
        )
        self.session_insights = self.memory.build_insights()

    def run(self, user_input: str) -> str:
        stripped = user_input.strip()
        if stripped == "/clear":
            return self._handle_clear()
        if stripped.startswith("/rewind"):
            return self._handle_rewind(stripped)

        self.config.ensure_ready()
        self.wrote_files_in_run = False
        self.verification_requested = False
        self.verification_succeeded = False
        self.validation_in_progress = False
        self.validation_failed = False
        self.last_validation_error = ""
        self.dependency_install_completed = False
        self.progress_step = 0
        self.user_input = user_input
        self.task_profile = TaskProfile.from_user_input(user_input)
        self.session_insights = self.memory.build_insights()
        self.assistant_turns = 0
        self.tool_calls_executed = 0
        self.context_rounds_without_action = 0
        self.last_validation_command = ""
        is_build_request = self._looks_like_build_request(user_input) or self.task_profile.should_act
        history = self.memory.load_messages(limit=self.config.history_message_limit)
        history_digest = self.memory.build_history_digest(
            recent_message_limit=self.config.history_digest_recent_limit,
            item_limit=self.config.history_digest_item_limit,
        )
        system_prompt = {
            "role": "system",
            "content": self.config.build_system_prompt(
                user_input=user_input,
                extra_instructions=self._runtime_instructions(user_input),
                session_context=self.session_insights.render_for_prompt(),
                history_digest=history_digest,
            ),
        }
        user_message = {"role": "user", "content": user_input}
        self.memory.append_message(user_message)
        messages = [system_prompt, *history, user_message]

        for _ in range(self.config.max_tool_rounds):
            response = self._call_model(messages)
            assistant_message = self._extract_assistant_message(response)
            self.assistant_turns += 1
            self._emit_progress_updates(assistant_message.get("content", ""))
            self.memory.append_message(assistant_message)
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                if self._needs_action_nudge():
                    nudge_message = self._build_action_nudge_message()
                    messages.append(nudge_message)
                    self.memory.append_message(nudge_message)
                    continue
                if is_build_request and self._needs_required_project_files():
                    required_files_message = self._build_required_project_files_message()
                    messages.append(required_files_message)
                    self.memory.append_message(required_files_message)
                    continue
                if is_build_request and self._needs_dependency_install():
                    install_message = self._build_dependency_install_message()
                    messages.append(install_message)
                    self.memory.append_message(install_message)
                    continue
                if is_build_request and self._needs_repair_after_failed_validation():
                    repair_message = self._build_repair_message()
                    messages.append(repair_message)
                    self.memory.append_message(repair_message)
                    self.validation_in_progress = False
                    continue
                if is_build_request and self._needs_verification():
                    verification_message = self._build_verification_message()
                    messages.append(verification_message)
                    self.memory.append_message(verification_message)
                    self.verification_requested = True
                    self.validation_in_progress = True
                    continue
                return assistant_message.get("content", "")

            for tool_call in tool_calls:
                tool_result_message = self._execute_tool_call(tool_call)
                self.memory.append_message(tool_result_message)
                messages.append(tool_result_message)
                self.tool_calls_executed += 1
                if tool_result_message.get("name") == "write_file":
                    self.wrote_files_in_run = True
                    self.context_rounds_without_action = 0
                    if self.validation_failed:
                        self.validation_failed = False
                        self.last_validation_error = ""
                elif tool_result_message.get("name") in {"read_file", "list_files"}:
                    self.context_rounds_without_action += 1
                self._update_validation_state(tool_result_message)
                if self._tool_result_installed_dependencies(tool_result_message):
                    self.dependency_install_completed = True
                follow_up_message = self._maybe_build_phase_follow_up(tool_result_message)
                if follow_up_message:
                    messages.append(follow_up_message)
                    self.memory.append_message(follow_up_message)

        raise RuntimeError("Agent loop exceeded maximum tool rounds.")

    def _runtime_instructions(self, user_input: str) -> list[str]:
        instructions = [f"Sub-agent depth: {self.depth}"]
        failure_patterns = list(self.session_insights.recent_failure_patterns)
        if failure_patterns:
            instructions.append(
                "Recent failure patterns from this session that should not be repeated blindly:"
            )
            instructions.extend(f"- {pattern}" for pattern in failure_patterns)
        instructions.extend(
            [
                "Work in explicit phases: gather context, take action, then verify.",
                "Prefer short plans with 2 to 5 steps before making broad changes.",
                "If you have already inspected enough context, stop reading and start editing or executing the next concrete step.",
                "After any failed command, explain the likely cause briefly, adjust the plan, and try a narrower repair.",
            ]
        )
        if self.session_insights.successful_commands:
            instructions.append("Relevant successful shell commands from this session may be reused when appropriate:")
            instructions.extend(f"- {command}" for command in self.session_insights.successful_commands)
        if self._looks_like_build_request(user_input):
            instructions.extend(
                [
                    "This request is a build/scaffold task.",
                    "Ground yourself in PROJECT_ROOT before editing: inspect files and read relevant code first.",
                    "Do not read from or operate on files outside PROJECT_ROOT.",
                    "Use make_dir, write_file, read_file, list_files, and bash as needed to create a working project in PROJECT_ROOT.",
                    "When creating a project, you must create README.md in PROJECT_ROOT.",
                    "When creating a project, you must create requirements.txt in PROJECT_ROOT, even if it is minimal.",
                    "README.md must explain what the project does, how to launch it, and the launch algorithm step by step.",
                    "Do not stop at a text-only answer when files or folders are expected.",
                    "While working, include short status markers in braces, for example {PROJECT SETUP} or {REPORT GENERATION}. Keep them short, uppercase, and focused on the current phase.",
                    "If you create dependency manifests such as requirements.txt, pyproject.toml, package.json, Pipfile, or similar, install the dependencies inside PROJECT_ROOT when PERMISSION_MODE allows shell execution.",
                    "After creating files, run functionality checks, and if a check fails, fix the project and run the checks again until they pass or you can explain the blocker precisely.",
                ]
            )
        return instructions

    @staticmethod
    def _looks_like_build_request(user_input: str) -> bool:
        lowered = user_input.lower()
        keywords = [
            "create project",
            "build project",
            "generate project",
            "scaffold",
            "create files",
            "create folders",
            "implement",
            "make project",
            "agent-based ai",
            "agentic ai",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _needs_verification(self) -> bool:
        return self.wrote_files_in_run and not self.verification_succeeded and not self.validation_in_progress

    def _needs_repair_after_failed_validation(self) -> bool:
        return self.validation_failed and not self.validation_in_progress

    def _needs_dependency_install(self) -> bool:
        if self.config.permission_mode == "plan":
            return False
        return self._project_has_dependency_manifest() and not self.dependency_install_completed

    def _needs_required_project_files(self) -> bool:
        if not self.wrote_files_in_run:
            return False
        required_files = ("README.md", "requirements.txt")
        return any(not (self.config.project_root / name).exists() for name in required_files)

    def _project_has_dependency_manifest(self) -> bool:
        manifests = [
            "requirements.txt",
            "pyproject.toml",
            "Pipfile",
            "package.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
        ]
        return any((self.config.project_root / name).exists() for name in manifests)

    def _build_dependency_install_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Dependency installation pass required. Check PROJECT_ROOT for dependency manifests "
                "such as requirements.txt, pyproject.toml, Pipfile, package.json, yarn.lock, pnpm-lock.yaml, "
                "or poetry.lock. If present and PERMISSION_MODE allows shell execution, install dependencies "
                "inside PROJECT_ROOT, report any install failures briefly, fix them if needed, and continue. "
                "Include a short progress marker in braces while you work."
            ),
        }

    def _build_required_project_files_message(self) -> dict[str, str]:
        missing_files = [
            name
            for name in ("README.md", "requirements.txt")
            if not (self.config.project_root / name).exists()
        ]
        joined = ", ".join(missing_files)
        return {
            "role": "user",
            "content": (
                "Project generation is not complete yet. "
                f"Create the missing required files inside PROJECT_ROOT now: {joined}. "
                "README.md must explain the project and how to run it. requirements.txt must exist even if it is minimal."
            ),
        }

    def _build_verification_message(self) -> dict[str, str]:
        validation_hints = ", ".join(self._suggest_validation_commands()) or "a project-appropriate check"
        return {
            "role": "user",
            "content": (
                "Validation pass required. Check the generated project for functionality now. "
                f"Run an appropriate functionality check inside PROJECT_ROOT, preferably with bash. Suggested commands: {validation_hints}. "
                "Only treat the project as validated if the check succeeds. Include a short progress marker in braces while you work."
            ),
        }

    def _build_repair_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "The last functionality check failed. Fix the project based on this validation error, then run the functionality check again inside PROJECT_ROOT. "
                f"Last validation error: {self.last_validation_error} "
                "Include a short progress marker in braces while you work."
            ),
        }

    def _emit_progress_updates(self, content: str) -> None:
        for marker in self._extract_progress_markers(content):
            self.progress_step += 1
            print(self._format_progress_marker(marker, self.progress_step), flush=True)

    @staticmethod
    def _extract_progress_markers(content: str) -> list[str]:
        markers = re.findall(r"\{[^{}\n]{1,120}\}", content)
        return markers

    @staticmethod
    def _format_progress_marker(marker: str, step: int) -> str:
        label = marker.strip("{} ").strip()
        if not label:
            label = "WORK IN PROGRESS"
        width = max(69, len(label) + 14)
        line = "-" * width
        return f"{line}\n {step} {label}\n{line}"

    @staticmethod
    def _tool_result_installed_dependencies(tool_result_message: dict[str, Any]) -> bool:
        if tool_result_message.get("name") != "bash":
            return False
        content = str(tool_result_message.get("content", "")).lower()
        install_markers = [
            "pip install",
            "python -m pip install",
            "poetry install",
            "pipenv install",
            "npm install",
            "npm ci",
            "yarn install",
            "pnpm install",
        ]
        return any(marker in content for marker in install_markers)

    def _update_validation_state(self, tool_result_message: dict[str, Any]) -> None:
        if tool_result_message.get("name") != "bash" or not self.validation_in_progress:
            return
        try:
            payload = json.loads(str(tool_result_message.get("content", "{}")))
        except json.JSONDecodeError:
            return
        command = str(payload.get("command", ""))
        if not self._looks_like_validation_command(command):
            return
        self.last_validation_command = command
        returncode = int(payload.get("returncode", 1))
        if returncode == 0:
            self.verification_succeeded = True
            self.validation_in_progress = False
            self.validation_failed = False
            self.last_validation_error = ""
            self._remember_successful_run()
            return
        self.verification_succeeded = False
        self.validation_in_progress = False
        self.validation_failed = True
        stderr = str(payload.get("stderr", "")).strip()
        stdout = str(payload.get("stdout", "")).strip()
        self.last_validation_error = stderr or stdout or f"Validation command failed: {command}"

    @staticmethod
    def _looks_like_validation_command(command: str) -> bool:
        lowered = command.lower()
        markers = [
            "pytest",
            "python -m pytest",
            "python -m unittest",
            "unittest",
            "npm test",
            "npm run test",
            "yarn test",
            "pnpm test",
            "cargo test",
            "go test",
            "--self-check",
            "python main.py",
            "python app.py",
            "uvicorn",
            "flask run",
        ]
        return any(marker in lowered for marker in markers)

    def _suggest_validation_commands(self) -> tuple[str, ...]:
        commands: list[str] = []
        project_root = self.config.project_root

        if (project_root / "tests").exists() or any(project_root.glob("test_*.py")):
            commands.append("python -m pytest")

        python_entrypoints = [
            project_root / "main.py",
            project_root / "app" / "main.py",
            project_root / "app.py",
        ]
        for entrypoint in python_entrypoints:
            if not entrypoint.exists():
                continue
            relative_path = entrypoint.relative_to(project_root).as_posix()
            if self._entrypoint_supports_self_check(entrypoint):
                commands.append(f"python {relative_path} --self-check")
            commands.append(f"python {relative_path}")

        if (project_root / "package.json").exists():
            commands.extend(["npm test", "npm run test"])

        if (project_root / "Cargo.toml").exists():
            commands.append("cargo test")

        if not commands:
            commands.extend(self.task_profile.preferred_validation_commands)

        deduped: list[str] = []
        for command in commands:
            if command not in deduped:
                deduped.append(command)
        return tuple(deduped)

    @staticmethod
    def _entrypoint_supports_self_check(path: Any) -> bool:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return False
        return "--self-check" in content

    def _needs_action_nudge(self) -> bool:
        if not self.task_profile.should_act:
            return False
        if self.tool_calls_executed > 0:
            return False
        return self.assistant_turns >= 1

    def _build_action_nudge_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "This request requires execution, not only analysis. Produce a short plan, then use tools to inspect or modify the workspace. "
                "Do not end with a text-only answer yet."
            ),
        }

    def _maybe_build_phase_follow_up(self, tool_result_message: dict[str, Any]) -> dict[str, str] | None:
        name = str(tool_result_message.get("name", ""))
        if name in {"read_file", "list_files"} and self.context_rounds_without_action >= 2 and not self.wrote_files_in_run:
            self.context_rounds_without_action = 0
            return {
                "role": "user",
                "content": (
                    "You have gathered initial context. Move to the next concrete action now: edit files, run the relevant command, or explain the missing blocker in one sentence."
                ),
            }
        if name == "bash":
            try:
                payload = json.loads(str(tool_result_message.get("content", "{}")))
            except json.JSONDecodeError:
                return None
            if int(payload.get("returncode", 0)) != 0 and not self.validation_in_progress:
                stderr = str(payload.get("stderr", "")).strip()
                stdout = str(payload.get("stdout", "")).strip()
                error = stderr or stdout or "Command failed."
                return {
                    "role": "user",
                    "content": (
                        "The last command failed. Explain the failure briefly, adjust the plan, and try the smallest repair that addresses it. "
                        f"Failure: {error[:600]}"
                    ),
                }
        return None

    def _remember_successful_run(self) -> None:
        touched_paths = list(self.memory.build_insights(path_limit=self.config.plan_cache_limit).touched_paths)
        self.config.plan_cache.remember(
            task=self.user_input,
            summary=self._summarize_successful_run(),
            validation_command=self.last_validation_command,
            touched_paths=touched_paths,
        )

    def _summarize_successful_run(self) -> str:
        if self.wrote_files_in_run:
            return "Implemented or updated project files and completed a successful validation pass."
        return "Completed the task and finished with a successful validation pass."

    def _handle_rewind(self, command: str) -> str:
        parts = command.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError("Usage: /rewind <message_index>")
        retained = self.memory.rewind(int(parts[1]))
        return f"Removed the first {parts[1]} messages. {retained} messages remain in session."

    def _handle_clear(self) -> str:
        removed = self.memory.count()
        self.memory.clear()
        return f"Cleared session history. Removed {removed} messages."

    def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": [tool.spec() for tool in self.tools.values()],
            "tool_choice": "auto",
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.model_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.config.request_retry_attempts + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.request_timeout_seconds,
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenRouter request failed: {exc.code} {body}") from exc
            except (TimeoutError, socket.timeout) as exc:
                last_error = exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", "")
                if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
                    last_error = exc
                else:
                    raise RuntimeError("OpenRouter is unreachable from this environment.") from exc

            if attempt < self.config.request_retry_attempts:
                time.sleep(self.config.request_retry_backoff_seconds)

        raise RuntimeError(
            "OpenRouter request timed out after multiple attempts. "
            "Try again, reduce prompt size, or increase request timeout/retry settings in settings.json."
        ) from last_error

    def _extract_assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Model response missing choices: {response}")
        message = choices[0].get("message")
        if not message:
            raise RuntimeError(f"Model response missing assistant message: {response}")
        normalized = {
            "role": message.get("role", "assistant"),
            "content": message.get("content", "") or "",
        }
        if message.get("tool_calls"):
            normalized["tool_calls"] = message["tool_calls"]
        return normalized

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        if name not in self.tools:
            result = f"Unknown tool requested: {name}"
        else:
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError as exc:
                arguments = {}
                result = f"Invalid tool arguments for {name}: {exc}"
            else:
                context = ToolContext(
                    project_root=self.config.project_root,
                    permission_mode=self.config.permission_mode,
                    memory=self.memory,
                    spawn_handler=self._spawn_subagent,
                    subagent_depth=self.depth,
                    max_subagent_depth=self.config.max_subagent_depth,
                )
                try:
                    result = self.tools[name].run(arguments, context)
                except ToolExecutionError as exc:
                    result = f"ToolExecutionError: {exc}"
                except OSError as exc:
                    result = f"OSError while running {name}: {exc}"

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": name,
            "content": result,
        }

    def _spawn_subagent(self, *, task: str, session_id: str, depth: int) -> str:
        child = AgentHarness(config=self.config, session_id=session_id, depth=depth, tools=self.tools)
        return child.run(task)
