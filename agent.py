from __future__ import annotations

import json
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import HarnessConfig
from memory import SessionInsights, SessionMemory
from task_orchestration import (
    PlannedSubtask,
    SubagentOutcome,
    TaskPlan,
    build_planning_messages,
    build_subtask_execution_prompt,
    fallback_task_plan,
    parse_task_plan,
)
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
        if any(
            keyword in lowered
            for keyword in ("why", "explain", "what is", "how does", "почему", "объясни", "что такое", "как работает")
        ):
            return cls(
                kind="analysis",
                should_act=False,
                requires_verification=False,
                preferred_validation_commands=(),
            )
        if any(
            keyword in lowered
            for keyword in ("test", "bug", "fix", "implement", "build", "create", "тест", "баг", "ошиб", "исправ", "реализ", "собер", "создай")
        ):
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
    gigachat_access_token: str = field(init=False, default="")
    streamed_model_output_in_run: bool = field(init=False, default=False)
    action_nudge_sent: bool = field(init=False, default=False)
    repeated_assistant_turns: int = field(init=False, default=0)
    last_assistant_fingerprint: str = field(init=False, default="")
    pseudo_tool_correction_sent: bool = field(init=False, default=False)

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
        self.streamed_model_output_in_run = False
        self.action_nudge_sent = False
        self.repeated_assistant_turns = 0
        self.last_assistant_fingerprint = ""
        self.pseudo_tool_correction_sent = False
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
        if is_build_request and self.depth == 0:
            plan = self._build_task_plan(user_input=user_input, history_digest=history_digest)
            return self._execute_task_plan(plan, user_input=user_input)
        messages = [system_prompt, *history, user_message]

        for _ in range(self.config.max_tool_rounds):
            response = self._call_model(messages)
            assistant_message = self._extract_assistant_message(response)
            self.assistant_turns += 1
            assistant_content = assistant_message.get("content", "")
            self._emit_progress_updates(assistant_content)
            self._emit_assistant_output(assistant_content, is_build_request=is_build_request)
            synthesized_tool_call = self._extract_textual_tool_call(assistant_content)
            if synthesized_tool_call:
                assistant_message["tool_calls"] = [synthesized_tool_call]
            self._track_assistant_repetition(assistant_content)
            self.memory.append_message(assistant_message)
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                if self._looks_like_textual_tool_imitation(assistant_content):
                    if self.pseudo_tool_correction_sent:
                        return self._build_pseudo_tool_abort_message(assistant_content)
                    correction_message = self._build_pseudo_tool_correction_message()
                    messages.append(correction_message)
                    self.memory.append_message(correction_message)
                    self.pseudo_tool_correction_sent = True
                    continue
                if self._should_abort_on_repetition():
                    return self._build_repetition_abort_message(assistant_content)
                if self._needs_action_nudge():
                    nudge_message = self._build_action_nudge_message()
                    messages.append(nudge_message)
                    self.memory.append_message(nudge_message)
                    self.action_nudge_sent = True
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

        raise RuntimeError("Цикл агента превысил допустимое число раундов работы с инструментами.")

    def _runtime_instructions(self, user_input: str) -> list[str]:
        instructions = [f"Глубина подагента: {self.depth}"]
        failure_patterns = list(self.session_insights.recent_failure_patterns)
        if failure_patterns:
            instructions.append(
                "Недавние шаблоны ошибок из этой сессии, которые не нужно повторять автоматически:"
            )
            instructions.extend(f"- {pattern}" for pattern in failure_patterns)
        instructions.extend(
            [
                "Работай явными фазами: собрать контекст, выполнить действие, затем проверить результат.",
                "Перед крупными изменениями предпочитай короткий план из 2-5 шагов.",
                "Если ты уже изучил достаточно контекста, прекращай чтение и переходи к следующему конкретному редактированию или выполнению команды.",
                "После любой неудачной команды кратко объясни вероятную причину, скорректируй план и попробуй более узкое исправление.",
            ]
        )
        if self.session_insights.successful_commands:
            instructions.append("Уместные успешные shell-команды из этой сессии можно переиспользовать:")
            instructions.extend(f"- {command}" for command in self.session_insights.successful_commands)
        if self._looks_like_build_request(user_input):
            instructions.extend(
                [
                    "Это задача на сборку или генерацию проекта.",
                    "Перед редактированием заземлись в PROJECT_ROOT: сначала изучи файлы и релевантный код.",
                    "Не читай и не изменяй файлы вне PROJECT_ROOT.",
                    "Используй make_dir, write_file, read_file, list_files, delete_file и bash по мере необходимости, чтобы создать рабочий проект внутри PROJECT_ROOT.",
                    "При создании проекта ты обязан создать README.md в PROJECT_ROOT.",
                    "При создании проекта ты обязан создать requirements.txt в PROJECT_ROOT, даже если он минимальный.",
                    "README.md должен объяснять, что делает проект, как его запускать и каков пошаговый алгоритм запуска.",
                    "Не останавливайся на текстовом ответе, если ожидаются реальные файлы или папки.",
                    "Во время работы добавляй короткие маркеры статуса в фигурных скобках, например {НАСТРОЙКА ПРОЕКТА} или {ГЕНЕРАЦИЯ ОТЧЁТА}. Они должны быть короткими, в верхнем регистре и отражать текущую фазу.",
                    "Если ты создаёшь файлы зависимостей, такие как requirements.txt, pyproject.toml, package.json, Pipfile и подобные, устанавливай зависимости внутри PROJECT_ROOT, если PERMISSION_MODE разрешает shell-команды.",
                    "После создания файлов запускай проверку работоспособности, и если она падает, исправляй проект и запускай проверку снова, пока она не пройдёт или пока ты не сможешь точно описать блокер.",
                    "Все содержательные ответы пользователю давай на русском языке.",
                    "Если нужно создать, изменить или удалить файл внутри Generated_Project, не описывай это текстом: обязательно вызывай соответствующую функцию.",
                    "Не печатай в обычном ответе строки вроде write_file, make_dir, read_file, list_files, delete_file или bash. Такие действия нужно выполнять только через function_call.",
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
            "создай проект",
            "сгенерируй проект",
            "собери проект",
            "создай файлы",
            "создай папки",
            "реализуй",
            "сделай проект",
            "агентный ии",
            "агентный ai",
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
                "Требуется этап установки зависимостей. Проверь PROJECT_ROOT на наличие файлов зависимостей, "
                "таких как requirements.txt, pyproject.toml, Pipfile, package.json, yarn.lock, pnpm-lock.yaml "
                "или poetry.lock. Если они есть и PERMISSION_MODE разрешает выполнение shell-команд, установи зависимости "
                "внутри PROJECT_ROOT, кратко сообщи о сбоях установки, исправь их при необходимости и продолжай работу. "
                "Во время работы добавляй короткий маркер прогресса в фигурных скобках."
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
                "Генерация проекта ещё не завершена. "
                f"Сейчас создай недостающие обязательные файлы внутри PROJECT_ROOT: {joined}. "
                "README.md должен объяснять проект и способ запуска. requirements.txt должен существовать, даже если он минимальный."
            ),
        }

    def _build_verification_message(self) -> dict[str, str]:
        validation_hints = ", ".join(self._suggest_validation_commands()) or "подходящая для проекта проверка"
        return {
            "role": "user",
            "content": (
                "Требуется этап проверки. Проверь работоспособность сгенерированного проекта прямо сейчас. "
                f"Запусти подходящую проверку внутри PROJECT_ROOT, предпочтительно через bash. Подходящие команды: {validation_hints}. "
                "Считай проект проверенным только если команда завершилась успешно. Во время работы добавляй короткий маркер прогресса в фигурных скобках."
            ),
        }

    def _build_repair_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Последняя проверка работоспособности завершилась ошибкой. Исправь проект с опорой на эту ошибку проверки, затем снова запусти проверку внутри PROJECT_ROOT. "
                f"Последняя ошибка проверки: {self.last_validation_error} "
                "Во время работы добавляй короткий маркер прогресса в фигурных скобках."
            ),
        }

    def _emit_progress_updates(self, content: str) -> None:
        for marker in self._extract_progress_markers(content):
            self.progress_step += 1
            print(self._format_progress_marker(marker, self.progress_step), flush=True)

    def _emit_assistant_output(self, content: str, *, is_build_request: bool) -> None:
        text = content.strip()
        if not is_build_request or not text:
            return
        self.streamed_model_output_in_run = True
        print(f"Model> {text}", flush=True)

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
        if self.config.uses_gigachat:
            return False
        if self.tool_calls_executed > 0:
            return False
        if self.action_nudge_sent:
            return False
        return self.assistant_turns >= 1

    def _track_assistant_repetition(self, content: str) -> None:
        fingerprint = self._fingerprint_text(content)
        if not fingerprint:
            self.repeated_assistant_turns = 0
            self.last_assistant_fingerprint = ""
            return
        if fingerprint == self.last_assistant_fingerprint:
            self.repeated_assistant_turns += 1
        else:
            self.repeated_assistant_turns = 0
            self.last_assistant_fingerprint = fingerprint

    @staticmethod
    def _fingerprint_text(content: str) -> str:
        compact = re.sub(r"\s+", " ", content).strip().lower()
        if not compact:
            return ""
        return compact[:800]

    def _should_abort_on_repetition(self) -> bool:
        return self.repeated_assistant_turns >= 1 and self.tool_calls_executed == 0

    @staticmethod
    def _looks_like_textual_tool_imitation(content: str) -> bool:
        if not content.strip():
            return False
        patterns = [
            r"(?mi)^\s*(make_dir|write_file|read_file|list_files|delete_file|bash|spawn_subagent)\b",
            r"(?mi)```(?:\w+)?\s*(?:.*\n)?\s*(make_dir|write_file|read_file|list_files|delete_file|bash|spawn_subagent)\b",
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    @classmethod
    def _extract_textual_tool_call(cls, content: str) -> dict[str, Any] | None:
        match = re.search(
            r"(?ms)(?:^|\n)\s*(?P<name>make_dir|write_file|read_file|list_files|delete_file|bash|spawn_subagent)\b(?P<body>.*?)(?=\n\s*(?:make_dir|write_file|read_file|list_files|delete_file|bash|spawn_subagent)\b|\Z)",
            content,
        )
        if not match:
            return None
        name = match.group("name")
        body = match.group("body").strip()
        arguments = cls._parse_textual_tool_arguments(name, body)
        if arguments is None:
            return None
        return {
            "id": f"textual-{name}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def _parse_textual_tool_arguments(cls, name: str, body: str) -> dict[str, Any] | None:
        if name == "list_files":
            path = cls._strip_wrapping_punctuation(body)
            return {} if not path or path == "{}" else {"path": path}
        if name in {"make_dir", "read_file", "delete_file"}:
            path = cls._strip_wrapping_punctuation(body)
            return {"path": path} if path else None
        if name == "bash":
            command = cls._strip_wrapping_punctuation(body)
            return {"command": command} if command else None
        if name == "spawn_subagent":
            task = cls._strip_wrapping_punctuation(body)
            return {"task": task} if task else None
        if name == "write_file":
            return cls._parse_textual_write_file_arguments(body)
        return None

    @staticmethod
    def _strip_wrapping_punctuation(value: str) -> str:
        stripped = value.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = re.sub(r"^```[\w-]*\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip().strip("`").strip()

    @classmethod
    def _parse_textual_write_file_arguments(cls, body: str) -> dict[str, str] | None:
        stripped = cls._strip_wrapping_punctuation(body)
        if not stripped:
            return None
        if stripped.startswith("{"):
            return cls._parse_object_style_write_file_arguments(stripped)
        triple_quote_match = re.match(
            r'(?s)^(?P<path>\S+)\s+"""\s*(?P<content>.*?)\s*"""$',
            stripped,
        )
        if triple_quote_match:
            return {
                "path": triple_quote_match.group("path").strip(),
                "content": triple_quote_match.group("content"),
            }
        line_match = re.match(r'(?s)^(?P<path>\S+)\s+(?P<content>.+)$', stripped)
        if line_match:
            return {
                "path": line_match.group("path").strip(),
                "content": line_match.group("content").strip().strip('"'),
            }
        return None

    @classmethod
    def _parse_object_style_write_file_arguments(cls, body: str) -> dict[str, str] | None:
        normalized = body.replace("<|superquote|>", '"')
        lines = [line.rstrip() for line in normalized.splitlines()]
        content_line_index = next((index for index, line in enumerate(lines) if '"content"' in line), None)
        path_line_index = next((index for index, line in enumerate(lines) if '"path"' in line), None)
        if content_line_index is None or path_line_index is None or path_line_index <= content_line_index:
            return None
        path_match = re.search(r'"path"\s*:\s*"(?P<path>[^"]+)"', lines[path_line_index])
        if not path_match:
            return None
        first_content_line = lines[content_line_index]
        if ":" not in first_content_line:
            return None
        _, initial_content = first_content_line.split(":", 1)
        content_lines = [initial_content.strip()]
        content_lines.extend(line for line in lines[content_line_index + 1:path_line_index] if line.strip())
        content = "\n".join(content_lines).strip().rstrip(",").strip()
        if content.startswith('"') and len(content) > 1 and content[1] in "{[":
            content = content[1:]
        return {
            "path": path_match.group("path").strip(),
            "content": content,
        }

    @staticmethod
    def _build_pseudo_tool_correction_message() -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Ты описал вызовы инструментов обычным текстом, но не вызвал функцию структурированно. "
                "Не печатай `write_file`, `make_dir`, `list_files`, `read_file`, `bash` или другие инструменты как текст ответа. "
                "Если нужно создать файл, папку или выполнить команду, используй именно function_call. "
                "Ответь следующим сообщением либо реальным вызовом функции, либо коротким объяснением блокера без псевдокоманд."
            ),
        }

    @staticmethod
    def _build_pseudo_tool_abort_message(assistant_content: str) -> str:
        text = assistant_content.strip()
        prefix = (
            "Генерация остановлена: модель дважды подряд имитировала вызовы инструментов текстом вместо реального function_call. "
            "Из-за этого проект не был создан на диске."
        )
        if not text:
            return prefix
        return f"{prefix}\n\nПоследний некорректный ответ:\n{text}"

    def _build_repetition_abort_message(self, assistant_content: str) -> str:
        text = assistant_content.strip()
        prefix = (
            "Генерация остановлена: модель начала повторять один и тот же ответ без вызова инструментов. "
            "Harness принудительно прервал цикл, чтобы не гонять одинаковые запросы бесконечно."
        )
        if not text:
            return prefix
        return f"{prefix}\n\nПоследний повторяющийся ответ:\n{text}"

    def _build_action_nudge_message(self) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Этот запрос требует выполнения, а не только анализа. Составь короткий план, затем используй инструменты, чтобы изучить или изменить рабочую область. "
                "Пока не завершай ответ только текстом."
            ),
        }

    def _maybe_build_phase_follow_up(self, tool_result_message: dict[str, Any]) -> dict[str, str] | None:
        name = str(tool_result_message.get("name", ""))
        if name in {"read_file", "list_files"} and self.context_rounds_without_action >= 2 and not self.wrote_files_in_run:
            self.context_rounds_without_action = 0
            return {
                "role": "user",
                "content": (
                    "Ты уже собрал начальный контекст. Теперь переходи к следующему конкретному действию: редактируй файлы, запускай нужную команду или в одном предложении объясни, чего не хватает."
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
                        "Последняя команда завершилась ошибкой. Кратко объясни сбой, скорректируй план и попробуй минимальное исправление, которое устраняет проблему. "
                        f"Ошибка: {error[:600]}"
                ),
            }
        return None

    def _build_task_plan(self, *, user_input: str, history_digest: str) -> TaskPlan:
        if self.depth > 0:
            return fallback_task_plan(user_input)
        messages = build_planning_messages(
            user_input=user_input,
            session_context=self.session_insights.render_for_prompt(),
            history_digest=history_digest,
        )
        response = self._call_planner_model(messages)
        plan = parse_task_plan(self._extract_assistant_message(response).get("content", ""))
        if plan is None:
            return fallback_task_plan(user_input)
        if not plan.subtasks:
            return fallback_task_plan(user_input)
        return plan

    def _execute_task_plan(self, plan: TaskPlan, *, user_input: str) -> str:
        outputs: list[str] = []
        wrote_files = False
        total = len(plan.subtasks)
        for index, subtask in enumerate(plan.subtasks, start=1):
            result = self._run_subtask(
                subtask,
                user_input=user_input,
                index=index,
                total=total,
            )
            outputs.append(f"[{subtask.domain}] {result.output}".strip())
            wrote_files = wrote_files or result.wrote_files
        self.wrote_files_in_run = wrote_files
        if self.wrote_files_in_run:
            self._remember_successful_run()
        return "\n\n".join(output for output in outputs if output.strip())

    def _run_subtask(self, subtask: PlannedSubtask, *, user_input: str, index: int, total: int) -> SubagentOutcome:
        prompt = build_subtask_execution_prompt(subtask, user_input=user_input, total=total, index=index)
        session_id = f"{self.session_id}-{subtask.domain}-{index}"
        child = AgentHarness(config=self.config, session_id=session_id, depth=self.depth + 1, tools=self.tools)
        output = child.run(prompt)
        return SubagentOutcome(output=output, wrote_files=child.wrote_files_in_run)

    def _call_planner_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = self._build_planner_payload(messages)
        return self._call_model_with_payload(payload)

    def _call_model_with_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        auth_header = self._build_authorization_header()
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": auth_header,
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
                    context=self._request_context(),
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                provider = "GigaChat" if self.config.uses_gigachat else "OpenRouter"
                raise RuntimeError(f"{provider} request failed: {exc.code} {body}") from exc
            except (TimeoutError, socket.timeout) as exc:
                last_error = exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", "")
                if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
                    last_error = exc
                else:
                    provider = "GigaChat" if self.config.uses_gigachat else "OpenRouter"
                    raise RuntimeError(f"{provider} is unreachable from this environment.") from exc

            if attempt < self.config.request_retry_attempts:
                time.sleep(self.config.request_retry_backoff_seconds)

        provider = "GigaChat" if self.config.uses_gigachat else "OpenRouter"
        raise RuntimeError(
            f"{provider} request timed out after multiple attempts. "
            "Повтори запрос, уменьши размер промпта или увеличь timeout/retry в settings.json."
        ) from last_error

    def _build_planner_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.config.uses_gigachat:
            return {
                "model": self.config.model,
                "messages": self._normalize_gigachat_messages(messages),
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
        }

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
            return "Созданы или обновлены файлы проекта, после чего проверка завершилась успешно."
        return "Задача выполнена, итоговая проверка завершилась успешно."

    def _handle_rewind(self, command: str) -> str:
        parts = command.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError("Использование: /rewind <message_index>")
        retained = self.memory.rewind(int(parts[1]))
        return f"Удалены первые {parts[1]} сообщений. В сессии осталось {retained} сообщений."

    def _handle_clear(self) -> str:
        removed = self.memory.count()
        self.memory.clear()
        return f"История сессии очищена. Удалено {removed} сообщений."

    def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = self._build_model_payload(messages)
        return self._call_model_with_payload(payload)

    def _build_model_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.config.uses_gigachat:
            return {
                "model": self.config.model,
                "messages": self._normalize_gigachat_messages(messages),
                "functions": self._gigachat_functions_payload(),
                "function_call": "auto",
                "temperature": 0.87,
                "max_tokens": 1024,
            }
        return {
            "model": self.config.model,
            "messages": messages,
            "tools": [tool.spec() for tool in self.tools.values()],
            "tool_choice": "auto",
        }

    def _gigachat_functions_payload(self) -> list[dict[str, Any]]:
        functions_payload: list[dict[str, Any]] = []
        for tool in self.tools.values():
            spec = tool.spec()["function"]
            functions_payload.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                }
            )
        return functions_payload

    def _normalize_gigachat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        previous_assistant_had_function_call = False
        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            if role == "tool":
                if not previous_assistant_had_function_call:
                    continue
                normalized.append(
                    {
                        "role": "function",
                        "name": str(message.get("name", "") or ""),
                        "content": self._gigachat_function_result_content(message),
                    }
                )
                previous_assistant_had_function_call = False
                continue
            if role not in {"system", "user", "assistant"}:
                continue
            if role == "assistant":
                function_call = self._gigachat_function_call_from_message(message)
                next_role = str(messages[index + 1].get("role", "")) if index + 1 < len(messages) else ""
                if function_call and next_role != "tool":
                    previous_assistant_had_function_call = False
                    continue
            else:
                previous_assistant_had_function_call = False
            payload: dict[str, Any] = {
                "role": role,
                "content": str(message.get("content", "") or ""),
            }
            if role == "assistant":
                function_call = self._gigachat_function_call_from_message(message)
                if function_call:
                    payload["function_call"] = function_call
                    previous_assistant_had_function_call = True
                else:
                    previous_assistant_had_function_call = False
                functions_state_id = message.get("functions_state_id") or message.get("function_state_id")
                if functions_state_id:
                    payload["functions_state_id"] = functions_state_id
            normalized.append(payload)
        return normalized

    @staticmethod
    def _gigachat_function_call_from_message(message: dict[str, Any]) -> dict[str, Any] | None:
        if message.get("function_call"):
            function_call = message["function_call"]
            if isinstance(function_call, dict):
                return function_call
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return None
        function = tool_calls[0].get("function", {})
        name = function.get("name")
        if not name:
            return None
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "name": name,
            "arguments": arguments,
        }

    @staticmethod
    def _gigachat_function_result_content(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            try:
                json.loads(content)
            except json.JSONDecodeError:
                return json.dumps({"result": content}, ensure_ascii=False)
            return content
        return json.dumps(content, ensure_ascii=False)

    def _build_authorization_header(self) -> str:
        if self.config.uses_gigachat:
            return f"Bearer {self._get_gigachat_access_token()}"
        return f"Bearer {self.config.model_token}"

    def _get_gigachat_access_token(self) -> str:
        if self.gigachat_access_token:
            return self.gigachat_access_token

        request = urllib.request.Request(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            data=urllib.parse.urlencode({"scope": "GIGACHAT_API_PERS"}).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {self.config.authorization_key}",
                "Content-Type": "application/x-www-form-urlencoded",
                "RqUID": str(uuid.uuid4()),
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.config.request_timeout_seconds,
            context=self._request_context(),
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        access_token = payload.get("access_token", "")
        if not access_token:
            raise RuntimeError("GigaChat OAuth response did not include access_token.")
        self.gigachat_access_token = access_token
        return access_token

    def _request_context(self) -> ssl.SSLContext | None:
        if self.config.uses_gigachat:
            return ssl._create_unverified_context()
        return None

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
        if self.config.uses_gigachat:
            function_call = message.get("function_call")
            if function_call:
                normalized["function_call"] = function_call
                normalized["tool_calls"] = [self._gigachat_function_call_to_tool_call(function_call)]
            functions_state_id = message.get("functions_state_id") or message.get("function_state_id")
            if functions_state_id:
                normalized["functions_state_id"] = functions_state_id
        elif message.get("tool_calls"):
            normalized["tool_calls"] = message["tool_calls"]
        return normalized

    @staticmethod
    def _gigachat_function_call_to_tool_call(function_call: dict[str, Any]) -> dict[str, Any]:
        name = str(function_call.get("name", "") or "")
        arguments = function_call.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_call_id = f"gigachat-{name or 'function'}"
        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        if name not in self.tools:
            result = f"Unknown tool requested: {name}"
        else:
            try:
                raw_arguments = function.get("arguments", {})
                if isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments)
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    raise TypeError(f"Unsupported tool argument payload type: {type(raw_arguments).__name__}")
            except (json.JSONDecodeError, TypeError) as exc:
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
