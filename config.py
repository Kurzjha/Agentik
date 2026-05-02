from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from context_profiles import detect_context_profile, load_profile_rules
from planning import PlanCache
from research import render_research_context

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


def _trim_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated>"


def _read_text_if_exists(path: Path, *, limit: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        return _trim_text(path.read_text(encoding="utf-8"), limit=limit)
    except OSError:
        return ""


def _trim_lines(value: str, *, max_lines: int) -> str:
    lines = value.splitlines()
    if len(lines) <= max_lines:
        return value
    trimmed = "\n".join(lines[:max_lines])
    return f"{trimmed}\n...<truncated>"


def _collect_rule_files(root: Path) -> list[Path]:
    rule_files: list[Path] = []
    for candidate in [root]:
        agents_path = candidate / "AGENTS.md"
        if agents_path.exists():
            rule_files.append(agents_path)
    return rule_files


def _collect_markdown_files(root: Path, *, limit: int, max_files: int) -> str:
    markdown_snippets: list[str] = []
    for path in sorted(root.glob("*.md"))[:max_files]:
        content = _read_text_if_exists(path, limit=limit)
        if content:
            markdown_snippets.append(f"## {path.name}\n{content}")
    return "\n\n".join(markdown_snippets)


def _safe_git_status(root: Path, *, max_lines: int) -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", str(root)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "git status unavailable"

    stdout = _trim_lines(completed.stdout.strip(), max_lines=max_lines)
    return stdout or "working tree clean or not a git repository"


def load_settings(path: Path | None = None) -> dict[str, object]:
    settings_path = path or (Path.cwd() / "settings.json")
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Settings file is missing: {settings_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Settings file is invalid JSON: {settings_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Settings file must contain a JSON object: {settings_path}")
    return payload


@dataclass(slots=True)
class HarnessConfig:
    model_token: str
    base_url: str
    model: str
    project_root: Path
    session_root: Path
    library_root: Path
    permission_mode: str
    max_tool_rounds: int
    max_subagent_depth: int
    request_timeout_seconds: int
    request_retry_attempts: int
    request_retry_backoff_seconds: int
    context_file_limit: int
    prompt_section_limit: int
    markdown_file_limit: int
    git_status_line_limit: int
    history_message_limit: int
    history_digest_recent_limit: int
    history_digest_item_limit: int
    research_paper_limit: int
    plan_cache_limit: int
    workspace_root: Path = field(default_factory=lambda: Path.cwd())

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
        workspace_root: Path | None = None,
        permission_mode: str = "plan",
    ) -> "HarnessConfig":
        load_dotenv()
        settings = load_settings((workspace_root or Path.cwd()) / "settings.json")
        repo_root = workspace_root or Path.cwd()
        output_root = project_root or (repo_root / str(settings["project_root"]))
        selected_permission_mode = permission_mode.lower() if permission_mode else str(settings["permission_mode"])
        return cls(
            model_token=os.getenv("MODEL_TOKEN", ""),
            base_url=str(settings["base_url"]),
            model=str(settings["model"]),
            workspace_root=repo_root,
            project_root=output_root,
            session_root=repo_root / str(settings["session_root"]),
            library_root=repo_root / str(settings["library_root"]),
            permission_mode=selected_permission_mode,
            max_tool_rounds=int(settings["max_tool_rounds"]),
            max_subagent_depth=int(settings["max_subagent_depth"]),
            request_timeout_seconds=int(settings["request_timeout_seconds"]),
            request_retry_attempts=int(settings["request_retry_attempts"]),
            request_retry_backoff_seconds=int(settings["request_retry_backoff_seconds"]),
            context_file_limit=int(settings["context_file_limit"]),
            prompt_section_limit=int(settings["prompt_section_limit"]),
            markdown_file_limit=int(settings["markdown_file_limit"]),
            git_status_line_limit=int(settings["git_status_line_limit"]),
            history_message_limit=int(settings["history_message_limit"]),
            history_digest_recent_limit=int(settings["history_digest_recent_limit"]),
            history_digest_item_limit=int(settings["history_digest_item_limit"]),
            research_paper_limit=int(settings["research_paper_limit"]),
            plan_cache_limit=int(settings["plan_cache_limit"]),
        )

    def ensure_ready(self) -> None:
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.session_root.mkdir(parents=True, exist_ok=True)
        if not self.model_token:
            raise RuntimeError("MODEL_TOKEN is missing in .env.")

    @property
    def plan_cache(self) -> PlanCache:
        return PlanCache(self.session_root / "plan_cache.json")

    def build_system_prompt(
        self,
        *,
        user_input: str = "",
        extra_instructions: Iterable[str] | None = None,
        session_context: str = "",
        history_digest: str = "",
    ) -> str:
        context_profile = detect_context_profile(user_input, self.project_root)
        rule_sections: list[str] = []
        for path in _collect_rule_files(self.project_root):
            content = _read_text_if_exists(path, limit=self.context_file_limit)
            if content:
                rule_sections.append(f"## Rules from {path}\n{content}")

        architecture_doc = ""
        if context_profile.include_architecture_doc:
            architecture_doc = _read_text_if_exists(
                self.project_root / "Architecture.md",
                limit=self.context_file_limit,
            )
        markdown_context = ""
        if context_profile.include_markdown_context:
            markdown_context = _collect_markdown_files(
                self.project_root,
                limit=max(1000, self.context_file_limit // 2),
                max_files=self.markdown_file_limit,
            )
        env_context = [
            f"WORKSPACE_ROOT={self.workspace_root}",
            f"PROJECT_ROOT={self.project_root}",
            f"LIBRARY_ROOT={self.library_root}",
            f"MODEL={self.model}",
            f"PERMISSION_MODE={self.permission_mode}",
        ]
        git_context = _safe_git_status(self.project_root, max_lines=self.git_status_line_limit)
        research_context = render_research_context(
            user_input,
            limit=self.research_paper_limit,
        )
        cached_plan_context = _trim_text(
            self.plan_cache.render_for_prompt(user_input),
            limit=self.prompt_section_limit,
        )
        profile_rules = _trim_text(
            load_profile_rules(self.workspace_root, context_profile),
            limit=self.context_file_limit,
        )

        sections = [
            "You are an agentic coding harness operating on a local project.",
            "Persist state externally and use tools when that materially helps.",
            "Prefer concrete actions over long explanations.",
            "You must only read, inspect, create, modify, and execute files inside PROJECT_ROOT.",
            "Do not rely on files outside PROJECT_ROOT for task execution or context gathering.",
            "Create any new files, folders, code, and generated artifacts inside PROJECT_ROOT unless the user explicitly overrides that requirement.",
            "If the user asks you to create, build, scaffold, implement, generate, or modify a project, you must use tools to create or update real files and directories rather than only describing what should be done.",
            "Before finishing a build-style request, verify the resulting files with list_files, read_file, or bash and then return a short summary of what was created.",
            f"## Selected Context Profile\n{context_profile.name} - {context_profile.description}",
            "## Environment\n" + "\n".join(env_context),
            f"## Git Status\n{git_context}",
            f"## Research Context\n{research_context}",
        ]
        if session_context:
            sections.append(
                f"## Session Working Memory\n{_trim_text(session_context, limit=self.prompt_section_limit)}"
            )
        if history_digest:
            sections.append(
                f"## Session History Digest\n{_trim_text(history_digest, limit=self.prompt_section_limit)}"
            )
        if cached_plan_context:
            sections.append(f"## Retrieved Prior Plans\n{cached_plan_context}")
        if profile_rules:
            sections.append(f"## Profile Rules\n{profile_rules}")
        if architecture_doc:
            sections.append(f"## Architecture.md\n{architecture_doc}")
        if rule_sections:
            sections.append("\n\n".join(rule_sections))
        if markdown_context:
            sections.append(f"## Local Markdown Context\n{markdown_context}")
        if extra_instructions:
            sections.append("## Runtime Instructions\n" + "\n".join(extra_instructions))
        return "\n\n".join(sections)
