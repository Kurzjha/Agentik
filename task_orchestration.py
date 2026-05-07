from __future__ import annotations

import json
import re
from dataclasses import dataclass


DOMAIN_PRIORITY: tuple[str, ...] = (
    "frontend",
    "backend",
    "tests",
    "docs",
    "infra",
    "cli",
    "general",
)

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "frontend": (
        "frontend",
        "ui",
        "menu",
        "interface",
        "page",
        "layout",
        "react",
        "vue",
        "html",
        "css",
        "tailwind",
        "tsx",
        "jsx",
    ),
    "backend": (
        "backend",
        "api",
        "server",
        "endpoint",
        "database",
        "sql",
        "flask",
        "fastapi",
        "django",
        "postgres",
        "service",
    ),
    "tests": (
        "test",
        "tests",
        "pytest",
        "unittest",
        "spec",
        "qa",
        "verification",
    ),
    "docs": (
        "readme",
        "docs",
        "documentation",
        "guide",
        "instruction",
    ),
    "infra": (
        "deploy",
        "docker",
        "ci",
        "cd",
        "infra",
        "deployment",
        "devops",
    ),
    "cli": (
        "cli",
        "terminal",
        "command line",
        "argparse",
        "console",
        "script",
    ),
}


@dataclass(frozen=True, slots=True)
class PlannedSubtask:
    domain: str
    title: str
    task: str
    files: tuple[str, ...] = ()
    profile_hint: str = ""


@dataclass(frozen=True, slots=True)
class TaskPlan:
    subtasks: tuple[PlannedSubtask, ...]


@dataclass(frozen=True, slots=True)
class SubagentOutcome:
    output: str
    wrote_files: bool


def build_planning_messages(*, user_input: str, session_context: str = "", history_digest: str = "") -> list[dict[str, str]]:
    sections = [
        "Ты планировщик задач для агентной системы разработки.",
        "Разбей запрос на 2-5 подзадач по зонам ответственности: frontend, backend, tests, docs, infra, cli или general.",
        "Верни ТОЛЬКО валидный JSON без markdown, пояснений и code fences.",
        "Формат ответа:",
        '{"subtasks":[{"domain":"frontend","title":"...","task":"...","files":["..."],"profileHint":"frontend"}]}',
        "Правила:",
        "Каждая подзадача должна быть конкретной и выполнимой одним подагентом.",
        "Для каждой coding-подзадачи перечисли реальные файлы, которые нужно создать или изменить.",
        "Если задача про генерацию проекта, обязательно добавь подзадачу, которая создаёт базовые файлы вроде README.md и requirements.txt, если они нужны.",
        "В task явно укажи, что исполнитель должен создавать реальные файлы и папки через инструменты, а не ограничиваться текстом.",
        "Старайся, чтобы frontend/backend/tests/docs шли отдельными подзадачами, если это применимо.",
    ]
    if session_context:
        sections.append(f"Рабочий контекст сессии:\n{session_context}")
    if history_digest:
        sections.append(f"Краткая история сессии:\n{history_digest}")
    sections.append(f"Запрос пользователя:\n{user_input}")
    return [
        {
            "role": "system",
            "content": "\n\n".join(sections),
        },
    ]


def parse_task_plan(content: str) -> TaskPlan | None:
    payload = _parse_json_blob(content)
    if payload is None:
        return None
    raw_subtasks = payload.get("subtasks")
    if not isinstance(raw_subtasks, list):
        return None
    subtasks: list[PlannedSubtask] = []
    for item in raw_subtasks:
        if not isinstance(item, dict):
            continue
        domain = _normalize_domain(str(item.get("domain", "") or item.get("profileHint", "") or "general"))
        title = str(item.get("title", "")).strip() or f"{domain.title()} work"
        task = str(item.get("task", "")).strip()
        if not task:
            continue
        files = tuple(
            str(path).strip()
            for path in item.get("files", [])
            if str(path).strip()
        )
        profile_hint = str(item.get("profileHint", "")).strip() or domain
        subtasks.append(
            PlannedSubtask(
                domain=domain,
                title=title,
                task=task,
                files=files,
                profile_hint=profile_hint,
            )
        )
    if not subtasks:
        return None
    return TaskPlan(subtasks=tuple(subtasks))


def fallback_task_plan(user_input: str) -> TaskPlan:
    lowered = user_input.lower()
    matches = [domain for domain in DOMAIN_PRIORITY if domain != "general" and any(keyword in lowered for keyword in DOMAIN_KEYWORDS[domain])]
    if not matches:
        matches = ["general"]
    subtasks = [_build_subtask_for_domain(domain, user_input) for domain in matches]
    if "docs" not in matches:
        subtasks.append(
            PlannedSubtask(
                domain="docs",
                title="Create project documentation",
                task=(
                    f"Создай README.md и, если у проекта есть зависимости, requirements.txt для задачи: {user_input}. "
                    "Обязательно создай реальные файлы и каталоги через make_dir/write_file."
                ),
                files=("README.md", "requirements.txt"),
                profile_hint="docs",
            )
        )
    return TaskPlan(subtasks=tuple(subtasks))


def build_subtask_execution_prompt(subtask: PlannedSubtask, *, user_input: str, total: int, index: int) -> str:
    files = ", ".join(subtask.files) if subtask.files else "определи необходимые файлы сам, но создай их явно"
    return (
        f"Подзадача {index}/{total}\n"
        f"Общая задача: {user_input}\n"
        f"Зона ответственности: {subtask.domain}\n"
        f"Профиль-подсказка: {subtask.profile_hint or subtask.domain}\n"
        f"Название: {subtask.title}\n"
        f"Описание: {subtask.task}\n"
        f"Файлы для создания или изменения: {files}\n\n"
        "Обязательно создавай реальные файлы и папки через tools, а не только описывай их текстом.\n"
        "Если нужны новые каталоги, сначала создай их. Если нужен код, сначала создай структуру файлов, затем содержимое.\n"
        "Возвращай краткую сводку после выполнения."
    )


def _build_subtask_for_domain(domain: str, user_input: str) -> PlannedSubtask:
    if domain == "frontend":
        return PlannedSubtask(
            domain=domain,
            title="Build the frontend",
            task=(
                f"Сделай фронтенд для задачи: {user_input}. "
                "Выбери подходящий фронтенд-профиль и создай реальные UI-файлы, а не только описание."
            ),
            files=("index.html", "styles.css", "src"),
            profile_hint="frontend",
        )
    if domain == "backend":
        return PlannedSubtask(
            domain=domain,
            title="Build the backend",
            task=(
                f"Сделай backend для задачи: {user_input}. "
                "Создай серверный код, маршруты и любые необходимые файлы проекта."
            ),
            files=("app.py", "requirements.txt"),
            profile_hint="backend",
        )
    if domain == "tests":
        return PlannedSubtask(
            domain=domain,
            title="Add tests",
            task=(
                f"Добавь тесты для задачи: {user_input}. "
                "Создай тестовые файлы и проверь ключевые сценарии."
            ),
            files=("tests",),
            profile_hint="tests",
        )
    if domain == "docs":
        return PlannedSubtask(
            domain=domain,
            title="Write documentation",
            task=(
                f"Добавь документацию для задачи: {user_input}. "
                "Создай README.md и опиши запуск проекта, структуру и шаги выполнения."
            ),
            files=("README.md",),
            profile_hint="docs",
        )
    if domain == "infra":
        return PlannedSubtask(
            domain=domain,
            title="Prepare infrastructure",
            task=(
                f"Подготовь инфраструктуру для задачи: {user_input}. "
                "Создай Dockerfile, CI или другие нужные файлы, если они уместны."
            ),
            files=("Dockerfile",),
            profile_hint="infra",
        )
    if domain == "cli":
        return PlannedSubtask(
            domain=domain,
            title="Build the CLI",
            task=(
                f"Сделай CLI-часть для задачи: {user_input}. "
                "Создай entrypoint, разбор аргументов и необходимые файлы."
            ),
            files=("main.py",),
            profile_hint="cli",
        )
    return PlannedSubtask(
        domain="general",
        title="Implement the request",
        task=(
            f"Выполни задачу: {user_input}. "
            "Создай реальные файлы и каталоги через tools, а не ограничивайся текстовым ответом."
        ),
        files=(),
        profile_hint="general",
    )


def _parse_json_blob(content: str) -> dict[str, object] | None:
    stripped = content.strip()
    if not stripped:
        return None
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _normalize_domain(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in DOMAIN_PRIORITY:
        return lowered
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return domain
    return "general"
