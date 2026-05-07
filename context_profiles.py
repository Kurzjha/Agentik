from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zа-яё0-9]{3,}", text.lower()))


@dataclass(frozen=True, slots=True)
class ContextProfile:
    name: str
    keywords: tuple[str, ...]
    file_name: str
    description: str
    include_architecture_doc: bool
    include_markdown_context: bool
    structure_markers: tuple[str, ...] = ()

    def score(self, user_input: str, project_root: Path | None = None) -> int:
        tokens = _tokenize(user_input)
        if not tokens:
            base_score = 0
        else:
            keyword_hits = sum(1 for keyword in self.keywords if keyword in user_input.lower())
            token_hits = len(tokens & _tokenize(" ".join(self.keywords)))
            base_score = keyword_hits * 2 + token_hits
        if project_root is None:
            return base_score
        return base_score + self._structure_score(project_root)

    def _structure_score(self, project_root: Path) -> int:
        score = 0
        files = list(project_root.rglob("*"))
        file_names = {path.name.lower() for path in files if path.is_file()}
        dir_names = {path.name.lower() for path in files if path.is_dir()}
        path_parts = {part.lower() for path in files for part in path.parts}
        text = " ".join(path.name.lower() for path in files)

        for marker in self.structure_markers:
            marker_lower = marker.lower()
            if marker_lower in file_names or marker_lower in dir_names or marker_lower in path_parts or marker_lower in text:
                score += 3

        if self.name == "frontend-react":
            if any(name.endswith((".tsx", ".jsx")) for name in file_names) or "react" in text:
                score += 6
        elif self.name == "frontend-vue":
            if any(name.endswith(".vue") for name in file_names) or "vue" in text:
                score += 6
        elif self.name == "frontend-tailwind":
            if "tailwind" in text or any("tailwind" in name for name in file_names):
                score += 6
        elif self.name == "frontend-html-css":
            if any(name.endswith(".html") for name in file_names) or any(name.endswith(".css") for name in file_names):
                score += 5
        elif self.name == "cli":
            if any(name in {"main.py", "app.py"} for name in file_names) or "argparse" in text:
                score += 4
        elif self.name == "backend":
            if any(name.endswith(".py") for name in file_names) and any(
                marker in text for marker in ("api", "fastapi", "flask", "django", "sql", "database")
            ):
                score += 4

        return score


PROFILES: tuple[ContextProfile, ...] = (
    ContextProfile(
        name="frontend-react",
        keywords=(
            "frontend",
            "фронтенд",
            "react",
            "jsx",
            "tsx",
            "next.js",
            "nextjs",
            "vite",
            "component",
            "hooks",
            "state",
            "props",
            "ui",
        ),
        file_name="frontend-react.md",
        description="Фронтенд на React и близких технологиях с компонентным UI, хуками и состоянием приложения.",
        include_architecture_doc=False,
        include_markdown_context=False,
        structure_markers=("package.json", "src", "components", "pages"),
    ),
    ContextProfile(
        name="frontend-vue",
        keywords=(
            "frontend",
            "фронтенд",
            "vue",
            "nuxt",
            "composition api",
            "component",
            "ui",
            "browser",
        ),
        file_name="frontend-vue.md",
        description="Фронтенд на Vue с компонентным браузерным интерфейсом.",
        include_architecture_doc=False,
        include_markdown_context=False,
        structure_markers=("package.json", "src", "components", "views"),
    ),
    ContextProfile(
        name="frontend-tailwind",
        keywords=(
            "frontend",
            "фронтенд",
            "tailwind",
            "utility",
            "css",
            "responsive",
            "design system",
            "ui",
        ),
        file_name="frontend-tailwind.md",
        description="Фронтенд-задачи с упором на Tailwind и utility-first стилизацию.",
        include_architecture_doc=False,
        include_markdown_context=False,
        structure_markers=("tailwind.config.js", "tailwind.config.ts", "tailwind.config.cjs"),
    ),
    ContextProfile(
        name="frontend-html-css",
        keywords=(
            "frontend",
            "фронтенд",
            "html",
            "css",
            "responsive",
            "адаптив",
            "layout",
            "вёрстка",
            "template",
            "ui",
        ),
        file_name="frontend-html-css.md",
        description="Фронтенд на HTML и CSS без компонентного фреймворка.",
        include_architecture_doc=False,
        include_markdown_context=False,
        structure_markers=("index.html", "styles.css", "style.css", "templates"),
    ),
    ContextProfile(
        name="backend",
        keywords=(
            "backend",
            "бэкенд",
            "api",
            "server",
            "сервер",
            "endpoint",
            "эндпоинт",
            "database",
            "база данных",
            "django",
            "flask",
            "fastapi",
            "postgres",
            "postgresql",
            "auth",
            "service",
            "model",
            "migration",
            "rest",
        ),
        file_name="backend.md",
        description="Правила для API, серверной логики, данных и сервисной разработки.",
        include_architecture_doc=False,
        include_markdown_context=False,
        structure_markers=("api", "server", "app.py", "main.py"),
    ),
    ContextProfile(
        name="cli",
        keywords=(
            "cli",
            "командная строка",
            "terminal",
            "терминал",
            "command line",
            "argparse",
            "shell",
            "скрипт",
            "script",
            "console",
            "консоль",
            "interactive",
            "интерактив",
            "headless",
            "subprocess",
        ),
        file_name="cli.md",
        description="Правила для CLI-инструментов, разбора аргументов и неинтерактивной автоматизации.",
        include_architecture_doc=False,
        include_markdown_context=False,
        structure_markers=("main.py", "app.py", "argparse", "click", "typer"),
    ),
    ContextProfile(
        name="general",
        keywords=(),
        file_name="general.md",
        description="Общие инженерные правила, когда ни один профиль не подходит достаточно точно.",
        include_architecture_doc=True,
        include_markdown_context=True,
    ),
)


def detect_context_profile(user_input: str, project_root: Path | None = None) -> ContextProfile:
    ranked = sorted(
        PROFILES,
        key=lambda profile: profile.score(user_input, project_root),
        reverse=True,
    )
    best = ranked[0]
    if best.name == "general":
        return best
    if best.score(user_input, project_root) <= 0:
        return next(profile for profile in PROFILES if profile.name == "general")
    return best


def load_profile_rules(root: Path, profile: ContextProfile) -> str:
    profile_dir = root / "profiles"
    lines: list[str] = []

    shared = profile_dir / "shared.md"
    if shared.exists():
        content = shared.read_text(encoding="utf-8").strip()
        if content:
            lines.append(f"## Общие правила\n{content}")

    profile_file = profile_dir / profile.file_name
    if profile_file.exists():
        content = profile_file.read_text(encoding="utf-8").strip()
        if content:
            lines.append(f"## Правила профиля {profile.name.title()}\n{content}")

    if not lines:
        return ""
    return "\n\n".join(lines)
