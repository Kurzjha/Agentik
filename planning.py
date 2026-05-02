from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


@dataclass(frozen=True, slots=True)
class CachedPlan:
    task: str
    summary: str
    validation_command: str
    touched_paths: tuple[str, ...]

    def score(self, task: str) -> int:
        task_tokens = _tokenize(task)
        if not task_tokens:
            return 0
        overlap = len(task_tokens & _tokenize(f"{self.task} {self.summary} {' '.join(self.touched_paths)}"))
        return overlap

    def render(self) -> str:
        lines = [f"- Prior task: {self.task}", f"  Summary: {self.summary}"]
        if self.validation_command:
            lines.append(f"  Validation: {self.validation_command}")
        if self.touched_paths:
            lines.append(f"  Paths: {', '.join(self.touched_paths)}")
        return "\n".join(lines)


class PlanCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def retrieve(self, task: str, *, limit: int = 2) -> list[CachedPlan]:
        ranked = sorted(self._load(), key=lambda item: item.score(task), reverse=True)
        return [item for item in ranked if item.score(task) > 0][:limit]

    def remember(
        self,
        *,
        task: str,
        summary: str,
        validation_command: str,
        touched_paths: list[str],
    ) -> None:
        if not task.strip() or not summary.strip():
            return

        plans = self._load()
        new_plan = CachedPlan(
            task=task.strip(),
            summary=summary.strip(),
            validation_command=validation_command.strip(),
            touched_paths=tuple(path for path in touched_paths if path.strip()),
        )
        plans.append(new_plan)
        deduped = self._dedupe(plans)
        serialized = [
            {
                "task": plan.task,
                "summary": plan.summary,
                "validation_command": plan.validation_command,
                "touched_paths": list(plan.touched_paths),
            }
            for plan in deduped[-20:]
        ]
        self.path.write_text(json.dumps(serialized, ensure_ascii=True, indent=2), encoding="utf-8")

    def render_for_prompt(self, task: str) -> str:
        matches = self.retrieve(task)
        if not matches:
            return ""
        lines = ["Similar successful plans from earlier sessions:"]
        lines.extend(plan.render() for plan in matches)
        return "\n".join(lines)

    def _load(self) -> list[CachedPlan]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        plans: list[CachedPlan] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            plans.append(
                CachedPlan(
                    task=str(item.get("task", "")),
                    summary=str(item.get("summary", "")),
                    validation_command=str(item.get("validation_command", "")),
                    touched_paths=tuple(str(path) for path in item.get("touched_paths", [])),
                )
            )
        return plans

    @staticmethod
    def _dedupe(plans: list[CachedPlan]) -> list[CachedPlan]:
        deduped: list[CachedPlan] = []
        seen: set[tuple[str, str]] = set()
        for plan in plans:
            key = (plan.task.lower(), plan.summary.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(plan)
        return deduped
