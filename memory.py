from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class SessionInsights:
    recent_failure_patterns: tuple[str, ...]
    successful_commands: tuple[str, ...]
    touched_paths: tuple[str, ...]
    tool_usage: tuple[str, ...]

    def render_for_prompt(self) -> str:
        lines: list[str] = []
        if self.recent_failure_patterns:
            lines.append("Недавние шаблоны ошибок, которые не стоит повторять:")
            lines.extend(f"- {pattern}" for pattern in self.recent_failure_patterns)
        if self.successful_commands:
            lines.append("Недавние успешные shell-команды, которые можно переиспользовать по ситуации:")
            lines.extend(f"- {command}" for command in self.successful_commands)
        if self.touched_paths:
            lines.append("Недавно изменённые пути проекта в этой сессии:")
            lines.extend(f"- {path}" for path in self.touched_paths)
        if self.tool_usage:
            lines.append("Самые используемые инструменты в этой сессии:")
            lines.extend(f"- {tool}" for tool in self.tool_usage)
        return "\n".join(lines)


@dataclass(slots=True)
class SessionMemory:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def load_messages(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        records = self._read_records()
        messages = [record["message"] for record in records if "message" in record]
        if limit is None or limit <= 0 or len(messages) <= limit:
            return messages
        return messages[-limit:]

    def append_message(self, message: dict[str, Any]) -> None:
        record = {
            "timestamp": _utc_now(),
            "message": message,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def rewind(self, message_index: int) -> int:
        records = self._read_records()
        if message_index < 0 or message_index > len(records):
            raise IndexError(
                f"Rewind index {message_index} is out of range for {len(records)} messages."
            )
        retained = records[message_index:]
        with self.path.open("w", encoding="utf-8") as handle:
            for record in retained:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return len(retained)

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def count(self) -> int:
        return len(self._read_records())

    def recent_failure_patterns(self, *, limit: int = 3) -> list[str]:
        patterns: Counter[str] = Counter()
        for record in self._read_records():
            message = record.get("message", {})
            if message.get("role") != "tool":
                continue

            content = str(message.get("content", ""))
            if "ToolExecutionError:" in content:
                patterns[self._normalize_failure(content)] += 1
                continue

            if message.get("name") != "bash":
                continue

            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue

            if int(payload.get("returncode", 0)) == 0:
                continue

            stderr = str(payload.get("stderr", "")).strip()
            stdout = str(payload.get("stdout", "")).strip()
            failure = stderr or stdout or f"Command failed: {payload.get('command', '')}"
            patterns[self._normalize_failure(failure)] += 1

        return [pattern for pattern, _ in patterns.most_common(limit)]

    def build_insights(
        self,
        *,
        failure_limit: int = 3,
        success_limit: int = 5,
        path_limit: int = 8,
        tool_limit: int = 5,
    ) -> SessionInsights:
        successful_commands: list[str] = []
        touched_paths: list[str] = []
        tool_counter: Counter[str] = Counter()

        for record in self._read_records():
            message = record.get("message", {})
            role = message.get("role")
            if role != "tool":
                continue

            tool_name = str(message.get("name", "")).strip()
            if tool_name:
                tool_counter[tool_name] += 1

            content = str(message.get("content", ""))
            if tool_name == "bash":
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if int(payload.get("returncode", 1)) == 0:
                    command = str(payload.get("command", "")).strip()
                    if command and command not in successful_commands:
                        successful_commands.append(command)
                continue

            touched_path = self._extract_touched_path(tool_name, content)
            if touched_path and touched_path not in touched_paths:
                touched_paths.append(touched_path)

        return SessionInsights(
            recent_failure_patterns=tuple(self.recent_failure_patterns(limit=failure_limit)),
            successful_commands=tuple(successful_commands[-success_limit:]),
            touched_paths=tuple(touched_paths[-path_limit:]),
            tool_usage=tuple(tool for tool, _ in tool_counter.most_common(tool_limit)),
        )

    def build_history_digest(
        self,
        *,
        recent_message_limit: int = 8,
        item_limit: int = 6,
    ) -> str:
        records = self._read_records()
        if not records:
            return ""

        messages = [record["message"] for record in records if "message" in record]
        if not messages:
            return ""

        older_messages = messages[:-recent_message_limit] if len(messages) > recent_message_limit else []
        recent_messages = messages[-recent_message_limit:]

        lines: list[str] = []
        if older_messages:
            lines.append(
                f"Старые сообщения сессии сжаты: из активного контекста исключено {len(older_messages)} сообщений."
            )

        recent_user_items = self._collect_recent_items(recent_messages, role="user", limit=item_limit)
        recent_assistant_items = self._collect_recent_items(recent_messages, role="assistant", limit=item_limit)
        recent_tool_items = self._collect_recent_tool_items(recent_messages, limit=item_limit)

        if recent_user_items:
            lines.append("Недавние запросы пользователя:")
            lines.extend(f"- {item}" for item in recent_user_items)
        if recent_assistant_items:
            lines.append("Недавние ответы ассистента:")
            lines.extend(f"- {item}" for item in recent_assistant_items)
        if recent_tool_items:
            lines.append("Недавняя активность инструментов:")
            lines.extend(f"- {item}" for item in recent_tool_items)

        return "\n".join(lines)

    def _read_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    @staticmethod
    def _normalize_failure(message: str) -> str:
        compact = " ".join(message.split())
        if len(compact) <= 200:
            return compact
        return f"{compact[:200]}...<truncated>"

    @staticmethod
    def _extract_touched_path(tool_name: str, content: str) -> str:
        if tool_name == "write_file":
            marker = " to "
            if marker in content:
                return content.split(marker, maxsplit=1)[1].strip()
        if tool_name == "list_files":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return ""
            return str(payload.get("path", "")).strip()
        if tool_name == "read_file":
            stripped = content.strip()
            if stripped:
                return stripped.splitlines()[0].strip()[:200]
        return ""

    @staticmethod
    def _shorten(text: str, *, limit: int = 180) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}...<truncated>"

    @classmethod
    def _collect_recent_items(cls, messages: list[dict[str, Any]], *, role: str, limit: int) -> list[str]:
        items: list[str] = []
        for message in reversed(messages):
            if message.get("role") != role:
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            items.append(cls._shorten(content))
            if len(items) >= limit:
                break
        items.reverse()
        return items

    @classmethod
    def _collect_recent_tool_items(cls, messages: list[dict[str, Any]], *, limit: int) -> list[str]:
        items: list[str] = []
        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            tool_name = str(message.get("name", "")).strip() or "tool"
            content = cls._shorten(str(message.get("content", "")))
            items.append(f"{tool_name}: {content}")
            if len(items) >= limit:
                break
        items.reverse()
        return items
