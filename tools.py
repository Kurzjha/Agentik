from __future__ import annotations

import json
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory import SessionMemory


class ToolExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolContext:
    project_root: Path
    permission_mode: str
    memory: SessionMemory
    spawn_handler: Any
    subagent_depth: int = 0
    max_subagent_depth: int = 2

    def resolve_path(self, candidate: str) -> Path:
        path = (self.project_root / candidate).resolve()
        project_root_resolved = self.project_root.resolve()
        if path != project_root_resolved and project_root_resolved not in path.parents:
            raise ToolExecutionError(f"Refusing to access path outside project: {path}")
        return path


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        raise NotImplementedError


class BashTool(BaseTool):
    name = "bash"
    description = "Выполнить shell-команду внутри корня проекта."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        command = arguments["command"]
        cwd = context.project_root
        if arguments.get("cwd"):
            cwd = context.resolve_path(arguments["cwd"])

        mode = context.permission_mode
        if mode == "plan":
            raise ToolExecutionError(
                f"Permission mode plan blocked shell execution. Planned command: {command}"
            )

        if mode == "auto" and self._looks_dangerous(command):
            raise ToolExecutionError(
                f"Permission mode auto rejected potentially destructive command: {command}"
            )

        completed = subprocess.run(
            command if isinstance(command, list) else shlex.split(command, posix=False),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            shell=False,
        )
        payload = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "cwd": str(cwd),
        }
        return json.dumps(payload, ensure_ascii=True)

    @staticmethod
    def _looks_dangerous(command: str) -> bool:
        lowered = command.lower()
        blocked_tokens = [
            "rm ",
            "rmdir ",
            "del ",
            "format ",
            "shutdown ",
            "restart-computer",
            "remove-item ",
        ]
        return any(token in lowered for token in blocked_tokens)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Прочитать текстовый файл из проекта."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        path = context.resolve_path(arguments["path"])
        return path.read_text(encoding="utf-8")


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "Показать файлы и папки внутри доступного для записи корня проекта."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        relative_path = arguments.get("path", ".")
        path = context.resolve_path(relative_path)
        if not path.exists():
            raise ToolExecutionError(f"Path does not exist: {path}")
        if path.is_file():
            return json.dumps(
                {
                    "path": str(path),
                    "type": "file",
                    "entries": [],
                },
                ensure_ascii=True,
            )
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            entries.append(
                {
                    "name": child.name,
                    "type": "file" if child.is_file() else "directory",
                }
            )
        return json.dumps(
            {
                "path": str(path),
                "type": "directory",
                "entries": entries,
            },
            ensure_ascii=True,
        )


class MakeDirectoryTool(BaseTool):
    name = "make_dir"
    description = "Создать директорию или дерево директорий внутри доступного для записи корня проекта."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        path = context.resolve_path(arguments["path"])
        path.mkdir(parents=True, exist_ok=True)
        return f"Created directory {path}"


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Создать новый или полностью перезаписать UTF-8 текстовый файл внутри проекта."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        path = context.resolve_path(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return f"Wrote {len(arguments['content'])} characters to {path}"


class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Удалить файл внутри проекта. Используй только для файлов, а не для папок."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        path = context.resolve_path(arguments["path"])
        if not path.exists():
            raise ToolExecutionError(f"File does not exist: {path}")
        if not path.is_file():
            raise ToolExecutionError(f"Refusing to delete non-file path: {path}")
        path.unlink()
        return f"Deleted file {path}"


class SpawnSubagentTool(BaseTool):
    name = "spawn_subagent"
    description = "Запустить подагента для узкой задачи и вернуть результат."
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "session_id": {"type": "string"},
        },
        "required": ["task"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any], context: ToolContext) -> str:
        if context.subagent_depth >= context.max_subagent_depth:
            raise ToolExecutionError("Maximum sub-agent depth reached.")
        session_id = arguments.get("session_id") or f"subagent-depth-{context.subagent_depth + 1}"
        return context.spawn_handler(
            task=arguments["task"],
            session_id=session_id,
            depth=context.subagent_depth + 1,
        )


def build_default_tools() -> dict[str, BaseTool]:
    tools: list[BaseTool] = [
        BashTool(),
        DeleteFileTool(),
        ListFilesTool(),
        MakeDirectoryTool(),
        ReadFileTool(),
        WriteFileTool(),
        SpawnSubagentTool(),
    ]
    return {tool.name: tool for tool in tools}
