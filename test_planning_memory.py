from __future__ import annotations

import json
import unittest
from pathlib import Path

from memory import SessionMemory
from planning import PlanCache


class WorkspaceTempDir:
    def __init__(self, name: str) -> None:
        self.path = Path.cwd() / ".test_tmp" / name

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for child in sorted(self.path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        self.path.rmdir()


class SessionMemoryInsightsTests(unittest.TestCase):
    def test_build_insights_collects_successes_failures_and_paths(self) -> None:
        with WorkspaceTempDir("memory_case") as tmpdir:
            path = tmpdir / "session.jsonl"
            memory = SessionMemory(path)
            memory.append_message(
                {
                    "role": "tool",
                    "name": "bash",
                    "content": json.dumps(
                        {
                            "command": "python -m pytest",
                            "returncode": 0,
                            "stdout": "ok",
                            "stderr": "",
                        }
                    ),
                }
            )
            memory.append_message(
                {
                    "role": "tool",
                    "name": "write_file",
                    "content": "Wrote 12 characters to C:\\tmp\\project\\main.py",
                }
            )
            memory.append_message(
                {
                    "role": "tool",
                    "name": "bash",
                    "content": json.dumps(
                        {
                            "command": "python main.py",
                            "returncode": 1,
                            "stdout": "",
                            "stderr": "ImportError: missing dependency",
                        }
                    ),
                }
            )

            insights = memory.build_insights()

            self.assertIn("python -m pytest", insights.successful_commands)
            self.assertIn("C:\\tmp\\project\\main.py", insights.touched_paths)
            self.assertTrue(
                any("ImportError: missing dependency" in pattern for pattern in insights.recent_failure_patterns)
            )


class PlanCacheTests(unittest.TestCase):
    def test_retrieve_prefers_similar_successful_plan(self) -> None:
        with WorkspaceTempDir("plan_case") as tmpdir:
            cache = PlanCache(tmpdir / "plan_cache.json")
            cache.remember(
                task="Build a pygame racing game",
                summary="Create modular files and validate with python main.py",
                validation_command="python main.py",
                touched_paths=["main.py", "track.py"],
            )
            cache.remember(
                task="Explain memory architecture",
                summary="Documentation-only answer",
                validation_command="",
                touched_paths=["Architecture.md"],
            )

            matches = cache.retrieve("Create a modular racing game", limit=1)

            self.assertEqual(1, len(matches))
            self.assertEqual("Build a pygame racing game", matches[0].task)


if __name__ == "__main__":
    unittest.main()
