from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent import AgentHarness
from config import HarnessConfig
from task_orchestration import PlannedSubtask, SubagentOutcome, TaskPlan, build_subtask_execution_prompt
from tools import build_default_tools


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


class StubHarness(AgentHarness):
    def __init__(
        self,
        *,
        config: HarnessConfig,
        responses: list[dict[str, object]],
        disable_follow_ups: bool = False,
    ) -> None:
        super().__init__(config=config, session_id="test-session", tools=build_default_tools())
        self._responses = list(responses)
        self._disable_follow_ups = disable_follow_ups

    def _call_model(self, messages: list[dict[str, object]]) -> dict[str, object]:
        if not self._responses:
            raise AssertionError("РњРѕРє-РѕС‚РІРµС‚С‹ РјРѕРґРµР»Рё Р·Р°РєРѕРЅС‡РёР»РёСЃСЊ.")
        return self._responses.pop(0)

    def _needs_required_project_files(self) -> bool:
        if self._disable_follow_ups:
            return False
        return super()._needs_required_project_files()

    def _needs_verification(self) -> bool:
        if self._disable_follow_ups:
            return False
        return super()._needs_verification()

    def _needs_dependency_install(self) -> bool:
        if self._disable_follow_ups:
            return False
        return super()._needs_dependency_install()


class PlannerStubHarness(StubHarness):
    def __init__(
        self,
        *,
        config: HarnessConfig,
        plan: TaskPlan,
    ) -> None:
        super().__init__(config=config, responses=[])
        self._plan = plan
        self.executed_subtasks: list[str] = []
        self.subtask_prompts: list[str] = []

    def _build_task_plan(self, *, user_input: str, history_digest: str) -> TaskPlan:
        return self._plan

    def _run_subtask(
        self,
        subtask: PlannedSubtask,
        *,
        user_input: str,
        index: int,
        total: int,
    ) -> SubagentOutcome:
        prompt = build_subtask_execution_prompt(subtask, user_input=user_input, total=total, index=index)
        self.executed_subtasks.append(subtask.domain)
        self.subtask_prompts.append(prompt)
        return SubagentOutcome(output=f"{subtask.domain} done", wrote_files=True)


def build_config(root: Path, *, uses_gigachat: bool) -> HarnessConfig:
    return HarnessConfig(
        model_token="dummy-token",
        authorization_key="dummy-auth",
        base_url="https://gigachat.devices.sberbank.ru/api/v1" if uses_gigachat else "https://openrouter.ai/api/v1",
        model="GigaChat" if uses_gigachat else "openai/test-model",
        project_root=root / "Generated_Project",
        session_root=root / "sessions",
        library_root=root / "library",
        permission_mode="plan",
        max_tool_rounds=6,
        max_subagent_depth=1,
        request_timeout_seconds=1,
        request_retry_attempts=1,
        request_retry_backoff_seconds=0,
        context_file_limit=1000,
        prompt_section_limit=1000,
        markdown_file_limit=2,
        git_status_line_limit=5,
        history_message_limit=5,
        history_digest_recent_limit=4,
        history_digest_item_limit=4,
        research_paper_limit=1,
        plan_cache_limit=1,
        workspace_root=root,
    )


def make_response(content: str) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


class AgentHarnessLoopProtectionTests(unittest.TestCase):
    def test_gigachat_payload_contains_functions_and_function_messages(self) -> None:
        with WorkspaceTempDir("agent_gigachat_payload") as root:
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[],
            )
            messages = [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "gigachat-write_file",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": "{\"path\":\"README.md\",\"content\":\"x\"}",
                            },
                        }
                    ],
                    "functions_state_id": "state-1",
                },
                {"role": "tool", "name": "write_file", "content": "ok"},
            ]

            payload = harness._build_model_payload(messages)

            self.assertEqual("auto", payload["function_call"])
            self.assertTrue(payload["functions"])
            self.assertIn("delete_file", {item["name"] for item in payload["functions"]})
            self.assertEqual("state-1", payload["messages"][1]["functions_state_id"])
            self.assertEqual("function", payload["messages"][2]["role"])
            self.assertEqual("write_file", payload["messages"][2]["name"])
            self.assertEqual({"result": "ok"}, json.loads(payload["messages"][2]["content"]))

    def test_gigachat_preserves_json_tool_result_content(self) -> None:
        with WorkspaceTempDir("agent_gigachat_json_tool_result") as root:
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[],
            )
            messages = [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "bash",
                        "arguments": "{\"command\":\"echo ok\"}",
                    },
                },
                {
                    "role": "tool",
                    "name": "bash",
                    "content": "{\"returncode\":0,\"stdout\":\"ok\",\"stderr\":\"\"}",
                },
            ]

            payload = harness._build_model_payload(messages)

            self.assertEqual(
                {"returncode": 0, "stdout": "ok", "stderr": ""},
                json.loads(payload["messages"][2]["content"]),
            )

    def test_gigachat_drops_orphan_tool_messages_from_trimmed_history(self) -> None:
        with WorkspaceTempDir("agent_gigachat_trimmed_history") as root:
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[],
            )
            messages = [
                {"role": "tool", "name": "write_file", "content": "ok"},
                {"role": "user", "content": "next"},
            ]

            payload = harness._build_model_payload(messages)

            self.assertEqual(
                [
                    {"role": "user", "content": "next"},
                ],
                payload["messages"],
            )

    def test_gigachat_drops_dangling_function_calls_from_trimmed_history(self) -> None:
        with WorkspaceTempDir("agent_gigachat_trimmed_call") as root:
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[],
            )
            messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "write_file",
                        "arguments": "{\"path\":\"README.md\",\"content\":\"x\"}",
                    },
                },
                {"role": "user", "content": "next"},
            ]

            payload = harness._build_model_payload(messages)

            self.assertEqual(
                [{"role": "user", "content": "next"}],
                payload["messages"],
            )

    def test_executes_gigachat_function_call_and_completes(self) -> None:
        with WorkspaceTempDir("agent_gigachat_function_call") as root:
            tool_call_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "function_call": {
                                "name": "write_file",
                                "arguments": "{\"path\":\"README.md\",\"content\":\"test\"}",
                            },
                            "functions_state_id": "state-1",
                        }
                    }
                ]
            }
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[tool_call_response, make_response("Р“РѕС‚РѕРІРѕ.")],
                disable_follow_ups=True,
            )

            result = harness.run("РЎРѕР·РґР°Р№ README С„Р°Р№Р»")

            self.assertEqual("Р“РѕС‚РѕРІРѕ.", result)
            self.assertTrue((root / "Generated_Project" / "README.md").exists())

    def test_stops_when_tool_capable_provider_repeats_same_text_without_actions(self) -> None:
        with WorkspaceTempDir("agent_repeat_loop") as root:
            repeated = "РџР»Р°РЅ: СЃРѕР·РґР°Рј СЃС‚СЂСѓРєС‚СѓСЂСѓ РїСЂРѕРµРєС‚Р° Рё Р·Р°С‚РµРј Р·Р°РїСѓС‰Сѓ РїСЂРёР»РѕР¶РµРЅРёРµ."
            harness = StubHarness(
                config=build_config(root, uses_gigachat=False),
                responses=[make_response(repeated), make_response(repeated)],
            )

            result = harness.run("РЎРѕР·РґР°Р№ РїСЂРѕРµРєС‚ СЃР°Р№С‚Р° РєРѕС„РµР№РЅРё РЅР° Flask")

            self.assertIn("Harness", result)
            self.assertIn(repeated, result)

    def test_executes_textual_tool_imitation_for_gigachat(self) -> None:
        with WorkspaceTempDir("agent_pseudo_tools") as root:
            pseudo_tools = 'write_file website/__init__.py """from flask import Flask"""'
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[make_response(pseudo_tools), make_response("Р“РѕС‚РѕРІРѕ.")],
                disable_follow_ups=True,
            )

            result = harness.run("РЎРѕР·РґР°Р№ СЃР°Р№С‚ РЅР° flask РґР»СЏ РєРѕС„РµР№РЅРё")

            self.assertEqual("Р“РѕС‚РѕРІРѕ.", result)
            self.assertTrue((root / "Generated_Project" / "website").exists())
            self.assertEqual(
                "from flask import Flask",
                (root / "Generated_Project" / "website" / "__init__.py").read_text(encoding="utf-8"),
            )

    def test_executes_superquote_textual_write_file_for_gigachat(self) -> None:
        with WorkspaceTempDir("agent_superquote_pseudo_tools") as root:
            pseudo_tool = (
                "write_file {\n"
                "    <|superquote|>content<|superquote|>: <|superquote|>{\n"
                "        <|superquote|>port<|superquote|>: 3000\n"
                "    }\n"
                "    <|superquote|>path<|superquote|>: <|superquote|>config/config.json<|superquote|>\n"
                "}\n"
            )
            harness = StubHarness(
                config=build_config(root, uses_gigachat=True),
                responses=[make_response(pseudo_tool), make_response("Готово.")],
                disable_follow_ups=True,
            )

            result = harness.run("Создай config")

            self.assertEqual("Готово.", result)
            self.assertEqual(
                '{\n        "port": 3000\n    }',
                (root / "Generated_Project" / "config" / "config.json").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
