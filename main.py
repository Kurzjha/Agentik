from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import AgentHarness
from config import HarnessConfig, load_settings


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings(Path.cwd() / "settings.json")
    parser = argparse.ArgumentParser(description="Agentic Harness CLI")
    parser.add_argument("prompt", nargs="*", help="User request for the agent.")
    parser.add_argument("--session", default=str(settings["default_session_id"]), help="Session id used for JSONL state.")
    parser.add_argument(
        "--permission-mode",
        choices=["plan", "auto", "bypass"],
        default=str(settings["permission_mode"]),
        help="Shell command execution policy.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd() / str(settings["project_root"])),
        help="Writable project directory exposed to the agent.",
    )
    parser.add_argument(
        "--headless-file",
        help="Optional file containing the prompt. Useful for subprocess and cron usage.",
    )
    return parser


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.headless_file:
        return Path(args.headless_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return " ".join(args.prompt).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def run_chat_shell(harness: AgentHarness) -> None:
    print("Interactive agent shell started.")
    print("Commands: /clear, /rewind <message_index>, /exit")
    while True:
        try:
            prompt = input("You> ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return

        try:
            result = harness.run(prompt)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Model> {result}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prompt = resolve_prompt(args)

    config = HarnessConfig.from_env(
        project_root=Path(args.project_root).resolve(),
        workspace_root=Path.cwd().resolve(),
        permission_mode=args.permission_mode,
    )
    harness = AgentHarness(config=config, session_id=args.session)
    if not prompt:
        run_chat_shell(harness)
        return
    result = harness.run(prompt)
    print(result)


if __name__ == "__main__":
    main()
