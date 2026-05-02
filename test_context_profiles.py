from __future__ import annotations

import unittest
from pathlib import Path

from context_profiles import detect_context_profile, load_profile_rules


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


class ContextProfileTests(unittest.TestCase):
    def test_detect_context_profile_prefers_frontend_for_ui_work(self) -> None:
        profile = detect_context_profile("Build a responsive frontend dashboard in React")
        self.assertEqual("frontend-react", profile.name)

    def test_detect_context_profile_prefers_cli_for_command_line_work(self) -> None:
        profile = detect_context_profile("Create a CLI tool with argparse and self-check")
        self.assertEqual("cli", profile.name)

    def test_detect_context_profile_uses_project_structure(self) -> None:
        with WorkspaceTempDir("profile_structure") as root:
            (root / "package.json").write_text("{}", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "App.tsx").write_text("export default function App() {}", encoding="utf-8")

            profile = detect_context_profile("Build a UI", project_root=root)

            self.assertEqual("frontend-react", profile.name)

    def test_load_profile_rules_includes_matching_rule_pack(self) -> None:
        root = Path.cwd()
        rules = load_profile_rules(root, detect_context_profile("Build frontend UI"))
        self.assertIn("React Frontend Rules", rules)

    def test_frontend_profiles_disable_extra_blocks(self) -> None:
        profile = detect_context_profile("Build a Tailwind UI")
        self.assertFalse(profile.include_markdown_context)
        self.assertFalse(profile.include_architecture_doc)


if __name__ == "__main__":
    unittest.main()
