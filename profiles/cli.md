# CLI Rules

- Prefer non-interactive commands for validation and automation.
- Support `--help` and a deterministic machine-readable check mode when possible.
- Keep command names and flags descriptive.
- Avoid blocking input loops unless the task truly needs interactivity.
- Make outputs concise and parseable.
- Prefer defaults that work in scripts, cron jobs, and subprocesses.
