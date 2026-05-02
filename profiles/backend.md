# Backend Rules

- Optimize for correctness, explicit validation, and stable boundaries.
- Keep I/O, persistence, and network code isolated from orchestration when possible.
- Validate inputs early and return precise errors.
- Prefer idempotent operations and clear failure handling.
- Preserve compatibility when changing request/response shapes.
- Add tests for edge cases, error paths, and invariants.
