# Agent Instructions (Copilot / AI assistants)

## 0) Where to find coding rules
- Coding rules (style, typing, testing, tooling) are defined in:
  - `docs/coding-instructions.md`
- Consult it before making stylistic decisions.

## 1) Priority / scope
- Follow this file first.
- Then follow `docs/coding-instructions.md`.
- Then follow other repository docs (`README.md`, `CONTRIBUTING.md`, `docs/*`).
- If instructions conflict, ask for clarification rather than guessing.

## 2) Repository structure (must follow)
Expected at project root:
- `pyproject.toml` (required; single source of truth)
- `src/<package_name>/` (package code; keep `src/` layout consistent)
- `tests/`
- `docs/`

## 3) Dependency management (uv required)
- Dependency management MUST use **uv**.
- Prefer `uv run ...` to execute tools in the managed environment.

## 4) What to include in proposals/patches
- Exact file paths to create/edit.
- Final code (not only pseudocode).
- Brief rationale.
- How to run formatting/lint/type-check/tests (uv-based commands).