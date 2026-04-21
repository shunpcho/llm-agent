# Contributing

Thanks for your interest in contributing!

## Development setup (uv)

This project uses **uv** for dependency and environment management.

1. Install uv (see official docs).
2. Create/update the environment and install dependencies:

```bash
uv sync
```

## Project structure

- Package code: `src/<package_name>/`
- Tests: `tests/`
- Coding rules: `docs/coding-instructions.md`
- Tooling config: `pyproject.toml` (repo root)

## Code quality (required)

Format, lint, type-check, and test your changes before opening a PR:

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest -q
```

## Pull requests

- Keep PRs focused and small when possible.
- Include tests for behavior changes.
- Describe what changed and why, and how to verify it.

## Reporting issues

- Provide clear reproduction steps.
- Include environment details (OS, Python version, relevant package versions).
- Include logs/tracebacks when applicable.