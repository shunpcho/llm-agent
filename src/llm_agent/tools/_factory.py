"""Factory that creates config-aware LangChain tool instances."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import (
    tool,  # pyright: ignore[reportUnknownVariableType]  # langchain-core stubs are incomplete
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from llm_agent.config import AgentConfig

_MAX_OUTPUT_CHARS = 8_000


# ---------------------------------------------------------------------------
# Per-tool implementation helpers (accept explicit config parameters so they
# can be shared between the module-level tools and the config-aware factory).
# ---------------------------------------------------------------------------


def _resolve(path: str, base: Path) -> Path:
    """Return *path* resolved relative to *base* if not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else base / path


def _read_file_impl(path: str, working_dir: Path) -> str:
    resolved = _resolve(path, working_dir)
    if not resolved.exists():
        return f"Error: file not found: {path}"
    if not resolved.is_file():
        return f"Error: path is not a file: {path}"
    try:
        return resolved.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def _write_file_impl(path: str, content: str, working_dir: Path) -> str:
    resolved = _resolve(path, working_dir)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error: failed to write file {path}: {exc}"
    return f"Successfully wrote {len(content)} characters to {path}"


def _list_directory_impl(path: str, working_dir: Path) -> str:
    resolved = _resolve(path, working_dir)
    if not resolved.exists():
        return f"Error: path not found: {path}"
    if not resolved.is_dir():
        return f"Error: path is not a directory: {path}"
    entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = [f"{'d' if entry.is_dir() else 'f'}  {entry.name}" for entry in entries]
    return "\n".join(lines) if lines else "(empty directory)"


def _run_shell_impl(command: str, timeout: int, cwd: str | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
        output = result.stdout + result.stderr
        return_code_line = f"\n[exit code: {result.returncode}]"
        truncation_marker = "\n...[output truncated]"
        max_output_chars = _MAX_OUTPUT_CHARS - len(return_code_line)

        if len(output) > max_output_chars:
            max_truncated_output_chars = max_output_chars - len(truncation_marker)
            if max_truncated_output_chars > 0:
                output = output[:max_truncated_output_chars] + truncation_marker
            else:
                output = output[:max_output_chars]
        return output + return_code_line
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except Exception as exc:  # noqa: BLE001
        return f"Error running command: {exc}"


def _search_code_impl(
    pattern: str, path: str, file_glob: str, working_dir: Path, timeout: int
) -> str:
    resolved = _resolve(path, working_dir)
    if not resolved.exists():
        return f"Error: path not found: {path}"

    command = [
        "grep",
        "--recursive",
        "--line-number",
        "--include",
        file_glob,
        "-e",
        pattern,
        str(resolved),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = result.stdout + result.stderr
        if not output.strip():
            return f"No matches found for pattern '{pattern}' in {path}"
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n...[output truncated]"
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as exc:  # noqa: BLE001
        return f"Error running search: {exc}"
    else:
        return output


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_tools(config: AgentConfig) -> list[BaseTool]:
    """Create tool instances bound to *config*.

    Relative paths passed to filesystem and search tools are resolved against
    ``config.working_directory``.  Shell commands are executed with
    ``cwd=config.working_directory`` and time out after
    ``config.shell_timeout`` seconds.
    """
    working_dir = Path(config.working_directory).resolve()
    shell_timeout = config.shell_timeout

    @tool
    def read_file(path: str) -> str:
        """Read and return the text content of the file at *path*."""
        return _read_file_impl(path, working_dir)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write *content* to the file at *path*, creating parent directories as needed."""
        return _write_file_impl(path, content, working_dir)

    @tool
    def list_directory(path: str = ".") -> str:
        """List the immediate children of the directory at *path*."""
        return _list_directory_impl(path, working_dir)

    @tool
    def run_shell(command: str, timeout: int = shell_timeout) -> str:
        """Execute *command* in a shell and return combined stdout/stderr.

        The process is killed after *timeout* seconds to prevent runaway commands.
        Output is capped at 8 000 characters to stay within LLM context limits.
        """
        return _run_shell_impl(command, timeout, cwd=str(working_dir))

    @tool
    def search_code(pattern: str, path: str = ".", file_glob: str = "*") -> str:
        """Search for *pattern* (regex) in files matching *file_glob* under *path*.

        Returns matching lines with file names and line numbers, capped at 8 000
        characters.
        """
        return _search_code_impl(pattern, path, file_glob, working_dir, shell_timeout)

    return [read_file, write_file, list_directory, run_shell, search_code]
