"""Search tool: grep for patterns inside a directory tree."""

import subprocess
from pathlib import Path

from langchain_core.tools import (
    tool,  # pyright: ignore[reportUnknownVariableType]  # langchain-core stubs are incomplete
)

_MAX_OUTPUT_CHARS = 8_000


@tool
def search_code(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Search for *pattern* (regex) in files matching *file_glob* under *path*.

    Returns matching lines with file names and line numbers, capped at 8 000
    characters.
    """
    search_path = Path(path)
    if not search_path.exists():
        return f"Error: path not found: {path}"

    command = [
        "grep",
        "--recursive",
        "--line-number",
        "--include",
        file_glob,
        "-e",
        pattern,
        str(search_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
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
