"""Filesystem tools: read, write, and list files."""

from pathlib import Path

from langchain_core.tools import (
    tool,  # pyright: ignore[reportUnknownVariableType]  # langchain-core stubs are incomplete
)


@tool
def read_file(path: str) -> str:
    """Read and return the text content of the file at *path*."""
    file_path = Path(path)
    if not file_path.exists():
        return f"Error: file not found: {path}"
    if not file_path.is_file():
        return f"Error: path is not a file: {path}"
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write *content* to the file at *path*, creating parent directories as needed."""
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error: failed to write file {path}: {exc}"
    return f"Successfully wrote {len(content)} characters to {path}"


@tool
def list_directory(path: str = ".") -> str:
    """List the immediate children of the directory at *path*."""
    dir_path = Path(path)
    if not dir_path.exists():
        return f"Error: path not found: {path}"
    if not dir_path.is_dir():
        return f"Error: path is not a directory: {path}"

    entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = [f"{'d' if entry.is_dir() else 'f'}  {entry.name}" for entry in entries]
    return "\n".join(lines) if lines else "(empty directory)"
