"""Tests for filesystem tools."""

from pathlib import Path

from llm_agent.tools.filesystem import list_directory, read_file, write_file


def test_read_file(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    result = read_file.invoke({"path": str(target)})
    assert result == "hello world"


def test_read_file_not_found(tmp_path: Path) -> None:
    result = read_file.invoke({"path": str(tmp_path / "missing.txt")})
    assert "Error" in result
    assert "not found" in result


def test_read_file_not_a_file(tmp_path: Path) -> None:
    result = read_file.invoke({"path": str(tmp_path)})
    assert "Error" in result
    assert "not a file" in result


def test_write_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    result = write_file.invoke({"path": str(target), "content": "data"})
    assert "Successfully wrote" in result
    assert target.read_text(encoding="utf-8") == "data"


def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    write_file.invoke({"path": str(target), "content": "nested"})
    assert target.read_text(encoding="utf-8") == "nested"


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    result = list_directory.invoke({"path": str(tmp_path)})
    assert "subdir" in result
    assert "file.txt" in result


def test_list_directory_not_found(tmp_path: Path) -> None:
    result = list_directory.invoke({"path": str(tmp_path / "no_such_dir")})
    assert "Error" in result
    assert "not found" in result


def test_list_directory_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "f.txt"
    file_path.write_text("x", encoding="utf-8")
    result = list_directory.invoke({"path": str(file_path)})
    assert "Error" in result
    assert "not a directory" in result


def test_list_directory_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = list_directory.invoke({"path": str(empty)})
    assert result == "(empty directory)"
