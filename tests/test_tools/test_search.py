"""Tests for the search_code tool."""

from pathlib import Path

from llm_agent.tools.search import search_code


def test_search_code_finds_match(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("def hello():\n    pass\n", encoding="utf-8")
    result = search_code.invoke({"pattern": "def hello", "path": str(tmp_path)})
    assert "def hello" in result
    assert "sample.py" in result


def test_search_code_no_match(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("x = 1\n", encoding="utf-8")
    result = search_code.invoke({"pattern": "def hello", "path": str(tmp_path)})
    assert "No matches found" in result


def test_search_code_path_not_found(tmp_path: Path) -> None:
    result = search_code.invoke({"pattern": "foo", "path": str(tmp_path / "missing")})
    assert "Error" in result
    assert "not found" in result


def test_search_code_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("import os\n", encoding="utf-8")
    result = search_code.invoke({"pattern": "import os", "path": str(tmp_path), "file_glob": "*.py"})
    assert "code.py" in result
    assert "readme.md" not in result
