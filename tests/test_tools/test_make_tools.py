"""Tests for the make_tools config-aware factory."""

from pathlib import Path

from llm_agent.config import AgentConfig
from llm_agent.tools import make_tools


def _config(tmp_path: Path, timeout: int = 30) -> AgentConfig:
    return AgentConfig(
        model_name="test-model",
        working_directory=str(tmp_path),
        shell_timeout=timeout,
    )


def test_make_tools_returns_five_tools(tmp_path: Path) -> None:
    tools = make_tools(_config(tmp_path))
    assert len(tools) == 5
    names = {t.name for t in tools}
    assert names == {"read_file", "write_file", "list_directory", "run_shell", "search_code"}


def test_read_file_resolves_relative_path(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("world", encoding="utf-8")
    tools = {t.name: t for t in make_tools(_config(tmp_path))}
    result = tools["read_file"].invoke({"path": "hello.txt"})
    assert result == "world"


def test_write_file_resolves_relative_path(tmp_path: Path) -> None:
    tools = {t.name: t for t in make_tools(_config(tmp_path))}
    tools["write_file"].invoke({"path": "out.txt", "content": "data"})
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "data"


def test_list_directory_defaults_to_working_dir(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    tools = {t.name: t for t in make_tools(_config(tmp_path))}
    result = tools["list_directory"].invoke({"path": "."})
    assert "a.txt" in result


def test_run_shell_executes_in_working_dir(tmp_path: Path) -> None:
    tools = {t.name: t for t in make_tools(_config(tmp_path))}
    result = tools["run_shell"].invoke({"command": "pwd"})
    assert str(tmp_path.resolve()) in result


def test_run_shell_respects_configured_timeout(tmp_path: Path) -> None:
    tools = {t.name: t for t in make_tools(_config(tmp_path, timeout=1))}
    result = tools["run_shell"].invoke({"command": "sleep 10"})
    assert "timed out" in result


def test_search_code_resolves_relative_path(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("def hello():\n    pass\n", encoding="utf-8")
    tools = {t.name: t for t in make_tools(_config(tmp_path))}
    result = tools["search_code"].invoke({"pattern": "def hello", "path": "."})
    assert "def hello" in result
