"""Tests for the shell tool."""

from llm_agent.tools.shell import run_shell


def test_run_shell_success() -> None:
    result = run_shell.invoke({"command": "echo hello"})
    assert "hello" in result
    assert "[exit code: 0]" in result


def test_run_shell_nonzero_exit() -> None:
    result = run_shell.invoke({"command": "exit 1"})
    assert "[exit code: 1]" in result


def test_run_shell_stderr() -> None:
    result = run_shell.invoke({"command": "echo err >&2"})
    assert "err" in result


def test_run_shell_timeout() -> None:
    result = run_shell.invoke({"command": "sleep 10", "timeout": 1})
    assert "timed out" in result


def test_run_shell_combined_output() -> None:
    result = run_shell.invoke({"command": "echo out && echo err >&2"})
    assert "out" in result
    assert "err" in result
