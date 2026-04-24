"""Shell tool: run a command inside a subprocess with a hard timeout."""

import subprocess

from langchain_core.tools import (
    tool,  # pyright: ignore[reportUnknownVariableType]  # langchain-core stubs are incomplete
)

_DEFAULT_TIMEOUT = 30
_MAX_OUTPUT_CHARS = 8_000


@tool
def run_shell(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute *command* in a shell and return combined stdout/stderr.

    The process is killed after *timeout* seconds to prevent runaway commands.
    Output is capped at 8 000 characters to stay within LLM context limits.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
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
