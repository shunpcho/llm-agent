# llm-agent

A local LLM coding agent powered by [Ollama](https://ollama.com/) and [LangGraph](https://github.com/langchain-ai/langgraph).

The agent receives a natural-language coding task and autonomously executes a **plan → act → observe** loop — reading files, writing code, running tests, and using shell tools — until the task is complete.

## Prerequisites

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** for dependency management
- **[Ollama](https://ollama.com/)** running locally with a model pulled, e.g.:
  ```bash
  ollama pull qwen2.5-coder:14b
  ```

## Installation

```bash
uv sync
```

## Usage

### Python API

```python
from llm_agent.agent import run_agent
from llm_agent.config import AgentConfig

# Run with defaults (qwen2.5-coder:14b, working directory = ".")
messages = run_agent("Add docstrings to every public function in src/")

# Print the agent's final response
print(messages[-1].content)
```

Custom configuration:

```python
config = AgentConfig(
    model_name="qwen2.5-coder:7b",
    working_directory="/path/to/your/repo",
    max_iterations=10,
)
messages = run_agent("Fix all type errors reported by pyright", config=config)
```

### Configuration

All settings can be provided via environment variables:

| Environment variable | Default | Description |
|---|---|---|
| `LLM_AGENT_MODEL_NAME` | `qwen2.5-coder:14b` | Ollama model tag |
| `LLM_AGENT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_AGENT_WORKING_DIRECTORY` | `.` | Root directory the agent operates in |
| `LLM_AGENT_MAX_ITERATIONS` | `20` | Maximum plan→act→observe cycles |
| `LLM_AGENT_SHELL_TIMEOUT` | `30` | Maximum seconds per shell command |

Example:

```bash
LLM_AGENT_MODEL_NAME=qwen2.5-coder:7b \
LLM_AGENT_WORKING_DIRECTORY=/my/project \
python -c "
from llm_agent.agent import run_agent
run_agent('Run tests and fix any failures')
"
```

## Available Tools

The agent can use the following tools during its loop:

| Tool | Description |
|---|---|
| `read_file` | Read the contents of a file |
| `write_file` | Create or overwrite a file |
| `list_directory` | List files and subdirectories at a path |
| `run_shell` | Execute a shell command and return stdout/stderr |
| `search_code` | Search for a pattern across files (grep) |

## Development

Format, lint, type-check, and test before opening a PR:

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full contribution guidelines.
