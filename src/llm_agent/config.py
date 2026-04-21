"""Agent configuration loaded from environment variables or defaults."""

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class AgentConfig:
    """Top-level settings for the coding agent.

    All fields can be overridden via environment variables with the
    ``LLM_AGENT_`` prefix (e.g. ``LLM_AGENT_MODEL_NAME=qwen2.5-coder:7b``).
    """

    model_name: str = field(default_factory=lambda: os.environ.get("LLM_AGENT_MODEL_NAME", "qwen2.5-coder:14b"))
    """Ollama model tag to use for inference."""

    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    )
    """Base URL of the running Ollama server."""

    shell_timeout: int = field(default_factory=lambda: int(os.environ.get("LLM_AGENT_SHELL_TIMEOUT", "30")))
    """Maximum seconds a shell command is allowed to run."""

    max_iterations: int = field(default_factory=lambda: int(os.environ.get("LLM_AGENT_MAX_ITERATIONS", "20")))
    """Maximum plan→act→observe iterations before the agent stops."""

    working_directory: str = field(default_factory=lambda: os.environ.get("LLM_AGENT_WORKING_DIRECTORY", "."))
    """Root directory the agent operates in."""
