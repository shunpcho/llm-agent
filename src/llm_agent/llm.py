"""LLM client factory: returns a LangChain chat model backed by Ollama."""

from langchain_ollama import ChatOllama

from llm_agent.config import AgentConfig


def create_llm(config: AgentConfig) -> ChatOllama:
    """Create a ChatOllama instance configured from *config*."""
    return ChatOllama(
        model=config.model_name,
        base_url=config.ollama_base_url,
    )
