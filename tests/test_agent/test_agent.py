"""Tests for the agent graph construction and tool dispatch."""

import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from llm_agent.agent import AgentState, build_graph, call_tools  # pyright: ignore[reportUnknownVariableType]
from llm_agent.config import AgentConfig


def _make_config() -> AgentConfig:
    return AgentConfig(
        model_name="test-model",
        ollama_base_url="http://localhost:11434",
        max_iterations=5,
    )


def test_call_tools_unknown_tool() -> None:
    state = AgentState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "nonexistent_tool", "args": {}}],
            )
        ]
    )
    result = call_tools(state)
    tool_messages = result["messages"]
    assert len(tool_messages) == 1  # pyright: ignore[reportArgumentType]
    assert "unknown tool" in tool_messages[0].content  # pyright: ignore[reportIndexIssue]


def test_call_tools_increments_iteration() -> None:
    state = AgentState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"id": "2", "name": "nonexistent_tool", "args": {}}],
            )
        ],
        iteration=3,
    )
    result = call_tools(state)
    assert result["iteration"] == 4


def test_call_tools_without_id_generates_fallback() -> None:
    state = AgentState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"id": None, "name": "nonexistent_tool", "args": {}}],
            )
        ]
    )
    result = call_tools(state)
    tool_messages = result["messages"]
    assert len(tool_messages) == 1  # pyright: ignore[reportArgumentType]
    generated_id = tool_messages[0].tool_call_id  # pyright: ignore[reportIndexIssue]
    assert generated_id is not None
    uuid.UUID(generated_id)  # raises ValueError if not a valid UUID


def test_call_tools_no_tool_calls() -> None:
    state = AgentState(messages=[AIMessage(content="done")])
    result = call_tools(state)
    assert result == {}


def test_build_graph_returns_state_graph() -> None:
    config = _make_config()
    with patch("llm_agent.agent.create_llm") as mock_create_llm:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_create_llm.return_value = mock_llm
        graph = build_graph(config)  # pyright: ignore[reportUnknownVariableType]
    assert graph is not None


def test_agent_state_messages_accumulate() -> None:
    state = AgentState(messages=[HumanMessage(content="task")])
    extra = HumanMessage(content="follow-up")
    updated = AgentState(messages=[*state.messages, extra])
    assert len(updated.messages) == 2
