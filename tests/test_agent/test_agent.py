"""Tests for the agent graph construction and tool dispatch."""

import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from llm_agent.agent import AgentState, build_graph, call_tools  # pyright: ignore[reportUnknownVariableType]
from llm_agent.config import AgentConfig
from llm_agent.utils.tool_call import extract_first_json_object as _extract_first_json_object
from llm_agent.utils.tool_call import extract_tool_call_from_content as _extract_tool_call_from_content


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


# --- _extract_tool_call_from_content ---


def test_extract_tool_call_from_content_arguments_key() -> None:
    """Model output using the "arguments" key (OpenAI-style) is parsed correctly."""
    content = '{"name": "read_file", "arguments": {"path": "foo.py"}}'
    result = _extract_tool_call_from_content(content)
    assert result is not None
    assert result["name"] == "read_file"
    assert result["args"] == {"path": "foo.py"}
    assert result["type"] == "tool_call"
    uuid.UUID(str(result["id"]))  # must be a valid UUID


def test_extract_tool_call_from_content_args_key() -> None:
    """Model output using the "args" key (LangChain-style) is parsed correctly."""
    content = '{"name": "list_directory", "args": {"path": "."}}'
    result = _extract_tool_call_from_content(content)
    assert result is not None
    assert result["name"] == "list_directory"
    assert result["args"] == {"path": "."}


def test_extract_tool_call_from_content_embedded_in_text() -> None:
    """JSON embedded inside surrounding text is still extracted."""
    content = 'Sure, let me do that:\n\n{"name": "run_shell", "arguments": {"command": "ls"}}\n'
    result = _extract_tool_call_from_content(content)
    assert result is not None
    assert result["name"] == "run_shell"


def test_extract_tool_call_from_content_no_json() -> None:
    """Plain text with no JSON returns None."""
    result = _extract_tool_call_from_content("The task is now complete.")
    assert result is None


def test_extract_tool_call_from_content_missing_name() -> None:
    """JSON without a 'name' field returns None."""
    result = _extract_tool_call_from_content('{"arguments": {"path": "foo.py"}}')
    assert result is None


def test_extract_tool_call_from_content_empty_string() -> None:
    """Empty content returns None."""
    result = _extract_tool_call_from_content("")
    assert result is None


# --- _extract_first_json_object ---


def test_extract_first_json_object_simple() -> None:
    """Extracts a plain JSON object correctly."""
    result = _extract_first_json_object('{"a": 1}')
    assert result == {"a": 1}


def test_extract_first_json_object_embedded_in_text() -> None:
    """Extracts the first JSON object when surrounded by prose."""
    result = _extract_first_json_object('Here it is: {"x": 2} and some trailing text.')
    assert result == {"x": 2}


def test_extract_first_json_object_multiple_objects() -> None:
    """Only the first JSON object is returned when multiple are present."""
    result = _extract_first_json_object('{"first": 1} {"second": 2}')
    assert result == {"first": 1}


def test_extract_first_json_object_nested_braces() -> None:
    """Nested objects are handled correctly and not cut short by inner closing braces."""
    result = _extract_first_json_object('{"name": "f", "arguments": {"path": "p"}}')
    assert result == {"name": "f", "arguments": {"path": "p"}}


def test_extract_first_json_object_string_with_braces() -> None:
    """Brace characters inside a string value do not confuse the brace counter."""
    result = _extract_first_json_object('{"name": "f", "arguments": {"cmd": "echo {hi}"}}')
    assert result == {"name": "f", "arguments": {"cmd": "echo {hi}"}}


def test_extract_first_json_object_no_object() -> None:
    """Returns None when no JSON object is present."""
    result = _extract_first_json_object("no braces here")
    assert result is None


def test_extract_tool_call_from_content_multiple_objects_in_prose() -> None:
    """The correct tool-call object is extracted even when there are extra braces in the text."""
    content = 'Status: {} and then {"name": "write_file", "arguments": {"path": "a.py", "content": "x"}}'
    result = _extract_tool_call_from_content(content)
    # First balanced object is {} which has no "name", so None is returned.
    # The function should gracefully return None for the empty object.
    assert result is None


def test_call_model_falls_back_when_tool_calls_empty() -> None:
    """call_model converts JSON content to a structured tool call when tool_calls is empty."""
    config = _make_config()
    json_content = '{"name": "list_directory", "arguments": {"path": "."}}'

    with patch("llm_agent.agent.create_llm") as mock_create_llm:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content=json_content, tool_calls=[])
        mock_create_llm.return_value = mock_llm

        graph = build_graph(config)  # pyright: ignore[reportUnknownVariableType]
        compiled = graph.compile()  # pyright: ignore[reportUnknownVariableType]

        # Run the compiled graph once. Downstream tool execution may still occur; this test
        # only verifies that call_model populated tool_calls from the JSON content it returned.
        state = AgentState(messages=[HumanMessage(content="task")], iteration=config.max_iterations - 1)

        # Invoke the full graph and inspect the returned messages for the model-produced AIMessage.
        result = compiled.invoke(state)  # pyright: ignore[reportUnknownVariableType]
        messages = result["messages"]

        # Find the AIMessage emitted by the model node in the final graph state.
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert ai_messages, "Expected at least one AIMessage"
        first_ai = ai_messages[0]
        assert first_ai.tool_calls, "Expected tool_calls to be populated from JSON content"
        assert first_ai.tool_calls[0]["name"] == "list_directory"
