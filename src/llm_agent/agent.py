"""LangGraph-based coding agent: plan → act → observe loop."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Literal, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from llm_agent.config import AgentConfig
from llm_agent.llm import create_llm
from llm_agent.prompts import build_system_prompt
from llm_agent.tools import list_directory, make_tools, read_file, run_shell, search_code, write_file

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class AgentState(BaseModel):
    """Mutable state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    """Conversation history including tool calls and results."""

    iteration: int = 0
    """Number of act→observe cycles completed."""


_ALL_TOOLS: list[BaseTool] = [read_file, write_file, list_directory, run_shell, search_code]
_TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in _ALL_TOOLS}


def _extract_first_json_object(text: str) -> object | None:
    """Scan *text* for the first balanced ``{...}`` block and try to parse it as JSON.

    Unlike a greedy regex, this correctly handles content that contains multiple JSON
    objects or extra ``{``/``}`` characters outside the target object.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_tool_call_from_content(content: str) -> dict[str, object] | None:
    """Try to extract a structured tool call from a model response that placed JSON in its content.

    Some Ollama-backed models emit ``{"name": ..., "arguments": ...}`` as plain text
    instead of using the structured ``tool_calls`` field.  This function parses such
    responses so the agent can still dispatch the call.

    Returns a tool-call dict compatible with ``AIMessage.tool_calls``, or ``None`` if
    no tool call could be extracted.
    """
    stripped = content.strip()

    # First try parsing the whole content as JSON; fall back to a balanced-brace scan
    # so that extra surrounding text or multiple JSON objects do not confuse extraction.
    data: object = None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = _extract_first_json_object(stripped)

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None

    # The model may use "arguments" (OpenAI-style) or "args" (LangChain-style).
    raw_args = data.get("arguments") or data.get("args") or {}
    args: dict[str, object] = raw_args if isinstance(raw_args, dict) else {}

    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "args": args,
        "type": "tool_call",
    }


def _should_continue(state: AgentState, config: AgentConfig) -> Literal["tools", "__end__"]:
    """Route to tool execution or END based on the last message."""
    last = state.messages[-1]
    if not isinstance(last, AIMessage):
        return END  # pyright: ignore[reportReturnType]  # END is the string "__end__" at runtime
    if last.tool_calls and state.iteration < config.max_iterations:
        return "tools"
    return END  # pyright: ignore[reportReturnType]  # END is the string "__end__" at runtime


def call_tools(state: AgentState) -> dict[str, object]:
    """Execute all tool calls present in the last AI message."""
    last = state.messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    tool_messages: list[ToolMessage] = []
    for tool_call in last.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            tool_call_id = str(uuid.uuid4())
            tool_call["id"] = tool_call_id
        tool = _TOOL_MAP.get(tool_name)
        result = f"Error: unknown tool '{tool_name}'" if tool is None else tool.invoke(tool_args)
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
            )
        )

    return {
        "messages": tool_messages,
        "iteration": state.iteration + 1,
    }


def build_graph(  # pyright: ignore[reportUnknownParameterType]  # LangGraph stubs are incomplete
    config: AgentConfig,
) -> StateGraph:  # pyright: ignore[reportMissingTypeArgument]  # LangGraph stubs are incomplete
    """Construct and return the coding-agent LangGraph."""
    config_tools = make_tools(config)
    config_tool_map: dict[str, BaseTool] = {t.name: t for t in config_tools}
    llm = create_llm(config).bind_tools(config_tools)
    system_prompt = build_system_prompt(config)

    def call_model(state: AgentState) -> dict[str, object]:
        """Invoke the LLM with the current conversation history."""
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *state.messages]
        response = llm.invoke(messages)

        # Some Ollama-backed models emit tool calls as JSON text rather than using the
        # structured tool_calls field.  When that happens, parse the content and inject
        # a synthetic tool call so the graph can continue to the tools node.
        if (
            isinstance(response, AIMessage)
            and not response.tool_calls
            and isinstance(response.content, str)
            and response.content.strip()
        ):
            parsed = _extract_tool_call_from_content(response.content)
            if parsed is not None:
                response = AIMessage(
                    # The original content was a JSON tool-call string; clearing it
                    # avoids surfacing raw JSON in the conversation history and ensures
                    # the message is recognisable as a pure tool-dispatch turn.
                    content="",
                    tool_calls=[parsed],  # pyright: ignore[reportArgumentType]  # dict is compatible with ToolCall TypedDict
                    response_metadata=response.response_metadata,
                    id=response.id,
                )

        return {"messages": [response]}

    def _call_tools_node(state: AgentState) -> dict[str, object]:
        """Execute all tool calls using config-aware tool instances."""
        last = state.messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        tool_messages: list[ToolMessage] = []
        for tool_call in last.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call.get("id")
            if not tool_call_id:
                tool_call_id = str(uuid.uuid4())
                tool_call["id"] = tool_call_id
            tool = config_tool_map.get(tool_name)
            result = f"Error: unknown tool '{tool_name}'" if tool is None else tool.invoke(tool_args)
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )

        return {
            "messages": tool_messages,
            "iteration": state.iteration + 1,
        }

    graph: StateGraph = StateGraph(AgentState)  # pyright: ignore[reportMissingTypeArgument]  # LangGraph stubs
    graph.add_node("model", call_model)
    graph.add_node("tools", _call_tools_node)

    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        lambda state: _should_continue(state, config),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "model")

    return graph  # pyright: ignore[reportUnknownVariableType]  # LangGraph stubs are incomplete


def run_agent(task: str, config: AgentConfig | None = None) -> list[BaseMessage]:
    """Run the coding agent on *task* and return the full message history.

    Args:
        task: Natural-language description of the coding task to perform.
        config: Optional agent configuration; defaults to ``AgentConfig()``.
    """
    if config is None:
        config = AgentConfig()

    graph = build_graph(config)  # pyright: ignore[reportUnknownVariableType]  # LangGraph stubs are incomplete
    compiled = graph.compile()  # pyright: ignore[reportUnknownVariableType]  # LangGraph stubs are incomplete
    initial_state = AgentState(messages=[HumanMessage(content=task)])
    final_state = compiled.invoke(initial_state)
    # LangGraph returns a dict-like object; the type is inferred as Unknown due to incomplete stubs.
    return final_state["messages"]
