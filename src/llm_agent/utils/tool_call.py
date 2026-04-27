"""Fallback helpers for parsing tool calls from raw model output."""

from __future__ import annotations

import json
import uuid


def extract_first_json_object(text: str) -> object | None:
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


def extract_tool_call_from_content(content: str) -> dict[str, object] | None:
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
        data = extract_first_json_object(stripped)

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
