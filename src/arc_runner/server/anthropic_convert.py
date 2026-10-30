"""Prompt rendering and tool-call handling for the Anthropic Messages routes.

Prompts use the same ChatML layout as /v1/chat/completions. Tools are described either by an engine
that renders its tokenizer chat template with tools (optional hook ``render_chat_with_tools``) or by a
compact deterministic instruction appended to the system prompt. Model output is read back with one
documented format only:

    <tool_call>{"name": "<tool name>", "arguments": {<JSON object>}}</tool_call>

Parsing is strict and all-or-nothing: anything that does not match exactly is returned as plain text.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from arc_runner.server.anthropic_schemas import ParsedRequest, ToolSpec, Turn

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
_CALL_PATTERN = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.DOTALL)


def new_message_id() -> str:
    """Return a fresh message id of the form msg_<24 hex>."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def new_tool_id() -> str:
    """Return a fresh tool call id of the form toolu_<24 hex>."""
    return f"toolu_{uuid.uuid4().hex[:24]}"


@dataclass
class Segment:
    """One output segment: plain text or a validated tool call."""

    kind: str
    text: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


def _tool_call_text(name: str, arguments: Dict[str, Any]) -> str:
    """Render a call in the documented format.

    Args:
        name: Tool name.
        arguments: Argument object.

    Returns:
        str: The tool_call block.
    """
    return f"{TOOL_OPEN}{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}{TOOL_CLOSE}"


def _render_turn_text(turn: Turn, names: Dict[str, str]) -> str:
    """Render one turn's blocks as text.

    Args:
        turn: Conversation turn.
        names: Tool names by tool_use id, updated as tool_use blocks are seen.

    Returns:
        str: Text of the turn, empty when it holds nothing renderable.
    """
    parts: List[str] = []
    for block in turn.blocks:
        if block.kind == "text":
            parts.append(block.text)
        elif block.kind == "tool_use":
            names[block.tool_id] = block.name
            parts.append(_tool_call_text(block.name, block.input))
        elif block.kind == "tool_result":
            attrs = f" id={json.dumps(block.tool_id)}"
            if block.tool_id in names:
                attrs += f" name={json.dumps(names[block.tool_id])}"
            if block.is_error:
                attrs += ' is_error="true"'
            parts.append(f"<tool_response{attrs}>\n{block.content}\n</tool_response>")
    return "\n".join(parts)


def render_chatml(system: str, turns: List[Turn]) -> str:
    """Render a conversation in the ChatML layout used by /v1/chat/completions.

    Args:
        system: System prompt, empty for none.
        turns: Conversation turns; a role of system is rendered in place.

    Returns:
        str: Prompt ending with an open assistant turn.
    """
    lines: List[str] = []
    if system:
        lines.append(f"<|im_start|>system\n{system}<|im_end|>")
    names: Dict[str, str] = {}
    for turn in turns:
        text = _render_turn_text(turn, names)
        if text:
            lines.append(f"<|im_start|>{turn.role}\n{text}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


def render_tools_instruction(tools: List[ToolSpec]) -> str:
    """Render the deterministic tool instruction placed in the system prompt.

    Args:
        tools: Declared tools in request order.

    Returns:
        str: Instruction text; identical input gives identical text.
    """
    listing = "\n".join(
        json.dumps(
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for tool in tools
    )
    return (
        "You can call tools. Available tools, one JSON object per line:\n"
        f"<tools>\n{listing}\n</tools>\n"
        'To call a tool reply with one block per call, exactly in this form: '
        f'{TOOL_OPEN}{{"name": "<tool name>", "arguments": {{<JSON object that follows the tool input_schema>}}}}{TOOL_CLOSE}\n'
        "Write nothing after the last block. If no tool is needed answer in plain text without any block. "
        "Tool results come back in <tool_response> blocks."
    )


def _chat_messages(system: str, turns: List[Turn]) -> List[Dict[str, str]]:
    """Render turns as role/content dicts for an engine chat template.

    Args:
        system: System prompt.
        turns: Conversation turns.

    Returns:
        List[Dict[str, str]]: Messages with tool calls and results rendered as text.
    """
    names: Dict[str, str] = {}
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in turns:
        text = _render_turn_text(turn, names)
        if text:
            messages.append({"role": turn.role, "content": text})
    return messages


def build_prompt(engine: Any, parsed: ParsedRequest) -> str:
    """Build the engine prompt for a validated request.

    Args:
        engine: Engine that will serve the request; may expose ``render_chat_with_tools(messages, tools)``
            returning the rendered prompt, or None when its chat template has no tool support.
        parsed: Validated request.

    Returns:
        str: Prompt text.
    """
    tools = [] if parsed.tool_choice_none else parsed.tools
    if tools:
        hook = getattr(engine, "render_chat_with_tools", None)
        if callable(hook):
            function_tools = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]
            rendered = hook(_chat_messages(parsed.system, parsed.messages), function_tools)
            if rendered is not None:
                return str(rendered)
    system = parsed.system
    if tools:
        instruction = render_tools_instruction(tools)
        system = f"{system}\n\n{instruction}" if system else instruction
    return render_chatml(system, parsed.messages)


def _reject_constant(token: str) -> Any:
    """Refuse NaN and Infinity, which are not JSON.

    Args:
        token: The constant found by the decoder.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-JSON constant {token}")


def _load_call(body: str, names: Set[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Validate the JSON inside one tool_call block.

    Args:
        body: Text between the tags.
        names: Declared tool names.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: (name, arguments) or None if anything is off.
    """
    try:
        data = json.loads(body.strip(), parse_constant=_reject_constant)
    except (ValueError, RecursionError, OverflowError):
        return None
    if not isinstance(data, dict) or set(data) != {"name", "arguments"}:
        return None
    name, arguments = data["name"], data["arguments"]
    if not isinstance(name, str) or name not in names or not isinstance(arguments, dict):
        return None
    return name, arguments


def parse_tool_calls(text: str, tool_names: Set[str]) -> Optional[List[Segment]]:
    """Split model output into text and tool-call segments, or report that it holds no valid call.

    Args:
        text: Complete model output.
        tool_names: Names of the tools declared in the request.

    Returns:
        Optional[List[Segment]]: Segments in order, or None when the output must be treated as plain text
        (no tools declared, no tag, an unclosed or malformed block, an unknown tool, non-object arguments).
    """
    if not tool_names or TOOL_OPEN not in text:
        return None
    matches = list(_CALL_PATTERN.finditer(text))
    if not matches or text.count(TOOL_OPEN) != len(matches):
        return None
    segments: List[Segment] = []
    position = 0
    for match in matches:
        call = _load_call(match.group(1), tool_names)
        if call is None:
            return None
        before = text[position:match.start()]
        if before.strip():
            segments.append(Segment(kind="text", text=before))
        segments.append(Segment(kind="tool_use", name=call[0], input=call[1]))
        position = match.end()
    after = text[position:]
    if after.strip():
        segments.append(Segment(kind="text", text=after))
    return segments


class StreamSegmenter:
    """Splits a token stream into text that is safe to emit now and a held tail that may be tool calls."""

    def __init__(self, enabled: bool) -> None:
        """Create a segmenter.

        Args:
            enabled: False passes every chunk through (no tools declared).
        """
        self._enabled = enabled
        self._pending = ""
        self._tail = ""
        self._in_call = False

    def feed(self, chunk: str) -> str:
        """Accept one chunk of model text.

        Args:
            chunk: Newly decoded text.

        Returns:
            str: Text that can be emitted immediately; a possible tag prefix at the end of the chunk is held back.
        """
        if not self._enabled:
            return chunk
        if self._in_call:
            self._tail += chunk
            return ""
        buffer = self._pending + chunk
        index = buffer.find(TOOL_OPEN)
        if index >= 0:
            self._in_call = True
            self._tail = buffer[index:]
            self._pending = ""
            return buffer[:index]
        hold = 0
        for size in range(min(len(TOOL_OPEN) - 1, len(buffer)), 0, -1):
            if buffer.endswith(TOOL_OPEN[:size]):
                hold = size
                break
        self._pending = buffer[len(buffer) - hold:] if hold else ""
        return buffer[:len(buffer) - hold]

    def finish(self) -> Tuple[str, str]:
        """Close the stream.

        Returns:
            Tuple[str, str]: Held-back text that turned out not to be a tag, and the tail starting at the
            first tool_call tag ("" if none was seen).
        """
        flushed, self._pending = self._pending, ""
        return flushed, self._tail
