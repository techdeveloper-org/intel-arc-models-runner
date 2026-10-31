"""Request validation, parsed shapes and error envelopes for the Anthropic Messages API subset.

The subset is what Claude Code needs from POST /v1/messages and POST /v1/messages/count_tokens.
Validation is written by hand instead of with a strict model so that unknown fields (thinking,
output_config, context_management, cache_control, beta tool fields) are ignored and every
failure leaves as an Anthropic-shaped 400 rather than a framework 422.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from arc_runner.core.exceptions import ArcRunnerError

MAX_SUPPORTED_MAX_TOKENS: int = 32768
MAX_SUPPORTED_TEMPERATURE: float = 2.0
VALID_ROLES = ("user", "assistant", "system")
IGNORED_BLOCK_TYPES = ("thinking", "redacted_thinking")
REFUSED_BLOCK_TYPES = ("image", "document")

_ERROR_TYPE_BY_STATUS: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def error_type_for_status(status: int) -> str:
    """Map an HTTP status to the Anthropic error type.

    Args:
        status: HTTP status code.

    Returns:
        str: The documented type for the status, invalid_request_error for other 4xx, api_error otherwise.
    """
    if status in _ERROR_TYPE_BY_STATUS:
        return _ERROR_TYPE_BY_STATUS[status]
    return "invalid_request_error" if 400 <= status < 500 else "api_error"


def new_request_id() -> str:
    """Return a fresh request id of the form req_<24 hex>."""
    return f"req_{uuid.uuid4().hex[:24]}"


def error_envelope(error_type: str, message: str, request_id: str) -> Dict[str, Any]:
    """Build the Anthropic error body.

    Args:
        error_type: Anthropic error type.
        message: Human-readable message.
        request_id: Request id echoed in the body and the request-id header.

    Returns:
        Dict[str, Any]: {"type": "error", "error": {"type", "message"}, "request_id"}.
    """
    return {"type": "error", "error": {"type": error_type, "message": message}, "request_id": request_id}


class AnthropicRequestError(ArcRunnerError):
    """Raised when a request cannot be served; carries the HTTP status and Anthropic error type."""

    def __init__(self, message: str, status_code: int = 400, error_type: Optional[str] = None) -> None:
        """Initialise the error.

        Args:
            message: Message shown to the client.
            status_code: HTTP status code.
            error_type: Anthropic error type; derived from the status when omitted.
        """
        super().__init__(message, "ANTHROPIC_REQUEST_ERROR", status_code)
        self.error_type = error_type or error_type_for_status(status_code)


@dataclass
class Block:
    """One renderable content block of a message."""

    kind: str
    text: str = ""
    tool_id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)
    content: str = ""
    is_error: bool = False


@dataclass
class Turn:
    """One conversation message with its renderable blocks."""

    role: str
    blocks: List[Block]


@dataclass
class ToolSpec:
    """A tool declared by the client."""

    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ParsedRequest:
    """A validated Messages or count_tokens request."""

    model: str
    max_tokens: Optional[int]
    messages: List[Turn]
    system: str
    stream: bool
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    stop_sequences: List[str]
    tools: List[ToolSpec]
    tool_choice_none: bool


def _is_number(value: Any) -> bool:
    """Return True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text_of_blocks(blocks: Any, path: str) -> str:
    """Join the text blocks of a system or tool_result content list.

    Args:
        blocks: List of content blocks.
        path: Field path used in error messages.

    Returns:
        str: Texts joined by newlines.

    Raises:
        AnthropicRequestError: If a block is not text.
    """
    texts: List[str] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise AnthropicRequestError(f"{path}.{index}: Input should be an object")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind in REFUSED_BLOCK_TYPES:
            raise AnthropicRequestError(f"{path}.{index}.type: '{kind}' content is not supported by this runner (text only)")
        else:
            raise AnthropicRequestError(f"{path}.{index}.type: unsupported block type '{kind}'; only text blocks are accepted here")
    return "\n".join(texts)


def _parse_system(value: Any) -> str:
    """Parse the system field (string or list of text blocks).

    Args:
        value: Raw system value.

    Returns:
        str: The system prompt, empty when absent.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(_text_of_blocks([block], f"system.{i}") for i, block in enumerate(value))
    raise AnthropicRequestError("system: Input should be a string or a list of text blocks")


def _parse_block(block: Any, path: str) -> Optional[Block]:
    """Parse one message content block.

    Args:
        block: Raw block.
        path: Field path used in error messages.

    Returns:
        Optional[Block]: The block, or None for blocks that are intentionally dropped.

    Raises:
        AnthropicRequestError: If the block is malformed or of an unsupported type.
    """
    if not isinstance(block, dict):
        raise AnthropicRequestError(f"{path}: Input should be an object")
    kind = block.get("type")
    if kind == "text":
        if not isinstance(block.get("text"), str):
            raise AnthropicRequestError(f"{path}.text: Input should be a valid string")
        return Block(kind="text", text=block["text"])
    if kind == "tool_use":
        if not isinstance(block.get("id"), str) or not block["id"]:
            raise AnthropicRequestError(f"{path}.id: Field required")
        if not isinstance(block.get("name"), str) or not block["name"]:
            raise AnthropicRequestError(f"{path}.name: Field required")
        tool_input = block.get("input", {})
        if not isinstance(tool_input, dict):
            raise AnthropicRequestError(f"{path}.input: Input should be an object")
        return Block(kind="tool_use", tool_id=block["id"], name=block["name"], input=tool_input)
    if kind == "tool_result":
        if not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]:
            raise AnthropicRequestError(f"{path}.tool_use_id: Field required")
        is_error = block.get("is_error", False)
        if not isinstance(is_error, bool):
            raise AnthropicRequestError(f"{path}.is_error: Input should be a valid boolean")
        content = block.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = _text_of_blocks(content, f"{path}.content")
        else:
            raise AnthropicRequestError(f"{path}.content: Input should be a string or a list of text blocks")
        return Block(kind="tool_result", tool_id=block["tool_use_id"], content=text, is_error=is_error)
    if kind in IGNORED_BLOCK_TYPES:
        return None
    if kind in REFUSED_BLOCK_TYPES:
        raise AnthropicRequestError(f"{path}.type: '{kind}' content is not supported by this runner (text only)")
    raise AnthropicRequestError(f"{path}.type: unsupported content block type '{kind}'")


def _parse_messages(value: Any) -> List[Turn]:
    """Parse the messages list.

    Args:
        value: Raw messages value.

    Returns:
        List[Turn]: Validated turns.

    Raises:
        AnthropicRequestError: If the list is missing, empty or malformed.
    """
    if value is None:
        raise AnthropicRequestError("messages: Field required")
    if not isinstance(value, list) or not value:
        raise AnthropicRequestError("messages: Input should be a non-empty list")
    turns: List[Turn] = []
    for index, message in enumerate(value):
        path = f"messages.{index}"
        if not isinstance(message, dict):
            raise AnthropicRequestError(f"{path}: Input should be an object")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise AnthropicRequestError(f"{path}.role: Input should be 'user', 'assistant' or 'system'")
        content = message.get("content")
        if isinstance(content, str):
            blocks = [Block(kind="text", text=content)]
        elif isinstance(content, list):
            blocks = [b for b in (_parse_block(item, f"{path}.content.{i}") for i, item in enumerate(content)) if b is not None]
        else:
            raise AnthropicRequestError(f"{path}.content: Input should be a string or a list of content blocks")
        turns.append(Turn(role=role, blocks=blocks))
    return turns


def _parse_tools(value: Any) -> List[ToolSpec]:
    """Parse the tools list.

    Args:
        value: Raw tools value.

    Returns:
        List[ToolSpec]: Declared tools in request order.

    Raises:
        AnthropicRequestError: If a tool is malformed.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnthropicRequestError("tools: Input should be a list")
    tools: List[ToolSpec] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
            raise AnthropicRequestError(f"tools.{index}.name: Field required")
        description = tool.get("description") or ""
        if not isinstance(description, str):
            raise AnthropicRequestError(f"tools.{index}.description: Input should be a valid string")
        schema = tool.get("input_schema", {})
        if schema is None:
            schema = {}
        if not isinstance(schema, dict):
            raise AnthropicRequestError(f"tools.{index}.input_schema: Input should be an object")
        tools.append(ToolSpec(name=tool["name"], description=description, input_schema=schema))
    return tools


def _parse_max_tokens(payload: Dict[str, Any], required: bool) -> Optional[int]:
    """Validate max_tokens.

    Args:
        payload: Request body.
        required: Whether the field must be present.

    Returns:
        Optional[int]: The value, or None when absent and not required.

    Raises:
        AnthropicRequestError: If missing, not an integer, below 1 or above the supported maximum.
    """
    value = payload.get("max_tokens")
    if value is None:
        if required:
            raise AnthropicRequestError("max_tokens: Field required")
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnthropicRequestError("max_tokens: Input should be a valid integer")
    if value < 1:
        raise AnthropicRequestError("max_tokens: Input should be greater than or equal to 1")
    if value > MAX_SUPPORTED_MAX_TOKENS:
        raise AnthropicRequestError(
            f"max_tokens: {value} exceeds the maximum of {MAX_SUPPORTED_MAX_TOKENS} this runner supports; "
            "lower it (Claude Code: CLAUDE_CODE_MAX_OUTPUT_TOKENS)"
        )
    return value


def _parse_sampling(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate temperature, top_p, top_k and stop_sequences.

    Args:
        payload: Request body.

    Returns:
        Dict[str, Any]: Validated values keyed by field name; absent fields are None or [].

    Raises:
        AnthropicRequestError: If a value is mistyped or out of range.
    """
    temperature = payload.get("temperature")
    if temperature is not None and (not _is_number(temperature) or not 0.0 <= temperature <= MAX_SUPPORTED_TEMPERATURE):
        raise AnthropicRequestError(f"temperature: Input should be a number from 0 to {MAX_SUPPORTED_TEMPERATURE}")
    top_p = payload.get("top_p")
    if top_p is not None and (not _is_number(top_p) or not 0.0 < top_p <= 1.0):
        raise AnthropicRequestError("top_p: Input should be a number greater than 0 and at most 1")
    top_k = payload.get("top_k")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise AnthropicRequestError("top_k: Input should be an integer greater than or equal to 0")
        top_k = top_k or None
    stops = payload.get("stop_sequences")
    if stops is None:
        stops = []
    if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
        raise AnthropicRequestError("stop_sequences: Input should be a list of non-empty strings")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k, "stop_sequences": list(stops)}


def parse_messages_request(payload: Any, require_max_tokens: bool = True) -> ParsedRequest:
    """Validate a /v1/messages or /v1/messages/count_tokens body.

    Unknown fields are ignored on purpose; Claude Code sends beta fields that change between releases.

    Args:
        payload: Decoded JSON body.
        require_max_tokens: True for /v1/messages, False for count_tokens.

    Returns:
        ParsedRequest: The validated request.

    Raises:
        AnthropicRequestError: With status 400 for any invalid field.
    """
    if not isinstance(payload, dict):
        raise AnthropicRequestError("Request body must be a JSON object")
    model = payload.get("model")
    if model is None:
        raise AnthropicRequestError("model: Field required")
    if not isinstance(model, str) or not model.strip():
        raise AnthropicRequestError("model: Input should be a non-empty string")
    max_tokens = _parse_max_tokens(payload, require_max_tokens)
    messages = _parse_messages(payload.get("messages"))
    stream = payload.get("stream", False)
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise AnthropicRequestError("stream: Input should be a valid boolean")
    sampling = _parse_sampling(payload)
    tool_choice = payload.get("tool_choice")
    return ParsedRequest(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=_parse_system(payload.get("system")),
        stream=stream,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        stop_sequences=sampling["stop_sequences"],
        tools=_parse_tools(payload.get("tools")),
        tool_choice_none=isinstance(tool_choice, dict) and tool_choice.get("type") == "none",
    )
