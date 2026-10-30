"""Unit tests for Anthropic request validation, prompt rendering, tool-call parsing and error envelopes."""

import json
import re
from typing import Any, Dict, List

import pytest

from arc_runner.core.exceptions import (
    ArcRunnerError,
    InferenceTimeoutError,
    VRAMExhaustedError,
)
from arc_runner.engine.models import GenerationConfig
from arc_runner.server.anthropic_convert import (
    StreamSegmenter,
    build_prompt,
    new_message_id,
    new_tool_id,
    parse_tool_calls,
    render_chatml,
    render_tools_instruction,
)
from arc_runner.server.anthropic_schemas import (
    MAX_SUPPORTED_MAX_TOKENS,
    AnthropicRequestError,
    error_envelope,
    error_type_for_status,
    parse_messages_request,
)
from tests.anthropic_support import MODEL, TOOL_WEATHER


def _payload(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not None}


def test_max_supported_max_tokens_matches_generation_config_limit() -> None:
    """The route limit is the engine config limit, so no accepted request can fail config validation."""
    GenerationConfig(max_new_tokens=MAX_SUPPORTED_MAX_TOKENS)
    with pytest.raises(Exception):
        GenerationConfig(max_new_tokens=MAX_SUPPORTED_MAX_TOKENS + 1)


@pytest.mark.parametrize("bad", [
    None, [], "x", 5, _payload(messages=[]), _payload(messages=None), _payload(messages="hi"), _payload(messages={}),
    _payload(model=None), _payload(model=""), _payload(model=3),
])
def test_parse_messages_request_non_object_or_empty_messages_raises(bad: Any) -> None:
    """Anything but an object with a model and a non-empty messages list is a 400."""
    with pytest.raises(AnthropicRequestError) as info:
        parse_messages_request(bad)
    assert info.value.status_code == 400


@pytest.mark.parametrize("value", [None, 0, -1, True, "5", 1.5, MAX_SUPPORTED_MAX_TOKENS + 1])
def test_parse_messages_request_bad_max_tokens_raises(value: Any) -> None:
    """max_tokens must be an integer from 1 to the supported maximum, and is required."""
    payload = _payload()
    payload.pop("max_tokens")
    if value is not None:
        payload["max_tokens"] = value
    with pytest.raises(AnthropicRequestError) as info:
        parse_messages_request(payload)
    assert "max_tokens" in info.value.message


def test_parse_messages_request_count_tokens_does_not_require_max_tokens() -> None:
    """count_tokens requests carry no max_tokens."""
    payload = _payload()
    payload.pop("max_tokens")
    assert parse_messages_request(payload, require_max_tokens=False).max_tokens is None


def test_parse_messages_request_unknown_fields_are_ignored() -> None:
    """Every beta field Claude Code may send is accepted."""
    parsed = parse_messages_request(_payload(
        thinking={"type": "adaptive"}, output_config={"effort": "high"}, context_management={"edits": []},
        metadata={"user_id": "u"}, service_tier="auto", future_field=[1, 2],
        system=[{"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}}],
        tools=[{**TOOL_WEATHER, "cache_control": {"type": "ephemeral"}, "defer_loading": True, "strict": True}],
    ))
    assert parsed.system == "A" and parsed.tools[0].name == "get_weather"


def test_parse_messages_request_system_blocks_are_joined_in_order() -> None:
    """Block-form system prompts (attribution block first) are joined with blank lines."""
    parsed = parse_messages_request(_payload(system=[{"type": "text", "text": "x-attr: 1"}, {"type": "text", "text": "You are X."}]))
    assert parsed.system == "x-attr: 1\n\nYou are X."


def test_parse_messages_request_system_non_text_block_raises() -> None:
    """A system block that is not text cannot be rendered and is refused."""
    with pytest.raises(AnthropicRequestError):
        parse_messages_request(_payload(system=[{"type": "image", "source": {}}]))


@pytest.mark.parametrize("role", ["tool", "developer", "", None, 3])
def test_parse_messages_request_bad_role_raises(role: Any) -> None:
    """Only user, assistant and (mid-conversation) system roles exist."""
    with pytest.raises(AnthropicRequestError) as info:
        parse_messages_request(_payload(messages=[{"role": role, "content": "x"}]))
    assert "role" in info.value.message


def test_parse_messages_request_role_system_is_accepted() -> None:
    """Claude Code can send role:system entries inside messages for models it treats as new."""
    parsed = parse_messages_request(_payload(messages=[{"role": "user", "content": "a"}, {"role": "system", "content": "mid"}, {"role": "user", "content": "b"}]))
    assert [t.role for t in parsed.messages] == ["user", "system", "user"]


@pytest.mark.parametrize("block_type", ["image", "document"])
def test_parse_messages_request_image_and_document_blocks_raise(block_type: str) -> None:
    """No vision path exists, so image and document blocks are a 400, never silently dropped."""
    with pytest.raises(AnthropicRequestError) as info:
        parse_messages_request(_payload(messages=[{"role": "user", "content": [{"type": block_type, "source": {}}]}]))
    assert block_type in info.value.message and info.value.error_type == "invalid_request_error"


def test_parse_messages_request_image_inside_tool_result_raises() -> None:
    """An image nested in a tool_result is refused as well."""
    content = [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "image", "source": {}}]}]
    with pytest.raises(AnthropicRequestError):
        parse_messages_request(_payload(messages=[{"role": "user", "content": content}]))


def test_parse_messages_request_thinking_blocks_in_history_are_dropped() -> None:
    """Thinking blocks carry signatures only the real API can use; they are not rendered."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"}, {"type": "redacted_thinking", "data": "x"}, {"type": "text", "text": "a"}]}, {"role": "user", "content": "q2"}]
    parsed = parse_messages_request(_payload(messages=messages))
    assert [b.kind for b in parsed.messages[1].blocks] == ["text"]


def test_parse_messages_request_unknown_block_type_raises() -> None:
    """A block type the runner cannot render is refused instead of dropped."""
    with pytest.raises(AnthropicRequestError) as info:
        parse_messages_request(_payload(messages=[{"role": "user", "content": [{"type": "mystery"}]}]))
    assert "mystery" in info.value.message


@pytest.mark.parametrize("block", [
    {"type": "text"}, {"type": "text", "text": 3},
    {"type": "tool_use", "name": "n", "input": {}}, {"type": "tool_use", "id": "i", "input": {}},
    {"type": "tool_use", "id": "i", "name": "n", "input": []}, {"type": "tool_result"},
    {"type": "tool_result", "tool_use_id": "t", "is_error": "yes"},
])
def test_parse_messages_request_malformed_blocks_raise(block: Dict[str, Any]) -> None:
    """Required block fields are checked."""
    with pytest.raises(AnthropicRequestError):
        parse_messages_request(_payload(messages=[{"role": "user", "content": [block]}]))


def test_parse_messages_request_tool_result_content_forms() -> None:
    """tool_result content may be a string, text blocks or absent."""
    blocks = [
        {"type": "tool_result", "tool_use_id": "a", "content": "plain"},
        {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "is_error": True},
        {"type": "tool_result", "tool_use_id": "c"},
    ]
    results = parse_messages_request(_payload(messages=[{"role": "user", "content": blocks}])).messages[0].blocks
    assert [(b.tool_id, b.content, b.is_error) for b in results] == [("a", "plain", False), ("b", "one\ntwo", True), ("c", "", False)]


@pytest.mark.parametrize("field,value", [
    ("temperature", -0.1), ("temperature", 2.5), ("temperature", True), ("temperature", "1"),
    ("top_p", 0), ("top_p", 1.5), ("top_k", -1), ("top_k", 1.5),
    ("stop_sequences", "x"), ("stop_sequences", [""]), ("stop_sequences", [1]),
    ("stream", "yes"), ("tools", {}), ("tools", [{"description": "x"}]), ("tools", [{"name": "n", "input_schema": []}]),
])
def test_parse_messages_request_bad_optional_fields_raise(field: str, value: Any) -> None:
    """Out-of-range or mistyped sampling, stream and tool fields are 400s."""
    with pytest.raises(AnthropicRequestError):
        parse_messages_request(_payload(**{field: value}))


def test_parse_messages_request_top_k_zero_means_unset() -> None:
    """top_k of 0 is the API's 'disabled', not a config value."""
    assert parse_messages_request(_payload(top_k=0)).top_k is None


def test_parse_messages_request_tool_choice_none_is_flagged_and_others_ignored() -> None:
    """Only tool_choice none changes behaviour."""
    assert parse_messages_request(_payload(tool_choice={"type": "none"}, tools=[TOOL_WEATHER])).tool_choice_none is True
    assert parse_messages_request(_payload(tool_choice={"type": "any"}, tools=[TOOL_WEATHER])).tool_choice_none is False


def test_error_type_for_status_covers_documented_statuses() -> None:
    """Each status maps to the documented Anthropic error type."""
    expected = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
                413: "request_too_large", 429: "rate_limit_error", 500: "api_error", 504: "timeout_error", 529: "overloaded_error"}
    assert {s: error_type_for_status(s) for s in expected} == expected
    assert error_type_for_status(422) == "invalid_request_error" and error_type_for_status(507) == "api_error"


def test_error_envelope_has_exact_shape() -> None:
    """type, error{type,message} and request_id, nothing else."""
    assert error_envelope("api_error", "boom", "req_1") == {"type": "error", "error": {"type": "api_error", "message": "boom"}, "request_id": "req_1"}


def test_typed_arc_errors_keep_their_status_in_request_error_mapping() -> None:
    """Typed engine errors expose statuses the route maps (507 memory, 504 timeout)."""
    assert error_type_for_status(VRAMExhaustedError(10, 5).status_code) == "api_error"
    assert error_type_for_status(InferenceTimeoutError(2.0, 1.0).status_code) == "timeout_error"
    assert issubclass(AnthropicRequestError, ArcRunnerError)


def test_ids_have_documented_prefixes_and_are_unique() -> None:
    """msg_ and toolu_ ids are well formed and do not repeat."""
    ids = {new_tool_id() for _ in range(200)}
    assert len(ids) == 200 and all(re.fullmatch(r"toolu_[0-9a-f]{24}", i) for i in ids)
    assert re.fullmatch(r"msg_[0-9a-f]{24}", new_message_id())


def test_render_tools_instruction_is_deterministic_and_lists_every_tool() -> None:
    """Same tools, same text; each tool's name, description and schema appear verbatim."""
    parsed = parse_messages_request(_payload(tools=[TOOL_WEATHER, {"name": "b_tool", "description": "B", "input_schema": {"type": "object"}}]))
    first = render_tools_instruction(parsed.tools)
    assert first == render_tools_instruction(parsed.tools)
    assert first.index("get_weather") < first.index("b_tool")
    assert json.dumps(TOOL_WEATHER["input_schema"], sort_keys=True, separators=(",", ":")) in first
    assert "<tool_call>" in first and "</tool_call>" in first


def test_render_chatml_matches_openai_route_template() -> None:
    """Plain conversations use the same ChatML layout as /v1/chat/completions."""
    parsed = parse_messages_request(_payload(system="S", messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "v"}]))
    assert render_chatml(parsed.system, parsed.messages) == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n"
        "<|im_start|>user\nv<|im_end|>\n<|im_start|>assistant\n"
    )


def test_render_chatml_tool_round_trip_renders_call_and_response() -> None:
    """History tool_use becomes a tool_call block and tool_result a tool_response block with id, name and error flag."""
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Pune"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "31C", "is_error": False}, {"type": "text", "text": "and?"}]},
    ]
    parsed = parse_messages_request(_payload(messages=messages, tools=[TOOL_WEATHER]))
    text = render_chatml(parsed.system, parsed.messages)
    assert '<tool_call>{"name": "get_weather", "arguments": {"city": "Pune"}}</tool_call>' in text
    assert '<tool_response id="toolu_1" name="get_weather">\n31C\n</tool_response>\nand?' in text
    error = parse_messages_request(_payload(messages=[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "zz", "content": "bad", "is_error": True}]}]))
    assert '<tool_response id="zz" is_error="true">' in render_chatml("", error.messages)


def test_render_chatml_duplicate_tool_use_ids_in_history_do_not_crash() -> None:
    """Duplicate ids in client history are rendered, the later name wins, nothing is rejected."""
    use = {"type": "tool_use", "id": "dup", "name": "a", "input": {}}
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [use, {**use, "name": "b"}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "dup", "content": "r"}]}]
    parsed = parse_messages_request(_payload(messages=messages))
    assert 'name="b"' in render_chatml("", parsed.messages)


def test_build_prompt_uses_engine_template_hook_when_it_supports_tools() -> None:
    """An engine exposing render_chat_with_tools renders the prompt; its tools arrive in function form."""
    seen: Dict[str, Any] = {}

    class Hooked:
        def render_chat_with_tools(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
            seen.update(messages=messages, tools=tools)
            return "TEMPLATED"

    parsed = parse_messages_request(_payload(system="S", tools=[TOOL_WEATHER]))
    assert build_prompt(Hooked(), parsed) == "TEMPLATED"
    assert seen["tools"] == [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": TOOL_WEATHER["input_schema"]}}]
    assert seen["messages"][0] == {"role": "system", "content": "S"}


def test_build_prompt_hook_returning_none_falls_back_to_instruction() -> None:
    """A template without tool support yields None and the deterministic instruction is used."""

    class NoTools:
        def render_chat_with_tools(self, messages: Any, tools: Any) -> None:
            return None

    parsed = parse_messages_request(_payload(tools=[TOOL_WEATHER]))
    assert "get_weather" in build_prompt(NoTools(), parsed) and "<tool_call>" in build_prompt(NoTools(), parsed)


def test_build_prompt_without_hook_or_tools_is_plain_chatml() -> None:
    """No tools means no instruction at all, with or without a hook."""
    parsed = parse_messages_request(_payload())
    assert "<tool_call>" not in build_prompt(object(), parsed)


def test_build_prompt_tool_choice_none_omits_the_instruction() -> None:
    """tool_choice none removes the tool instruction."""
    parsed = parse_messages_request(_payload(tools=[TOOL_WEATHER], tool_choice={"type": "none"}))
    assert "get_weather" not in build_prompt(object(), parsed)


NAMES = {"get_weather", "read_file"}
CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Pune"}}</tool_call>'


def test_parse_tool_calls_single_valid_call_returns_tool_use_segment() -> None:
    """The documented format yields one tool_use with a dict input."""
    segments = parse_tool_calls(CALL, NAMES)
    assert segments is not None and [(s.kind, s.name, s.input) for s in segments] == [("tool_use", "get_weather", {"city": "Pune"})]


def test_parse_tool_calls_keeps_text_before_between_and_after() -> None:
    """Prose around calls is preserved in order; whitespace-only gaps are dropped."""
    text = f"Checking.\n{CALL}\n{CALL.replace('Pune', 'Delhi')}\nDone."
    segments = parse_tool_calls(text, NAMES)
    assert [s.kind for s in segments] == ["text", "tool_use", "tool_use", "text"]
    assert segments[0].text == "Checking.\n" and segments[3].text == "\nDone."
    assert segments[2].input == {"city": "Delhi"}


@pytest.mark.parametrize("text", [
    "plain answer with no markup",
    "I would call get_weather with city Pune.",
    '{"name": "get_weather", "arguments": {"city": "Pune"}}',
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Pune"}}',
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Pu',
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Pune"}}</tool_call><tool_call>{"name": "get_weather", "argum',
    '<tool_call>{"name": "nope", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "get_weather", "arguments": []}</tool_call>',
    '<tool_call>{"name": "get_weather", "arguments": "{}"}</tool_call>',
    '<tool_call>{"name": "get_weather"}</tool_call>',
    '<tool_call>{"arguments": {}}</tool_call>',
    '<tool_call>{"name": 3, "arguments": {}}</tool_call>',
    '<tool_call>{"name": "get_weather", "arguments": {}, "extra": 1}</tool_call>',
    '<tool_call>[1, 2]</tool_call>',
    '<tool_call>not json</tool_call>',
    '<tool_call></tool_call>',
    '<tool_call>{"name": "get_weather", "arguments": {"a": NaN}}</tool_call>',
    '<tool_call><tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
    '</tool_call><tool_call>',
])
def test_parse_tool_calls_invalid_or_prose_returns_none(text: str) -> None:
    """Anything that is not exactly the documented format is plain text; no call is invented."""
    assert parse_tool_calls(text, NAMES) is None


def test_parse_tool_calls_one_bad_call_voids_the_whole_response() -> None:
    """All-or-nothing: a valid call next to a bad one is not executed."""
    text = CALL + '<tool_call>{"name": "ghost", "arguments": {}}</tool_call>'
    assert parse_tool_calls(text, NAMES) is None


def test_parse_tool_calls_empty_tool_set_never_parses() -> None:
    """With no declared tools nothing can be a call."""
    assert parse_tool_calls(CALL, set()) is None


def test_parse_tool_calls_empty_arguments_object_is_valid() -> None:
    """A call with no arguments carries an empty object."""
    segments = parse_tool_calls('<tool_call>{"name": "read_file", "arguments": {}}</tool_call>', NAMES)
    assert segments[0].input == {}


def test_parse_tool_calls_identical_calls_are_two_segments() -> None:
    """Repeated identical calls stay separate (ids are assigned by the route, unique per call)."""
    assert len(parse_tool_calls(CALL + CALL, NAMES)) == 2


def _feed_all(text: str, size: int, enabled: bool = True) -> "tuple[str, str]":
    seg = StreamSegmenter(enabled=enabled)
    safe = ""
    for i in range(0, len(text), size):
        safe += seg.feed(text[i:i + size])
    flushed, tail = seg.finish()
    return safe + flushed, tail


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 1000])
def test_stream_segmenter_chunking_does_not_change_the_split(size: int) -> None:
    """Text before the first call tag is emitted, the rest is held, whatever the chunk size."""
    text = f"Sure. {CALL} tail"
    safe, tail = _feed_all(text, size)
    assert safe == "Sure. " and tail == f"{CALL} tail"


@pytest.mark.parametrize("size", [1, 2, 4])
def test_stream_segmenter_partial_tag_prefix_is_released_when_it_is_not_a_tag(size: int) -> None:
    """A '<' that never becomes <tool_call> comes out as ordinary text, in order, nothing lost."""
    for text in ("a < b and <tool and <", "<tool_cal", "x<", "<<tool_call>y"):
        safe, tail = _feed_all(text, size)
        assert safe + tail == text
    assert _feed_all("<<tool_call>y", size) == ("<", "<tool_call>y")


def test_stream_segmenter_disabled_passes_everything_through() -> None:
    """Without tools the tag is ordinary text and nothing is held back."""
    seg = StreamSegmenter(enabled=False)
    assert seg.feed("<tool_call>") == "<tool_call>" and seg.finish() == ("", "")
