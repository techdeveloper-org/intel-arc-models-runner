"""HTTP-level tests for POST /v1/messages, /v1/messages/count_tokens and the /api/hello probe (fake engines only)."""

import hmac
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from arc_runner.core.exceptions import ArcRunnerError, ContextWindowExceededError, InferenceTimeoutError, VRAMExhaustedError
from arc_runner.engine.models import EngineTelemetry
from arc_runner.server import routes_anthropic
from tests.anthropic_support import (
    CALL_PIECES,
    FIXTURES,
    MODEL,
    TOOL_WEATHER,
    ScriptedEngine,
    body,
    make_client,
    measured,
    normalize_ids,
    parse_sse,
)

MSG = "/v1/messages"


def _post(client: TestClient, payload: Dict[str, Any], **kwargs: Any) -> Any:
    return client.post(MSG, json=payload, **kwargs)


def _assert_error(response: Any, status: int, error_type: str, prefix: str = "") -> Dict[str, Any]:
    assert response.status_code == status, response.text
    data = response.json()
    assert set(data) == {"type", "error", "request_id"} and data["type"] == "error"
    assert data["error"]["type"] == error_type and data["error"]["message"].startswith(prefix), data
    assert data["request_id"].startswith("req_") and response.headers["request-id"] == data["request_id"]
    return data


@pytest.fixture
def engine() -> ScriptedEngine:
    """A scripted engine that answers 'Hello there' with no measured counts."""
    return ScriptedEngine()


@pytest.fixture
def client(tmp_path: Path, engine: ScriptedEngine) -> TestClient:
    """The real app over the scripted engine."""
    return make_client(tmp_path, engine)


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("messages", [[], None, "hi", {}])
def test_messages_empty_or_invalid_messages_returns_400_envelope(client: TestClient, messages: Any) -> None:
    """An empty or non-list messages field is an Anthropic invalid_request_error, not FastAPI's 422."""
    _assert_error(_post(client, body(messages=messages)), 400, "invalid_request_error", "messages")


def test_messages_bad_role_returns_400(client: TestClient) -> None:
    """A role outside user, assistant and system is refused."""
    _assert_error(_post(client, body(messages=[{"role": "tool", "content": "x"}])), 400, "invalid_request_error", "messages.0.role")


@pytest.mark.parametrize("value", [None, 0, -5, "9", True, 32769, 100000])
def test_messages_bad_max_tokens_returns_400(client: TestClient, engine: ScriptedEngine, value: Any) -> None:
    """max_tokens is required, positive, an integer and within what the engine config accepts; the engine is never called."""
    _assert_error(_post(client, body(max_tokens=value)), 400, "invalid_request_error", "max_tokens")
    assert engine.prompts == []


def test_messages_huge_max_tokens_message_names_the_limit(client: TestClient) -> None:
    """The refusal says what the supported maximum is rather than clamping silently."""
    data = _post(client, body(max_tokens=100000)).json()
    assert "32768" in data["error"]["message"]


def test_messages_max_tokens_at_limit_is_accepted_and_passed_to_engine(client: TestClient, engine: ScriptedEngine) -> None:
    """The boundary value works and reaches the engine config unchanged."""
    assert _post(client, body(max_tokens=32768)).status_code == 200
    assert engine.configs[-1].max_new_tokens == 32768


def test_messages_unknown_fields_are_tolerated(client: TestClient, engine: ScriptedEngine) -> None:
    """thinking, output_config, context_management, metadata, cache_control and beta tool fields never cause a 400."""
    payload = body(
        thinking={"type": "adaptive"}, output_config={"effort": "high", "format": {"type": "json_schema"}},
        context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}, metadata={"user_id": "u1"}, speed="fast",
        system=[{"type": "text", "text": "x-anthropic-billing-header: cc_version=1", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "You are a coding agent."}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "Say hi", "cache_control": {"type": "ephemeral"}}]}],
        tools=[{**TOOL_WEATHER, "cache_control": {"type": "ephemeral"}, "strict": True, "defer_loading": False, "eager_input_streaming": True}],
    )
    response = _post(client, payload, headers={"anthropic-version": "2023-06-01", "anthropic-beta": "made-up-beta-1,another-2", "x-claude-code-session-id": "s"})
    assert response.status_code == 200, response.text
    prompt = engine.prompts[-1]
    assert "x-anthropic-billing-header" in prompt and "You are a coding agent." in prompt and "Say hi" in prompt


def test_messages_content_string_and_blocks_render_the_same_text(tmp_path: Path) -> None:
    """A string content and an equivalent single text block produce the same prompt."""
    prompts = []
    for index, content in enumerate(["Say hi", [{"type": "text", "text": "Say hi"}]]):
        engine = ScriptedEngine()
        sub = tmp_path / str(index)
        sub.mkdir()
        assert _post(make_client(sub, engine), body(messages=[{"role": "user", "content": content}])).status_code == 200
        prompts.append(engine.prompts[-1])
    assert prompts[0] == prompts[1]


def test_messages_system_string_is_the_first_chatml_turn(client: TestClient, engine: ScriptedEngine) -> None:
    """A string system prompt becomes the leading system turn."""
    _post(client, body(system="Be brief."))
    assert engine.prompts[-1].startswith("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nSay hi<|im_end|>")


def test_messages_role_system_inside_messages_is_accepted_in_place(client: TestClient, engine: ScriptedEngine) -> None:
    """Claude Code's mid-conversation system entries render as system turns where they occur."""
    messages = [{"role": "user", "content": "a"}, {"role": "system", "content": "reminder"}, {"role": "user", "content": "b"}]
    assert _post(client, body(messages=messages)).status_code == 200
    assert "<|im_start|>user\na<|im_end|>\n<|im_start|>system\nreminder<|im_end|>\n<|im_start|>user\nb" in engine.prompts[-1]


@pytest.mark.parametrize("content", [
    [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}],
    [{"type": "text", "text": "look"}, {"type": "image", "source": {"type": "url", "url": "http://x/y.png"}}],
    [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {}}]}],
    [{"type": "document", "source": {}}],
])
def test_messages_image_or_document_returns_400_and_engine_not_called(client: TestClient, engine: ScriptedEngine, content: Any) -> None:
    """No vision path exists, so the request is refused before any generation."""
    _assert_error(_post(client, body(messages=[{"role": "user", "content": content}])), 400, "invalid_request_error")
    assert engine.prompts == []


def test_messages_thinking_blocks_in_history_are_accepted(client: TestClient) -> None:
    """Replayed thinking blocks do not fail the request."""
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "text", "text": "a"}]}, {"role": "user", "content": "q2"}]
    assert _post(client, body(messages=history)).status_code == 200


@pytest.mark.parametrize("raw", [b"", b"{", b"[1]", b"\xff\xfe", b"null"])
def test_messages_invalid_json_body_returns_400(client: TestClient, raw: bytes) -> None:
    """Garbage bodies get an Anthropic-shaped 400."""
    _assert_error(client.post(MSG, content=raw, headers={"content-type": "application/json"}), 400, "invalid_request_error")


def test_messages_oversized_body_returns_413(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bodies above the policy limit are refused with request_too_large."""
    monkeypatch.setattr(routes_anthropic, "MAX_REQUEST_BYTES", 50)
    _assert_error(_post(client, body()), 413, "request_too_large")


def test_messages_sampling_fields_reach_the_engine_config(client: TestClient, engine: ScriptedEngine) -> None:
    """temperature, top_p, top_k, stop_sequences and max_tokens are mapped onto GenerationConfig."""
    assert _post(client, body(temperature=0.2, top_p=0.8, top_k=20, stop_sequences=["END", "STOP"], max_tokens=77)).status_code == 200
    config = engine.configs[-1]
    assert (config.temperature, config.top_p, config.top_k, config.stop_strings, config.max_new_tokens) == (0.2, 0.8, 20, ["END", "STOP"], 77)


@pytest.mark.parametrize("field,value", [("temperature", 3), ("top_p", 0), ("stop_sequences", ["", "x"]), ("stream", "true"), ("top_k", -2)])
def test_messages_invalid_sampling_returns_400(client: TestClient, field: str, value: Any) -> None:
    """Out-of-range sampling values are 400s that name the field."""
    _assert_error(_post(client, body(**{field: value})), 400, "invalid_request_error", field)


def test_messages_query_string_beta_true_is_ignored(client: TestClient) -> None:
    """Claude Code posts /v1/messages?beta=true; the query string never affects routing."""
    for query in ("?beta=true", "?beta=true&x=1", "?"):
        assert client.post(MSG + query, json=body()).status_code == 200


def test_messages_missing_model_returns_400(client: TestClient) -> None:
    """model is required."""
    _assert_error(_post(client, body(model=None)), 400, "invalid_request_error", "model")


def test_messages_unknown_model_returns_404_listing_local_models_without_fallback(client: TestClient, engine: ScriptedEngine) -> None:
    """An unknown model is a not_found_error that names the available local models; no other model answers."""
    data = _assert_error(_post(client, body(model="claude-sonnet-4-5")), 404, "not_found_error", "model: claude-sonnet-4-5")
    assert MODEL in data["error"]["message"] and engine.prompts == []


def test_messages_unknown_model_with_no_local_models_says_so(tmp_path: Path) -> None:
    """An empty model directory is reported, not hidden."""
    from arc_runner.server.app import create_app
    from arc_runner.server.model_manager import ModelManager

    empty = tmp_path / "empty"
    empty.mkdir()
    client = TestClient(create_app(model_manager=ModelManager(model_dirs=[empty])))
    data = _assert_error(_post(client, body()), 404, "not_found_error")
    assert "no local models" in data["error"]["message"]


# ---------------------------------------------------------------- non-stream responses

def test_messages_non_stream_response_shape_and_ids(client: TestClient) -> None:
    """The message object carries msg_ id, role, echoed model, one text block and null stop_sequence."""
    response = _post(client, body())
    data = response.json()
    assert response.headers["content-type"].startswith("application/json")
    assert data["id"].startswith("msg_") and data["type"] == "message" and data["role"] == "assistant"
    assert data["model"] == MODEL and data["content"] == [{"type": "text", "text": "Hello there"}]
    assert data["stop_reason"] == "end_turn" and data["stop_sequence"] is None
    assert "thinking" not in response.text


def test_messages_non_stream_matches_golden_fixture(tmp_path: Path) -> None:
    """Request and response of the plain text shape equal the committed golden files."""
    engine = ScriptedEngine(telemetry=measured(9, 2))
    request = json.loads((FIXTURES / "text_request.json").read_text(encoding="utf-8"))
    response = _post(make_client(tmp_path, engine), request)
    assert json.loads(normalize_ids(response.text)) == json.loads((FIXTURES / "text_response.json").read_text(encoding="utf-8"))


def test_messages_non_stream_usage_is_measured_or_absent(tmp_path: Path) -> None:
    """usage carries engine-measured counts only; unmeasured keys are absent, never zero-filled."""
    cases = [(measured(31, 17), {"input_tokens": 31, "output_tokens": 17}), (EngineTelemetry(model_id=MODEL), {}), (measured(9, 0), {"input_tokens": 9, "output_tokens": 0})]
    for index, (telemetry, expected) in enumerate(cases):
        sub = tmp_path / str(index)
        sub.mkdir()
        usage = _post(make_client(sub, ScriptedEngine(telemetry=telemetry)), body()).json()["usage"]
        assert usage == expected


def test_messages_non_stream_stop_reason_max_tokens_derived_from_measured_count(tmp_path: Path) -> None:
    """max_tokens is reported only when the measured completion count reached the limit."""
    hit = _post(make_client(tmp_path, ScriptedEngine(telemetry=measured(5, 16))), body(max_tokens=16)).json()
    assert hit["stop_reason"] == "max_tokens"
    sub = tmp_path / "b"
    sub.mkdir()
    short = _post(make_client(sub, ScriptedEngine(telemetry=measured(5, 15))), body(max_tokens=16)).json()
    assert short["stop_reason"] == "end_turn"
    sub2 = tmp_path / "c"
    sub2.mkdir()
    unknown = _post(make_client(sub2, ScriptedEngine()), body(max_tokens=1)).json()
    assert unknown["stop_reason"] == "end_turn"


def test_messages_non_stream_empty_generation_returns_one_empty_text_block(tmp_path: Path) -> None:
    """An empty completion is still a well-formed message."""
    data = _post(make_client(tmp_path, ScriptedEngine(pieces=[])), body()).json()
    assert data["content"] == [{"type": "text", "text": ""}]


# ---------------------------------------------------------------- streaming (buffered TestClient)

def _stream(client: TestClient, payload: Dict[str, Any], **kwargs: Any) -> Any:
    return client.post(MSG, json={**payload, "stream": True}, **kwargs)


def test_messages_stream_headers_and_event_order(client: TestClient) -> None:
    """text/event-stream, no caching, and exactly: start, block start, deltas, block stop, message_delta, stop."""
    response = _stream(client, body())
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    assert [name for name, _ in events] == [
        "message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["id"].startswith("msg_") and start["content"] == [] and start["stop_reason"] is None and start["model"] == MODEL
    assert [e[1]["delta"]["text"] for e in events[2:4]] == ["Hello", " there"]
    assert events[1][1] == {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    assert events[5][1]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}


def test_messages_stream_matches_golden_fixture_byte_for_byte(tmp_path: Path) -> None:
    """The plain text stream equals the committed golden file exactly (ids normalised)."""
    engine = ScriptedEngine(telemetry=measured(9, 2))
    request = json.loads((FIXTURES / "text_request.json").read_text(encoding="utf-8"))
    response = _stream(make_client(tmp_path, engine), request)
    assert normalize_ids(response.text) == (FIXTURES / "text_stream.sse").read_text(encoding="utf-8")


def test_messages_stream_usage_in_message_delta_only_when_measured(tmp_path: Path) -> None:
    """message_delta usage is the engine's measured counts; unmeasured means an empty usage object."""
    measured_events = parse_sse(_stream(make_client(tmp_path, ScriptedEngine(telemetry=measured(11, 4))), body()).text)
    assert measured_events[-2][1]["usage"] == {"input_tokens": 11, "output_tokens": 4}
    sub = tmp_path / "u"
    sub.mkdir()
    unmeasured = parse_sse(_stream(make_client(sub, ScriptedEngine()), body()).text)
    assert unmeasured[-2][1]["usage"] == {} and "usage" in unmeasured[0][1]["message"]


def test_messages_stream_stop_reason_max_tokens_when_engine_reports_length(tmp_path: Path) -> None:
    """finish_reason length maps to max_tokens."""
    events = parse_sse(_stream(make_client(tmp_path, ScriptedEngine(finish="length")), body()).text)
    assert events[-2][1]["delta"]["stop_reason"] == "max_tokens"


def test_messages_stream_empty_generation_still_emits_a_complete_message(tmp_path: Path) -> None:
    """No tokens still yields one empty text block and the closing events."""
    events = parse_sse(_stream(make_client(tmp_path, ScriptedEngine(pieces=[])), body()).text)
    assert [n for n, _ in events] == ["message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"]


def test_messages_stream_stop_sequences_are_passed_to_the_engine(client: TestClient, engine: ScriptedEngine) -> None:
    """Streaming requests carry stop_sequences into the config."""
    _stream(client, body(stop_sequences=["END"]))
    assert engine.configs[-1].stop_strings == ["END"] and engine.configs[-1].stream is True


def test_messages_stream_query_string_and_probe_headers(client: TestClient) -> None:
    """The streaming path also ignores ?beta=true."""
    assert client.post(MSG + "?beta=true", json={**body(), "stream": True}).status_code == 200


def test_messages_stream_context_exceeded_before_first_token_is_http_400(tmp_path: Path) -> None:
    """A prompt that does not fit fails as an HTTP 400 (so Claude Code can compact), not as a 200 with an error event."""
    engine = ScriptedEngine(error=ContextWindowExceededError(9000, 4096))
    response = _stream(make_client(tmp_path, engine), body())
    _assert_error(response, 400, "invalid_request_error", "prompt is too long")


def test_messages_stream_mid_stream_engine_error_emits_error_event_and_no_message_stop(tmp_path: Path) -> None:
    """After bytes were sent a failure becomes 'event: error' and the stream ends without message_stop."""
    engine = ScriptedEngine(pieces=["a", "b", "c"], error=VRAMExhaustedError(100, 10), error_after=1)
    events = parse_sse(_stream(make_client(tmp_path, engine), body()).text)
    names = [n for n, _ in events]
    assert names[-1] == "error" and "message_stop" not in names and "message_delta" not in names
    assert events[-1][1] == {"type": "error", "error": {"type": "api_error", "message": events[-1][1]["error"]["message"]}}
    assert "Insufficient memory" in events[-1][1]["error"]["message"]


def test_messages_stream_unexpected_exception_does_not_leak_its_text(tmp_path: Path) -> None:
    """A non-typed failure produces a generic api_error event without the exception text."""
    engine = ScriptedEngine(pieces=["a", "b"], error=RuntimeError("secret-internal-detail"), error_after=1)
    response = _stream(make_client(tmp_path, engine), body())
    events = parse_sse(response.text)
    assert events[-1][0] == "error" and events[-1][1]["error"]["type"] == "api_error"
    assert "secret-internal-detail" not in response.text


# ---------------------------------------------------------------- errors (non-stream)

def test_messages_context_exceeded_message_begins_with_prompt_is_too_long(tmp_path: Path) -> None:
    """ContextWindowExceededError becomes a 400 whose message starts with 'prompt is too long' and carries both numbers."""
    engine = ScriptedEngine(error=ContextWindowExceededError(9000, 4096))
    data = _assert_error(_post(make_client(tmp_path, engine), body()), 400, "invalid_request_error", "prompt is too long")
    assert "9000" in data["error"]["message"] and "4096" in data["error"]["message"]


@pytest.mark.parametrize("error,status,error_type", [
    (VRAMExhaustedError(100, 10), 507, "api_error"),
    (InferenceTimeoutError(2.0, 1.0), 504, "timeout_error"),
    (ArcRunnerError("busy", "BUSY", 429), 429, "rate_limit_error"),
    (ArcRunnerError("busy", "BUSY", 529), 529, "overloaded_error"),
    (ArcRunnerError("nope", "NOPE", 403), 403, "permission_error"),
    (RuntimeError("secret-internal-detail"), 500, "api_error"),
])
def test_messages_engine_errors_map_to_anthropic_types(tmp_path: Path, error: Exception, status: int, error_type: str) -> None:
    """Typed engine errors keep their status; anything else is a generic 500 api_error without internals."""
    response = _post(make_client(tmp_path, ScriptedEngine(error=error)), body())
    data = _assert_error(response, status, error_type)
    assert "secret-internal-detail" not in response.text and data["error"]["message"]


def test_messages_prompts_are_never_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """Logs carry key-value metadata only, never the prompt text."""
    caplog.set_level(logging.DEBUG)
    _post(client, body(system="SYS-ZEBRA-7731", messages=[{"role": "user", "content": "USER-ZEBRA-7731"}]))
    _stream(client, body(messages=[{"role": "user", "content": "USER-ZEBRA-7731"}]))
    assert "ZEBRA-7731" not in caplog.text


# ---------------------------------------------------------------- tools

def test_messages_tools_are_converted_into_a_deterministic_system_instruction(client: TestClient, engine: ScriptedEngine) -> None:
    """Without a tools-aware template the prompt gets the compact instruction; identical requests give identical prompts."""
    _post(client, body(tools=[TOOL_WEATHER], system="Sys."))
    _post(client, body(tools=[TOOL_WEATHER], system="Sys."))
    prompt = engine.prompts[0]
    assert prompt == engine.prompts[1]
    assert prompt.startswith("<|im_start|>system\nSys.\n\n") and "get_weather" in prompt and '"required":["city"]' in prompt and "<tool_call>" in prompt


def test_messages_tool_choice_none_hides_tools_and_disables_parsing(client: TestClient, engine: ScriptedEngine, tmp_path: Path) -> None:
    """tool_choice none: no instruction in the prompt and a tool-call-shaped answer stays text."""
    sub = tmp_path / "n"
    sub.mkdir()
    eng = ScriptedEngine(pieces=CALL_PIECES)
    data = _post(make_client(sub, eng), body(tools=[TOOL_WEATHER], tool_choice={"type": "none"})).json()
    assert "get_weather" not in eng.prompts[-1].split("<|im_start|>user")[0]
    assert data["stop_reason"] == "end_turn" and all(b["type"] == "text" for b in data["content"])


def test_messages_tool_choice_any_is_accepted_and_not_enforced(client: TestClient) -> None:
    """tool_choice any/tool are accepted and ignored (documented), never a 400."""
    for choice in ({"type": "any"}, {"type": "tool", "name": "get_weather"}, {"type": "auto"}):
        assert _post(client, body(tools=[TOOL_WEATHER], tool_choice=choice)).status_code == 200


def test_messages_engine_template_hook_is_used_when_available(tmp_path: Path) -> None:
    """An engine whose tokenizer template supports tools renders the prompt itself."""
    class TemplatedEngine(ScriptedEngine):
        def render_chat_with_tools(self, messages: Any, tools: Any) -> str:
            return f"TEMPLATE|{len(messages)}|{tools[0]['function']['name']}"

    eng = TemplatedEngine()
    assert _post(make_client(tmp_path, eng), body(tools=[TOOL_WEATHER])).status_code == 200
    assert eng.prompts[-1] == "TEMPLATE|1|get_weather"


def test_messages_tool_result_round_trip_reaches_the_prompt(client: TestClient, engine: ScriptedEngine) -> None:
    """An assistant tool_use followed by a user tool_result is rendered as call and response, then answered."""
    messages = [
        {"role": "user", "content": "weather in Pune?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "Pune"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": [{"type": "text", "text": "31C clear"}]}]},
    ]
    response = _post(client, body(messages=messages, tools=[TOOL_WEATHER]))
    assert response.status_code == 200
    prompt = engine.prompts[-1]
    assert '<tool_call>{"name": "get_weather", "arguments": {"city": "Pune"}}</tool_call>' in prompt
    assert '<tool_response id="toolu_abc" name="get_weather">\n31C clear\n</tool_response>' in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_messages_tool_call_non_stream_returns_tool_use_block(tmp_path: Path) -> None:
    """A model answer in the documented format becomes a tool_use block with toolu_ id and stop_reason tool_use."""
    engine = ScriptedEngine(pieces=CALL_PIECES, telemetry=measured(20, 12))
    request = json.loads((FIXTURES / "tool_request.json").read_text(encoding="utf-8"))
    response = _post(make_client(tmp_path, engine), request)
    assert json.loads(normalize_ids(response.text)) == json.loads((FIXTURES / "tool_response.json").read_text(encoding="utf-8"))
    block = response.json()["content"][1]
    assert block["id"].startswith("toolu_") and isinstance(block["input"], dict)


def test_messages_tool_call_stream_matches_golden_fixture_byte_for_byte(tmp_path: Path) -> None:
    """tool_use start with empty input, input_json_delta, block stop, stop_reason tool_use: exact bytes."""
    engine = ScriptedEngine(pieces=CALL_PIECES, telemetry=measured(20, 12))
    request = json.loads((FIXTURES / "tool_request.json").read_text(encoding="utf-8"))
    response = _stream(make_client(tmp_path, engine), request)
    assert normalize_ids(response.text) == (FIXTURES / "tool_stream.sse").read_text(encoding="utf-8")


def test_messages_tool_call_stream_partial_json_concatenates_to_the_input_object(tmp_path: Path) -> None:
    """The input_json_delta strings concatenate to the JSON object of the call."""
    engine = ScriptedEngine(pieces=['<tool_call>{"name": "get_weather", "arguments": {"city": "Pune", "unit": "c"}}</tool_call>'])
    events = parse_sse(_stream(make_client(tmp_path, engine), body(tools=[TOOL_WEATHER])).text)
    joined = "".join(e["delta"]["partial_json"] for n, e in events if n == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
    assert json.loads(joined) == {"city": "Pune", "unit": "c"}
    start = [e for n, e in events if n == "content_block_start"][0]
    assert start["index"] == 0 and start["content_block"]["input"] == {}


def test_messages_tool_call_stream_split_across_many_tiny_chunks(tmp_path: Path) -> None:
    """The tag and JSON split at arbitrary points still parse; no tag text leaks as a text delta."""
    text = 'Sure.<tool_call>{"name": "get_weather", "arguments": {"city": "Pune"}}</tool_call>'
    engine = ScriptedEngine(pieces=list(text))
    events = parse_sse(_stream(make_client(tmp_path, engine), body(tools=[TOOL_WEATHER])).text)
    texts = "".join(e["delta"]["text"] for n, e in events if n == "content_block_delta" and e["delta"]["type"] == "text_delta")
    assert texts == "Sure." and events[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_messages_tool_call_stream_multiple_calls_get_distinct_ids_and_indexes(tmp_path: Path) -> None:
    """Two calls give two tool_use blocks, indexes 0 and 1, distinct ids."""
    one = '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}}</tool_call>'
    engine = ScriptedEngine(pieces=[one, "\n", one.replace('"A"', '"B"')])
    events = parse_sse(_stream(make_client(tmp_path, engine), body(tools=[TOOL_WEATHER])).text)
    starts = [e for n, e in events if n == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1] and starts[0]["content_block"]["id"] != starts[1]["content_block"]["id"]
    assert [n for n, _ in events].count("content_block_stop") == 2


def test_messages_tool_call_non_stream_multiple_calls_have_unique_ids(tmp_path: Path) -> None:
    """Identical calls still receive different ids."""
    one = '<tool_call>{"name": "get_weather", "arguments": {"city": "A"}}</tool_call>'
    data = _post(make_client(tmp_path, ScriptedEngine(pieces=[one + one])), body(tools=[TOOL_WEATHER])).json()
    ids = [b["id"] for b in data["content"] if b["type"] == "tool_use"]
    assert len(ids) == 2 and len(set(ids)) == 2 and data["stop_reason"] == "tool_use"


@pytest.mark.parametrize("pieces", [
    ["I would use get_weather for Pune."],
    ['<tool_call>{"name": "ghost", "arguments": {}}</tool_call>'],
    ['<tool_call>{"name": "get_weather", "arguments": {"city": "Pu'],
    ['<tool_call>{"name": "get_weather", "arguments": []}</tool_call>'],
    ['<tool_call>not json</tool_call>'],
    ['<tool_call>{"name": "get_weather", "arguments": {}}</tool_call><tool_call>{"name"'],
])
def test_messages_tool_call_invalid_output_is_plain_text_in_both_modes(tmp_path: Path, pieces: List[str]) -> None:
    """Prose, unknown names, partial or malformed JSON never become calls; the text is returned verbatim."""
    text = "".join(pieces)
    sub = tmp_path / "ns"
    sub.mkdir()
    data = _post(make_client(sub, ScriptedEngine(pieces=pieces)), body(tools=[TOOL_WEATHER])).json()
    assert data["content"] == [{"type": "text", "text": text}] and data["stop_reason"] == "end_turn"
    sub2 = tmp_path / "st"
    sub2.mkdir()
    events = parse_sse(_stream(make_client(sub2, ScriptedEngine(pieces=pieces)), body(tools=[TOOL_WEATHER])).text)
    streamed = "".join(e["delta"]["text"] for n, e in events if n == "content_block_delta")
    assert streamed == text and events[-2][1]["delta"]["stop_reason"] == "end_turn"
    assert not any(n == "content_block_start" and e["content_block"]["type"] == "tool_use" for n, e in events)


def test_messages_tool_tag_without_declared_tools_is_plain_text(tmp_path: Path) -> None:
    """With no tools in the request the markup is ordinary output in both modes."""
    pieces = ['<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>']
    sub = tmp_path / "a"
    sub.mkdir()
    data = _post(make_client(sub, ScriptedEngine(pieces=pieces)), body()).json()
    assert data["content"][0]["type"] == "text" and data["stop_reason"] == "end_turn"
    sub2 = tmp_path / "b"
    sub2.mkdir()
    events = parse_sse(_stream(make_client(sub2, ScriptedEngine(pieces=pieces)), body()).text)
    assert "".join(e["delta"]["text"] for n, e in events if n == "content_block_delta") == pieces[0]


# ---------------------------------------------------------------- count_tokens

def test_count_tokens_returns_measured_count_matching_golden(tmp_path: Path) -> None:
    """input_tokens is the engine tokenizer's count of the exact prompt the engine would get."""
    request = json.loads((FIXTURES / "text_request.json").read_text(encoding="utf-8"))
    request.pop("max_tokens")
    response = make_client(tmp_path, ScriptedEngine()).post("/v1/messages/count_tokens", json=request)
    assert response.json() == json.loads((FIXTURES / "count_tokens_response.json").read_text(encoding="utf-8"))


def test_count_tokens_includes_the_tool_instruction(client: TestClient) -> None:
    """Declaring tools increases the measured count because the instruction is part of the prompt."""
    plain = client.post("/v1/messages/count_tokens", json={"model": MODEL, "messages": body()["messages"]}).json()["input_tokens"]
    with_tools = client.post("/v1/messages/count_tokens", json={"model": MODEL, "messages": body()["messages"], "tools": [TOOL_WEATHER]}).json()["input_tokens"]
    assert with_tools > plain


def test_count_tokens_accepts_query_string_and_unknown_fields(client: TestClient) -> None:
    """?beta=true and beta body fields are tolerated."""
    response = client.post("/v1/messages/count_tokens?beta=true", json={"model": MODEL, "messages": body()["messages"], "thinking": {"type": "adaptive"}, "context_management": {}})
    assert response.status_code == 200 and isinstance(response.json()["input_tokens"], int)


@pytest.mark.parametrize("has_tokenizer", [False])
def test_count_tokens_without_tokenizer_is_an_error_not_an_estimate(tmp_path: Path, has_tokenizer: bool) -> None:
    """An engine with no tokenizer yields a not_found_error envelope and no number."""
    client = make_client(tmp_path, ScriptedEngine(tokenizer=has_tokenizer))
    response = client.post("/v1/messages/count_tokens", json={"model": MODEL, "messages": body()["messages"]})
    data = _assert_error(response, 404, "not_found_error", "token counting is unavailable")
    assert "input_tokens" not in response.text and "tokenizer" in data["error"]["message"]


def test_count_tokens_engine_without_tokenize_attribute_is_an_error(tmp_path: Path) -> None:
    """An engine class lacking tokenize() is handled the same way."""
    engine = ScriptedEngine()
    engine.tokenize = None  # type: ignore[assignment]
    response = make_client(tmp_path, engine).post("/v1/messages/count_tokens", json={"model": MODEL, "messages": body()["messages"]})
    _assert_error(response, 404, "not_found_error", "token counting is unavailable")


def test_count_tokens_unknown_model_lists_local_models(client: TestClient) -> None:
    """Unknown model: 404 with the local model list, same as /v1/messages."""
    response = client.post("/v1/messages/count_tokens", json={"model": "nope", "messages": body()["messages"]})
    data = _assert_error(response, 404, "not_found_error", "model: nope")
    assert MODEL in data["error"]["message"]


def test_count_tokens_validation_errors_use_the_envelope(client: TestClient) -> None:
    """Bad bodies get the same 400 envelope; max_tokens is not required."""
    _assert_error(client.post("/v1/messages/count_tokens", json={"model": MODEL, "messages": []}), 400, "invalid_request_error", "messages")
    _assert_error(client.post("/v1/messages/count_tokens", json={"model": MODEL, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}), 400, "invalid_request_error")


# ---------------------------------------------------------------- probe

def test_api_hello_head_and_get_return_200(client: TestClient) -> None:
    """Claude Code's connection-warming HEAD /api/hello is answered; GET works too."""
    head = client.head("/api/hello")
    assert head.status_code == 200 and head.content == b""
    assert client.get("/api/hello").status_code == 200


def test_get_on_messages_is_not_a_success(client: TestClient) -> None:
    """Only POST creates a message."""
    assert client.get(MSG).status_code == 405


# ---------------------------------------------------------------- auth

KEY = "s3cret-key"


@pytest.fixture
def secured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The app with ARC_API_KEY set."""
    monkeypatch.setenv("ARC_API_KEY", KEY)
    return make_client(tmp_path, ScriptedEngine())


@pytest.mark.parametrize("headers,ok", [
    ({"x-api-key": KEY}, True),
    ({"Authorization": f"Bearer {KEY}"}, True),
    ({"authorization": f"bearer {KEY}"}, True),
    ({"x-api-key": KEY, "Authorization": f"Bearer {KEY}"}, True),
    ({"x-api-key": "wrong"}, False),
    ({"Authorization": "Bearer wrong"}, False),
    ({"Authorization": KEY}, False),
    ({"x-api-key": ""}, False),
    ({}, False),
])
def test_api_key_messages_accepts_x_api_key_and_bearer(secured: TestClient, headers: Dict[str, str], ok: bool) -> None:
    """With ARC_API_KEY set both credential headers Claude Code can send are honoured; anything else is a 401."""
    response = secured.post(MSG, json=body(), headers=headers)
    if ok:
        assert response.status_code == 200
    else:
        _assert_error(response, 401, "authentication_error")


def test_api_key_count_tokens_requires_the_key_with_anthropic_envelope(secured: TestClient) -> None:
    """count_tokens is protected and answers in the Anthropic error shape."""
    _assert_error(secured.post("/v1/messages/count_tokens", json=body()), 401, "authentication_error")
    assert secured.post("/v1/messages/count_tokens", json=body(), headers={"x-api-key": KEY}).status_code == 200


def test_api_key_non_anthropic_routes_keep_their_existing_401_shape(secured: TestClient) -> None:
    """Other routes still answer {error, code}."""
    response = secured.get("/api/tags")
    assert response.status_code == 401 and response.json() == {"error": "missing or invalid API key", "code": "UNAUTHORIZED"}


def test_api_key_comparison_is_constant_time_for_both_headers(secured: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """hmac.compare_digest sees the bytes of each presented credential, even when the first already matches."""
    calls: List[Any] = []
    real = hmac.compare_digest

    def spy(a: Any, b: Any) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    secured.post(MSG, json=body(), headers={"x-api-key": KEY, "Authorization": f"Bearer {KEY}"})
    assert len(calls) == 2 and all(isinstance(a, bytes) and b == KEY.encode() for a, b in calls)
    calls.clear()
    secured.post(MSG, json=body(), headers={"x-api-key": "short"})
    assert len(calls) == 2


def test_api_key_health_path_is_exempt_and_probe_is_not(secured: TestClient) -> None:
    """/health is never answered with 401 (it is exempt; no route exists yet so it is a 404); /api/hello is protected."""
    assert secured.get("/health").status_code != 401
    assert secured.head("/api/hello").status_code == 401
    assert secured.head("/api/hello", headers={"x-api-key": KEY}).status_code == 200
    assert secured.get("/api/tags").status_code == 401


def test_api_key_unset_leaves_messages_open(client: TestClient) -> None:
    """Without ARC_API_KEY no credential is needed (Claude Code's placeholder token is ignored)."""
    assert client.post(MSG, json=body(), headers={"Authorization": "Bearer local"}).status_code == 200
