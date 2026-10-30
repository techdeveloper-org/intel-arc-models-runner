"""Anthropic Messages API routes so Claude Code can use the runner the way it uses Ollama.

Serves POST /v1/messages (blocking and server-sent events), POST /v1/messages/count_tokens and the
HEAD/GET /api/hello connection probe. Engine work reuses the OpenAI route machinery (prompt layout,
the thread-to-async stream bridge and measured-usage rules); this module adds Anthropic framing,
tool-call handling, ping events and the Anthropic error envelope.

Token counts on the wire are engine-measured or absent. A stream carries ping events while the
engine is silent so the client's idle watchdog does not abort a long prefill.
"""

import asyncio
import json
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple
import weakref

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from arc_runner.core.exceptions import ArcRunnerError, ContextWindowExceededError, ModelNotFoundError
from arc_runner.core.logging import get_logger
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.anthropic_convert import (
    Segment,
    StreamSegmenter,
    build_prompt,
    new_message_id,
    new_tool_id,
    parse_tool_calls,
)
from arc_runner.server.anthropic_schemas import (
    AnthropicRequestError,
    ParsedRequest,
    error_envelope,
    error_type_for_status,
    new_request_id,
    parse_messages_request,
)
from arc_runner.server.model_manager import ModelManager
from arc_runner.server.routes_openai import _stream_token_chunks, get_model_manager, measured_usage

logger = get_logger("server.routes_anthropic")

router = APIRouter(prefix="/v1", tags=["anthropic"])
probe_router = APIRouter(tags=["anthropic"])

STREAM_PING_INTERVAL_SECONDS: float = 10.0
MAX_REQUEST_BYTES: int = 64 * 1024 * 1024

_ENGINE_LOCKS: "weakref.WeakKeyDictionary[Any, threading.Lock]" = weakref.WeakKeyDictionary()
_ENGINE_LOCKS_GUARD = threading.Lock()
_END = object()
_PING = object()


def _engine_lock(engine: Any) -> threading.Lock:
    """Return the lock that serialises work on one engine instance.

    Args:
        engine: Loaded engine.

    Returns:
        threading.Lock: The engine's lock, created on first use.
    """
    with _ENGINE_LOCKS_GUARD:
        lock = _ENGINE_LOCKS.get(engine)
        if lock is None:
            lock = threading.Lock()
            _ENGINE_LOCKS[engine] = lock
        return lock


class _GuardedEngine:
    """Engine view for the stream bridge that serialises runs, stops on cancel and captures telemetry."""

    def __init__(self, engine: Any, cancel: threading.Event) -> None:
        """Wrap an engine.

        Args:
            engine: Loaded engine.
            cancel: Set by the consumer when the client left or the request ended.
        """
        self._engine = engine
        self._cancel = cancel
        self.telemetry: Optional[EngineTelemetry] = None

    def generate_stream(self, prompt: str, config: GenerationConfig) -> Any:
        """Yield engine chunks while holding the engine lock, closing the engine stream on cancel.

        Args:
            prompt: Prompt text.
            config: Generation parameters.

        Yields:
            TokenChunk: Chunks from the wrapped engine.
        """
        with _engine_lock(self._engine):
            if self._cancel.is_set():
                return
            inner = self._engine.generate_stream(prompt, config)
            try:
                for chunk in inner:
                    if self._cancel.is_set():
                        return
                    yield chunk
                self.telemetry = self._engine.get_telemetry()
            finally:
                close = getattr(inner, "close", None)
                if callable(close):
                    close()


def _describe_error(exc: BaseException) -> Tuple[int, str, str]:
    """Map an exception to (HTTP status, Anthropic error type, message).

    Args:
        exc: The exception raised while serving a request.

    Returns:
        Tuple[int, str, str]: Status, error type and a message that never contains prompt text.
    """
    if isinstance(exc, AnthropicRequestError):
        return exc.status_code, exc.error_type, exc.message
    if isinstance(exc, ContextWindowExceededError):
        message = (
            f"prompt is too long: {exc.prompt_tokens} tokens (prompt plus max_tokens) > {exc.max_context} maximum context window"
        )
        return 400, "invalid_request_error", message
    if isinstance(exc, ArcRunnerError):
        return exc.status_code, error_type_for_status(exc.status_code), exc.message
    logger.error("anthropic request failed", extra={"error_class": type(exc).__name__}, exc_info=exc)
    return 500, "api_error", f"internal error while serving the request ({type(exc).__name__}); see the server log"


def _model_not_found(manager: ModelManager, model: str) -> AnthropicRequestError:
    """Build the 404 for an unknown model, listing what is installed.

    Args:
        manager: Model manager.
        model: Requested model name.

    Returns:
        AnthropicRequestError: not_found_error naming the available local models.
    """
    names = [m["name"] for m in manager.list_available_models()]
    if names:
        message = f"model: {model} (not found; available local models: {', '.join(names)})"
    else:
        message = f"model: {model} (not found; no local models are installed, pull one with `arc-runner pull`)"
    return AnthropicRequestError(message, 404)


def _error_response(exc: BaseException, request_id: str) -> JSONResponse:
    """Build the Anthropic error response for an exception.

    Args:
        exc: Exception to report.
        request_id: Request id for the body and header.

    Returns:
        JSONResponse: Envelope with the mapped status.
    """
    status, error_type, message = _describe_error(exc)
    return JSONResponse(status_code=status, content=error_envelope(error_type, message, request_id), headers={"request-id": request_id})


async def _read_request(raw_request: Request, require_max_tokens: bool) -> ParsedRequest:
    """Read, size-check, decode and validate a request body.

    Args:
        raw_request: Incoming request.
        require_max_tokens: True for /v1/messages.

    Returns:
        ParsedRequest: Validated request.

    Raises:
        AnthropicRequestError: 413 for oversized bodies, 400 for invalid JSON or fields.
    """
    declared = raw_request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise AnthropicRequestError(f"request body exceeds {MAX_REQUEST_BYTES} bytes", 413)
    raw = await raw_request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        raise AnthropicRequestError(f"request body exceeds {MAX_REQUEST_BYTES} bytes", 413)
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise AnthropicRequestError("Request body must be valid JSON") from exc
    return parse_messages_request(payload, require_max_tokens=require_max_tokens)


async def _load_engine(manager: ModelManager, model: str) -> Any:
    """Load or fetch the engine for a model off the event loop.

    Args:
        manager: Model manager.
        model: Requested model name.

    Returns:
        Any: Ready engine.

    Raises:
        AnthropicRequestError: 404 when the model is unknown.
    """
    try:
        return await asyncio.to_thread(manager.get_or_load_engine, model, "GPU")
    except ModelNotFoundError as exc:
        raise _model_not_found(manager, model) from exc


def _build_config(parsed: ParsedRequest, stream: bool) -> GenerationConfig:
    """Map validated sampling fields onto the engine config.

    Args:
        parsed: Validated request.
        stream: Whether the request streams.

    Returns:
        GenerationConfig: Engine parameters.

    Raises:
        AnthropicRequestError: If the engine config rejects a value.
    """
    values: Dict[str, Any] = {"max_new_tokens": parsed.max_tokens, "stop_strings": parsed.stop_sequences, "stream": stream}
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if getattr(parsed, source) is not None:
            values[target] = getattr(parsed, source)
    try:
        return GenerationConfig(**values)
    except ValidationError as exc:
        raise AnthropicRequestError(f"invalid generation parameters: {exc.errors()[0]['loc']}") from exc


def _tool_names(parsed: ParsedRequest) -> Set[str]:
    """Return the tool names the output may call (none when tool_choice is none).

    Args:
        parsed: Validated request.

    Returns:
        Set[str]: Declared tool names.
    """
    return set() if parsed.tool_choice_none else {tool.name for tool in parsed.tools}


def _usage(telemetry: Optional[EngineTelemetry], prompt_tokens: Optional[int] = None) -> Dict[str, int]:
    """Build an Anthropic usage object from measured counts only.

    Args:
        telemetry: Telemetry of the finished run, or None.
        prompt_tokens: Prompt token count reported by the engine's final chunk, if any (0 means unmeasured).

    Returns:
        Dict[str, int]: input_tokens and output_tokens when measured; keys that were not measured are absent.
    """
    usage: Dict[str, int] = {}
    measured = measured_usage(telemetry) if telemetry is not None else None
    if measured and "prompt_tokens" in measured:
        usage["input_tokens"] = measured["prompt_tokens"]
    elif prompt_tokens is not None and prompt_tokens > 0:
        usage["input_tokens"] = prompt_tokens
    if measured and "completion_tokens" in measured:
        usage["output_tokens"] = measured["completion_tokens"]
    return usage


def _segments_to_blocks(segments: Optional[List[Segment]], text: str) -> List[Dict[str, Any]]:
    """Turn parsed segments into response content blocks.

    Args:
        segments: Parsed segments, or None when the output is plain text.
        text: The complete output text.

    Returns:
        List[Dict[str, Any]]: Content blocks; one (possibly empty) text block for plain text.
    """
    if segments is None:
        return [{"type": "text", "text": text}]
    blocks: List[Dict[str, Any]] = []
    for segment in segments:
        if segment.kind == "text":
            blocks.append({"type": "text", "text": segment.text})
        else:
            blocks.append({"type": "tool_use", "id": new_tool_id(), "name": segment.name, "input": segment.input})
    return blocks


def _generate_blocking(engine: Any, prompt: str, config: GenerationConfig) -> Tuple[str, EngineTelemetry]:
    """Run one blocking generation under the engine lock.

    Args:
        engine: Loaded engine.
        prompt: Prompt text.
        config: Generation parameters.

    Returns:
        Tuple[str, EngineTelemetry]: Output text and the telemetry of that same run.
    """
    with _engine_lock(engine):
        text = engine.generate(prompt, config)
        return text, engine.get_telemetry()


def _event(name: str, data: Dict[str, Any]) -> bytes:
    """Frame one server-sent event.

    Args:
        name: Event name; repeated as the data's type.
        data: Event payload.

    Returns:
        bytes: 'event: <name>', 'data: <json>' and a blank line, UTF-8.
    """
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class _BlockWriter:
    """Tracks content block indexes and produces the block events of one stream."""

    def __init__(self) -> None:
        """Start with no blocks."""
        self._index = -1
        self._text_open = False
        self.blocks = 0
        self.tool_blocks = 0

    def text(self, text: str) -> List[bytes]:
        """Emit a text delta, opening a text block first when needed.

        Args:
            text: Text to append; empty text emits nothing.

        Returns:
            List[bytes]: Events.
        """
        if not text:
            return []
        events: List[bytes] = []
        if not self._text_open:
            self._index += 1
            self.blocks += 1
            self._text_open = True
            events.append(_event("content_block_start", {"type": "content_block_start", "index": self._index, "content_block": {"type": "text", "text": ""}}))
        events.append(_event("content_block_delta", {"type": "content_block_delta", "index": self._index, "delta": {"type": "text_delta", "text": text}}))
        return events

    def close_text(self) -> List[bytes]:
        """Close the open text block, if any.

        Returns:
            List[bytes]: Events.
        """
        if not self._text_open:
            return []
        self._text_open = False
        return [_event("content_block_stop", {"type": "content_block_stop", "index": self._index})]

    def tool(self, name: str, tool_input: Dict[str, Any]) -> List[bytes]:
        """Emit a complete tool_use block (start with empty input, one input_json_delta, stop).

        Args:
            name: Tool name.
            tool_input: Validated argument object.

        Returns:
            List[bytes]: Events.
        """
        events = self.close_text()
        self._index += 1
        self.blocks += 1
        self.tool_blocks += 1
        events.append(_event("content_block_start", {"type": "content_block_start", "index": self._index, "content_block": {"type": "tool_use", "id": new_tool_id(), "name": name, "input": {}}}))
        events.append(_event("content_block_delta", {"type": "content_block_delta", "index": self._index, "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input, ensure_ascii=False)}}))
        events.append(_event("content_block_stop", {"type": "content_block_stop", "index": self._index}))
        return events

    def empty_block_if_none(self) -> List[bytes]:
        """Emit one empty text block when the message would otherwise have no content.

        Returns:
            List[bytes]: Events.
        """
        if self.blocks:
            return []
        self.blocks = 1
        return [
            _event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ]


async def _next_chunk(chunks: AsyncIterator[TokenChunk]) -> Any:
    """Await the next chunk, returning a sentinel at the end of the stream.

    Args:
        chunks: Bridge iterator.

    Returns:
        Any: A TokenChunk, or the end sentinel.
    """
    try:
        return await chunks.__anext__()
    except StopAsyncIteration:
        return _END


async def _sse_body(
    raw_request: Request,
    manager: ModelManager,
    guarded: _GuardedEngine,
    chunks: AsyncIterator[TokenChunk],
    first: "asyncio.Future[Any]",
    cancel: threading.Event,
    parsed: ParsedRequest,
) -> AsyncIterator[bytes]:
    """Produce the server-sent event stream for one message.

    Args:
        raw_request: Incoming request, used to notice a departed client while the engine is silent.
        manager: Model manager, told the measured token count.
        guarded: Engine view that captured telemetry for this run.
        chunks: Bridge iterator over engine chunks.
        first: Task already waiting for the first chunk.
        cancel: Cancel flag shared with the engine thread.
        parsed: Validated request.

    Yields:
        bytes: Framed events.
    """
    names = _tool_names(parsed)
    writer = _BlockWriter()
    segmenter = StreamSegmenter(enabled=bool(names))
    finish_reason: Optional[str] = None
    final_prompt_tokens: Optional[int] = None
    pending = first
    yield _event("message_start", {"type": "message_start", "message": {
        "id": new_message_id(), "type": "message", "role": "assistant", "model": parsed.model,
        "content": [], "stop_reason": None, "stop_sequence": None, "usage": {},
    }})
    try:
        while True:
            while not pending.done():
                await asyncio.wait({pending}, timeout=STREAM_PING_INTERVAL_SECONDS)
                if pending.done():
                    break
                if await raw_request.is_disconnected():
                    return
                yield _event("ping", {"type": "ping"})
            chunk = pending.result()
            if chunk is _END:
                break
            if chunk.is_finished:
                finish_reason = chunk.finish_reason
                final_prompt_tokens = chunk.prompt_tokens
            else:
                for event in writer.text(segmenter.feed(chunk.text)):
                    yield event
            pending = asyncio.ensure_future(_next_chunk(chunks))

        flushed, tail = segmenter.finish()
        for event in writer.text(flushed):
            yield event
        if tail:
            segments = parse_tool_calls(tail, names)
            for segment in segments if segments is not None else [Segment(kind="text", text=tail)]:
                for event in writer.text(segment.text) if segment.kind == "text" else writer.tool(segment.name, segment.input):
                    yield event
        for event in writer.close_text() + writer.empty_block_if_none():
            yield event
        usage = _usage(guarded.telemetry, final_prompt_tokens)
        if guarded.telemetry is not None and guarded.telemetry.metrics_source != "unavailable":
            manager.record_generation_metrics(guarded.telemetry.total_tokens)
        if writer.tool_blocks:
            stop_reason = "tool_use"
        else:
            stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
        yield _event("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": usage})
        yield _event("message_stop", {"type": "message_stop"})
    except Exception as exc:
        _, error_type, message = _describe_error(exc)
        yield _event("error", {"type": "error", "error": {"type": error_type, "message": message}})
    finally:
        cancel.set()
        if not pending.done():
            pending.cancel()


async def _open_stream(
    raw_request: Request,
    manager: ModelManager,
    engine: Any,
    prompt: str,
    parsed: ParsedRequest,
    request_id: str,
) -> StreamingResponse:
    """Start the engine and return the streaming response.

    The first engine event is awaited for up to one ping interval before the HTTP status is committed, so a
    prompt that does not fit the context window fails as an HTTP 400 that the client can act on.

    Args:
        raw_request: Incoming request.
        manager: Model manager.
        engine: Loaded engine.
        prompt: Prompt text.
        parsed: Validated request.
        request_id: Request id for the response header.

    Returns:
        StreamingResponse: text/event-stream response.

    Raises:
        Exception: Whatever the engine raised before producing its first event.
    """
    cancel = threading.Event()
    guarded = _GuardedEngine(engine, cancel)
    chunks = _stream_token_chunks(guarded, prompt, _build_config(parsed, True))
    first = asyncio.ensure_future(_next_chunk(chunks))
    done, _ = await asyncio.wait({first}, timeout=STREAM_PING_INTERVAL_SECONDS)
    if first in done and first.exception() is not None:
        cancel.set()
        raise first.exception()  # type: ignore[misc]
    return StreamingResponse(
        _sse_body(raw_request, manager, guarded, chunks, first, cancel, parsed),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "request-id": request_id},
    )


@router.post("/messages")
async def create_message(raw_request: Request, manager: ModelManager = Depends(get_model_manager)) -> Response:
    """Create a message (POST /v1/messages), blocking or as server-sent events.

    Args:
        raw_request: Incoming request; the body is validated by hand so unknown fields are ignored.
        manager: Injected ModelManager.

    Returns:
        Response: Message JSON, an event stream, or an Anthropic error envelope.
    """
    request_id = new_request_id()
    try:
        parsed = await _read_request(raw_request, require_max_tokens=True)
        logger.info("anthropic messages", extra={
            "model": parsed.model, "stream": parsed.stream, "turns": len(parsed.messages),
            "tools": len(parsed.tools), "max_tokens": parsed.max_tokens,
        })
        engine = await _load_engine(manager, parsed.model)
        prompt = build_prompt(engine, parsed)
        if parsed.stream:
            return await _open_stream(raw_request, manager, engine, prompt, parsed, request_id)
        config = _build_config(parsed, False)
        text, telemetry = await asyncio.to_thread(_generate_blocking, engine, prompt, config)
    except Exception as exc:
        return _error_response(exc, request_id)

    if telemetry.metrics_source != "unavailable":
        manager.record_generation_metrics(telemetry.total_tokens)
    usage = _usage(telemetry)
    segments = parse_tool_calls(text, _tool_names(parsed))
    blocks = _segments_to_blocks(segments, text)
    if segments is not None:
        stop_reason = "tool_use"
    elif parsed.max_tokens is not None and usage.get("output_tokens", 0) >= parsed.max_tokens:
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return JSONResponse(
        {
            "id": new_message_id(), "type": "message", "role": "assistant", "model": parsed.model, "content": blocks,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": usage,
        },
        headers={"request-id": request_id},
    )


@router.post("/messages/count_tokens")
async def count_tokens(raw_request: Request, manager: ModelManager = Depends(get_model_manager)) -> Response:
    """Count the prompt tokens of a request with the engine's own tokenizer (POST /v1/messages/count_tokens).

    Args:
        raw_request: Incoming request.
        manager: Injected ModelManager.

    Returns:
        Response: {"input_tokens": N} measured on the exact prompt the engine would receive, or an error
        envelope; no estimate is ever returned.
    """
    request_id = new_request_id()
    try:
        parsed = await _read_request(raw_request, require_max_tokens=False)
        engine = await _load_engine(manager, parsed.model)
        tokenize = getattr(engine, "tokenize", None)
        token_ids: Optional[List[int]] = None
        if callable(tokenize):
            prompt = build_prompt(engine, parsed)

            def run() -> Optional[List[int]]:
                with _engine_lock(engine):
                    return tokenize(prompt)

            token_ids = await asyncio.to_thread(run)
        if token_ids is None:
            raise AnthropicRequestError(
                "token counting is unavailable: the loaded engine has no usable tokenizer, so no count is reported "
                "(Claude Code falls back to its own estimate)",
                404,
            )
    except Exception as exc:
        return _error_response(exc, request_id)
    return JSONResponse({"input_tokens": len(token_ids)}, headers={"request-id": request_id})


@probe_router.api_route("/api/hello", methods=["GET", "HEAD"])
async def api_hello(request: Request) -> Response:
    """Answer the connection-warming probe Claude Code sends at startup (HEAD /api/hello).

    Args:
        request: Incoming request.

    Returns:
        Response: 200 with no body for HEAD, a small JSON status for GET.
    """
    if request.method == "HEAD":
        return Response(status_code=200)
    return JSONResponse({"status": "ok"})
