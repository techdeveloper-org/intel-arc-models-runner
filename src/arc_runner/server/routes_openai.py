"""FastAPI router implementing 100% OpenAI v1 wire protocol parity.

Provides /v1/chat/completions with Server-Sent Events (SSE) streaming and non-streaming fallback,
as well as /v1/models catalog discovery for integration with OpenAI-compatible clients.
"""

import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from arc_runner.server.prefix_probe import match_prefix_with_real_tokens
from arc_runner.core.constants import DEFAULT_MAX_NEW_TOKENS
from arc_runner.core.exceptions import EngineUnavailableError, ModelNotFoundError
from arc_runner.core.logging import get_logger
from arc_runner.engine.models import GenerationConfig
from arc_runner.server.errors import hidden_error, load_failure_to_http, model_not_found_detail
from arc_runner.server.model_manager import ModelManager
from arc_runner.server.worker import AsyncTokenStream, JobContext, LoadPhaseError, get_worker
from arc_runner.server.schemas import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatMessage,
    OpenAIChunkChoice,
    OpenAIChunkDelta,
    OpenAIModelCard,
    OpenAIModelListResponse,
)

logger = get_logger("server.routes_openai")

router = APIRouter(prefix="/v1", tags=["openai"])


def get_model_manager(request: Request) -> ModelManager:
    """Extract registered ModelManager instance from application state.

    Args:
        request: Active FastAPI HTTP request.

    Returns:
        Configured ModelManager instance.
    """
    return request.app.state.model_manager


@router.get("/models", response_model=OpenAIModelListResponse)
async def list_models(manager: ModelManager = Depends(get_model_manager)) -> OpenAIModelListResponse:
    """List discoverable models in OpenAI API catalog format.

    Args:
        manager: Injected ModelManager instance.

    Returns:
        OpenAIModelListResponse populated with model cards.
    """
    raw_models = await asyncio.to_thread(manager.list_available_models)
    cards = [
        OpenAIModelCard(
            id=m["name"],
            created=int(datetime.fromisoformat(m["modified_at"]).timestamp()),
            owned_by="intel-arc-models-runner",
        )
        for m in raw_models
    ]
    return OpenAIModelListResponse(data=cards)


@router.get("/models/{model:path}", response_model=OpenAIModelCard)
async def get_model(
    model: str,
    manager: ModelManager = Depends(get_model_manager),
) -> OpenAIModelCard:
    """Retrieve metadata card for a specific model identifier.

    Args:
        model: Target model lookup name or path.
        manager: Injected ModelManager instance.

    Returns:
        OpenAIModelCard descriptor.

    Raises:
        HTTPException: If model is not found in search paths.
    """
    try:
        resolved = manager.resolve_model_path(model)
        mtime = int(resolved.stat().st_mtime) if resolved.exists() else int(time.time())
        return OpenAIModelCard(id=resolved.name, created=mtime, owned_by="intel-arc-models-runner")
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=model_not_found_detail(model)) from exc


def measured_usage(telemetry: Any) -> Optional[Dict[str, int]]:
    """Build an OpenAI usage object from engine-measured counts only.

    Args:
        telemetry: EngineTelemetry of the finished request.

    Returns:
        Optional[Dict[str, int]]: prompt/completion/total tokens, or None when the engine measured neither.
        A prompt count of 0 means unmeasured; completion counts are valid when metrics_source is not unavailable.
    """
    prompt = telemetry.prompt_tokens if telemetry.prompt_tokens > 0 else None
    completion = telemetry.total_tokens if telemetry.metrics_source != "unavailable" else None
    if prompt is None and completion is None:
        return None
    usage: Dict[str, int] = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if prompt is not None and completion is not None:
        usage["total_tokens"] = prompt + completion
    return usage


def _chat_prompt(request: OpenAIChatCompletionRequest) -> str:
    """Render the request messages into the ChatML prompt the engines expect.

    Args:
        request: Validated chat completion request.

    Returns:
        str: Prompt ending with the assistant turn opener.
    """
    formatted_lines: List[str] = [f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>" for msg in request.messages]
    formatted_lines.append("<|im_start|>assistant\n")
    return "\n".join(formatted_lines)


def _generation_config(request: OpenAIChatCompletionRequest) -> GenerationConfig:
    """Map OpenAI sampling fields onto the engine GenerationConfig.

    Args:
        request: Validated chat completion request.

    Returns:
        GenerationConfig: Config with defaults for every field the client left unset.
    """
    stops: List[str] = []
    if request.stop:
        stops = [request.stop] if isinstance(request.stop, str) else list(request.stop)
    rep_penalty = 1.1
    if request.frequency_penalty is not None and request.frequency_penalty > 0:
        rep_penalty = 1.0 + float(request.frequency_penalty)
    return GenerationConfig(
        max_new_tokens=request.max_tokens or DEFAULT_MAX_NEW_TOKENS,
        temperature=request.temperature if request.temperature is not None else 1.0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        repetition_penalty=rep_penalty,
        stop_strings=stops,
        stream=bool(request.stream),
    )


def _sse_chunk(
    completion_id: str,
    created_ts: int,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str],
) -> str:
    """Serialise one OpenAI chat.completion.chunk as an SSE data line.

    Args:
        completion_id: Stable id shared by every chunk of the response.
        created_ts: Unix timestamp shared by every chunk.
        model: Requested model name.
        delta: The delta object (role, content, or empty for the final chunk).
        finish_reason: Completion reason on the final chunk, else None.

    Returns:
        str: ``data: {...}`` line terminated by a blank line.
    """
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _forward_job_failure(job: "asyncio.Task[Any]", stream: AsyncTokenStream, loop: asyncio.AbstractEventLoop) -> None:
    """Push a job failure that never reached the job body (for example a refused submit) into the stream.

    Args:
        job: The finished submit() task.
        stream: Stream the SSE generator is waiting on.
        loop: Event loop that owns the stream.
    """
    if job.cancelled():
        return
    exc = job.exception()
    if exc is not None:
        stream.push_error(exc, loop)


def _stream_failure(exc: BaseException, model: str) -> BaseException:
    """Translate a failure that arrived before the first token into the exception to raise.

    Args:
        exc: Exception carried by the stream's error event.
        model: Requested model name.

    Returns:
        BaseException: EngineUnavailableError unchanged, or an HTTPException for load and other failures.
    """
    if isinstance(exc, EngineUnavailableError):
        return exc
    if isinstance(exc, LoadPhaseError):
        return load_failure_to_http(exc, model)
    return hidden_error("generation failed", exc, "Generation failed", model=model)


@router.post("/chat/completions", response_model_exclude_none=True)
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    raw_request: Request,
    manager: ModelManager = Depends(get_model_manager),
) -> Any:
    """Execute chat completion compliant with OpenAI POST /v1/chat/completions.

    Every engine call runs on the serial inference worker, so the event loop stays free.

    Args:
        request: OpenAIChatCompletionRequest payload.
        raw_request: Injected FastAPI HTTP Request instance.
        manager: Injected ModelManager instance.

    Returns:
        StreamingResponse (text/event-stream) or OpenAIChatCompletionResponse.

    Raises:
        HTTPException: If messages array is empty, the model is unknown or cannot be loaded.
        EngineUnavailableError: If the worker is stopping or wedged (HTTP 503).
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="Messages array must not be empty")

    worker = get_worker(raw_request.app)
    loop = asyncio.get_running_loop()
    chat_prompt = _chat_prompt(request)
    gen_config = _generation_config(request)
    batch_pipeline = getattr(raw_request.app.state, "batch_pipeline", None)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    def load_engine(ctx: JobContext) -> Any:
        """Load the engine and prime the prefix cache; runs on the worker thread."""
        engine = ctx.load(lambda: manager.get_or_load_engine(request.model, device="GPU"))
        if batch_pipeline and hasattr(batch_pipeline, "prefix_cache"):
            match_prefix_with_real_tokens(batch_pipeline, engine, chat_prompt)
        return engine

    if request.stream:
        stream = AsyncTokenStream()

        def run_stream(ctx: JobContext) -> None:
            """Generate on the worker thread and push every token into the stream."""
            try:
                engine = load_engine(ctx)
                finish_reason: Optional[str] = None
                for chunk in engine.generate_stream(chat_prompt, gen_config):
                    if stream.is_cancelled:
                        return
                    if chunk.is_finished:
                        finish_reason = chunk.finish_reason
                    else:
                        stream.push_chunk(chunk.text, loop)
                stream.push_done((engine.get_telemetry(), finish_reason), loop)
            except Exception as exc:
                stream.push_error(exc, loop)

        job = asyncio.create_task(worker.submit(run_stream))
        job.add_done_callback(lambda finished: _forward_job_failure(finished, stream, loop))
        events = stream.__aiter__()
        try:
            first = await events.__anext__()
        except BaseException:
            stream.cancel()
            raise
        if first[0] == "error":
            stream.cancel()
            raise _stream_failure(first[1], request.model)

        async def merged() -> AsyncIterator[Any]:
            """Yield the already-read first event, then the rest of the stream."""
            yield first
            if first[0] == "chunk":
                async for item in events:
                    yield item

        async def sse_generator() -> AsyncIterator[str]:
            tokens_generated = 0
            role_sent = False
            try:
                async for kind, payload in merged():
                    if await raw_request.is_disconnected():
                        break
                    if not role_sent:
                        role_sent = True
                        yield _sse_chunk(completion_id, created_ts, request.model, {"role": "assistant"}, None)
                    if kind == "chunk":
                        tokens_generated += 1
                        yield _sse_chunk(completion_id, created_ts, request.model, {"content": payload}, None)
                    elif kind == "done":
                        manager.record_generation_metrics(tokens_generated)
                        yield _sse_chunk(completion_id, created_ts, request.model, {}, payload[1] or "stop")
                    else:
                        raise payload
                yield "data: [DONE]\n\n"
            finally:
                stream.cancel()

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    def run(ctx: JobContext) -> Any:
        """Generate the full completion on the worker thread."""
        engine = load_engine(ctx)
        text = engine.generate(chat_prompt, gen_config)
        return text, engine.get_telemetry()

    try:
        text, telemetry = await worker.submit(run)
    except LoadPhaseError as exc:
        raise load_failure_to_http(exc, request.model) from exc.__cause__

    usage = measured_usage(telemetry)
    if telemetry.metrics_source != "unavailable":
        manager.record_generation_metrics(telemetry.total_tokens)

    return OpenAIChatCompletionResponse(
        id=completion_id,
        created=created_ts,
        model=request.model,
        choices=[
            OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
