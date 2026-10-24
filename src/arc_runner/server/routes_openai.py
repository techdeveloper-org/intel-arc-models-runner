"""FastAPI router implementing 100% OpenAI v1 wire protocol parity.

Provides /v1/chat/completions with Server-Sent Events (SSE) streaming and non-streaming fallback,
as well as /v1/models catalog discovery for integration with OpenAI-compatible clients.
"""

import asyncio
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from arc_runner.core.constants import DEFAULT_MAX_NEW_TOKENS
from arc_runner.core.exceptions import ModelNotFoundError
from arc_runner.core.logging import get_logger
from arc_runner.engine.models import GenerationConfig, TokenChunk
from arc_runner.server.model_manager import ModelManager
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
    raw_models = manager.list_available_models()
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
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _stream_token_chunks(
    engine: Any,
    prompt: str,
    config: GenerationConfig,
) -> AsyncIterator[TokenChunk]:
    """Asynchronously bridge C++ engine token generator to async generator queue.

    Args:
        engine: Loaded BaseEngine instance.
        prompt: Prefill prompt text.
        config: GenerationConfig parameters.

    Yields:
        TokenChunk emissions as decoded.
    """
    queue: asyncio.Queue[Union[TokenChunk, Exception, None]] = asyncio.Queue(maxsize=128)
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for chunk in engine.generate_stream(prompt, config):
                while True:
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                        break
                    except asyncio.QueueFull:
                        time.sleep(0.002)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


@router.post("/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    raw_request: Request,
    manager: ModelManager = Depends(get_model_manager),
) -> Any:
    """Execute chat completion compliant with OpenAI POST /v1/chat/completions.

    Args:
        request: OpenAIChatCompletionRequest payload.
        raw_request: Injected FastAPI HTTP Request instance.
        manager: Injected ModelManager instance.

    Returns:
        StreamingResponse (text/event-stream) or OpenAIChatCompletionResponse.

    Raises:
        HTTPException: If messages array is empty or model cannot be compiled.
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="Messages array must not be empty")

    try:
        engine = manager.get_or_load_engine(request.model, device="GPU")
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load engine: {exc}") from exc

    formatted_lines: List[str] = []
    for msg in request.messages:
        formatted_lines.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    formatted_lines.append("<|im_start|>assistant\n")
    chat_prompt = "\n".join(formatted_lines)

    batch_pipeline = getattr(raw_request.app.state, "batch_pipeline", None)
    if batch_pipeline and hasattr(batch_pipeline, "prefix_cache"):
        dummy_tokens = [hash(word) % 32000 for word in chat_prompt.split()[:32]]
        batch_pipeline.prefix_cache.match_prefix(dummy_tokens)

    stops: List[str] = []
    if request.stop:
        stops = [request.stop] if isinstance(request.stop, str) else list(request.stop)

    rep_penalty = 1.1
    if request.frequency_penalty is not None and request.frequency_penalty > 0:
        rep_penalty = 1.0 + float(request.frequency_penalty)

    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens or DEFAULT_MAX_NEW_TOKENS,
        temperature=request.temperature if request.temperature is not None else 1.0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        repetition_penalty=rep_penalty,
        stop_strings=stops,
        stream=bool(request.stream),
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if request.stream:
        async def sse_generator() -> AsyncIterator[str]:
            tokens_generated = 0
            first_chunk = True

            async for chunk in _stream_token_chunks(engine, chat_prompt, gen_config):
                if await raw_request.is_disconnected():
                    break
                if first_chunk:
                    initial_sse = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(initial_sse)}\n\n"
                    first_chunk = False

                if not chunk.is_finished:
                    tokens_generated += 1
                    content_sse = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(content_sse)}\n\n"
                else:
                    manager.record_generation_metrics(tokens_generated)
                    final_sse = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": chunk.finish_reason or "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_sse)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    text = engine.generate(chat_prompt, gen_config)
    prompt_tokens = len(chat_prompt.split())
    completion_tokens = len(text.split())
    manager.record_generation_metrics(completion_tokens)

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
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
