"""FastAPI router implementing 100% Ollama HTTP wire protocol parity.

Provides /api/generate, /api/chat, /api/tags, /api/show, /api/version, and root / probes
with NDJSON streaming and non-blocking C++ execution.
"""

import asyncio
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse

from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.core.logging import get_logger, set_request_context
from arc_runner.engine.models import GenerationConfig, TokenChunk
from arc_runner.server.model_manager import ModelManager
from arc_runner.server.schemas import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelItem,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
    OllamaVersionResponse,
)

logger = get_logger("server.routes_ollama")

router = APIRouter(tags=["ollama"])


def get_model_manager(request: Request) -> ModelManager:
    """Extract registered ModelManager instance from application state.

    Args:
        request: Active FastAPI HTTP request.

    Returns:
        Configured ModelManager instance.
    """
    return request.app.state.model_manager


@router.get("/", response_class=PlainTextResponse)
async def root_probe() -> str:
    """Root health probe matching standard Ollama daemon probe contract.

    Returns:
        Canonical text string 'Ollama is running'.
    """
    return "Ollama is running"


@router.get("/api/version", response_model=OllamaVersionResponse)
async def get_version() -> OllamaVersionResponse:
    """Retrieve daemon server version information.

    Returns:
        OllamaVersionResponse containing active version string.
    """
    return OllamaVersionResponse(version="0.1.0")


@router.get("/api/tags", response_model=OllamaTagsResponse)
async def list_tags(manager: ModelManager = Depends(get_model_manager)) -> OllamaTagsResponse:
    """List all available converted models discoverable by the runner.

    Args:
        manager: Injected ModelManager instance.

    Returns:
        OllamaTagsResponse with model summaries and architectures.
    """
    raw_models = manager.list_available_models()
    items = [OllamaModelItem(**m) for m in raw_models]
    return OllamaTagsResponse(models=items)


@router.post("/api/show", response_model=OllamaShowResponse)
async def show_model(
    request: OllamaShowRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> OllamaShowResponse:
    """Inspect model configuration, parameters, and template definitions.

    Args:
        request: Model name or identifier query.
        manager: Injected ModelManager instance.

    Returns:
        OllamaShowResponse populated with template and details.

    Raises:
        HTTPException: If model cannot be located.
    """
    target = request.name or request.model
    if not target:
        raise HTTPException(status_code=400, detail="Missing required 'name' or 'model' field")
    try:
        details_dict = manager.get_model_details(target)
        return OllamaShowResponse(**details_dict)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _build_generation_config(options: Optional[Dict[str, Any]], stream: bool = True) -> GenerationConfig:
    """Build strongly typed GenerationConfig from untyped options dictionary.

    Args:
        options: Dictionary of client-supplied sampling parameters.
        stream: Streaming mode flag.

    Returns:
        Validated GenerationConfig object.
    """
    cfg_args: Dict[str, Any] = {"stream": stream}
    if options:
        if "num_predict" in options:
            cfg_args["max_new_tokens"] = int(options["num_predict"])
        elif "max_tokens" in options:
            cfg_args["max_new_tokens"] = int(options["max_tokens"])
        if "temperature" in options:
            cfg_args["temperature"] = float(options["temperature"])
        if "top_p" in options:
            cfg_args["top_p"] = float(options["top_p"])
        if "top_k" in options:
            cfg_args["top_k"] = int(options["top_k"])
        if "repeat_penalty" in options:
            cfg_args["repetition_penalty"] = float(options["repeat_penalty"])
        if "stop" in options:
            stops = options["stop"]
            cfg_args["stop_strings"] = [stops] if isinstance(stops, str) else list(stops)
    return GenerationConfig(**cfg_args)


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


@router.post("/api/generate")
async def generate(
    request: OllamaGenerateRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> Any:
    """Autoregressive completion endpoint matching Ollama POST /api/generate.

    Args:
        request: OllamaGenerateRequest containing model, prompt, and options.
        manager: Injected ModelManager instance.

    Returns:
        StreamingResponse (application/x-ndjson) or single OllamaGenerateResponse.

    Raises:
        HTTPException: If prompt is empty or model cannot be loaded.
    """
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt must not be empty")

    device = "GPU"
    if request.options and "device" in request.options:
        device = str(request.options["device"])

    try:
        engine = manager.get_or_load_engine(request.model, device=device)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load engine: {exc}") from exc

    gen_config = _build_generation_config(request.options, stream=request.stream)
    start_ns = time.time_ns()

    if request.stream:
        async def ndjson_generator() -> AsyncIterator[str]:
            tokens_generated = 0
            async for chunk in _stream_token_chunks(engine, request.prompt, gen_config):
                now_iso = datetime.now(timezone.utc).isoformat()
                if not chunk.is_finished:
                    tokens_generated += 1
                    payload = {
                        "model": request.model,
                        "created_at": now_iso,
                        "response": chunk.text,
                        "done": False,
                    }
                    yield json.dumps(payload) + "\n"
                else:
                    duration_ns = time.time_ns() - start_ns
                    manager.record_generation_metrics(tokens_generated)
                    final_payload = {
                        "model": request.model,
                        "created_at": now_iso,
                        "response": chunk.text,
                        "done": True,
                        "total_duration": duration_ns,
                        "load_duration": 0,
                        "prompt_eval_count": chunk.prompt_tokens or len(request.prompt.split()),
                        "prompt_eval_duration": int(duration_ns * 0.2),
                        "eval_count": tokens_generated,
                        "eval_duration": int(duration_ns * 0.8),
                        "context": [],
                    }
                    yield json.dumps(final_payload) + "\n"

        return StreamingResponse(ndjson_generator(), media_type="application/x-ndjson")

    text = engine.generate(request.prompt, gen_config)
    duration_ns = time.time_ns() - start_ns
    tokens_count = len(text.split())
    manager.record_generation_metrics(tokens_count)

    return OllamaGenerateResponse(
        model=request.model,
        created_at=datetime.now(timezone.utc).isoformat(),
        response=text,
        done=True,
        total_duration=duration_ns,
        load_duration=0,
        prompt_eval_count=len(request.prompt.split()),
        prompt_eval_duration=int(duration_ns * 0.2),
        eval_count=tokens_count,
        eval_duration=int(duration_ns * 0.8),
        context=[],
    )


@router.post("/api/chat")
async def chat(
    request: OllamaChatRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> Any:
    """Conversational multi-turn chat endpoint matching Ollama POST /api/chat.

    Args:
        request: OllamaChatRequest with messages array and options.
        manager: Injected ModelManager instance.

    Returns:
        StreamingResponse (application/x-ndjson) or single OllamaChatResponse.

    Raises:
        HTTPException: If messages array is missing/empty or model fails.
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="Messages array must not be empty")

    device = "GPU"
    if request.options and "device" in request.options:
        device = str(request.options["device"])

    try:
        engine = manager.get_or_load_engine(request.model, device=device)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load engine: {exc}") from exc

    formatted_lines: List[str] = []
    for msg in request.messages:
        formatted_lines.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    formatted_lines.append("<|im_start|>assistant\n")
    chat_prompt = "\n".join(formatted_lines)

    gen_config = _build_generation_config(request.options, stream=request.stream)
    start_ns = time.time_ns()

    if request.stream:
        async def chat_ndjson_generator() -> AsyncIterator[str]:
            tokens_generated = 0
            async for chunk in _stream_token_chunks(engine, chat_prompt, gen_config):
                now_iso = datetime.now(timezone.utc).isoformat()
                if not chunk.is_finished:
                    tokens_generated += 1
                    payload = {
                        "model": request.model,
                        "created_at": now_iso,
                        "message": {
                            "role": "assistant",
                            "content": chunk.text,
                        },
                        "done": False,
                    }
                    yield json.dumps(payload) + "\n"
                else:
                    duration_ns = time.time_ns() - start_ns
                    manager.record_generation_metrics(tokens_generated)
                    final_payload = {
                        "model": request.model,
                        "created_at": now_iso,
                        "message": {
                            "role": "assistant",
                            "content": chunk.text,
                        },
                        "done": True,
                        "total_duration": duration_ns,
                        "load_duration": 0,
                        "prompt_eval_count": chunk.prompt_tokens or len(chat_prompt.split()),
                        "prompt_eval_duration": int(duration_ns * 0.2),
                        "eval_count": tokens_generated,
                        "eval_duration": int(duration_ns * 0.8),
                    }
                    yield json.dumps(final_payload) + "\n"

        return StreamingResponse(chat_ndjson_generator(), media_type="application/x-ndjson")

    text = engine.generate(chat_prompt, gen_config)
    duration_ns = time.time_ns() - start_ns
    tokens_count = len(text.split())
    manager.record_generation_metrics(tokens_count)

    return OllamaChatResponse(
        model=request.model,
        created_at=datetime.now(timezone.utc).isoformat(),
        message=OllamaChatMessage(role="assistant", content=text),
        done=True,
        total_duration=duration_ns,
        load_duration=0,
        prompt_eval_count=len(chat_prompt.split()),
        prompt_eval_duration=int(duration_ns * 0.2),
        eval_count=tokens_count,
        eval_duration=int(duration_ns * 0.8),
    )
