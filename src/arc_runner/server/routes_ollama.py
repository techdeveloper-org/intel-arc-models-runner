"""FastAPI router implementing 100% Ollama HTTP wire protocol parity.

Provides /api/generate, /api/chat, /api/tags, /api/show, /api/version, and root / probes
with NDJSON streaming and non-blocking C++ execution.
"""

from arc_runner._version import __version__
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from arc_runner.server.prefix_probe import match_prefix_with_real_tokens
from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.core.logging import get_logger, set_request_context
from arc_runner.engine.models import GenerationConfig, LoRAAdapterSpec, TokenChunk
from arc_runner.hub.downloader import DownloadProgress
from arc_runner.hub.manager import HubManager
from arc_runner.server.errors import hidden_error, model_not_found_detail
from arc_runner.server.model_manager import ModelManager
from arc_runner.server.schemas import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaCopyRequest,
    OllamaDeleteRequest,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelItem,
    OllamaPullProgressResponse,
    OllamaPullRequest,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaStatusResponse,
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


def get_hub_manager(request: Request) -> HubManager:
    """Extract or lazily initialize HubManager instance from application state.

    Args:
        request: Active FastAPI HTTP request.

    Returns:
        Configured HubManager instance.
    """
    if not hasattr(request.app.state, "hub_manager") or request.app.state.hub_manager is None:
        model_mgr = getattr(request.app.state, "model_manager", None)
        base_dir = model_mgr.model_dirs[0] if (model_mgr and model_mgr.model_dirs) else None
        request.app.state.hub_manager = HubManager(base_models_dir=base_dir, model_manager=model_mgr)
    return request.app.state.hub_manager


@router.get("/")
async def root_probe(request: Request) -> Response:
    """Root health probe matching standard Ollama daemon probe contract.

    Serves embedded Intel Arc Model Studio Web UI when accessed from browsers,
    or returns canonical string 'Ollama is running' for CLI clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return PlainTextResponse("Ollama is running")


@router.get("/api/version", response_model=OllamaVersionResponse)
async def get_version() -> OllamaVersionResponse:
    """Retrieve daemon server version information.

    Returns:
        OllamaVersionResponse containing active version string.
    """
    return OllamaVersionResponse(version=__version__)


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
        raise HTTPException(status_code=404, detail=model_not_found_detail(target)) from exc


def _build_generation_config(
    options: Optional[Dict[str, Any]],
    stream: bool = True,
    adapters: Optional[List[Dict[str, Any]]] = None,
    adapter: Optional[str] = None,
) -> GenerationConfig:
    """Build strongly typed GenerationConfig from untyped options dictionary and LoRA adapter specs.

    Args:
        options: Dictionary of client-supplied sampling parameters.
        stream: Streaming mode flag.
        adapters: Optional list of adapter specification dictionaries.
        adapter: Optional single adapter shortcut name or path.

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

    if (options is None or "temperature" not in options) and cfg_args.get("temperature", 0.0) == 0.0:
        cfg_args["temperature"] = 0.6
    if options is None or not ("num_predict" in options or "max_tokens" in options):
        cfg_args["max_new_tokens"] = 256 if cfg_args.get("temperature", 0.0) > 0 else 128

    lora_specs: List[LoRAAdapterSpec] = []
    if adapter:
        lora_specs.append(LoRAAdapterSpec(name=adapter, scale=1.0))
    if adapters:
        for a in adapters:
            if isinstance(a, dict) and "name" in a:
                lora_specs.append(
                    LoRAAdapterSpec(
                        name=str(a["name"]),
                        scale=float(a.get("scale", 1.0)),
                        path=str(a["path"]) if "path" in a and a["path"] else None,
                    )
                )
    if lora_specs:
        cfg_args["adapters"] = lora_specs

    return GenerationConfig(**cfg_args)


def _format_chat_prompt(messages: List[Any], model_id: str) -> str:
    """Format chat messages using the model family's expected prompt tokens."""
    is_deepseek_r1 = "deepseek" in model_id.lower()
    formatted_lines: List[str] = []
    for msg in messages:
        content = msg.content
        if is_deepseek_r1:
            if msg.role == "system":
                formatted_lines.append(content)
            elif msg.role == "user":
                formatted_lines.append(f"<|User|>{content}")
            elif msg.role == "assistant":
                # Keep prior visible assistant responses as context, excluding private reasoning.
                if "</think>" in content:
                    content = content.rsplit("</think>", 1)[-1].lstrip()
                formatted_lines.append(f"<|Assistant|>{content}<|end_of_sentence|>")
        else:
            formatted_lines.append(f"<|im_start|>{msg.role}\n{content}<|im_end|>")

    if is_deepseek_r1:
        formatted_lines.append("<|Assistant|><think>\n")
    else:
        formatted_lines.append("<|im_start|>assistant\n")
    return "\n".join(formatted_lines)


class _ChatOutputParser:
    """Separate DeepSeek reasoning and answer text across arbitrary stream chunks."""

    def __init__(self, model_id: str) -> None:
        self._reasoning_model = "deepseek" in model_id.lower()
        self._in_thinking = self._reasoning_model
        self._pending = ""

    def feed(self, text: str, *, final: bool = False) -> Dict[str, str]:
        if not self._reasoning_model:
            return {"thinking": "", "content": text}

        data = self._pending + text
        self._pending = ""
        output = {"thinking": "", "content": ""}
        marker = "</think>"

        if self._in_thinking:
            end = data.find(marker)
            if end >= 0:
                output["thinking"] = data[:end]
                output["content"] = data[end + len(marker):].lstrip("\r\n ")
                self._in_thinking = False
            elif final:
                output["thinking"] = data
            else:
                keep = min(len(data), len(marker) - 1)
                split_at = len(data) - keep
                output["thinking"] = data[:split_at]
                self._pending = data[split_at:]
        else:
            output["content"] = data
        return output


def _split_chat_output(text: str, model_id: str) -> Dict[str, str]:
    """Parse a complete model result into optional thinking and answer fields."""
    parser = _ChatOutputParser(model_id)
    return parser.feed(text, final=True)


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


def _eval_timing_ns(engine: Any, duration_ns: int) -> Dict[str, int]:
    """Split a request duration into prompt and decode nanoseconds using measured TTFT only.

    Args:
        engine: Engine that served the request.
        duration_ns: Total wall-clock request duration in nanoseconds.

    Returns:
        Dict[str, int]: prompt_eval_duration and eval_duration, both 0 when TTFT was not measured.
    """
    telemetry = engine.get_telemetry()
    if not telemetry.is_ttft_valid:
        return {"prompt_eval_duration": 0, "eval_duration": 0}
    prompt_ns = min(int(telemetry.ttft_ms * 1_000_000), duration_ns)
    return {"prompt_eval_duration": prompt_ns, "eval_duration": max(duration_ns - prompt_ns, 0)}


def _measured_prompt_count(engine: Any) -> Optional[int]:
    """Return the engine-measured prompt token count, or None when it was not measured.

    Args:
        engine: Engine that served the request.

    Returns:
        Optional[int]: Telemetry prompt_tokens when positive, else None (0 means not measured).
    """
    count = engine.get_telemetry().prompt_tokens
    return count if count > 0 else None


def _without_unmeasured(response: Any, *fields: str) -> JSONResponse:
    """Serialize a response model, dropping the named fields when they are None.

    Args:
        response: Pydantic response model.
        *fields: Field names omitted from the wire payload when unmeasured.

    Returns:
        JSONResponse: JSON payload without the unmeasured fields.
    """
    payload = response.model_dump(mode="json")
    for name in fields:
        if payload.get(name) is None:
            payload.pop(name, None)
    return JSONResponse(content=payload)


@router.post("/api/generate")
async def generate(
    request: OllamaGenerateRequest,
    raw_request: Request,
    manager: ModelManager = Depends(get_model_manager),
) -> Any:
    """Autoregressive completion endpoint matching Ollama POST /api/generate.

    Args:
        request: OllamaGenerateRequest containing model, prompt, and options.
        raw_request: Injected FastAPI HTTP Request instance.
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
        raise HTTPException(status_code=404, detail=model_not_found_detail(request.model)) from exc
    except Exception as exc:
        raise hidden_error("engine load failed", exc, "Failed to load engine", model=request.model) from exc

    batch_pipeline = getattr(raw_request.app.state, "batch_pipeline", None)
    if batch_pipeline and hasattr(batch_pipeline, "prefix_cache"):
        match_prefix_with_real_tokens(batch_pipeline, engine, request.prompt)

    gen_config = _build_generation_config(
        request.options,
        stream=request.stream,
        adapters=request.adapters,
        adapter=request.adapter,
    )
    start_ns = time.time_ns()

    if request.stream:
        async def ndjson_generator() -> AsyncIterator[str]:
            tokens_generated = 0
            async for chunk in _stream_token_chunks(engine, request.prompt, gen_config):
                if await raw_request.is_disconnected():
                    break
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
                        **({} if chunk.prompt_tokens is None else {"prompt_eval_count": chunk.prompt_tokens}),
                        "eval_count": tokens_generated,
                        **_eval_timing_ns(engine, duration_ns),
                        "context": [],
                    }
                    yield json.dumps(final_payload) + "\n"

        return StreamingResponse(ndjson_generator(), media_type="application/x-ndjson")

    text = engine.generate(request.prompt, gen_config)
    duration_ns = time.time_ns() - start_ns
    tokens_count = engine.get_telemetry().total_tokens
    manager.record_generation_metrics(tokens_count)

    return _without_unmeasured(
        OllamaGenerateResponse(
            model=request.model,
            created_at=datetime.now(timezone.utc).isoformat(),
            response=text,
            done=True,
            total_duration=duration_ns,
            load_duration=0,
            prompt_eval_count=_measured_prompt_count(engine),
            eval_count=tokens_count,
            **_eval_timing_ns(engine, duration_ns),
            context=[],
        ),
        "prompt_eval_count",
    )


@router.post("/api/chat")
async def chat(
    request: OllamaChatRequest,
    raw_request: Request,
    manager: ModelManager = Depends(get_model_manager),
) -> Any:
    """Conversational multi-turn chat endpoint matching Ollama POST /api/chat.

    Args:
        request: OllamaChatRequest with messages array and options.
        raw_request: Injected FastAPI HTTP Request instance.
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
        raise HTTPException(status_code=404, detail=model_not_found_detail(request.model)) from exc
    except Exception as exc:
        raise hidden_error("engine load failed", exc, "Failed to load engine", model=request.model) from exc

    chat_prompt = _format_chat_prompt(request.messages, request.model)

    batch_pipeline = getattr(raw_request.app.state, "batch_pipeline", None)
    if batch_pipeline and hasattr(batch_pipeline, "prefix_cache"):
        match_prefix_with_real_tokens(batch_pipeline, engine, chat_prompt)

    gen_config = _build_generation_config(
        request.options,
        stream=request.stream,
        adapters=request.adapters,
        adapter=request.adapter,
    )
    start_ns = time.time_ns()

    if request.stream:
        async def chat_ndjson_generator() -> AsyncIterator[str]:
            tokens_generated = 0
            output_parser = _ChatOutputParser(request.model)
            async for chunk in _stream_token_chunks(engine, chat_prompt, gen_config):
                if await raw_request.is_disconnected():
                    break
                now_iso = datetime.now(timezone.utc).isoformat()
                if not chunk.is_finished:
                    tokens_generated += 1
                    parsed_text = output_parser.feed(chunk.text)
                    payload = {
                        "model": request.model,
                        "created_at": now_iso,
                        "message": {
                            "role": "assistant",
                            "content": parsed_text["content"],
                            "thinking": parsed_text["thinking"],
                        },
                        "done": False,
                    }
                    yield json.dumps(payload) + "\n"
                else:
                    parsed_text = output_parser.feed(chunk.text, final=True)
                    duration_ns = time.time_ns() - start_ns
                    manager.record_generation_metrics(tokens_generated)
                    final_payload = {
                        "model": request.model,
                        "created_at": now_iso,
                        "message": {
                            "role": "assistant",
                            "content": parsed_text["content"],
                            "thinking": parsed_text["thinking"],
                        },
                        "done": True,
                        "total_duration": duration_ns,
                        "load_duration": 0,
                        **({} if chunk.prompt_tokens is None else {"prompt_eval_count": chunk.prompt_tokens}),
                        "eval_count": tokens_generated,
                        **_eval_timing_ns(engine, duration_ns),
                    }
                    yield json.dumps(final_payload) + "\n"

        return StreamingResponse(chat_ndjson_generator(), media_type="application/x-ndjson")

    parsed_text = _split_chat_output(engine.generate(chat_prompt, gen_config), request.model)
    duration_ns = time.time_ns() - start_ns
    tokens_count = engine.get_telemetry().total_tokens
    manager.record_generation_metrics(tokens_count)

    return _without_unmeasured(
        OllamaChatResponse(
            model=request.model,
            created_at=datetime.now(timezone.utc).isoformat(),
            message=OllamaChatMessage(
                role="assistant",
                content=parsed_text["content"],
                thinking=parsed_text["thinking"] or None,
            ),
            done=True,
            total_duration=duration_ns,
            load_duration=0,
            prompt_eval_count=_measured_prompt_count(engine),
            eval_count=tokens_count,
            **_eval_timing_ns(engine, duration_ns),
        ),
        "prompt_eval_count",
    )


@router.post("/api/pull")
async def pull_model(
    request: OllamaPullRequest,
    hub: HubManager = Depends(get_hub_manager),
) -> Any:
    """Download and register a model from Hugging Face matching Ollama POST /api/pull.

    Args:
        request: OllamaPullRequest specifying model name and streaming flag.
        hub: Injected HubManager instance.

    Returns:
        StreamingResponse (application/x-ndjson) or OllamaStatusResponse.

    Raises:
        HTTPException: If model name is empty or download fails.
    """
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=422, detail="Model name must not be empty")
    hub.prepare_pull(request.name)

    if request.stream:
        async def ndjson_pull_generator() -> AsyncIterator[str]:
            queue: asyncio.Queue[Union[DownloadProgress, Exception, None]] = asyncio.Queue(maxsize=128)
            loop = asyncio.get_running_loop()

            def worker() -> None:
                try:
                    for prog in hub.pull(request.name, stream=True):
                        while True:
                            try:
                                loop.call_soon_threadsafe(queue.put_nowait, prog)
                                break
                            except asyncio.QueueFull:
                                time.sleep(0.005)
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
                    raise hidden_error("model pull failed", item, "Model pull failed", model=request.name)
                yield json.dumps(item.model_dump(exclude_none=True)) + "\n"

            yield json.dumps({"status": "success"}) + "\n"

        return StreamingResponse(ndjson_pull_generator(), media_type="application/x-ndjson")

    try:
        hub.pull(request.name, stream=False)
        return OllamaStatusResponse(status="success")
    except Exception as exc:
        raise hidden_error("model pull failed", exc, "Model pull failed", model=request.name) from exc


@router.delete("/api/delete", response_model=OllamaStatusResponse)
async def delete_model(
    request: OllamaDeleteRequest,
    hub: HubManager = Depends(get_hub_manager),
) -> OllamaStatusResponse:
    """Delete a local model repository from disk matching Ollama DELETE /api/delete.

    Args:
        request: OllamaDeleteRequest payload.
        hub: Injected HubManager instance.

    Returns:
        OllamaStatusResponse with status 'success'.

    Raises:
        HTTPException: If model is missing or not found.
    """
    target = request.name or request.model
    if not target or not target.strip():
        raise HTTPException(status_code=400, detail="Missing required 'name' or 'model' field")

    try:
        hub.delete(target)
        return OllamaStatusResponse(status="success")
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=model_not_found_detail(target)) from exc
    except Exception as exc:
        raise hidden_error("model delete failed", exc, "Model delete failed", model=target) from exc


@router.post("/api/copy", response_model=OllamaStatusResponse)
async def copy_model(
    request: OllamaCopyRequest,
    hub: HubManager = Depends(get_hub_manager),
) -> OllamaStatusResponse:
    """Clone an existing local model under a new alias matching Ollama POST /api/copy.

    Args:
        request: OllamaCopyRequest containing source and destination aliases.
        hub: Injected HubManager instance.

    Returns:
        OllamaStatusResponse with status 'success'.

    Raises:
        HTTPException: If source is missing or destination already exists.
    """
    try:
        hub.copy(request.source, request.destination)
        return OllamaStatusResponse(status="success")
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=model_not_found_detail(request.source)) from exc
    except ArcRunnerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise hidden_error("model copy failed", exc, "Model copy failed", 409, source=request.source) from exc

