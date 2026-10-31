"""Web UI and dashboard endpoints for Intel Arc Model Studio.

Serves embedded single-page application assets, local Ollama model catalogs,
binary GGUF inspection data, and Ahead-of-Time compilation triggers.
"""

import asyncio
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from arc_runner.core.exceptions import GGUFParseError, InvalidGGUFError
from arc_runner.core.logging import get_logger
from arc_runner.core.path_policy import allowed_roots, ensure_path_allowed
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.engine.embedding_engine import EmbeddingEngine
from arc_runner.hub.ollama_importer import OllamaImporter
from arc_runner.server.errors import hidden_error
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.transpiler import (
    GGUFTranspiler,
    TranspilationError,
    UnsupportedBitWidthError,
)

logger = get_logger(__name__)

router = APIRouter(tags=["UI & Studio"])


def _models_dirs(request: Request) -> list:
    """Return the models directories of the running server.

    Args:
        request: Active request.

    Returns:
        list: Model directories, defaulting to ./models.
    """
    manager = getattr(request.app.state, "model_manager", None)
    dirs = list(getattr(manager, "model_dirs", []) or [])
    return dirs or [Path("models")]


def _path_roots(request: Request) -> list:
    """Return the roots request-supplied filesystem paths may point into.

    Args:
        request: Active request.

    Returns:
        list: Models directories, the Ollama directory and ARC_ALLOWED_PATHS entries.
    """
    return allowed_roots([*_models_dirs(request), OllamaImporter().ollama_dir])


class InspectRequest(BaseModel):
    """Payload specifying model identifier or file path for binary GGUF inspection."""

    model: Optional[str] = None
    path: Optional[str] = None


class AOTCompileRequest(BaseModel):
    """Payload requesting ahead-of-time compilation for a model on target silicon."""

    model_id: str
    device: str = "GPU"
    prewarm: bool = True


class ConvertRequest(BaseModel):
    """Payload requesting GGUF to OpenVINO IR transpilation."""

    path: str
    output_dir: str
    precision: str = "int4"
    device: str = "GPU"


class EmbeddingTestRequest(BaseModel):
    """Payload for comparing two texts with cosine similarity on NPU/GPU."""

    query_a: str
    query_b: str
    model: str = "nomic-embed-text"
    device: str = "NPU"


@router.get("/ui", response_class=HTMLResponse)
async def get_ui_dashboard(request: Request) -> HTMLResponse:
    """Render and serve the Intel Arc Model Studio single-page application.

    Args:
        request: FastAPI HTTP request.

    Returns:
        HTMLResponse containing index.html dashboard markup.
    """
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Studio template index.html not found")
    content = template_path.read_text(encoding="utf-8")
    routes = {
        "/": "tab-cockpit",
        "/ui": "tab-cockpit",
        "/cockpit": "tab-cockpit",
        "/chat": "tab-chat",
        "/models/ollama": "tab-ollama",
        "/models/hub": "tab-hub",
        "/benchmarks": "tab-benchmark",
        "/embeddings": "tab-embeddings",
        "/studio": "tab-studio",
    }
    titles = {
        "tab-cockpit": "Heterogeneous Silicon Cockpit",
        "tab-chat": "Dual-Silicon Chat Playground",
        "tab-ollama": "Ollama & GGUF Quantization Studio",
        "tab-hub": "OpenVINO Model Hub & Downloader",
        "tab-benchmark": "Hardware Benchmarking Suite",
        "tab-embeddings": "Intel® AI Boost Vector Embeddings Studio",
        "tab-studio": "Intel® Arc™ Multi-Modal Autonomous Studio",
    }
    active_tab = routes.get(request.url.path, "tab-cockpit")

    def set_nav_active(match: re.Match[str]) -> str:
        prefix, current, suffix, tab_id = match.groups()
        active = tab_id == active_tab
        return f'{prefix}{" active" if active else ""}{suffix}'

    content = re.sub(
        r'(<a class="nav-item)( active)?("[^>]*data-tab="([^"]+)")',
        set_nav_active,
        content,
    )
    content = re.sub(
        r'(<section id=")(tab-[^"]+)(" class="tab-pane)( active)?(")',
        lambda match: f'{match.group(1)}{match.group(2)}{match.group(3)}{" active" if match.group(2) == active_tab else ""}{match.group(5)}',
        content,
    )
    content = content.replace(
        '<h1 id="page-title">Heterogeneous Silicon Cockpit</h1>',
        f'<h1 id="page-title">{titles[active_tab]}</h1>',
    )
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store"})


@router.get("/api/ollama/local")
async def get_local_ollama_models() -> dict[str, Any]:
    """Scan and return locally available Ollama models with blob resolution status.

    Returns:
        Dictionary containing list of discovered Ollama manifests.
    """
    importer = OllamaImporter()
    manifests = importer.discover_models()
    result = []
    for m in manifests:
        rec = importer.map_to_openvino_profile(m.model_name)
        result.append({
            "model_name": m.model_name,
            "tag": m.tag,
            "manifest_path": m.manifest_path,
            "model_digest": m.model_digest,
            "model_blob_path": m.model_blob_path,
            "model_size_bytes": m.model_size_bytes,
            "recommended_openvino": rec.model_id if rec else None,
        })
    return {"models": result, "count": len(result)}


@router.post("/api/ollama/inspect")
async def inspect_gguf_model(payload: InspectRequest, request: Request) -> dict[str, Any]:
    """Parse and return binary GGUF container metadata and tensor tables.

    Args:
        payload: InspectRequest containing model identifier or physical file path.

    Returns:
        Dictionary containing architecture, parameters, and quant distribution.
    """
    blob_path: Optional[Path] = None
    importer = OllamaImporter()

    if payload.path:
        blob_path = ensure_path_allowed(payload.path, _path_roots(request))
    elif payload.model:
        manifest = importer.find_manifest(payload.model)
        if manifest and manifest.model_blob_path:
            blob_path = Path(manifest.model_blob_path)

    if not blob_path or not blob_path.exists():
        raise HTTPException(status_code=404, detail=f"GGUF container or blob not found for '{payload.model or payload.path}'")

    try:
        parser = GGUFParser(blob_path)
        summary = parser.get_summary()
        return {
            "file_path": summary.file_path,
            "file_size_bytes": summary.file_size_bytes,
            "architecture": summary.architecture,
            "context_length": summary.context_length,
            "embedding_length": summary.embedding_length,
            "block_count": summary.block_count,
            "head_count": summary.head_count,
            "tensor_count": summary.tensor_count,
            "total_parameters": summary.total_parameters,
            "quant_distribution": summary.quant_distribution,
            "primary_quant_type": summary.primary_quant_type,
        }
    except (GGUFParseError, InvalidGGUFError) as e:
        logger.error(f"Failed to inspect GGUF container at '{blob_path}': {e}")
        raise HTTPException(status_code=400, detail="Not a valid GGUF container")
    except Exception as e:
        logger.error(f"Failed to inspect GGUF container at '{blob_path}': {e}")
        raise HTTPException(status_code=400, detail="Could not inspect the container")


@router.post("/api/aot/compile")
async def compile_aot(payload: AOTCompileRequest, request: Request) -> dict[str, Any]:
    """Execute ahead-of-time compilation for specified model on target silicon.

    Args:
        payload: AOTCompileRequest specifying model and target accelerator.

    Returns:
        Dictionary detailing compile latency, warm load latency, and speedup multiplier.
    """
    importer = OllamaImporter()
    aot_mgr = AOTCompilationManager()

    resolved_path: Optional[Path] = None
    target_path = Path(payload.model_id)
    if target_path.exists():
        resolved_path = ensure_path_allowed(target_path, _path_roots(request))
    else:
        from arc_runner.core.exceptions import ModelNotFoundError
        from arc_runner.hub.manager import HubManager
        hub_mgr = HubManager()
        try:
            resolved_path = hub_mgr.resolve_model_dir(payload.model_id)
        except ModelNotFoundError:
            pass

        if not resolved_path:
            manifest = importer.find_manifest(payload.model_id)
            if manifest and manifest.model_blob_path:
                resolved_path = Path(manifest.model_blob_path)

    if not resolved_path or not resolved_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{payload.model_id}' not found locally in hub or Ollama directory",
        )

    try:
        res = aot_mgr.compile_model(
            model_path=resolved_path,
            device=payload.device,
            prewarm=payload.prewarm,
            model_id=payload.model_id,
        )
        return {
            "model_id": res.model_id,
            "device": res.device,
            "compilation_time_ms": res.compilation_time_ms,
            "warm_load_time_ms": res.warm_load_time_ms,
            "blob_size_bytes": res.blob_size_bytes,
            "cache_path": res.cache_path,
            "speedup_multiplier": res.speedup_multiplier,
        }
    except Exception as e:
        raise hidden_error("aot compilation failed", e, "AOT compilation failed", model=payload.model_id) from e


@router.post("/api/gguf/convert")
async def convert_gguf(payload: ConvertRequest, request: Request) -> dict[str, Any]:
    """Transpile a local GGUF container into OpenVINO IR format.

    Args:
        payload: ConvertRequest containing source file path and output directory.

    Returns:
        Dictionary summarizing transpilation metadata and output artifacts.
    """
    input_path = ensure_path_allowed(payload.path, _path_roots(request))
    ensure_path_allowed(payload.output_dir, _models_dirs(request))
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="GGUF container not found")

    try:
        transpiler = GGUFTranspiler(str(input_path), payload.output_dir)
        res = transpiler.transpile(precision=payload.precision, target_device=payload.device)
        return {
            "model_name": res.model_name,
            "total_tensors": res.total_tensors,
            "source_format": res.source_format,
            "target_precision": res.target_precision,
            "xml_path": str(res.xml_path),
            "bin_path": str(res.bin_path),
            "total_bytes": res.total_bytes,
            "duration_ms": res.duration_ms,
        }
    except Exception as e:
        raise hidden_error("gguf transpilation failed", e, "GGUF conversion failed") from e


@router.get("/api/gguf/convert/stream")
async def convert_gguf_stream(
    request: Request,
    path: str = Query(..., description="Filesystem path to source GGUF container"),
    output_dir: str = Query(..., description="Target export directory"),
    precision: str = Query("int4", description="Quantization precision ('int4', 'int8', 'fp16')"),
    device: str = Query("GPU", description="Target hardware ('GPU', 'NPU', 'CPU')"),
) -> StreamingResponse:
    """Stream real-time GGUF to OpenVINO IR transpilation progress via Server-Sent Events.

    Args:
        path: Filesystem path to source GGUF file.
        output_dir: Target directory for OpenVINO IR files.
        precision: Quantization precision string.
        device: Hardware device identifier.

    Returns:
        StreamingResponse with media type 'text/event-stream' transmitting 'progress',
        'log', 'complete', and 'error' SSE frames.
    """
    resolved_path = ensure_path_allowed(path, _path_roots(request))
    resolved_out = ensure_path_allowed(output_dir, _models_dirs(request))

    async def sse_event_generator():

        if not resolved_path.exists():
            err_payload = json.dumps({"error": f"Source GGUF file does not exist: {path}"})
            yield f"event: error\ndata: {err_payload}\n\n"
            return

        init_msg = json.dumps({"timestamp": time.strftime("%H:%M:%S"), "level": "INFO", "message": f"Initializing transpiler for {resolved_path.name}"})
        yield f"event: log\ndata: {init_msg}\n\n"

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker_thread() -> None:
            try:
                transpiler = GGUFTranspiler(str(resolved_path), str(resolved_out))
                for progress in transpiler.transpile_stream(
                    precision=precision,
                    target_device=device,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("progress", progress))
                loop.call_soon_threadsafe(queue.put_nowait, ("complete", transpiler.last_result))
            except UnsupportedBitWidthError:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", "Unsupported target precision. Must be int4, int8, or fp16."))
            except Exception as exc:
                logger.error("gguf stream transpilation failed", extra={"error_class": type(exc).__name__, "error_detail": str(exc)[:1000]}, exc_info=exc)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", "GGUF conversion failed"))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        worker_task = asyncio.to_thread(worker_thread)
        asyncio.create_task(worker_task)

        while True:
            item_type, item_data = await queue.get()
            if item_type == "done":
                break
            elif item_type == "progress":
                payload = {
                    "tensor_index": item_data.tensor_index,
                    "total_tensors": item_data.total_tensors,
                    "current_tensor_name": item_data.current_tensor_name,
                    "current_tensor_type": item_data.current_tensor_type,
                    "tensor_bytes": item_data.tensor_bytes,
                    "processed_bytes": item_data.processed_bytes,
                    "total_bytes": item_data.total_bytes,
                    "progress_pct": item_data.progress_pct,
                    "throughput_mb_s": item_data.throughput_mb_s,
                    "elapsed_ms": item_data.elapsed_ms,
                    "eta_seconds": item_data.eta_seconds,
                    "sqnr_db": item_data.sqnr_db,
                    "sha256_partial": item_data.sha256_partial,
                }
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            elif item_type == "complete":
                payload = {
                    "model_name": item_data.model_name,
                    "total_tensors": item_data.total_tensors,
                    "target_precision": item_data.target_precision,
                    "xml_path": str(item_data.xml_path),
                    "bin_path": str(item_data.bin_path),
                    "total_bytes": item_data.total_bytes,
                    "duration_ms": item_data.duration_ms,
                    "final_sqnr_db": item_data.average_sqnr_db,
                    "sha256_hash": item_data.sha256_hash,
                }
                yield f"event: complete\ndata: {json.dumps(payload)}\n\n"
            elif item_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': item_data})}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/embeddings/test")
async def test_embeddings(payload: EmbeddingTestRequest, request: Request) -> dict[str, Any]:
    """Compute vector embeddings and cosine similarity between two queries on Intel NPU/GPU.

    Args:
        payload: EmbeddingTestRequest with two input queries.
        request: FastAPI HTTP request with application state.

    Returns:
        Dictionary detailing vectors, dimensions, and cosine similarity metric.
    """
    model_mgr = request.app.state.model_manager
    t_start = time.perf_counter()

    try:
        engine = model_mgr.get_or_load_embedding_engine(payload.model, device=payload.device)
        vec_a = engine.embed_query(payload.query_a)
        vec_b = engine.embed_query(payload.query_b)
        sim = engine.compute_similarity(vec_a, vec_b)
        duration_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "model": payload.model,
            "device": engine.device,
            "dimension": len(vec_a),
            "cosine_similarity": sim,
            "angular_distance": math.acos(max(-1.0, min(1.0, sim))) / math.pi,
            "duration_ms": duration_ms,
            "sample_vec_a": vec_a[:5],
            "sample_vec_b": vec_b[:5],
        }
    except Exception as e:
        raise hidden_error("embedding test failed", e, "Embedding test failed", model=payload.model) from e
