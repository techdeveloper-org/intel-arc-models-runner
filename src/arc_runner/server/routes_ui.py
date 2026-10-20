"""Web UI and dashboard endpoints for Intel Arc Model Studio.

Serves embedded single-page application assets, local Ollama model catalogs,
binary GGUF inspection data, and Ahead-of-Time compilation triggers.
"""

import math
from pathlib import Path
import time
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from arc_runner.core.logging import get_logger
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.engine.embedding_engine import EmbeddingEngine
from arc_runner.hub.ollama_importer import OllamaImporter
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.transpiler import GGUFTranspiler

logger = get_logger(__name__)

router = APIRouter(tags=["UI & Studio"])


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
    return HTMLResponse(content=content)


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
async def inspect_gguf_model(payload: InspectRequest) -> dict[str, Any]:
    """Parse and return binary GGUF container metadata and tensor tables.

    Args:
        payload: InspectRequest containing model identifier or physical file path.

    Returns:
        Dictionary containing architecture, parameters, and quant distribution.
    """
    blob_path: Optional[Path] = None
    importer = OllamaImporter()

    if payload.path:
        blob_path = Path(payload.path)
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
    except Exception as e:
        logger.error(f"Failed to inspect GGUF container at '{blob_path}': {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/aot/compile")
async def compile_aot(payload: AOTCompileRequest) -> dict[str, Any]:
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
        resolved_path = target_path
    else:
        from arc_runner.hub.manager import HubManager
        hub_mgr = HubManager()
        local_models = hub_mgr.list_local()
        for m in local_models:
            if m["model_id"] == payload.model_id:
                resolved_path = Path(m["path"])
                break

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
        logger.error(f"AOT compilation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/gguf/convert")
async def convert_gguf(payload: ConvertRequest) -> dict[str, Any]:
    """Transpile a local GGUF container into OpenVINO IR format.

    Args:
        payload: ConvertRequest containing source file path and output directory.

    Returns:
        Dictionary summarizing transpilation metadata and output artifacts.
    """
    input_path = Path(payload.path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"GGUF container not found at '{payload.path}'")

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
        logger.error(f"GGUF transpilation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        logger.error(f"Embedding test bench failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
