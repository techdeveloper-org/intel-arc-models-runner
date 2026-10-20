"""Web UI and dashboard endpoints for Intel Arc Model Studio.

Serves embedded single-page application assets, local Ollama model catalogs,
binary GGUF inspection data, and Ahead-of-Time compilation triggers.
"""

from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from arc_runner.core.logging import get_logger
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.hub.ollama_importer import OllamaImporter
from arc_runner.quant.gguf_parser import GGUFParser

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
