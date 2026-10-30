"""FastAPI routes for Multi-LoRA adapter discovery, loading, and hot-swap lifecycle.

Exposes RESTful endpoints for listing local LoRA adapters, pre-loading into memory,
evicting adapters from LRU cache, and inspecting adapter metadata (Proofs M23, M24, P10).
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.core.path_policy import allowed_roots, ensure_path_allowed
from arc_runner.engine.lora_manager import (
    AdapterConfigError,
    AdapterNotFoundError,
    LoRAAdapterMetadata,
)
from arc_runner.server.errors import hidden_error
from arc_runner.server.schemas import (
    LoRAAdapterResponse,
    LoRAInspectRequest,
    LoRALoadRequest,
    LoRAUnloadRequest,
)

logger = get_logger("arc_runner.server.routes_lora")
router = APIRouter(tags=["LoRA Adapters"])


def _to_response_dto(meta: LoRAAdapterMetadata) -> LoRAAdapterResponse:
    """Map internal LoRAAdapterMetadata domain object to Pydantic wire DTO.

    Args:
        meta: Populated metadata instance.

    Returns:
        LoRAAdapterResponse: Serialized wire schema.
    """
    return LoRAAdapterResponse(
        adapter_id=meta.adapter_id,
        adapter_path=str(meta.adapter_path),
        base_model_name=meta.base_model_name,
        rank=meta.rank,
        alpha=meta.alpha,
        target_modules=meta.target_modules,
        file_size_bytes=meta.file_size_bytes,
        is_cached=meta.is_cached,
    )


def _validate_safe_path(path_str: str, request: Request, allow_name: bool = False) -> Path:
    """Validate that a path has no traversal segments and lies inside an allowed root.

    Args:
        path_str: Raw path input string.
        request: Active request, for the models directories.
        allow_name: If True a bare adapter name (no separator, not absolute) is returned unresolved; the
            LoRA manager resolves it below its own directories.

    Returns:
        Path: Resolved filesystem path.

    Raises:
        HTTPException: If path contains directory traversal patterns.
    """
    if ".." in path_str.replace("\\", "/").split("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path traversal not permitted: {path_str}",
        )
    if allow_name and "/" not in path_str.replace("\\", "/") and not Path(path_str).is_absolute():
        return Path(path_str)
    model_mgr_dirs = getattr(request.app.state.model_manager, "model_dirs", [])
    try:
        return ensure_path_allowed(path_str, allowed_roots(model_mgr_dirs or [Path("models")]))
    except ArcRunnerError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.get("/api/adapters", response_model=List[LoRAAdapterResponse])
async def list_adapters(request: Request) -> List[LoRAAdapterResponse]:
    """List all discovered local LoRA adapters and their memory caching status.

    Args:
        request: FastAPI HTTP request containing application state.

    Returns:
        List of LoRAAdapterResponse descriptors.
    """
    model_mgr = request.app.state.model_manager
    discovered = await asyncio.to_thread(model_mgr.lora_manager.discover_adapters)
    return [_to_response_dto(m) for m in discovered]


@router.post("/api/adapters/load", response_model=LoRAAdapterResponse)
async def load_adapter(payload: LoRALoadRequest, request: Request) -> LoRAAdapterResponse:
    """Load an adapter into host/USM memory and insert into LRU cache (Proof P10).

    Args:
        payload: LoRALoadRequest containing adapter identifier or path.
        request: FastAPI HTTP request containing application state.

    Returns:
        LoRAAdapterResponse: Metadata of loaded adapter.

    Raises:
        HTTPException: If adapter is not found or config is corrupted.
    """
    _validate_safe_path(payload.adapter, request, allow_name=True)
    model_mgr = request.app.state.model_manager

    try:
        meta = await asyncio.to_thread(model_mgr.lora_manager.load_adapter, payload.adapter)
        return _to_response_dto(meta)
    except AdapterNotFoundError as exc:
        raise hidden_error("lora adapter not found", exc, f"LoRA adapter '{payload.adapter[:128]}' not found", 404, adapter=payload.adapter) from exc
    except AdapterConfigError as exc:
        raise hidden_error("lora adapter config invalid", exc, "LoRA adapter configuration is invalid", 400, adapter=payload.adapter) from exc
    except Exception as exc:
        raise hidden_error("lora adapter load failed", exc, "Failed to load LoRA adapter", adapter=payload.adapter) from exc


@router.post("/api/adapters/unload")
async def unload_adapter(payload: LoRAUnloadRequest, request: Request) -> Dict[str, Any]:
    """Evict an adapter from memory and release Level Zero USM buffers (Proof P11).

    Args:
        payload: LoRAUnloadRequest containing adapter identifier to evict.
        request: FastAPI HTTP request containing application state.

    Returns:
        Dictionary indicating unload success and adapter identifier.
    """
    model_mgr = request.app.state.model_manager
    success = await asyncio.to_thread(model_mgr.lora_manager.unload_adapter, payload.adapter)
    return {"adapter": payload.adapter, "unloaded": success}


@router.post("/api/adapters/inspect", response_model=LoRAAdapterResponse)
async def inspect_adapter(payload: LoRAInspectRequest, request: Request) -> LoRAAdapterResponse:
    """Inspect adapter container headers without loading weight matrices into memory.

    Args:
        payload: LoRAInspectRequest with filesystem path to inspect.
        request: FastAPI HTTP request containing application state.

    Returns:
        LoRAAdapterResponse: Metadata summary.

    Raises:
        HTTPException: If path is invalid or files are corrupt.
    """
    safe_path = _validate_safe_path(payload.path, request, allow_name=True)
    model_mgr = request.app.state.model_manager

    if not safe_path.is_absolute():
        in_dir = model_mgr.lora_manager.adapters_dir / payload.path
        if in_dir.exists():
            safe_path = in_dir.resolve()

    try:
        meta = await asyncio.to_thread(model_mgr.lora_manager.parse_adapter_metadata, safe_path)
        return _to_response_dto(meta)
    except AdapterNotFoundError as exc:
        raise hidden_error("lora adapter not found", exc, f"LoRA adapter '{payload.path[:128]}' not found", 404, adapter=payload.path) from exc
    except AdapterConfigError as exc:
        raise hidden_error("lora adapter config invalid", exc, "LoRA adapter configuration is invalid", 400, adapter=payload.path) from exc
    except Exception as exc:
        raise hidden_error("lora adapter inspect failed", exc, "Failed to inspect LoRA adapter", adapter=payload.path) from exc
