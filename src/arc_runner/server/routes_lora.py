"""FastAPI routes for Multi-LoRA adapter discovery, loading, and hot-swap lifecycle.

Exposes RESTful endpoints for listing local LoRA adapters, pre-loading into memory,
evicting adapters from LRU cache, and inspecting adapter metadata (Proofs M23, M24, P10).
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from arc_runner.core.logging import get_logger
from arc_runner.engine.lora_manager import (
    AdapterConfigError,
    AdapterNotFoundError,
    LoRAAdapterMetadata,
)
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


def _validate_safe_path(path_str: str) -> Path:
    """Validate that path does not contain illegal traversal segments.

    Args:
        path_str: Raw path input string.

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
    return Path(path_str).resolve()


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
    _validate_safe_path(payload.adapter)
    model_mgr = request.app.state.model_manager

    try:
        meta = await asyncio.to_thread(model_mgr.lora_manager.load_adapter, payload.adapter)
        return _to_response_dto(meta)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AdapterConfigError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to load LoRA adapter '{payload.adapter}': {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


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
    safe_path = _validate_safe_path(payload.path)
    model_mgr = request.app.state.model_manager

    if not safe_path.exists():
        in_dir = model_mgr.lora_manager.adapters_dir / payload.path
        if in_dir.exists():
            safe_path = in_dir.resolve()

    try:
        meta = await asyncio.to_thread(model_mgr.lora_manager.parse_adapter_metadata, safe_path)
        return _to_response_dto(meta)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AdapterConfigError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to inspect adapter at '{payload.path}': {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
