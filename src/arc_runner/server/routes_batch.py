"""FastAPI route controllers for Continuous Batching telemetry and sequence preemption.

Exposes GET /api/batch/status to monitor queue depth, VRAM saturation, and prefix cache hits,
and POST /api/batch/preempt to administratively evict running sequences.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request, status

from arc_runner.core.logging import get_logger
from arc_runner.server.schemas import BatchPreemptRequest, BatchStatusResponse

logger = get_logger("arc_runner.server.routes_batch")

router = APIRouter(tags=["batch"])


@router.get("/api/batch/status", response_model=BatchStatusResponse)
async def get_batch_status(request: Request) -> BatchStatusResponse:
    """Retrieve live continuous batching scheduler metrics and VRAM block table telemetry.

    Args:
        request: FastAPI HTTP request containing application state.

    Returns:
        BatchStatusResponse populated with active sequences, queue depth, and utilization.
    """
    pipeline = getattr(request.app.state, "batch_pipeline", None)
    if pipeline is None:
        return BatchStatusResponse(
            running_sequences=0,
            waiting_sequences=0,
            completed_requests=0,
            total_tokens_produced=0,
            used_kv_blocks=0,
            free_kv_blocks=1024,
            vram_block_utilization_pct=0.0,
            prefix_cache_hit_rate=0.0,
            prefix_cache_hits=0,
            prefix_cache_misses=0,
            active_client_connections=0,
            running_sequence_ids=[],
            waiting_sequence_ids=[],
        )

    status_dict = pipeline.get_status()
    return BatchStatusResponse(**status_dict)


@router.post("/api/batch/preempt")
async def preempt_sequence(payload: BatchPreemptRequest, request: Request) -> Dict[str, Any]:
    """Preempt an active sequence from the running batch, reclaiming physical blocks (Proof M32).

    Args:
        payload: BatchPreemptRequest containing sequence identifier.
        request: FastAPI HTTP request containing application state.

    Returns:
        Dictionary confirming preemption outcome.

    Raises:
        HTTPException: If sequence is not found in running batch.
    """
    pipeline = getattr(request.app.state, "batch_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Batch execution pipeline is not initialized.",
        )

    success = pipeline.preempt_sequence(payload.seq_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sequence '{payload.seq_id}' not found in running batch.",
        )

    logger.info(f"Administratively preempted sequence '{payload.seq_id}'.")
    return {"seq_id": payload.seq_id, "preempted": True}
