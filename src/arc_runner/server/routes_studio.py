"""FastAPI REST and Server-Sent Events (SSE) streaming routes for Arc MultiModal Studio.

Provides endpoints for launching studio production pipelines, streaming real-time stage progress,
executing automated multi-model co-benchmarks, and retrieving completed MP4 video artifacts.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1-M5.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from arc_runner.benchmark.multimodal_runner import MultiModelCoBenchmarker
from arc_runner.core.logging import get_logger
from arc_runner.orchestrator.models import (
    BenchmarkRequest,
    MultiModelBenchmarkReport,
    PipelineStage,
    StudioArtifactPayload,
    StudioProjectRequest,
)
from arc_runner.orchestrator.pipeline import StudioPipeline

logger = get_logger("server.routes_studio")
router = APIRouter(prefix="/v1/studio", tags=["Arc MultiModal Studio"])

_PIPELINE = StudioPipeline()
_BENCHMARKER = MultiModelCoBenchmarker(pipeline=_PIPELINE)
_PROJECT_CACHE: dict[str, StudioArtifactPayload] = {}


@router.post(
    "/generate",
    response_model=StudioArtifactPayload,
    summary="Dispatch Multi-Modal Video Production Pipeline",
)
async def generate_studio_project(
    request: StudioProjectRequest,
    stream: bool = Query(default=False, description="Whether to stream real-time SSE progress events"),
) -> Any:
    """Execute end-to-end multi-modal video generation from creative brief prompt.

    Args:
        request: Validated StudioProjectRequest specifying prompt, duration, and styling.
        stream: Query flag enabling Server-Sent Events (SSE) streaming format.

    Returns:
        StudioArtifactPayload JSON response or StreamingResponse of SSE event objects.

    Raises:
        HTTPException: If pipeline execution encounters fatal generation failure.
    """
    if stream:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress_cb(stg: PipelineStage, pct: float, msg: str) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"stage": stg.value, "progress": pct, "message": msg},
            )

        async def worker() -> None:
            try:
                payload = await asyncio.to_thread(_PIPELINE.execute, request, progress_cb)
                _PROJECT_CACHE[payload.project_id] = payload
                await queue.put(
                    {
                        "stage": "COMPLETED",
                        "progress": 100.0,
                        "project_id": payload.project_id,
                        "video_url": f"/v1/studio/projects/{payload.project_id}/video",
                        "payload": payload.model_dump(),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"SSE pipeline failed: {exc}")
                await queue.put({"stage": "FAILED", "error": str(exc)})
            finally:
                await queue.put(None)

        asyncio.create_task(worker())

        async def event_generator() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'stage': 'INITIATED', 'progress': 0.0, 'message': 'Project accepted by orchestrator'})}\n\n"
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        payload = _PIPELINE.execute(request)
        _PROJECT_CACHE[payload.project_id] = payload
        return payload
    except Exception as exc:
        logger.error(f"Failed to generate studio project: {exc}")
        raise HTTPException(status_code=500, detail=f"Studio pipeline failed: {exc}") from exc


@router.post(
    "/benchmark",
    response_model=MultiModelBenchmarkReport,
    summary="Run Multi-Model Co-Benchmarking Suite",
)
async def run_multimodal_benchmark(
    request: BenchmarkRequest,
) -> MultiModelBenchmarkReport:
    """Execute automated hardware co-benchmarking across multi-modal models and compute Cloud ROI.

    Args:
        request: Validated BenchmarkRequest specifying iterations, warmup passes, and silicon strategy.

    Returns:
        MultiModelBenchmarkReport with sample means, 95% confidence intervals, and Cloud cost savings.

    Raises:
        HTTPException: If benchmark execution fails.
    """
    try:
        report = _BENCHMARKER.run_co_benchmark(
            prompt=request.prompt,
            iterations=request.iterations,
            warmup_runs=request.warmup_runs,
            strategy=request.strategy,
        )
        return report
    except Exception as exc:
        logger.error(f"Co-benchmark failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Co-benchmark failed: {exc}") from exc


@router.get(
    "/projects/{project_id}",
    response_model=StudioArtifactPayload,
    summary="Retrieve Completed Project Artifact Metadata",
)
async def get_project_metadata(
    project_id: str,
) -> StudioArtifactPayload:
    """Retrieve metadata, telemetry, and script plan for completed production project.

    Args:
        project_id: Unique project run identifier.

    Returns:
        StudioArtifactPayload for requested project.

    Raises:
        HTTPException: If project identifier is not found.
    """
    if project_id in _PROJECT_CACHE:
        return _PROJECT_CACHE[project_id]

    raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found in active session cache.")


@router.get(
    "/projects/{project_id}/video",
    summary="Download Rendered MP4 Video Binary Container",
)
async def download_project_video(
    project_id: str,
) -> FileResponse:
    """Stream binary MP4 container file for completed production project.

    Args:
        project_id: Unique project run identifier.

    Returns:
        FileResponse streaming MP4 video file.

    Raises:
        HTTPException: If video file does not exist.
    """
    if project_id in _PROJECT_CACHE:
        path = Path(_PROJECT_CACHE[project_id].video_output_path)
        if path.exists():
            return FileResponse(path=str(path), media_type="video/mp4", filename=path.name)

    raise HTTPException(status_code=404, detail=f"Video file for project '{project_id}' not found.")
