"""FastAPI application factory and middleware orchestration for Intel Arc Models Runner daemon.

Provides centralized request tracing, CORS policy enforcement, structured domain exception handling,
and hardware telemetry exposition.
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.core.logging import (
    clear_request_context,
    get_logger,
    set_request_context,
)
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import DeviceType
from arc_runner.server.model_manager import ModelManager
from arc_runner.server.routes_embeddings import embeddings_router
from arc_runner.server.routes_lora import router as lora_router
from arc_runner.server.routes_ollama import router as ollama_router
from arc_runner.server.routes_openai import router as openai_router
from arc_runner.server.routes_ui import router as ui_router

logger = get_logger("server.app")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Intercepts requests to manage MDC correlation IDs and log latency telemetry."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Process incoming request within an MDC tracing scope.

        Args:
            request: Incoming HTTP request instance.
            call_next: ASGI downstream application handler.

        Returns:
            HTTP response with injected X-Request-ID header.
        """
        req_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:8]}"
        client_ip = request.client.host if request.client else "-"
        set_request_context(request_id=req_id, client_ip=client_ip)

        logger.info(f"Ingress: {request.method} {request.url.path}")
        start_time = time.perf_counter()

        try:
            response: Response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = req_id
            logger.info(f"Egress: {request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.1f}ms)")
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Failed: {request.method} {request.url.path} -> {exc} ({elapsed_ms:.1f}ms)", exc_info=True)
            raise
        finally:
            clear_request_context()


def create_app(
    model_manager: Optional[ModelManager] = None,
    debug: bool = False,
) -> FastAPI:
    """Instantiate and configure the FastAPI daemon server application.

    Args:
        model_manager: Optional pre-configured ModelManager instance.
        debug: If True, enables API docs endpoints and verbose tracebacks.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Intel Arc Models Runner Daemon",
        description="High-throughput local LLM runner optimized for Intel Arc GPUs and AI Boost NPUs",
        version="0.1.0",
        docs_url="/docs" if debug else None,
        redoc_url=None,
    )

    manager = model_manager or ModelManager()
    app.state.model_manager = manager

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        """Handle ModelNotFoundError with RFC 7807 compliant error JSON."""
        return JSONResponse(
            status_code=404,
            content={"error": exc.message, "code": exc.error_code},
        )

    @app.exception_handler(ArcRunnerError)
    async def arc_runner_error_handler(request: Request, exc: ArcRunnerError) -> JSONResponse:
        """Handle typed domain exceptions with matching HTTP status codes."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "code": exc.error_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle Pydantic schema validation failures with HTTP 422 payload."""
        return JSONResponse(
            status_code=422,
            content={"error": "Request validation failed", "details": exc.errors()},
        )

    @app.get("/api/devices")
    async def get_devices() -> Dict[str, Any]:
        """Hardware topology telemetry endpoint returning live silicon states.

        Returns:
            Dictionary containing detected GPU, NPU, and CPU capabilities.
        """
        detector = DeviceDetector()
        topology = detector.detect_topology()
        return {
            "status": "healthy",
            "device_summary": topology.model_dump(),
            "usm_shared_memory_ready": any(d.has_usm for d in topology.devices if d.device_type == DeviceType.GPU),
            "npu_boost_ready": topology.has_intel_npu,
        }

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(ui_router)
    app.include_router(embeddings_router)
    app.include_router(lora_router)
    app.include_router(ollama_router)
    app.include_router(openai_router)

    return app
