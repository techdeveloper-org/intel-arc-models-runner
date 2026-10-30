"""FastAPI application factory and middleware orchestration for Intel Arc Models Runner daemon.

Provides centralized request tracing, CORS policy enforcement, structured domain exception handling,
and hardware telemetry exposition.
"""

from arc_runner._version import __version__
import hmac
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
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
from arc_runner.server.anthropic_schemas import error_envelope, new_request_id
from arc_runner.server.batch_pipeline import BatchExecutionPipeline
from arc_runner.server.model_manager import ModelManager
from arc_runner.server.routes_anthropic import probe_router as anthropic_probe_router
from arc_runner.server.routes_anthropic import router as anthropic_router
from arc_runner.server.routes_batch import router as batch_router
from arc_runner.server.routes_embeddings import embeddings_router
from arc_runner.server.routes_lora import router as lora_router
from arc_runner.server.routes_ollama import router as ollama_router
from arc_runner.server.routes_openai import router as openai_router
from arc_runner.server.routes_studio import router as studio_router
from arc_runner.server.routes_ui import router as ui_router

logger = get_logger("server.app")


API_KEY_ENV = "ARC_API_KEY"
CORS_ORIGINS_ENV = "ARC_CORS_ORIGINS"
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_OPEN_PATHS = ("/", "/docs", "/openapi.json", "/health")
_ANTHROPIC_PATH_PREFIX = "/v1/messages"


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Requires the configured API key (Authorization: Bearer or X-API-Key) on every route except the UI page, docs and /health.

    Either header may carry the key, because Claude Code sends the token in Authorization (ANTHROPIC_AUTH_TOKEN)
    or in x-api-key (ANTHROPIC_API_KEY). Both are always compared, in constant time.
    """

    def __init__(self, app: Any, api_key: str) -> None:
        """Store the expected key.

        Args:
            app: Downstream ASGI application.
            api_key: Expected key; compared in constant time.
        """
        super().__init__(app)
        self._api_key = api_key.encode("utf-8")

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        """Reject requests without the right key.

        Args:
            request: Incoming request.
            call_next: Downstream handler.

        Returns:
            Response: 401 JSON for a missing or wrong key, otherwise the downstream response.
        """
        if request.method == "OPTIONS" or request.url.path in _OPEN_PATHS or request.url.path.startswith("/static"):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        bearer = header[7:] if header.lower().startswith("bearer ") else ""
        header_key = request.headers.get("x-api-key", "")
        matches = [hmac.compare_digest(supplied.encode("utf-8"), self._api_key) for supplied in (bearer, header_key)]
        if not any(matches):
            if request.url.path.startswith(_ANTHROPIC_PATH_PREFIX):
                request_id = new_request_id()
                return JSONResponse(
                    status_code=401,
                    content=error_envelope("authentication_error", "missing or invalid API key", request_id),
                    headers={"request-id": request_id},
                )
            return JSONResponse(status_code=401, content={"error": "missing or invalid API key", "code": "UNAUTHORIZED"})
        return await call_next(request)


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
    batch_pipeline: Optional[BatchExecutionPipeline] = None,
    debug: bool = False,
) -> FastAPI:
    """Instantiate and configure the FastAPI daemon server application.

    Args:
        model_manager: Optional pre-configured ModelManager instance.
        batch_pipeline: Optional pre-configured BatchExecutionPipeline instance.
        debug: If True, enables API docs endpoints and verbose tracebacks.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Intel Arc Models Runner Daemon",
        description="High-throughput local LLM runner optimized for Intel Arc GPUs and AI Boost NPUs",
        version=__version__,
        docs_url="/docs" if debug else None,
        redoc_url=None,
    )

    manager = model_manager or ModelManager()
    app.state.model_manager = manager
    pipeline = batch_pipeline or BatchExecutionPipeline()
    app.state.batch_pipeline = pipeline

    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if api_key:
        app.add_middleware(ApiKeyMiddleware, api_key=api_key)
    app.add_middleware(LoggingMiddleware)
    extra_origins = [item.strip() for item in os.environ.get(CORS_ORIGINS_ENV, "").split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=extra_origins,
        allow_origin_regex=LOCAL_ORIGIN_REGEX,
        allow_credentials=False,
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
            content={"error": "Request validation failed", "details": jsonable_encoder(exc.errors())},
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
    app.include_router(batch_router)
    app.include_router(ollama_router)
    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.include_router(anthropic_probe_router)
    app.include_router(studio_router)

    return app
