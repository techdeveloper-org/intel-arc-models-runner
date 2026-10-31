"""FastAPI REST endpoints for Ollama and OpenAI vector embeddings parity.

Exposes POST /api/embeddings, POST /api/embed, and POST /v1/embeddings routes
leveraging the Intel AI Boost NPU and Arc GPU embedding engine.
"""

import time
from typing import Any, Callable, List, Optional, TypeVar

from fastapi import APIRouter, HTTPException, Request, status

from arc_runner.core.exceptions import EngineUnavailableError, ModelNotFoundError
from arc_runner.core.logging import get_logger, get_request_id
from arc_runner.server.errors import hidden_error
from arc_runner.server.schemas import (
    OllamaEmbeddingRequest,
    OllamaEmbeddingResponse,
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    OpenAIEmbeddingData,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIEmbeddingUsage,
)
from arc_runner.server.worker import JobContext, LoadPhaseError, get_worker

logger = get_logger("server.routes.embeddings")

embeddings_router = APIRouter(tags=["Embeddings"])

T = TypeVar("T")


async def _run_embedding(
    request: Request,
    model: str,
    job: Callable[[Any], T],
    not_found_detail: str,
    failure_public: str,
) -> T:
    """Run an embedding job on the serial worker and map failures to the route's error shapes.

    Args:
        request: FastAPI request holding the application state.
        model: Embedding model name, used for logging.
        job: Callable taking the loaded embedding engine and returning the route result.
        not_found_detail: Client message for an unknown model (HTTP 404).
        failure_public: Client message for any other failure (HTTP 500).

    Returns:
        T: Value returned by ``job``.

    Raises:
        HTTPException: 404 for an unknown model, 500 for any other failure.
        EngineUnavailableError: When the worker is stopping or wedged (HTTP 503).
    """
    model_mgr = request.app.state.model_manager
    req_id = get_request_id()

    def run(ctx: JobContext) -> T:
        """Load the embedding engine and execute the job on the worker thread."""
        engine = ctx.load(lambda: model_mgr.get_or_load_embedding_engine(model))
        return job(engine)

    try:
        return await get_worker(request.app).submit(run)
    except EngineUnavailableError:
        raise
    except LoadPhaseError as exc:
        cause = exc.__cause__
        if isinstance(cause, ModelNotFoundError):
            logger.warning(f"Embedding model not found: {model} [request_id={req_id}]")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail) from cause
        raise hidden_error("embedding request failed", cause, failure_public, model=model, request_id=req_id) from cause
    except Exception as exc:
        raise hidden_error("embedding request failed", exc, failure_public, model=model, request_id=req_id) from exc


@embeddings_router.post(
    "/api/embeddings",
    response_model=OllamaEmbeddingResponse,
    summary="Generate vector embedding (Ollama legacy format)",
)
async def api_embeddings(req: OllamaEmbeddingRequest, request: Request) -> OllamaEmbeddingResponse:
    """Generate dense vector embedding for single prompt matching Ollama specification.

    Args:
        req: Validated OllamaEmbeddingRequest containing model identifier and prompt.
        request: FastAPI HTTP request containing active application state.

    Returns:
        OllamaEmbeddingResponse containing single 1D vector float array.

    Raises:
        HTTPException: 404 if model not found, 500 on execution error.
    """
    vector = await _run_embedding(
        request,
        req.model,
        lambda engine: engine.embed_query(req.prompt),
        f"Embedding model '{req.model}' not found in registered directories.",
        "Embedding generation failed",
    )
    return OllamaEmbeddingResponse(embedding=vector)


def _reported_tokens(engine: object) -> Optional[int]:
    """Return the engine-reported prompt token count, or None when the engine reports none.

    Args:
        engine: Embedding engine that served the request.

    Returns:
        Optional[int]: Non-negative integer count, else None.
    """
    value = getattr(engine, "last_prompt_tokens", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


@embeddings_router.post(
    "/api/embed",
    response_model=OllamaEmbedResponse,
    response_model_exclude_none=True,
    summary="Generate batch vector embeddings (Ollama v2 format)",
)
async def api_embed(req: OllamaEmbedRequest, request: Request) -> OllamaEmbedResponse:
    """Generate batch dense vector embeddings matching Ollama /api/embed specification.

    Args:
        req: Validated OllamaEmbedRequest containing model and text inputs.
        request: FastAPI HTTP request with application state.

    Returns:
        OllamaEmbedResponse containing 2D vector float array and execution timings.

    Raises:
        HTTPException: 404 if model not found, 500 on execution error.
    """
    t_start = time.perf_counter_ns()
    inputs: List[str] = [req.input] if isinstance(req.input, str) else list(req.input)

    def job(engine: Any) -> Any:
        """Embed the documents and capture the engine-reported token count on the worker thread."""
        return engine.embed_documents(inputs), _reported_tokens(engine)

    vectors, reported = await _run_embedding(
        request,
        req.model,
        job,
        f"Embedding model '{req.model}' not found in registered directories.",
        "Batch embedding generation failed",
    )
    return OllamaEmbedResponse(
        model=req.model,
        embeddings=vectors,
        total_duration=time.perf_counter_ns() - t_start,
        load_duration=0,
        prompt_eval_count=reported,
    )


@embeddings_router.post(
    "/v1/embeddings",
    response_model=OpenAIEmbeddingResponse,
    response_model_exclude_none=True,
    summary="Generate vector embeddings (OpenAI v1 format)",
)
async def v1_embeddings(req: OpenAIEmbeddingRequest, request: Request) -> OpenAIEmbeddingResponse:
    """Generate dense vector embeddings matching OpenAI specification.

    Args:
        req: Validated OpenAIEmbeddingRequest containing model and input array.
        request: FastAPI HTTP request with application state.

    Returns:
        OpenAIEmbeddingResponse containing data list with indexed embeddings.

    Raises:
        HTTPException: 404 if model not found, 500 on execution error.
    """
    inputs: List[str] = [req.input] if isinstance(req.input, str) else [str(x) for x in req.input]

    def job(engine: Any) -> Any:
        """Embed the documents and capture the engine-reported token count on the worker thread."""
        return engine.embed_documents(inputs), _reported_tokens(engine)

    vectors, reported = await _run_embedding(
        request,
        req.model,
        job,
        f"Model '{req.model}' not found.",
        "Embedding generation failed",
    )
    data = [OpenAIEmbeddingData(object="embedding", index=idx, embedding=vec) for idx, vec in enumerate(vectors)]
    usage = None if reported is None else OpenAIEmbeddingUsage(prompt_tokens=reported, total_tokens=reported)
    return OpenAIEmbeddingResponse(object="list", data=data, model=req.model, usage=usage)
