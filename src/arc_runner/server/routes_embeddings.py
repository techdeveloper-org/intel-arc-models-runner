"""FastAPI REST endpoints for Ollama and OpenAI vector embeddings parity.

Exposes POST /api/embeddings, POST /api/embed, and POST /v1/embeddings routes
leveraging the Intel AI Boost NPU and Arc GPU embedding engine.
"""

import time
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException, Request, status

from arc_runner.core.exceptions import ModelNotFoundError
from arc_runner.core.logging import get_logger, get_request_id
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

logger = get_logger("server.routes.embeddings")

embeddings_router = APIRouter(tags=["Embeddings"])


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
    model_mgr = request.app.state.model_manager
    req_id = get_request_id()

    try:
        engine = model_mgr.get_or_load_embedding_engine(req.model)
        vector = engine.embed_query(req.prompt)
        return OllamaEmbeddingResponse(embedding=vector)
    except ModelNotFoundError as exc:
        logger.warning(f"Embedding model not found: {req.model} [request_id={req_id}]")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding model '{req.model}' not found in registered directories.",
        ) from exc
    except Exception as exc:
        logger.error(f"Failed to generate embedding for {req.model}: {exc} [request_id={req_id}]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {exc}",
        ) from exc


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
    model_mgr = request.app.state.model_manager
    req_id = get_request_id()
    t_start = time.perf_counter_ns()

    try:
        engine = model_mgr.get_or_load_embedding_engine(req.model)

        inputs: List[str]
        if isinstance(req.input, str):
            inputs = [req.input]
        else:
            inputs = list(req.input)

        vectors = engine.embed_documents(inputs)
        t_duration = time.perf_counter_ns() - t_start

        return OllamaEmbedResponse(
            model=req.model,
            embeddings=vectors,
            total_duration=t_duration,
            load_duration=0,
            prompt_eval_count=_reported_tokens(engine),
        )
    except ModelNotFoundError as exc:
        logger.warning(f"Embedding model not found: {req.model} [request_id={req_id}]")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding model '{req.model}' not found in registered directories.",
        ) from exc
    except Exception as exc:
        logger.error(f"Batch embedding generation failed for {req.model}: {exc} [request_id={req_id}]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding generation failed: {exc}",
        ) from exc


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
    model_mgr = request.app.state.model_manager
    req_id = get_request_id()

    try:
        engine = model_mgr.get_or_load_embedding_engine(req.model)

        inputs: List[str]
        if isinstance(req.input, str):
            inputs = [req.input]
        else:
            inputs = [str(x) for x in req.input]

        vectors = engine.embed_documents(inputs)

        data = [
            OpenAIEmbeddingData(
                object="embedding",
                index=idx,
                embedding=vec,
            )
            for idx, vec in enumerate(vectors)
        ]

        reported = _reported_tokens(engine)
        usage = None if reported is None else OpenAIEmbeddingUsage(prompt_tokens=reported, total_tokens=reported)

        return OpenAIEmbeddingResponse(
            object="list",
            data=data,
            model=req.model,
            usage=usage,
        )
    except ModelNotFoundError as exc:
        logger.warning(f"OpenAI embedding model not found: {req.model} [request_id={req_id}]")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{req.model}' not found.",
        ) from exc
    except Exception as exc:
        logger.error(f"OpenAI embedding generation failed for {req.model}: {exc} [request_id={req_id}]")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {exc}",
        ) from exc
