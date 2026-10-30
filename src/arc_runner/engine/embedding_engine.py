"""Vector embedding engine optimized for Intel AI Boost NPU and Arc GPU architectures.

Provides L2-normalized dense vector embeddings, cosine similarity metrics, and attention-masked
pooling for Retrieval-Augmented Generation (RAG) pipelines and semantic search.
"""

import math
from pathlib import Path
from typing import Any, List, Optional, Sequence

from arc_runner.core.exceptions import ArcRunnerError, ModelLoadError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.embedding")


class EmbeddingEngineError(ArcRunnerError):
    """Raised when an embedding pipeline fails to initialize or generate vectors."""


class DimensionMismatchError(EmbeddingEngineError):
    """Raised when two vectors have differing dimensions for metric computation."""


class EmptyMaskError(EmbeddingEngineError):
    """Raised when an attention mask is completely empty (all zero tokens)."""


class EmbeddingEngine:
    """High-throughput vector embedding engine targeting Intel NPU, GPU, or CPU."""

    def __init__(
        self,
        model_path: str,
        device: str = "NPU",
        normalize: bool = True,
        max_length: int = 512,
    ) -> None:
        """Initialize the embedding pipeline on the requested Intel device.

        Args:
            model_path: Path to directory containing OpenVINO IR files and tokenizer.
            device: Target hardware device string ('NPU', 'GPU', 'CPU').
            normalize: Whether to apply strict L2 unit sphere normalization.
            max_length: Maximum sequence token truncation boundary.

        Raises:
            ModelLoadError: If the model path is missing or fails to load.
        """
        self._model_path = Path(model_path)
        self._device = device.upper()
        self._normalize = normalize
        self._max_length = max_length
        self._pipeline: Any = None
        self._dimension: int = 0

        self._initialize_pipeline()

    @property
    def model_path(self) -> Path:
        """Return the underlying model directory path."""
        return self._model_path

    @property
    def device(self) -> str:
        """Return the active silicon compute device."""
        return self._device

    @property
    def last_prompt_tokens(self) -> Optional[int]:
        """Tokenizer-counted prompt tokens of the last request; None because this engine does not measure them."""
        return None

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    def _initialize_pipeline(self) -> None:
        """Construct the OpenVINO TextEmbeddingPipeline or set up fallback execution.

        Raises:
            ModelLoadError: If initialization fails on the selected device and fallback fails.
        """
        if not self._model_path.exists():
            raise ModelLoadError(f"Embedding model directory does not exist: {self._model_path}")

        try:
            import openvino_genai as genai

            cfg = genai.TextEmbeddingPipeline.Config()
            cfg.max_length = self._max_length
            cfg.normalize = self._normalize
            cfg.pooling_type = genai.TextEmbeddingPipeline.PoolingType.MEAN

            try:
                self._pipeline = genai.TextEmbeddingPipeline(
                    str(self._model_path),
                    self._device,
                    config=cfg,
                )
                logger.info(
                    "Initialized TextEmbeddingPipeline",
                    device=self._device,
                    model_path=str(self._model_path),
                )
            except Exception as device_exc:
                if self._device == "NPU":
                    logger.warning(
                        "NPU embedding initialization failed, falling back to GPU",
                        error=str(device_exc),
                    )
                    self._device = "GPU"
                    self._pipeline = genai.TextEmbeddingPipeline(
                        str(self._model_path),
                        self._device,
                        config=cfg,
                    )
                else:
                    raise device_exc

            sample = self._pipeline.embed_query("probe")
            self._dimension = len(sample)

        except Exception as exc:
            logger.error(
                f"Failed to construct OpenVINO embedding pipeline at {self._model_path} on {self._device}: {exc}"
            )
            raise ModelLoadError(
                f"Failed to load embedding model at '{self._model_path}' on device '{self._device}': {exc}"
            ) from exc

    def embed_query(self, text: str) -> List[float]:
        """Compute an L2-normalized vector embedding for a single text query.

        Args:
            text: Query string to encode.

        Returns:
            List of floating point numbers representing the dense vector.

        Raises:
            EmbeddingEngineError: If encoding fails or input is malformed.
        """
        if not isinstance(text, str):
            raise EmbeddingEngineError(f"Expected string input for embed_query, got {type(text).__name__}")

        if not text.strip():
            return [0.0] * max(self._dimension, 1)

        try:
            vector = self._pipeline.embed_query(text)
            if self._normalize:
                return self.l2_normalize(vector)
            return list(vector)
        except Exception as exc:
            raise EmbeddingEngineError(f"Embedding generation failed: {exc}") from exc

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Compute embeddings for a sequence of document texts in a single batch.

        Args:
            texts: Sequence of document strings to encode.

        Returns:
            List of dense vector embeddings.

        Raises:
            EmbeddingEngineError: If batched encoding fails.
        """
        if not texts:
            return []

        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]

        try:
            vectors = self._pipeline.embed_documents(clean_texts)
            if self._normalize:
                return [self.l2_normalize(vec) for vec in vectors]
            return [list(vec) for vec in vectors]
        except Exception as exc:
            raise EmbeddingEngineError(f"Batched document embedding failed: {exc}") from exc

    @staticmethod
    def l2_normalize(vector: Sequence[float], eps: float = 1e-12) -> List[float]:
        """Apply strict L2 unit sphere normalization to a vector (Proof M1).

        Args:
            vector: Raw vector components.
            eps: Regularization threshold for zero magnitude detection.

        Returns:
            Unit normalized vector where sum(v_i^2) == 1.0.

        Raises:
            ValueError: If the vector has zero elements.
        """
        if not vector:
            raise ValueError("Cannot normalize an empty vector.")

        squared_sum = sum(x * x for x in vector)
        if squared_sum == 0.0:
            return [0.0] * len(vector)

        norm = math.sqrt(squared_sum)
        return [float(x / norm) for x in vector]

    @staticmethod
    def masked_mean_pooling(
        hidden_states: Sequence[Sequence[float]],
        attention_mask: Sequence[int],
        eps: float = 1e-12,
    ) -> List[float]:
        """Compute attention-masked mean pooling across token hidden states (Proof M2).

        Args:
            hidden_states: Matrix of token representations of shape (T, d).
            attention_mask: Binary mask of shape (T,) where 1 is valid and 0 is pad.
            eps: Regularization constant preventing division by zero.

        Returns:
            Pooled vector representation of dimension d.

        Raises:
            EmptyMaskError: If all tokens in the attention mask are zero.
            ValueError: If dimensions do not match.
        """
        seq_len = len(hidden_states)
        if seq_len != len(attention_mask):
            raise ValueError("Length of hidden_states must match length of attention_mask.")

        mask_sum = sum(attention_mask)
        if mask_sum == 0:
            raise EmptyMaskError("Attention mask has zero valid tokens; cannot compute masked pool.")

        dim = len(hidden_states[0])
        pooled = [0.0] * dim

        for t in range(seq_len):
            if attention_mask[t] != 0:
                h = hidden_states[t]
                for d in range(dim):
                    pooled[d] += h[d]

        divisor = float(mask_sum) + eps
        return [val / divisor for val in pooled]

    @classmethod
    def compute_similarity(cls, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors (Proofs M1, M14).

        Args:
            vec_a: First vector components.
            vec_b: Second vector components.

        Returns:
            Cosine similarity score in the interval [-1.0, 1.0].

        Raises:
            DimensionMismatchError: If the vectors have unequal length.
            ValueError: If either vector is empty.
        """
        if not vec_a or not vec_b:
            raise ValueError("Cannot compute similarity of empty vectors.")

        if len(vec_a) != len(vec_b):
            raise DimensionMismatchError(
                f"Dimension mismatch in similarity computation: {len(vec_a)} != {len(vec_b)}"
            )

        norm_a = cls.l2_normalize(vec_a)
        norm_b = cls.l2_normalize(vec_b)

        dot_product = sum(a * b for a, b in zip(norm_a, norm_b))
        return float(max(-1.0, min(1.0, dot_product)))
