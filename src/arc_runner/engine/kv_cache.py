"""Dynamic Paged KV-Cache Manager for context budgeting and OOM protection.

Implements token sequence reservations, memory footprint derivations, and context
window headroom checks based on the PagedAttention block model.
"""

import uuid
from typing import Dict, Optional

from arc_runner.core.constants import DEFAULT_CONTEXT_WINDOW, PAGE_BLOCK_SIZE
from arc_runner.core.exceptions import ContextWindowExceededError
from arc_runner.core.logging import get_logger
from arc_runner.engine.model_config import KVGeometry
from arc_runner.engine.models import KVAllocation

logger = get_logger(__name__)

_ASSUMED_LAYERS = 28
_ASSUMED_KV_HEADS = 8
_ASSUMED_HEAD_DIM = 128
_PRECISION_BYTES = {"": 2, "f16": 2, "fp16": 2, "bf16": 2, "u8": 1, "i8": 1, "int8": 1, "f32": 4, "fp32": 4}


class KVCacheManager:
    """Manages Key-Value cache block tables, token headroom, and memory budgeting."""

    def __init__(
        self,
        max_context_length: int = DEFAULT_CONTEXT_WINDOW,
        block_size: int = PAGE_BLOCK_SIZE,
        geometry: Optional[KVGeometry] = None,
        kv_precision: str = "",
    ) -> None:
        """Initialize the KV cache manager with context boundaries and block dimensions.

        Args:
            max_context_length: Maximum allowed sequence token length.
            block_size: Number of tokens allocated per paged block.
            geometry: KV shape read from the model config; None means it is unknown.
            kv_precision: KV cache precision setting ('' is the f16 runtime default).
        """
        self.max_context_length = max_context_length
        self.block_size = block_size
        self.geometry = geometry
        self.kv_precision = kv_precision
        self._allocations: Dict[str, KVAllocation] = {}

    def check_headroom(self, prompt_tokens: int) -> int:
        """Verify that prompt tokens do not exceed the context window boundary.

        Args:
            prompt_tokens: Token count of input prompt.

        Returns:
            int: Remaining token capacity available for completion generation.

        Raises:
            ContextWindowExceededError: If prompt tokens exceed max_context_length.
        """
        if prompt_tokens > self.max_context_length:
            raise ContextWindowExceededError(prompt_tokens, self.max_context_length)
        return self.max_context_length - prompt_tokens

    def calculate_footprint_mb(
        self,
        num_tokens: int,
        num_layers: int = 28,
        num_heads: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ) -> float:
        """Calculate the exact theoretical memory footprint in megabytes for a given token count.

        Formula:
            M_KV = 2 * num_layers * num_heads * head_dim * num_tokens * dtype_bytes / (1024 * 1024)

        Args:
            num_tokens: Number of active sequence tokens.
            num_layers: Number of transformer decoder layers.
            num_heads: Number of key-value attention heads (GQA heads).
            head_dim: Dimension size per attention head.
            dtype_bytes: Storage bytes per element (default 2 bytes for FP16).

        Returns:
            float: Key-value cache memory requirement in megabytes.
        """
        total_bytes = 2 * num_layers * num_heads * head_dim * num_tokens * dtype_bytes
        return total_bytes / (1024.0 * 1024.0)

    def _dtype_bytes(self) -> int:
        """Return the KV element width for the configured precision.

        Returns:
            int: Bytes per element; 2 (f16) with a WARNING when the precision is not recognised.
        """
        key = self.kv_precision.strip().lower()
        if key in _PRECISION_BYTES:
            return _PRECISION_BYTES[key]
        logger.warning("Unrecognised KV precision; footprint assumes 2 bytes per element", extra={"kv_precision": self.kv_precision})
        return 2

    def reserve_allocation(
        self,
        prompt_tokens: int,
        max_new_tokens: int,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ) -> KVAllocation:
        """Reserve a validated context allocation for an inference request.

        Shape precedence: explicit arguments, then the model-config geometry, then the legacy
        28/8/128 assumption (logged at WARNING and flagged on the allocation). This reserves
        accounting only; no cache memory is allocated here.

        Args:
            prompt_tokens: Number of prompt tokens to allocate.
            max_new_tokens: Requested maximum completion tokens.
            num_layers: Transformer layers count override.
            num_heads: KV attention heads count override.
            head_dim: Head dimension override.

        Returns:
            KVAllocation: Active reservation payload containing allocation identifier and footprint.

        Raises:
            ContextWindowExceededError: If prompt alone exceeds maximum context window.
        """
        self.check_headroom(prompt_tokens)
        safe_new_tokens = min(max_new_tokens, self.max_context_length - prompt_tokens)
        total_tokens = prompt_tokens + safe_new_tokens
        from_config = self.geometry is not None
        if from_config:
            layers = num_layers or self.geometry.num_layers
            heads = num_heads or self.geometry.num_kv_heads
            dim = head_dim or self.geometry.head_dim
        else:
            logger.warning(
                "KV footprint assumes 28 layers, 8 KV heads and head dim 128 because the model config did not provide them",
                extra={"total_tokens": total_tokens},
            )
            layers, heads, dim = num_layers or _ASSUMED_LAYERS, num_heads or _ASSUMED_KV_HEADS, head_dim or _ASSUMED_HEAD_DIM
        footprint_mb = self.calculate_footprint_mb(total_tokens, layers, heads, dim, self._dtype_bytes())

        allocation_id = uuid.uuid4().hex[:12]
        allocation = KVAllocation(
            allocation_id=allocation_id,
            prompt_tokens=prompt_tokens,
            reserved_new_tokens=safe_new_tokens,
            total_tokens=total_tokens,
            memory_footprint_mb=footprint_mb,
            is_active=True,
            footprint_from_model_config=from_config and not (num_layers or num_heads or head_dim),
        )
        self._allocations[allocation_id] = allocation
        return allocation

    def release_allocation(self, allocation_id: str) -> None:
        """Release an active KV reservation and free virtual cache blocks.

        Args:
            allocation_id: Unique allocation identifier to deactivate.
        """
        if allocation_id in self._allocations:
            alloc = self._allocations.pop(allocation_id)
            alloc.is_active = False

    @property
    def active_allocated_tokens(self) -> int:
        """Return the sum of all currently active reserved tokens.

        Returns:
            int: Total tokens currently tracked in active allocations.
        """
        return sum(a.total_tokens for a in self._allocations.values() if a.is_active)
