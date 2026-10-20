"""Dynamic Paged KV-Cache Manager for context budgeting and OOM protection.

Implements token sequence reservations, memory footprint derivations, and context
window headroom checks based on the PagedAttention block model.
"""

import uuid
from typing import Dict

from arc_runner.core.constants import DEFAULT_CONTEXT_WINDOW, PAGE_BLOCK_SIZE
from arc_runner.core.exceptions import ContextWindowExceededError
from arc_runner.engine.models import KVAllocation


class KVCacheManager:
    """Manages Key-Value cache block tables, token headroom, and memory budgeting."""

    def __init__(
        self,
        max_context_length: int = DEFAULT_CONTEXT_WINDOW,
        block_size: int = PAGE_BLOCK_SIZE,
    ) -> None:
        """Initialize the KV cache manager with context boundaries and block dimensions.

        Args:
            max_context_length: Maximum allowed sequence token length.
            block_size: Number of tokens allocated per paged block.
        """
        self.max_context_length = max_context_length
        self.block_size = block_size
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

    def reserve_allocation(
        self,
        prompt_tokens: int,
        max_new_tokens: int,
        num_layers: int = 28,
        num_heads: int = 8,
        head_dim: int = 128,
    ) -> KVAllocation:
        """Reserve a validated context allocation for an inference request.

        Args:
            prompt_tokens: Number of prompt tokens to allocate.
            max_new_tokens: Requested maximum completion tokens.
            num_layers: Transformer layers count.
            num_heads: KV attention heads count.
            head_dim: Head dimension.

        Returns:
            KVAllocation: Active reservation payload containing allocation identifier and footprint.

        Raises:
            ContextWindowExceededError: If prompt alone exceeds maximum context window.
        """
        self.check_headroom(prompt_tokens)
        safe_new_tokens = min(max_new_tokens, self.max_context_length - prompt_tokens)
        total_tokens = prompt_tokens + safe_new_tokens
        footprint_mb = self.calculate_footprint_mb(total_tokens, num_layers, num_heads, head_dim)

        allocation_id = uuid.uuid4().hex[:12]
        allocation = KVAllocation(
            allocation_id=allocation_id,
            prompt_tokens=prompt_tokens,
            reserved_new_tokens=safe_new_tokens,
            total_tokens=total_tokens,
            memory_footprint_mb=footprint_mb,
            is_active=True,
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
