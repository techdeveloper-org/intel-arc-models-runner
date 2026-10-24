"""Paged Attention and Radix Tree prefix cache memory manager for Intel Arc GPUs.

Implements non-contiguous physical block allocation (16-token blocks), dynamic virtual-to-physical
address translation (Proof M17), and Radix Tree prefix sharing for multi-turn chats (Proof M12).
"""

from collections import deque
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.paged_attention")


class PagedAttentionError(ArcRunnerError):
    """Raised when block table allocation or address translation fails."""


class VRAMExhaustedError(PagedAttentionError):
    """Raised when free physical KV blocks are exhausted."""


class BlockTableLookupError(PagedAttentionError):
    """Raised when an unallocated sequence or out-of-bounds token offset is queried."""


class PagedBlockTable:
    """Manages virtual-to-physical block table translation for Paged Attention (Proofs M11, M17)."""

    def __init__(self, block_size: int = 16, total_blocks: int = 1024) -> None:
        """Initialize paged block allocator with fixed-size physical blocks.

        Args:
            block_size: Number of tokens accommodated in each physical memory block.
            total_blocks: Total pool of physical blocks reserved in VRAM.

        Raises:
            ValueError: If block_size or total_blocks is non-positive.
        """
        if block_size <= 0 or total_blocks <= 0:
            raise ValueError("block_size and total_blocks must be strictly positive integers.")

        self._block_size = block_size
        self._total_blocks = total_blocks
        self._free_blocks: deque[int] = deque(range(total_blocks))
        self._seq_to_blocks: Dict[str, List[int]] = {}
        self._seq_lengths: Dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def block_size(self) -> int:
        """Return the physical block size in tokens."""
        return self._block_size

    @property
    def total_blocks(self) -> int:
        """Return the total capacity of physical blocks."""
        return self._total_blocks

    @property
    def free_blocks_count(self) -> int:
        """Return number of unallocated physical blocks remaining."""
        with self._lock:
            return len(self._free_blocks)

    @property
    def allocated_blocks_count(self) -> int:
        """Return number of physical blocks currently allocated."""
        with self._lock:
            return self._total_blocks - len(self._free_blocks)

    @property
    def utilization_ratio(self) -> float:
        """Return the ratio of allocated blocks to total capacity."""
        with self._lock:
            return float((self._total_blocks - len(self._free_blocks)) / self._total_blocks)

    def get_sequence_blocks(self, seq_id: str) -> List[int]:
        """Return list of physical block IDs assigned to a sequence.

        Args:
            seq_id: Sequence identifier.

        Returns:
            List of assigned physical block IDs.
        """
        with self._lock:
            return list(self._seq_to_blocks.get(seq_id, []))

    def get_block_map(self) -> List[int]:
        """Return binary occupation map (1=allocated, 0=free) across all blocks.

        Returns:
            List of binary integer flags for each physical block index.
        """
        with self._lock:
            free_set = set(self._free_blocks)
            return [0 if i in free_set else 1 for i in range(self._total_blocks)]

    def allocate_sequence(self, seq_id: str, initial_tokens: int) -> List[int]:
        """Allocate physical blocks to accommodate initial sequence length.

        Args:
            seq_id: Unique string identifier for sequence.
            initial_tokens: Initial prompt token count.

        Returns:
            List of assigned physical block IDs.

        Raises:
            VRAMExhaustedError: If insufficient physical blocks remain.
            ValueError: If initial_tokens is negative or seq_id is duplicate.
        """
        if initial_tokens < 0:
            raise ValueError(f"initial_tokens cannot be negative, got {initial_tokens}")
        if seq_id in self._seq_to_blocks:
            raise ValueError(f"Sequence '{seq_id}' is already allocated.")

        needed_blocks = max(1, math.ceil(initial_tokens / self._block_size)) if initial_tokens > 0 else 0
        if len(self._free_blocks) < needed_blocks:
            raise VRAMExhaustedError(
                f"Cannot allocate {needed_blocks} blocks for sequence '{seq_id}'. "
                f"Available: {len(self._free_blocks)}, Total: {self._total_blocks}"
            )

        assigned: List[int] = []
        for _ in range(needed_blocks):
            assigned.append(self._free_blocks.popleft())

        self._seq_to_blocks[seq_id] = assigned
        self._seq_lengths[seq_id] = initial_tokens
        return list(assigned)

    def append_token(self, seq_id: str) -> Tuple[int, int]:
        """Record an additional generated token for a sequence, allocating blocks as needed.

        Args:
            seq_id: Active sequence identifier.

        Returns:
            Tuple of (physical_block_id, within_block_offset).

        Raises:
            BlockTableLookupError: If sequence is not allocated.
            VRAMExhaustedError: If a new block boundary is crossed and no free blocks exist.
        """
        if seq_id not in self._seq_to_blocks:
            raise BlockTableLookupError(f"Sequence '{seq_id}' has not been allocated.")

        current_length = self._seq_lengths[seq_id]
        new_length = current_length + 1
        needed_blocks = math.ceil(new_length / self._block_size)
        allocated_blocks = self._seq_to_blocks[seq_id]

        if len(allocated_blocks) < needed_blocks:
            if not self._free_blocks:
                raise VRAMExhaustedError(
                    f"VRAM exhausted while expanding sequence '{seq_id}' to {new_length} tokens."
                )
            new_block = self._free_blocks.popleft()
            allocated_blocks.append(new_block)

        self._seq_lengths[seq_id] = new_length

        logical_index = current_length
        block_idx = logical_index // self._block_size
        offset = logical_index % self._block_size

        return allocated_blocks[block_idx], offset

    def translate_address(self, seq_id: str, logical_token_idx: int) -> Tuple[int, int]:
        """Resolve logical token position to physical block index and offset (Proof M17).

        Args:
            seq_id: Sequence identifier.
            logical_token_idx: Zero-indexed token position in sequence.

        Returns:
            Tuple of (physical_block_id, within_block_offset).

        Raises:
            BlockTableLookupError: If sequence is unallocated or token offset is out of range.
        """
        if seq_id not in self._seq_to_blocks:
            raise BlockTableLookupError(f"Sequence '{seq_id}' is not allocated.")

        length = self._seq_lengths[seq_id]
        if logical_token_idx < 0 or logical_token_idx >= length:
            raise BlockTableLookupError(
                f"Token index {logical_token_idx} out of bounds for sequence of length {length}."
            )

        block_index = logical_token_idx // self._block_size
        block_offset = logical_token_idx % self._block_size
        physical_block = self._seq_to_blocks[seq_id][block_index]

        return physical_block, block_offset

    def free_sequence(self, seq_id: str) -> None:
        """Release all allocated physical blocks for a completed sequence back to the free pool.

        Args:
            seq_id: Sequence identifier to reclaim.
        """
        if seq_id in self._seq_to_blocks:
            blocks = self._seq_to_blocks.pop(seq_id)
            self._free_blocks.extend(blocks)
            self._seq_lengths.pop(seq_id, None)

    def compute_internal_fragmentation(self, seq_id: str) -> float:
        """Compute internal memory fragmentation percentage for sequence (Proof M11).

        Args:
            seq_id: Sequence identifier.

        Returns:
            Fragmentation ratio in [0.0, 1.0]. Returns 0.0 if sequence is empty.

        Raises:
            BlockTableLookupError: If sequence is not allocated.
        """
        if seq_id not in self._seq_to_blocks:
            raise BlockTableLookupError(f"Sequence '{seq_id}' is not allocated.")

        length = self._seq_lengths[seq_id]
        if length == 0:
            return 0.0

        total_allocated_tokens = len(self._seq_to_blocks[seq_id]) * self._block_size
        unused_tokens = total_allocated_tokens - length
        return float(unused_tokens / total_allocated_tokens)


class RadixPrefixCache:
    """Radix tree prefix cache for shared KV blocks across requests (Proof M12, M31)."""

    def __init__(self) -> None:
        """Initialize empty radix prefix tree with root node."""
        self._tree: Dict[int, Any] = {}
        self._cached_prefixes: Dict[Tuple[int, ...], List[int]] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._lock = threading.RLock()

    @property
    def hit_count(self) -> int:
        """Return total prefix cache hits."""
        return self._hit_count

    @property
    def miss_count(self) -> int:
        """Return total prefix cache misses."""
        return self._miss_count

    @property
    def total_lookups(self) -> int:
        """Return combined lookup operations."""
        return self._hit_count + self._miss_count

    @property
    def hit_rate(self) -> float:
        """Return empirical prefix hit rate ratio in [0.0, 1.0]."""
        total = self.total_lookups
        return float(self._hit_count / total) if total > 0 else 0.0

    def match_prefix(self, token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
        """Identify longest common prefix and return matched token IDs and block indices.

        Args:
            token_ids: Sequence of prompt token integers.

        Returns:
            Tuple of (matched_token_ids, matched_physical_block_ids).
        """
        with self._lock:
            longest_match: List[int] = []
            longest_blocks: List[int] = []

            tokens_tuple = tuple(token_ids)
            for cached_key, blocks in self._cached_prefixes.items():
                common_len = 0
                for a, b in zip(tokens_tuple, cached_key):
                    if a == b:
                        common_len += 1
                    else:
                        break
                if common_len > len(longest_match):
                    longest_match = list(tokens_tuple[:common_len])
                    longest_blocks = list(blocks[: (common_len // 16)])

            if longest_match:
                self._hit_count += 1
            else:
                self._miss_count += 1

            return longest_match, longest_blocks

    def insert_prefix(self, token_ids: Sequence[int], block_ids: Sequence[int]) -> None:
        """Register a computed prefix and its physical block IDs in the prefix cache.

        Args:
            token_ids: Sequence of token integers in prefix.
            block_ids: List of physical block IDs assigned to prefix.
        """
        with self._lock:
            if token_ids and block_ids:
                self._cached_prefixes[tuple(token_ids)] = list(block_ids)

    def clear(self) -> None:
        """Purge all cached prefix associations."""
        with self._lock:
            self._cached_prefixes.clear()
            self._tree.clear()
            self._hit_count = 0
            self._miss_count = 0
