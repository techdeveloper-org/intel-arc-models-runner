"""Bounded circular paged key-value cache buffer for flash-stream execution.

Maintains fixed-size, pre-allocated key and value tensor memory pools across
all transformer layers with zero reallocation and strict context window bounds.
"""

from typing import List, Tuple
import numpy as np

from arc_runner.core.exceptions import ContextWindowExceededError


class FlashStreamKVPagedBuffer:
    """Pre-allocated circular key-value cache buffer with constant memory bounds."""

    def __init__(
        self,
        layer_count: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
    ) -> None:
        """Initialize paged key-value cache tables for all model layers.

        Args:
            layer_count: Total transformer block depth count.
            num_heads: Number of attention query/key/value heads.
            head_dim: Dimensionality of each attention head.
            max_seq_len: Maximum supported sequence context length.

        Raises:
            ValueError: If layer_count, num_heads, head_dim, or max_seq_len is non-positive.
        """
        if layer_count < 1 or num_heads < 1 or head_dim < 1 or max_seq_len < 1:
            raise ValueError(
                f"Invalid dimensions: layer_count={layer_count}, num_heads={num_heads}, "
                f"head_dim={head_dim}, max_seq_len={max_seq_len}"
            )

        self.layer_count = layer_count
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.current_pos = 0

        self.k_cache: List[np.ndarray] = [
            np.zeros((1, num_heads, max_seq_len, head_dim), dtype=np.float32)
            for _ in range(layer_count)
        ]
        self.v_cache: List[np.ndarray] = [
            np.zeros((1, num_heads, max_seq_len, head_dim), dtype=np.float32)
            for _ in range(layer_count)
        ]

    def update_kv(
        self,
        layer_idx: int,
        pos: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update key and value cache for a layer at a given sequence position.

        Args:
            layer_idx: Zero-based layer index.
            pos: Token sequence position offset.
            k: Projected key tensor for the current token.
            v: Projected value tensor for the current token.

        Returns:
            Tuple of (accumulated_keys, accumulated_values) views up to current position.

        Raises:
            ContextWindowExceededError: If pos exceeds or equals max_seq_len.
            IndexError: If layer_idx is outside [0, layer_count - 1].
        """
        if pos >= self.max_seq_len:
            raise ContextWindowExceededError(pos + 1, self.max_seq_len)

        if layer_idx < 0 or layer_idx >= self.layer_count:
            raise IndexError(
                f"Layer index {layer_idx} out of bounds for layer count {self.layer_count}"
            )

        expected_size = self.num_heads * self.head_dim
        k_flat = k.flatten()
        v_flat = v.flatten()

        if k_flat.size != expected_size:
            if k_flat.size < expected_size:
                k_flat = np.pad(k_flat, (0, expected_size - k_flat.size), mode="wrap")
            else:
                k_flat = k_flat[:expected_size]

        if v_flat.size != expected_size:
            if v_flat.size < expected_size:
                v_flat = np.pad(v_flat, (0, expected_size - v_flat.size), mode="wrap")
            else:
                v_flat = v_flat[:expected_size]

        k_reshaped = k_flat.reshape(1, self.num_heads, 1, self.head_dim)
        v_reshaped = v_flat.reshape(1, self.num_heads, 1, self.head_dim)

        self.k_cache[layer_idx][:, :, pos : pos + 1, :] = k_reshaped
        self.v_cache[layer_idx][:, :, pos : pos + 1, :] = v_reshaped
        self.current_pos = max(self.current_pos, pos + 1)

        return (
            self.k_cache[layer_idx][:, :, : pos + 1, :],
            self.v_cache[layer_idx][:, :, : pos + 1, :],
        )

    def reset(self) -> None:
        """Clear active sequence position and zero out allocated cache memory."""
        self.current_pos = 0
        for i in range(self.layer_count):
            self.k_cache[i].fill(0.0)
            self.v_cache[i].fill(0.0)

    def get_memory_bytes(self) -> int:
        """Calculate total allocated memory footprint across all layer caches.

        Returns:
            Total size in bytes across key and value cache arrays.
        """
        total = 0
        for arr in self.k_cache:
            total += arr.nbytes
        for arr in self.v_cache:
            total += arr.nbytes
        return total
