"""Python wrapper for lock-free circular ring buffer in native USM / shared memory."""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger(__name__)


class RingBufferError(ArcRunnerError):
    """Exception raised on ring buffer error or corruption."""


class _USMRingBuffer(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("pad0", ctypes.c_uint8 * 60),
        ("tail", ctypes.c_uint32),
        ("pad1", ctypes.c_uint8 * 60),
        ("capacity", ctypes.c_uint32),
        ("mask", ctypes.c_uint32),
        ("tokens", ctypes.POINTER(ctypes.c_int64)),
        ("latencies_ms", ctypes.POINTER(ctypes.c_float)),
    ]


class USMTokenStreamer:
    """Zero-copy lock-free token streaming queue."""

    _dll: Optional[ctypes.CDLL] = None

    def __init__(self, capacity: int = 4096) -> None:
        """Initialize the ring buffer queue.

        Args:
            capacity: Buffer token capacity (rounded to power of two).
        """
        self._capacity = capacity
        self._ring: Optional[ctypes.c_void_p] = None
        self._init_native()
        self._create_ring()

    def _init_native(self) -> None:
        """Load native DLL and bind C-ABI functions."""
        if USMTokenStreamer._dll is not None:
            return

        dll_name = "libarc_native_sys.dll" if sys.platform == "win32" else "libarc_native_sys.so"
        dll_path = Path(__file__).resolve().parent / dll_name

        if not dll_path.exists():
            logger.warning("Native library %s not found at %s", dll_name, dll_path)
            return

        try:
            dll = ctypes.CDLL(str(dll_path))

            dll.arc_usm_ring_create.argtypes = [ctypes.c_size_t]
            dll.arc_usm_ring_create.restype = ctypes.c_void_p

            dll.arc_usm_ring_push.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_float]
            dll.arc_usm_ring_push.restype = ctypes.c_int

            dll.arc_usm_ring_pop.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_float),
            ]
            dll.arc_usm_ring_pop.restype = ctypes.c_int

            dll.arc_usm_ring_destroy.argtypes = [ctypes.c_void_p]
            dll.arc_usm_ring_destroy.restype = None

            USMTokenStreamer._dll = dll
        except Exception as exc:
            logger.warning("Failed to bind native ring buffer functions: %s", exc)

    def _create_ring(self) -> None:
        """Allocate native ring buffer."""
        if USMTokenStreamer._dll is not None:
            self._ring = USMTokenStreamer._dll.arc_usm_ring_create(self._capacity)

    def push(self, token_id: int, latency_ms: float = 0.0) -> bool:
        """Push a token into the ring buffer.

        Args:
            token_id: Integer token ID.
            latency_ms: Microsecond latency delta in milliseconds.

        Returns:
            bool: True if push succeeded, False on overflow.
        """
        if self._ring is None or USMTokenStreamer._dll is None:
            return False
        ret = USMTokenStreamer._dll.arc_usm_ring_push(self._ring, ctypes.c_int64(token_id), ctypes.c_float(latency_ms))
        return ret == 0

    def pop(self) -> Optional[Tuple[int, float]]:
        """Pop the next available token.

        Returns:
            Tuple of (token_id, latency_ms) or None if buffer is empty.
        """
        if self._ring is None or USMTokenStreamer._dll is None:
            return None

        out_tok = ctypes.c_int64()
        out_lat = ctypes.c_float()
        ret = USMTokenStreamer._dll.arc_usm_ring_pop(self._ring, ctypes.byref(out_tok), ctypes.byref(out_lat))
        if ret == 0:
            return (out_tok.value, round(out_lat.value, 4))
        return None

    def drain(self) -> List[Tuple[int, float]]:
        """Drain all tokens currently in the ring buffer.

        Returns:
            List of (token_id, latency_ms) tuples.
        """
        items = []
        while True:
            popped = self.pop()
            if popped is None:
                break
            items.append(popped)
        return items

    def close(self) -> None:
        """Deallocate native ring buffer memory."""
        if self._ring is not None and USMTokenStreamer._dll is not None:
            USMTokenStreamer._dll.arc_usm_ring_destroy(self._ring)
            self._ring = None

    def __del__(self) -> None:
        """Destructor cleanup."""
        self.close()
