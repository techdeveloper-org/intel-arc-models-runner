"""Stage ring-buffer layers into Level Zero Unified Shared Memory through the native bridge.

Staging copies each tensor of a resident layer into one USM buffer (host USM by default, which an
integrated Arc GPU reads without a second copy). Every figure reported is counted from the copies
actually issued; the module never estimates bandwidth.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from arc_runner.core.exceptions import NativeLayerError
from arc_runner.core.logging import get_logger

logger = get_logger("engine.device_stager")

STAGE_ALIGNMENT_BYTES = 64
_COPY_FELL_BACK = 1
_GPU_MAY_STILL_USE_MEMORY = frozenset({-9, -8})


@dataclass
class StagedLayer:
    """A layer resident in one USM buffer.

    Attributes:
        layer_idx: Layer index.
        ptr: Address of the USM buffer.
        size_bytes: Allocated buffer size.
        extents: Tensor short name -> (offset into the buffer, byte length).
        generation: Ring-slot generation of the load this copy belongs to, if the caller tracks one.
    """

    layer_idx: int
    ptr: int
    size_bytes: int
    extents: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    generation: Optional[int] = None


@dataclass(frozen=True)
class StagingStats:
    """Counters accumulated by a DeviceLayerStager.

    Attributes:
        layers_staged: Layers successfully staged.
        bytes_staged: Bytes copied into USM.
        seconds_total: Wall-clock seconds spent in allocate + copy + synchronize.
        fallback_copies: Copies the bridge performed on the CPU instead of the GPU command list.
    """

    layers_staged: int
    bytes_staged: int
    seconds_total: float
    fallback_copies: int


def _aligned(value: int) -> int:
    """Round a byte count up to the staging alignment.

    Args:
        value: Byte count.

    Returns:
        int: value rounded up to a multiple of STAGE_ALIGNMENT_BYTES.
    """
    return (value + STAGE_ALIGNMENT_BYTES - 1) // STAGE_ALIGNMENT_BYTES * STAGE_ALIGNMENT_BYTES


class DeviceLayerStager:
    """Copies a layer's tensors into a USM buffer and tracks the buffers it owns."""

    def __init__(self, bridge: Any, usm_type: str = "host") -> None:
        """Bind the stager to a Level Zero bridge.

        Args:
            bridge: NativeLevelZeroBridge (or an object with the same allocate/copy/sync/free surface).
            usm_type: USM residency passed to the bridge: host, device or shared.
        """
        self._bridge = bridge
        self._usm_type = usm_type
        self._lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._retained: List[Tuple[int, Any]] = []
        self._staged: Dict[int, StagedLayer] = {}
        self._layers = 0
        self._bytes = 0
        self._seconds = 0.0
        self._fallbacks = 0

    def stage_layer(self, layer_idx: int, arrays: Mapping[str, np.ndarray], generation: Optional[int] = None) -> StagedLayer:
        """Copy every tensor of a layer into one USM buffer and wait (bounded) for completion.

        Args:
            layer_idx: Layer index; a previously staged copy of the same layer is replaced.
            arrays: Short tensor name -> array holding the layer's weights.
            generation: Optional ring-slot generation recorded with the buffer.

        Returns:
            StagedLayer: The resident buffer description.

        Raises:
            NativeLayerError: If the bridge is unavailable, the allocation is refused, or a copy or
                the synchronize fails; the buffer is freed before the error propagates.
        """
        if not self._bridge.is_available:
            raise NativeLayerError("Level Zero bridge unavailable; cannot stage layer", -10)

        extents: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        contiguous = {name: np.ascontiguousarray(array) for name, array in arrays.items()}
        for name, array in contiguous.items():
            extents[name] = (cursor, int(array.nbytes))
            cursor += _aligned(int(array.nbytes))
        total = max(cursor, STAGE_ALIGNMENT_BYTES)

        started = time.monotonic()
        with self._submit_lock:
            ptr = self._bridge.allocate_usm(total, self._usm_type)
            if not ptr:
                raise NativeLayerError(f"USM allocation of {total} bytes refused for layer {layer_idx}", -5)

            fallbacks = 0
            try:
                for name, array in contiguous.items():
                    offset, size = extents[name]
                    if size == 0:
                        continue
                    status = int(self._bridge.dma_copy_async(ptr + offset, array.ctypes.data, size))
                    if status < 0:
                        raise NativeLayerError(f"DMA copy of '{name}' failed for layer {layer_idx}", status)
                    if status == _COPY_FELL_BACK:
                        fallbacks += 1
                sync_status = int(self._bridge.synchronize())
                if sync_status < 0:
                    raise NativeLayerError(f"Synchronize failed while staging layer {layer_idx}", sync_status)
            except NativeLayerError as exc:
                if exc.native_status in _GPU_MAY_STILL_USE_MEMORY:
                    self._retained.append((ptr, contiguous))
                else:
                    self._bridge.free_usm(ptr)
                raise
        elapsed = time.monotonic() - started

        staged = StagedLayer(layer_idx=layer_idx, ptr=ptr, size_bytes=total, extents=extents, generation=generation)
        with self._lock:
            replaced = self._staged.get(layer_idx)
            self._staged[layer_idx] = staged
            self._layers += 1
            self._bytes += sum(size for _, size in extents.values())
            self._seconds += elapsed
            self._fallbacks += fallbacks
        if replaced is not None:
            self._bridge.free_usm(replaced.ptr)
        return staged

    def get(self, layer_idx: int) -> Optional[StagedLayer]:
        """Return the staged buffer of a layer, if any.

        Args:
            layer_idx: Layer index.

        Returns:
            Optional[StagedLayer]: The buffer, or None if the layer is not staged.
        """
        with self._lock:
            return self._staged.get(layer_idx)

    def release_layer(self, layer_idx: int, generation: Optional[int] = None) -> None:
        """Free a layer's USM buffer; unknown layers are ignored.

        Args:
            layer_idx: Layer index.
            generation: If given, only a buffer staged for that generation is freed; a buffer staged for
                a newer load of the same layer is left alone.
        """
        with self._lock:
            current = self._staged.get(layer_idx)
            if current is None or (generation is not None and current.generation != generation):
                return
            staged = self._staged.pop(layer_idx)
        if staged is not None:
            self._bridge.free_usm(staged.ptr)

    def retained_after_failure(self) -> int:
        """Count buffers deliberately kept after a timeout or dead context.

        After a synchronize timeout the GPU may still read the source arrays and write the buffer, so
        neither is released; the memory is reclaimed when the Level Zero context is destroyed.

        Returns:
            int: Number of retained buffers.
        """
        return len(self._retained)

    def stats(self) -> StagingStats:
        """Return the counters accumulated so far.

        Returns:
            StagingStats: Snapshot of staged layers, bytes, seconds and CPU-fallback copies.
        """
        with self._lock:
            return StagingStats(self._layers, self._bytes, self._seconds, self._fallbacks)

    def close(self) -> None:
        """Free every staged buffer."""
        with self._lock:
            staged, self._staged = list(self._staged.values()), {}
        for item in staged:
            self._bridge.free_usm(item.ptr)
