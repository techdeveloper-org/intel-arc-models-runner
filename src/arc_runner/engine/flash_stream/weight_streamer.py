"""Bare-metal flash-stream weight manager with constant-RAM ring buffering.

Streams model layer weights on-demand from non-volatile storage via zero-copy
memory-mapped I/O, maintaining an O(1) memory bound through fixed-slot ring
buffering and asynchronous prefetching.
"""

import gc
import mmap
from pathlib import Path
import re
import threading
import time
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np

from arc_runner.core.exceptions import ModelNotFoundError
from arc_runner.core.logging import get_logger
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.models import GGMLType, GGUFTensorInfo

logger = get_logger(__name__)


class FlashStreamWeightManager:
    """Manages constant-memory ring-buffered streaming of model weights from disk."""

    def __init__(self, model_path: Union[str, Path], ring_buffer_slots: int = 3) -> None:
        """Initialize the flash-stream weight manager targeting a model container.

        Args:
            model_path: Filesystem path to the model container file.
            ring_buffer_slots: Number of resident layer slots in the memory ring buffer.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            ValueError: If ring_buffer_slots is less than 1.
        """
        if ring_buffer_slots < 1:
            raise ValueError(f"Ring buffer slots must be at least 1, got {ring_buffer_slots}")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.ring_buffer_slots = ring_buffer_slots
        self.parser = GGUFParser(self.model_path)
        self.parser.parse_all()

        self._file_size = self.model_path.stat().st_size
        self._file_handle = open(self.model_path, "rb")
        self._mmap_handle: Optional[mmap.mmap] = None
        if self._file_size > 0:
            self._mmap_handle = mmap.mmap(
                self._file_handle.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )

        self._layer_tensor_map: Dict[int, List[GGUFTensorInfo]] = {}
        self._ring_slots: List[Optional[Dict[str, np.ndarray]]] = [None] * self.ring_buffer_slots
        self._slot_layer_idx: List[int] = [-1] * self.ring_buffer_slots
        self._slot_access_time: List[float] = [0.0] * self.ring_buffer_slots
        self._lock = threading.Lock()

        self.index_layers()

    def index_layers(self) -> Dict[int, int]:
        """Index all layer tensors from parsed metadata descriptors without loading weights.

        Returns:
            Dictionary mapping layer index integers to the count of associated tensors.
        """
        with self._lock:
            self._layer_tensor_map.clear()
            layer_counts: Dict[int, int] = {}
            layer_pattern = re.compile(r"(?:blk|layers|model\.layers)\.(\d+)\.")

            for tensor in self.parser.tensor_info:
                match = layer_pattern.search(tensor.name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in self._layer_tensor_map:
                        self._layer_tensor_map[layer_idx] = []
                    self._layer_tensor_map[layer_idx].append(tensor)
                    layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1
                else:
                    if -1 not in self._layer_tensor_map:
                        self._layer_tensor_map[-1] = []
                    self._layer_tensor_map[-1].append(tensor)

            return layer_counts

    def _calculate_tensor_byte_size(self, tensor: GGUFTensorInfo) -> int:
        """Calculate total byte footprint required for a tensor payload.

        Args:
            tensor: Tensor descriptor record.

        Returns:
            Total byte length required on disk.
        """
        if tensor.size_bytes > 0:
            return tensor.size_bytes

        num_elements = 1
        for dim in tensor.dimensions:
            num_elements *= dim

        if tensor.tensor_type == GGMLType.F32:
            return num_elements * 4
        elif tensor.tensor_type in (GGMLType.F16, GGMLType.BF16, GGMLType.I16):
            return num_elements * 2
        elif tensor.tensor_type in (GGMLType.I8, GGMLType.Q8_0):
            return (num_elements // 32) * 34 if tensor.tensor_type == GGMLType.Q8_0 else num_elements
        elif tensor.tensor_type == GGMLType.Q4_0:
            return (num_elements // 32) * 18
        elif tensor.tensor_type == GGMLType.Q4_1:
            return (num_elements // 32) * 20
        elif tensor.tensor_type == GGMLType.I32:
            return num_elements * 4
        return max(num_elements, 1)

    def _read_tensor_array(self, tensor: GGUFTensorInfo) -> np.ndarray:
        """Extract a single tensor array from the memory-mapped file buffer.

        Args:
            tensor: Descriptor defining tensor dimensions, type, and payload offset.

        Returns:
            Constructed NumPy array containing the tensor payload.

        Raises:
            ValueError: If tensor offsets or calculated bounds exceed container length.
        """
        if self._mmap_handle is None:
            raise ValueError(f"Memory map unavailable for container: {self.model_path}")

        start_offset = self.parser.data_offset + tensor.offset
        byte_size = self._calculate_tensor_byte_size(tensor)
        end_offset = start_offset + byte_size

        if start_offset < 0 or start_offset >= self._file_size or end_offset > self._file_size:
            raise ValueError(
                f"Corrupted tensor '{tensor.name}' offset {start_offset} with size "
                f"{byte_size} exceeds container length {self._file_size}"
            )

        raw_bytes = self._mmap_handle[start_offset:end_offset]

        if tensor.tensor_type == GGMLType.F32 and len(raw_bytes) >= 4:
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
            try:
                return arr.reshape(tensor.dimensions)
            except ValueError:
                return arr
        elif tensor.tensor_type in (GGMLType.F16, GGMLType.BF16) and len(raw_bytes) >= 2:
            arr = np.frombuffer(raw_bytes, dtype=np.float16)
            try:
                return arr.reshape(tensor.dimensions)
            except ValueError:
                return arr
        elif tensor.tensor_type in (GGMLType.I8,) and len(raw_bytes) >= 1:
            arr = np.frombuffer(raw_bytes, dtype=np.int8)
            try:
                return arr.reshape(tensor.dimensions)
            except ValueError:
                return arr

        raw_uint8 = np.frombuffer(raw_bytes, dtype=np.uint8)
        return raw_uint8.astype(np.float32)

    def _extract_short_tensor_name(self, full_name: str) -> str:
        """Derive standard short parameter identifier from full container key.

        Args:
            full_name: Canonical tensor name from container metadata.

        Returns:
            Normalized tensor identifier within the layer scope.
        """
        cleaned = re.sub(r"^(?:blk|layers|model\.layers)\.\d+\.", "", full_name)
        return cleaned

    def _load_layer_into_slot(self, layer_idx: int, slot_idx: int) -> None:
        """Load weights for the given layer directly into the designated ring buffer slot.

        Args:
            layer_idx: Zero-based layer index to load.
            slot_idx: Ring buffer slot index to occupy.
        """
        tensors = self._layer_tensor_map.get(layer_idx, [])
        layer_weights: Dict[str, np.ndarray] = {}

        for t in tensors:
            short_name = self._extract_short_tensor_name(t.name)
            layer_weights[short_name] = self._read_tensor_array(t)

        with self._lock:
            self._ring_slots[slot_idx] = layer_weights
            self._slot_layer_idx[slot_idx] = layer_idx
            self._slot_access_time[slot_idx] = time.time()

    def acquire_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Obtain tensor weights for a layer, loading into the ring buffer if not resident.

        Args:
            layer_idx: Target layer index.

        Returns:
            Dictionary mapping layer tensor identifiers to NumPy weight arrays.
        """
        with self._lock:
            for s in range(self.ring_buffer_slots):
                if self._slot_layer_idx[s] == layer_idx and self._ring_slots[s] is not None:
                    self._slot_access_time[s] = time.time()
                    return self._ring_slots[s]

            free_slot = -1
            for s in range(self.ring_buffer_slots):
                if self._ring_slots[s] is None:
                    free_slot = s
                    break

            if free_slot == -1:
                oldest_time = float("inf")
                target_slot = 0
                for s in range(self.ring_buffer_slots):
                    if self._slot_access_time[s] < oldest_time:
                        oldest_time = self._slot_access_time[s]
                        target_slot = s
                free_slot = target_slot
                self._ring_slots[free_slot] = None
                self._slot_layer_idx[free_slot] = -1
                gc.collect()

        self._load_layer_into_slot(layer_idx, free_slot)

        with self._lock:
            result = self._ring_slots[free_slot]
            assert result is not None
            return result

    def prefetch_layer_async(self, layer_idx: int, slot_idx: int) -> threading.Thread:
        """Initiate asynchronous background prefetch of layer weights into a target slot.

        Args:
            layer_idx: Zero-based layer index to prefetch.
            slot_idx: Designated ring buffer slot index.

        Returns:
            The started background worker Thread instance.
        """
        target_slot = slot_idx % self.ring_buffer_slots

        def worker() -> None:
            self._load_layer_into_slot(layer_idx, target_slot)

        thread = threading.Thread(
            target=worker,
            daemon=True,
            name=f"prefetch-layer-{layer_idx}-slot-{target_slot}",
        )
        thread.start()
        return thread

    def release_layer_slot(self, slot_idx: int) -> None:
        """Evict and free memory buffers for a specific ring buffer slot.

        Args:
            slot_idx: Slot index to release.
        """
        with self._lock:
            if 0 <= slot_idx < self.ring_buffer_slots:
                self._ring_slots[slot_idx] = None
                self._slot_layer_idx[slot_idx] = -1
                self._slot_access_time[slot_idx] = 0.0
                gc.collect()

    def get_active_memory_bytes(self) -> int:
        """Calculate current total resident memory consumed by resident ring buffer slots.

        Returns:
            Total size in bytes across all resident weight arrays.
        """
        total = 0
        with self._lock:
            for slot in self._ring_slots:
                if slot is not None:
                    for arr in slot.values():
                        if isinstance(arr, np.ndarray):
                            total += arr.nbytes
        return total

    def close(self) -> None:
        """Release memory-mapped handles, underlying files, and cached ring slots."""
        with self._lock:
            for s in range(self.ring_buffer_slots):
                self._ring_slots[s] = None
                self._slot_layer_idx[s] = -1
                self._slot_access_time[s] = 0.0

            if self._mmap_handle is not None:
                self._mmap_handle.close()
                self._mmap_handle = None

            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None

            gc.collect()

    def __enter__(self) -> "FlashStreamWeightManager":
        """Enter execution runtime context.

        Returns:
            FlashStreamWeightManager instance.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit runtime context and cleanly release resources.

        Args:
            exc_type: Exception type if raised during context.
            exc_val: Exception instance if raised.
            exc_tb: Traceback object if raised.
        """
        self.close()
