"""Bare-metal flash-stream weight manager with constant-RAM ring buffering.

Streams model layer weights on-demand from non-volatile storage via zero-copy
memory-mapped I/O, maintaining an O(1) memory bound through fixed-slot ring
buffering and asynchronous prefetching.
"""

import ctypes
from dataclasses import dataclass, field
import enum
import gc
import mmap
import os
from pathlib import Path
import re
import sys
import threading
from types import TracebackType
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError, NativeBoundsError
from arc_runner.core.logging import get_logger
from arc_runner.native.bridge import NativeMemoryEngine, dtype_for_tensor_type, view_shape_for_dims
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.models import GGMLType, GGUFTensorInfo

logger = get_logger(__name__)

SLOT_WAIT_TIMEOUT_SECONDS = 30.0
PREFETCH_JOIN_TIMEOUT_SECONDS = 5.0
PIN_RING_ENV = "ARC_PIN_RING"

_PLAIN_ITEM_BYTES: Dict[GGMLType, int] = {
    GGMLType.F32: 4,
    GGMLType.F16: 2,
    GGMLType.BF16: 2,
    GGMLType.F64: 8,
    GGMLType.I8: 1,
    GGMLType.I16: 2,
    GGMLType.I32: 4,
    GGMLType.I64: 8,
}

_BLOCK_LAYOUT: Dict[GGMLType, Tuple[int, int]] = {
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 36),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
}


class SlotState(enum.Enum):
    """Lifecycle of a ring buffer slot; mirrors docs/formal/RingSlots.tla."""

    EMPTY = "empty"
    PENDING = "pending"
    READY = "ready"


@dataclass
class _RingSlot:
    """One direct-mapped ring buffer slot; layer L always occupies slot L % ring_buffer_slots."""

    layer_idx: int = -1
    state: SlotState = SlotState.EMPTY
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    ready_event: threading.Event = field(default_factory=threading.Event)


def process_resident_bytes() -> Optional[int]:
    """Return the resident set size (working set) of the current process in bytes.

    Returns:
        Optional[int]: Resident bytes, or None if the platform offers no cheap source.
    """
    if sys.platform == "win32":
        return _windows_working_set_bytes()
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _windows_working_set_bytes() -> Optional[int]:
    """Query the Windows working set size of the current process via psapi.

    Returns:
        Optional[int]: Working set size in bytes, or None if the call fails.
    """

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    process = windll.kernel32.GetCurrentProcess()
    if not windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


class FlashStreamWeightManager:
    """Manages constant-memory ring-buffered streaming of model weights from disk."""

    def __init__(
        self,
        model_path: Union[str, Path],
        ring_buffer_slots: int = 3,
        pin_resident_layers: Optional[bool] = None,
    ) -> None:
        """Initialize the flash-stream weight manager targeting a model container.

        Args:
            model_path: Filesystem path to the model container file.
            ring_buffer_slots: Number of resident layer slots in the memory ring buffer.
            pin_resident_layers: Pin each resident layer into physical RAM (best effort, native mapping
                only). None reads the ARC_PIN_RING environment variable; unset means off.

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
        self._slots: List[_RingSlot] = [_RingSlot() for _ in range(self.ring_buffer_slots)]
        self._closing = threading.Event()
        self._prefetch_threads: Set[threading.Thread] = set()
        self._slot_wait_timeout_s = SLOT_WAIT_TIMEOUT_SECONDS
        self._lock = threading.Lock()
        self._pin_lock = threading.Lock()
        if pin_resident_layers is None:
            pin_resident_layers = os.environ.get(PIN_RING_ENV, "").strip() == "1"
        self._pin_enabled = pin_resident_layers
        self._pinned: Dict[int, Tuple[int, int]] = {}
        self._pending_unpin: List[int] = []
        self._native_engine = NativeMemoryEngine()
        if self._native_engine.is_available:
            self._native_engine.open_model(self.model_path)

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

            if hasattr(self, "_native_engine") and self._native_engine.is_available and self._native_engine.is_open:
                for l_idx, t_list in self._layer_tensor_map.items():
                    if l_idx >= 0:
                        self._register_layer_natively(l_idx, t_list)

            return layer_counts

    def _register_layer_natively(self, layer_idx: int, tensors: List[GGUFTensorInfo]) -> None:
        """Register one layer's tensors with the native mapping, falling back to Python on rejection.

        Args:
            layer_idx: Layer index.
            tensors: Tensor descriptors belonging to the layer.
        """
        try:
            records = [
                {
                    "name": t.name,
                    "tensor_type": t.tensor_type,
                    "dimensions": t.dimensions,
                    "offset": self.parser.data_offset + t.offset,
                    "size_bytes": self._calculate_tensor_byte_size(t),
                }
                for t in tensors
            ]
            self._native_engine.register_layer(layer_idx, records)
        except (NativeBoundsError, ValueError) as exc:
            logger.warning(
                "Layer not registered natively; using the Python read path",
                extra={"layer": layer_idx, "reason": str(exc)},
            )

    def _calculate_tensor_byte_size(self, tensor: GGUFTensorInfo) -> int:
        """Calculate total byte footprint required for a tensor payload.

        Args:
            tensor: Tensor descriptor record.

        Returns:
            Total byte length required on disk.

        Raises:
            ValueError: If the tensor type has no known layout and the container gave no size.
        """
        if tensor.size_bytes > 0:
            return tensor.size_bytes

        num_elements = 1
        for dim in tensor.dimensions:
            num_elements *= dim

        tensor_type = GGMLType(tensor.tensor_type)
        if tensor_type in _PLAIN_ITEM_BYTES:
            return num_elements * _PLAIN_ITEM_BYTES[tensor_type]
        if tensor_type in _BLOCK_LAYOUT:
            block_elements, block_bytes = _BLOCK_LAYOUT[tensor_type]
            return (num_elements // block_elements) * block_bytes
        raise ValueError(f"Unknown byte layout for tensor '{tensor.name}' of type {tensor_type.name}")

    def _read_tensor_array(self, tensor: GGUFTensorInfo) -> np.ndarray:
        """Extract a single tensor array from the memory-mapped file buffer.

        Args:
            tensor: Descriptor defining tensor dimensions, type, and payload offset.

        Returns:
            Read-only NumPy array; plain types use their own dtype, bfloat16 its raw uint16 bits,
            and block-quantized types raw uint8 bytes. Identical to the native zero-copy path.

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

        dtype = dtype_for_tensor_type(tensor.tensor_type)
        raw_bytes = self._mmap_handle[start_offset:end_offset]
        if len(raw_bytes) % dtype.itemsize != 0:
            dtype = np.dtype(np.uint8)
        array = np.frombuffer(raw_bytes, dtype=dtype)
        shape = view_shape_for_dims(tensor.dimensions, array.size) if tensor.tensor_type in _PLAIN_ITEM_BYTES else None
        return array.reshape(shape) if shape else array

    def _extract_short_tensor_name(self, full_name: str) -> str:
        """Derive standard short parameter identifier from full container key.

        Args:
            full_name: Canonical tensor name from container metadata.

        Returns:
            Normalized tensor identifier within the layer scope.
        """
        cleaned = re.sub(r"^(?:blk|layers|model\.layers)\.\d+\.", "", full_name)
        return cleaned

    def _slot_index_for(self, layer_idx: int) -> int:
        """Return the only slot a layer may occupy (direct mapping).

        Args:
            layer_idx: Layer index.

        Returns:
            Slot index in [0, ring_buffer_slots).
        """
        return layer_idx % self.ring_buffer_slots

    def _load_layer_arrays(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Read every tensor of a layer, preferring the native zero-copy mapping.

        Args:
            layer_idx: Zero-based layer index to load.

        Returns:
            Dictionary mapping short tensor names to read-only arrays.
        """
        native = self._native_engine
        use_native = native is not None and native.is_available and native.is_open
        layer_weights: Dict[str, np.ndarray] = {}

        for t in self._layer_tensor_map.get(layer_idx, []):
            arr = None
            if use_native:
                dims = tuple(reversed(t.dimensions)) if t.tensor_type in _PLAIN_ITEM_BYTES else None
                arr = native.get_tensor_buffer(layer_idx, t.name, dims=dims, dtype=dtype_for_tensor_type(t.tensor_type))
            layer_weights[self._extract_short_tensor_name(t.name)] = arr if arr is not None else self._read_tensor_array(t)
        return layer_weights

    def _layer_span(self, layer_idx: int) -> Optional[Tuple[int, int]]:
        """Return the file range (offset, size) covering every tensor of a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Optional[Tuple[int, int]]: Offset and size, or None if the layer has no tensors.
        """
        tensors = self._layer_tensor_map.get(layer_idx, [])
        if not tensors:
            return None
        starts = [self.parser.data_offset + t.offset for t in tensors]
        ends = [start + self._calculate_tensor_byte_size(t) for start, t in zip(starts, tensors)]
        return min(starts), max(ends) - min(starts)

    def _pin_layer(self, layer_idx: int) -> None:
        """Best-effort pin of a resident layer; the first OS refusal disables further attempts.

        Args:
            layer_idx: Layer that was just published as READY.
        """
        native = self._native_engine
        if not self._pin_enabled or native is None or not (native.is_available and native.is_open):
            return
        with self._pin_lock:
            with self._lock:
                slot = self._slots[self._slot_index_for(layer_idx)]
                still_resident = slot.state is SlotState.READY and slot.layer_idx == layer_idx
            span = self._layer_span(layer_idx)
            if not still_resident or span is None or layer_idx in self._pinned:
                return
            if native.lock_range(span[0], span[1]):
                self._pinned[layer_idx] = span
            else:
                self._pin_enabled = False
                logger.info("Layer pinning disabled after the OS refused a lock", extra={"layer": layer_idx})

    def _flush_unpins(self) -> None:
        """Unlock every layer queued for unpinning by an eviction."""
        with self._pin_lock:
            with self._lock:
                queued, self._pending_unpin = self._pending_unpin, []
            native = self._native_engine
            for layer_idx in queued:
                span = self._pinned.pop(layer_idx, None)
                if span is not None and native is not None:
                    native.unlock_range(span[0], span[1])

    def _reserve_slot(self, layer_idx: int) -> Tuple[str, Optional[Dict[str, np.ndarray]], Optional[threading.Event]]:
        """Decide, under the lock, whether to hit, wait for, or start loading a layer.

        Args:
            layer_idx: Layer being requested.

        Returns:
            ("hit", arrays, None) if resident; ("wait", None, event) if a load is in flight and
            the caller must wait on the event and retry; ("load", None, None) if the caller became
            the single loader of the slot.

        Raises:
            ArcRunnerError: If the manager is closing or closed.
        """
        with self._lock:
            if self._closing.is_set():
                raise ArcRunnerError("Weight manager is closed", "FLASH_MANAGER_CLOSED", 400)
            slot = self._slots[self._slot_index_for(layer_idx)]
            if slot.state is SlotState.READY and slot.layer_idx == layer_idx:
                return "hit", slot.arrays, None
            if slot.state is SlotState.PENDING:
                return "wait", None, slot.ready_event
            if slot.state is SlotState.READY:
                self._pending_unpin.append(slot.layer_idx)
            slot.layer_idx = layer_idx
            slot.state = SlotState.PENDING
            slot.arrays = {}
            slot.ready_event = threading.Event()
            return "load", None, None

    def _publish_load(self, layer_idx: int, arrays: Optional[Dict[str, np.ndarray]]) -> None:
        """Publish a finished load (or reset the slot after a failed one) and wake waiters.

        Args:
            layer_idx: Layer that was loaded.
            arrays: Loaded arrays, or None if the load failed.
        """
        with self._lock:
            slot = self._slots[self._slot_index_for(layer_idx)]
            if arrays is None:
                slot.layer_idx = -1
                slot.state = SlotState.EMPTY
                slot.arrays = {}
            else:
                slot.arrays = arrays
                slot.state = SlotState.READY
            slot.ready_event.set()

    def acquire_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Obtain tensor weights for a layer, loading into its slot exactly once if not resident.

        Concurrent callers for the same layer share one load; callers whose slot is busy with
        another layer wait for that load to finish. Returned arrays stay valid after the slot is
        recycled because they keep the underlying mapping alive.

        Args:
            layer_idx: Target layer index.

        Returns:
            Dictionary mapping layer tensor identifiers to NumPy weight arrays.

        Raises:
            ArcRunnerError: If the manager is closed.
            TimeoutError: If a load in flight does not finish within the wait timeout.
        """
        while True:
            action, arrays, event = self._reserve_slot(layer_idx)
            if action == "hit" and arrays is not None:
                return arrays
            if action == "load":
                self._flush_unpins()
                try:
                    loaded = self._load_layer_arrays(layer_idx)
                except BaseException:
                    self._publish_load(layer_idx, None)
                    raise
                self._publish_load(layer_idx, loaded)
                self._pin_layer(layer_idx)
                return loaded
            if event is not None and not event.wait(self._slot_wait_timeout_s):
                raise TimeoutError(f"Timed out waiting for layer load in slot {self._slot_index_for(layer_idx)}")

    def prefetch_layer_async(self, layer_idx: int, slot_idx: int) -> threading.Thread:
        """Start a background load of a layer into its direct-mapped slot.

        Args:
            layer_idx: Zero-based layer index to prefetch.
            slot_idx: Ignored; the slot is always layer_idx % ring_buffer_slots. Kept for compatibility.

        Returns:
            The started background worker Thread instance.
        """
        thread: Optional[threading.Thread] = None

        def worker() -> None:
            try:
                if not self._closing.is_set():
                    self.acquire_layer_weights(layer_idx)
            except (ArcRunnerError, TimeoutError, ValueError) as exc:
                logger.warning("Prefetch failed", extra={"layer": layer_idx, "reason": str(exc)})
            finally:
                with self._lock:
                    self._prefetch_threads.discard(threading.current_thread())

        thread = threading.Thread(target=worker, daemon=True, name=f"prefetch-layer-{layer_idx}")
        with self._lock:
            self._prefetch_threads.add(thread)
        thread.start()
        return thread

    def release_layer_slot(self, slot_idx: int) -> None:
        """Evict a loaded slot and drop its array references; a load in flight is left alone.

        Args:
            slot_idx: Slot index to release.
        """
        dropped = False
        with self._lock:
            if 0 <= slot_idx < self.ring_buffer_slots:
                slot = self._slots[slot_idx]
                if slot.state is SlotState.READY:
                    self._pending_unpin.append(slot.layer_idx)
                    slot.layer_idx = -1
                    slot.state = SlotState.EMPTY
                    slot.arrays = {}
                    dropped = True
        if dropped:
            self._flush_unpins()
            gc.collect()

    def get_occupied_slot_count(self) -> int:
        """Count ring buffer slots that currently hold a loaded layer.

        Returns:
            Number of occupied slots, never more than ring_buffer_slots.
        """
        with self._lock:
            return sum(1 for slot in self._slots if slot.state is SlotState.READY)

    def get_active_memory_bytes(self) -> int:
        """Return the bytes of mapped (virtual) address space viewed by the resident slots.

        This is not resident RAM: mapped file pages may or may not be in the working set. Use
        get_resident_memory_bytes for process memory.

        Returns:
            Total size in bytes across all resident weight arrays.
        """
        with self._lock:
            return sum(
                arr.nbytes
                for slot in self._slots
                if slot.state is SlotState.READY
                for arr in slot.arrays.values()
                if isinstance(arr, np.ndarray)
            )

    def get_resident_memory_bytes(self) -> Optional[int]:
        """Return the resident set size of this process in bytes.

        Returns:
            Optional[int]: Process working set / RSS, or None if unavailable on this platform.
        """
        return process_resident_bytes()

    def close(self) -> None:
        """Stop prefetching, wait for loads in flight, drop cached slots, and release handles."""
        with self._lock:
            if self._closing.is_set():
                return
            self._closing.set()
            threads = list(self._prefetch_threads)
            pending_events = [slot.ready_event for slot in self._slots if slot.state is SlotState.PENDING]

        for thread in threads:
            thread.join(PREFETCH_JOIN_TIMEOUT_SECONDS)
        for event in pending_events:
            event.wait(PREFETCH_JOIN_TIMEOUT_SECONDS)

        with self._lock:
            for slot in self._slots:
                if slot.state is SlotState.READY:
                    self._pending_unpin.append(slot.layer_idx)
                slot.layer_idx = -1
                slot.state = SlotState.EMPTY
                slot.arrays = {}
        self._flush_unpins()

        with self._lock:
            native, self._native_engine = getattr(self, "_native_engine", None), None
            mmap_handle, self._mmap_handle = self._mmap_handle, None
            file_handle, self._file_handle = self._file_handle, None

        if native is not None:
            native.close()
        if mmap_handle is not None:
            mmap_handle.close()
        if file_handle is not None:
            file_handle.close()
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
