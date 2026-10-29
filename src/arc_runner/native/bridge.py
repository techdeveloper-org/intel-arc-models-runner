"""Python foreign function interface bridging native C acceleration library.

Wraps libarc_native_sys.dll for high-performance zero-copy memory mapping,
working set physical RAM pinning, and Level Zero USM buffer allocations.
"""

import ctypes
from pathlib import Path
import sys
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.native.bridge")

ARC_MAX_NAME_LEN = 128
ARC_MAX_TENSORS_PER_LAYER = 64
ARC_MAX_LAYERS = 256


class ArcTensorDescriptor(ctypes.Structure):
    """C-binary tensor descriptor holding pre-calculated file offsets."""

    _fields_ = [
        ("name", ctypes.c_char * ARC_MAX_NAME_LEN),
        ("tensor_type", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint64 * 4),
        ("file_offset", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint64),
    ]


class ArcLayerDescriptor(ctypes.Structure):
    """C-binary layer descriptor mapping a layer index to constituent tensors."""

    _fields_ = [
        ("layer_idx", ctypes.c_int32),
        ("tensor_count", ctypes.c_uint32),
        ("total_layer_bytes", ctypes.c_uint64),
        ("tensors", ArcTensorDescriptor * ARC_MAX_TENSORS_PER_LAYER),
    ]


def _load_native_library() -> Optional[ctypes.CDLL]:
    """Locate and load libarc_native_sys shared object across environments.

    Returns:
        Optional[ctypes.CDLL]: Loaded library handle, or None if unavailable.
    """
    native_dir = Path(__file__).resolve().parent
    lib_name = "libarc_native_sys.dll" if sys.platform == "win32" else "libarc_native_sys.so"
    lib_path = native_dir / lib_name

    if not lib_path.exists():
        return None

    try:
        dll = ctypes.CDLL(str(lib_path))
        _configure_signatures(dll)
        return dll
    except Exception as exc:
        logger.warning("Failed loading native library %s: %s", lib_path, exc)
        return None


def _configure_signatures(dll: ctypes.CDLL) -> None:
    """Configure parameter and return types for exported native functions.

    Args:
        dll: Loaded shared library instance.
    """
    dll.arc_mmap_open.argtypes = [ctypes.c_char_p]
    dll.arc_mmap_open.restype = ctypes.c_void_p

    dll.arc_mmap_register_layer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ArcLayerDescriptor)]
    dll.arc_mmap_register_layer.restype = ctypes.c_int

    dll.arc_mmap_get_tensor_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    dll.arc_mmap_get_tensor_ptr.restype = ctypes.POINTER(ctypes.c_uint8)

    dll.arc_mmap_lock_range.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    dll.arc_mmap_lock_range.restype = ctypes.c_int

    dll.arc_mmap_unlock_range.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    dll.arc_mmap_unlock_range.restype = ctypes.c_int

    dll.arc_mmap_close.argtypes = [ctypes.c_void_p]
    dll.arc_mmap_close.restype = None

    dll.arc_ze_init.argtypes = []
    dll.arc_ze_init.restype = ctypes.c_void_p

    dll.arc_ze_alloc_usm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    dll.arc_ze_alloc_usm.restype = ctypes.c_void_p

    dll.arc_ze_dma_copy_async.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    dll.arc_ze_dma_copy_async.restype = ctypes.c_int

    dll.arc_ze_synchronize.argtypes = [ctypes.c_void_p]
    dll.arc_ze_synchronize.restype = ctypes.c_int

    dll.arc_ze_free_usm.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.arc_ze_free_usm.restype = None

    dll.arc_ze_destroy.argtypes = [ctypes.c_void_p]
    dll.arc_ze_destroy.restype = None


class NativeMemoryEngine:
    """Bare-metal Win32 memory-mapped model streaming manager."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the native memory engine.

        Args:
            file_path: Optional filesystem path to open immediately.
        """
        self._dll: Optional[ctypes.CDLL] = _load_native_library()
        self._ctx: Optional[int] = None
        self._file_path: Optional[Path] = Path(file_path).resolve() if file_path else None
        if self._file_path and self.is_available:
            self.open_model(self._file_path)

    @property
    def is_available(self) -> bool:
        """Check if native acceleration library is loaded.

        Returns:
            bool: True if libarc_native_sys is active.
        """
        return self._dll is not None

    @property
    def is_open(self) -> bool:
        """Check if an active memory-mapped context exists.

        Returns:
            bool: True if container file is currently mapped.
        """
        return self._ctx is not None

    def open_model(self, file_path: Union[str, Path]) -> bool:
        """Map a model container file into system memory.

        Args:
            file_path: Target filesystem path to container.

        Returns:
            bool: True if mapped successfully, False otherwise.
        """
        if not self.is_available or self._dll is None:
            return False

        if self.is_open:
            self.close()

        resolved = Path(file_path).resolve()
        if not resolved.exists():
            return False

        path_bytes = str(resolved).encode("utf-8")
        raw_ctx = self._dll.arc_mmap_open(path_bytes)
        if not raw_ctx:
            return False

        self._ctx = raw_ctx
        self._file_path = resolved
        return True

    def register_layer(self, layer_idx: int, tensors: List[Any]) -> bool:
        """Register tensor descriptors for a layer into the native C lookup table.

        Args:
            layer_idx: Zero-based layer index integer.
            tensors: List of tensor metadata records.

        Returns:
            bool: True on successful registration.
        """
        if not self.is_open or self._dll is None:
            return False

        c_layer = ArcLayerDescriptor()
        c_layer.layer_idx = layer_idx
        c_layer.tensor_count = min(len(tensors), ARC_MAX_TENSORS_PER_LAYER)

        total_bytes = 0
        for i, t in enumerate(tensors[:ARC_MAX_TENSORS_PER_LAYER]):
            raw_name = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", str(t))
            name_bytes = raw_name.encode("utf-8")[:ARC_MAX_NAME_LEN - 1]
            c_layer.tensors[i].name = name_bytes
            t_type = t.get("tensor_type", 0) if isinstance(t, dict) else getattr(t, "tensor_type", 0)
            c_layer.tensors[i].tensor_type = int(t_type)
            dims = t.get("dimensions", [1]) if isinstance(t, dict) else getattr(t, "dimensions", [1])
            c_layer.tensors[i].n_dims = len(dims)
            for d_idx, d_val in enumerate(dims[:4]):
                c_layer.tensors[i].dims[d_idx] = d_val
            c_layer.tensors[i].file_offset = t.get("offset", 0) if isinstance(t, dict) else getattr(t, "offset", 0)
            sz = t.get("size_bytes", 0) if isinstance(t, dict) else getattr(t, "size_bytes", 0)
            c_layer.tensors[i].size_bytes = sz
            total_bytes += sz

        c_layer.total_layer_bytes = total_bytes
        res = self._dll.arc_mmap_register_layer(self._ctx, ctypes.byref(c_layer))
        return res == 0

    def get_tensor_buffer(
        self,
        layer_idx: int,
        tensor_name: str,
        dims: Optional[Tuple[int, ...]] = None,
        dtype: Any = np.float32,
    ) -> Optional[np.ndarray]:
        """Retrieve a direct zero-copy NumPy array view wrapped around the mapped memory.

        Args:
            layer_idx: Layer index identifier.
            tensor_name: Short name of tensor.
            dims: Optional dimension tuple to reshape.
            dtype: Target NumPy datatype.

        Returns:
            Optional[np.ndarray]: Zero-copy array view, or None on failure.
        """
        if not self.is_open or self._dll is None:
            return None

        out_size = ctypes.c_uint64(0)
        ptr = self._dll.arc_mmap_get_tensor_ptr(
            self._ctx,
            layer_idx,
            tensor_name.encode("utf-8"),
            ctypes.byref(out_size),
        )

        if not ptr or out_size.value == 0:
            return None

        addr = ctypes.cast(ptr, ctypes.c_void_p).value
        if not addr:
            return None

        itemsize = np.dtype(dtype).itemsize
        element_count = out_size.value // itemsize

        raw_array = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))),
            shape=(element_count,),
        )

        if dims:
            try:
                return raw_array.reshape(dims)
            except ValueError:
                return raw_array

        return raw_array

    def lock_range(self, offset: int, size: int) -> bool:
        """Lock virtual memory region into physical RAM working set.

        Args:
            offset: Byte offset into file.
            size: Number of bytes to pin.

        Returns:
            bool: True if locked successfully.
        """
        if not self.is_open or self._dll is None:
            return False
        return self._dll.arc_mmap_lock_range(self._ctx, offset, size) == 0

    def unlock_range(self, offset: int, size: int) -> bool:
        """Unlock previously pinned memory region.

        Args:
            offset: Byte offset into file.
            size: Number of bytes to unpin.

        Returns:
            bool: True on success.
        """
        if not self.is_open or self._dll is None:
            return False
        return self._dll.arc_mmap_unlock_range(self._ctx, offset, size) == 0

    def close(self) -> None:
        """Unmap views and release native OS handles."""
        if self._ctx and self._dll:
            self._dll.arc_mmap_close(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        """Ensure native context is cleanly closed upon garbage collection."""
        self.close()


class NativeLevelZeroBridge:
    """Bare-metal Level Zero Unified Shared Memory and DMA coordinator."""

    def __init__(self) -> None:
        """Initialize the Level Zero driver interface context."""
        self._dll: Optional[ctypes.CDLL] = _load_native_library()
        self._ctx: Optional[int] = None
        if self._dll:
            raw_ctx = self._dll.arc_ze_init()
            if raw_ctx:
                self._ctx = raw_ctx

    @property
    def is_available(self) -> bool:
        """Check whether Level Zero driver runtime is accessible.

        Returns:
            bool: True if ze_loader.dll initialized and context is valid.
        """
        return self._ctx is not None and self._dll is not None

    def allocate_usm(self, size_bytes: int, usm_type: str = "shared") -> Optional[int]:
        """Allocate a Unified Shared Memory buffer on Intel Arc GPU.

        Args:
            size_bytes: Size of allocation in bytes.
            usm_type: Residency type ('host', 'device', or 'shared').

        Returns:
            Optional[int]: Virtual memory pointer address, or None on failure.
        """
        if not self.is_available or self._dll is None:
            return None

        type_map = {"host": 0, "device": 1, "shared": 2}
        t_val = type_map.get(usm_type.lower(), 2)
        ptr = self._dll.arc_ze_alloc_usm(self._ctx, size_bytes, t_val)
        return ptr

    def dma_copy_async(self, dst_ptr: int, src_ptr: int, size_bytes: int) -> int:
        """Submit an asynchronous DMA memory copy on the GPU queue.

        Args:
            dst_ptr: Destination memory address.
            src_ptr: Source memory address.
            size_bytes: Byte count to copy.

        Returns:
            int: 0 on success, non-zero on error.
        """
        if not self.is_available or self._dll is None:
            return -1
        return self._dll.arc_ze_dma_copy_async(self._ctx, dst_ptr, src_ptr, size_bytes)

    def synchronize(self) -> int:
        """Block until pending hardware DMA operations complete.

        Returns:
            int: 0 on success.
        """
        if not self.is_available or self._dll is None:
            return -1
        return self._dll.arc_ze_synchronize(self._ctx)

    def free_usm(self, ptr: int) -> None:
        """Release a Unified Shared Memory buffer.

        Args:
            ptr: Pointer to free.
        """
        if self.is_available and self._dll and ptr:
            self._dll.arc_ze_free_usm(self._ctx, ptr)

    def destroy(self) -> None:
        """Release all driver resources and queues."""
        if self._ctx and self._dll:
            self._dll.arc_ze_destroy(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        """Ensure driver context is deallocated upon cleanup."""
        self.destroy()
