"""Python foreign function interface bridging native C acceleration library.

Wraps libarc_native_sys.dll for high-performance zero-copy memory mapping,
working set physical RAM pinning, and Level Zero USM buffer allocations.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from arc_runner.core.exceptions import NativeBoundsError, NativeLayerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.native.bridge")

ARC_MAX_NAME_LEN = 128
ARC_MAX_TENSORS_PER_LAYER = 64
ARC_MAX_LAYERS = 256

_U64_MAX = (1 << 64) - 1
_MAX_NAME_BYTES = ARC_MAX_NAME_LEN - 1
_MAX_DIMS = 4
_EXPECTED_ABI_VERSION = 2
_ENV_DISABLE_NATIVE = "ARC_NATIVE_DISABLE"
_BUILD_INFO_SUFFIX = ".build.json"

_ARC_ERR_RANGE = -2
_ARC_ERR_TOO_MANY_TENSORS = -3
_ARC_ERR_BAD_EXTENT = -4
_BOUNDS_STATUSES = (_ARC_ERR_RANGE, _ARC_ERR_TOO_MANY_TENSORS, _ARC_ERR_BAD_EXTENT)


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


if ctypes.sizeof(ArcTensorDescriptor) != 184 or ctypes.sizeof(ArcLayerDescriptor) != 11792:
    raise ImportError("ctypes descriptor layout does not match the native ABI (184 / 11792 bytes)")


def _validate_u64(value: Any, field: str) -> int:
    """Validate that a value is an integer representable as an unsigned 64-bit field.

    Args:
        value: Candidate value; bool and non-int types are rejected.
        field: Field name used in the error message.

    Returns:
        int: The validated value.

    Raises:
        NativeBoundsError: If value is not an int in [0, 2**64 - 1].
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeBoundsError(f"{field} must be an int, got {type(value).__name__}")
    if value < 0 or value > _U64_MAX:
        raise NativeBoundsError(f"{field}={value} is outside the unsigned 64-bit range")
    return value


def _sidecar_matches(lib_path: Path) -> bool:
    """Verify the library bytes against the sha256 recorded in its build sidecar, if present.

    Args:
        lib_path: Path to the shared library.

    Returns:
        bool: False only when a sidecar exists and its hash differs or is unreadable.
    """
    sidecar = lib_path.with_name(lib_path.name + _BUILD_INFO_SUFFIX)
    if not sidecar.exists():
        return True
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("sha256", "")
    except (OSError, ValueError):
        return False
    return recorded == hashlib.sha256(lib_path.read_bytes()).hexdigest()


def _load_native_library() -> Optional[ctypes.CDLL]:
    """Locate and load libarc_native_sys, refusing mismatched or tampered builds.

    Returns:
        Optional[ctypes.CDLL]: Loaded library handle, or None if disabled, missing,
        built with another ABI version, or failing its sidecar hash check.
    """
    if os.environ.get(_ENV_DISABLE_NATIVE, "") not in ("", "0"):
        logger.info("Native layer disabled by environment", extra={"reason": _ENV_DISABLE_NATIVE})
        return None

    native_dir = Path(__file__).resolve().parent
    lib_name = "libarc_native_sys.dll" if sys.platform == "win32" else "libarc_native_sys.so"
    lib_path = native_dir / lib_name

    if not lib_path.exists():
        return None

    if not _sidecar_matches(lib_path):
        logger.warning("Native library hash mismatch", extra={"reason": "sidecar_sha256", "path": str(lib_path)})
        return None

    try:
        dll = ctypes.CDLL(str(lib_path))
        dll.arc_native_abi_version.argtypes = []
        dll.arc_native_abi_version.restype = ctypes.c_uint32
        abi_version = int(dll.arc_native_abi_version())
        if abi_version != _EXPECTED_ABI_VERSION:
            logger.warning(
                "Native library ABI mismatch; rebuild with build_native.py",
                extra={"reason": "abi_version", "path": str(lib_path), "found": abi_version},
            )
            return None
        _configure_signatures(dll)
        return dll
    except (OSError, AttributeError) as exc:
        logger.warning("Failed loading native library", extra={"reason": str(exc), "path": str(lib_path)})
        return None


def _configure_signatures(dll: ctypes.CDLL) -> None:
    """Configure parameter and return types for exported native functions.

    Args:
        dll: Loaded shared library instance.
    """
    dll.arc_mmap_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
    dll.arc_mmap_open.restype = ctypes.c_void_p

    dll.arc_mmap_seal.argtypes = [ctypes.c_void_p]
    dll.arc_mmap_seal.restype = ctypes.c_int

    dll.arc_mmap_last_status.argtypes = [ctypes.c_void_p]
    dll.arc_mmap_last_status.restype = ctypes.c_int32

    dll.arc_mmap_last_os_error.argtypes = [ctypes.c_void_p]
    dll.arc_mmap_last_os_error.restype = ctypes.c_uint32

    dll.arc_mmap_locked_bytes.argtypes = [ctypes.c_void_p]
    dll.arc_mmap_locked_bytes.restype = ctypes.c_uint64

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


def _field(record: Any, key: str, attr: str, default: Any) -> Any:
    """Read a tensor field from a dict record or an attribute-style record.

    Args:
        record: Tensor metadata as a dict or object.
        key: Dict key to look up.
        attr: Attribute name to look up on objects.
        default: Value used when the field is absent.

    Returns:
        Any: The field value or the default.
    """
    return record.get(key, default) if isinstance(record, dict) else getattr(record, attr, default)


def _fill_tensor_descriptor(c_tensor: ArcTensorDescriptor, record: Any) -> int:
    """Populate one C tensor descriptor from a metadata record with strict validation.

    Args:
        c_tensor: Destination C descriptor.
        record: Tensor metadata as a dict or object.

    Returns:
        int: The validated payload size in bytes.

    Raises:
        NativeBoundsError: If the name, dimensions, or any integer is out of range.
    """
    raw_name = _field(record, "name", "name", str(record))
    name_bytes = str(raw_name).encode("utf-8")
    if len(name_bytes) > _MAX_NAME_BYTES:
        raise NativeBoundsError(f"tensor name exceeds {_MAX_NAME_BYTES} bytes: {raw_name[:32]!r}...")
    dims = list(_field(record, "dimensions", "dimensions", [1]))
    if len(dims) > _MAX_DIMS:
        raise NativeBoundsError(f"tensor {raw_name!r} has {len(dims)} dimensions, maximum is {_MAX_DIMS}")

    c_tensor.name = name_bytes
    c_tensor.tensor_type = _validate_u64(int(_field(record, "tensor_type", "tensor_type", 0)), "tensor_type")
    c_tensor.n_dims = len(dims)
    for index, dim in enumerate(dims):
        c_tensor.dims[index] = _validate_u64(dim, "dimension")
    c_tensor.file_offset = _validate_u64(_field(record, "offset", "offset", 0), "offset")
    size_bytes = _validate_u64(_field(record, "size_bytes", "size_bytes", 0), "size_bytes")
    c_tensor.size_bytes = size_bytes
    return size_bytes


def _build_c_layer(layer_idx: int, tensors: List[Any]) -> ArcLayerDescriptor:
    """Build a validated C layer descriptor without truncating any input.

    Args:
        layer_idx: Zero-based layer index.
        tensors: Tensor metadata records.

    Returns:
        ArcLayerDescriptor: Descriptor ready for arc_mmap_register_layer.

    Raises:
        NativeBoundsError: If the layer index or tensor count is out of range, or any tensor is invalid.
    """
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or not 0 <= layer_idx < ARC_MAX_LAYERS:
        raise NativeBoundsError(f"layer_idx must be an int in [0, {ARC_MAX_LAYERS}), got {layer_idx!r}")
    if len(tensors) > ARC_MAX_TENSORS_PER_LAYER:
        raise NativeBoundsError(f"layer {layer_idx} has {len(tensors)} tensors, maximum is {ARC_MAX_TENSORS_PER_LAYER}")

    c_layer = ArcLayerDescriptor()
    c_layer.layer_idx = layer_idx
    c_layer.tensor_count = len(tensors)
    total_bytes = 0
    for index, record in enumerate(tensors):
        total_bytes += _fill_tensor_descriptor(c_layer.tensors[index], record)
    c_layer.total_layer_bytes = _validate_u64(total_bytes, "total_layer_bytes")
    return c_layer


class NativeMemoryEngine:
    """Bounds-checked memory-mapped model streaming manager over the native library."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the native memory engine.

        Args:
            file_path: Optional filesystem path to open immediately.
        """
        self._dll: Optional[ctypes.CDLL] = _load_native_library()
        self._ctx: Optional[int] = None
        self._sealed: bool = False
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

    def last_error(self) -> Tuple[int, int]:
        """Return the last native status and operating-system error codes.

        Returns:
            Tuple[int, int]: (ArcStatus, OS error); (0, 0) when no context is open.
        """
        if not self.is_open or self._dll is None:
            return 0, 0
        return int(self._dll.arc_mmap_last_status(self._ctx)), int(self._dll.arc_mmap_last_os_error(self._ctx))

    def _raise_for_status(self, status: int, action: str) -> None:
        """Raise the exception matching a negative native status code.

        Args:
            status: ArcStatus value returned by the native call.
            action: Short description of the attempted operation.

        Raises:
            NativeBoundsError: For range, extent, and count violations.
            NativeLayerError: For every other failure.
        """
        _, os_error = self.last_error()
        message = f"{action} failed with native status {status} (os error {os_error})"
        if status in _BOUNDS_STATUSES:
            raise NativeBoundsError(message, status, os_error)
        raise NativeLayerError(message, status, os_error)

    def open_model(self, file_path: Union[str, Path]) -> bool:
        """Map a model container file into system memory.

        Args:
            file_path: Target filesystem path to container.

        Returns:
            bool: True if mapped successfully, False if the file is missing, empty, or unmappable.
        """
        if not self.is_available or self._dll is None:
            return False

        if self.is_open:
            self.close()

        resolved = Path(file_path).resolve()
        if not resolved.exists():
            return False

        status = ctypes.c_int32(0)
        raw_ctx = self._dll.arc_mmap_open(str(resolved).encode("utf-8"), ctypes.byref(status))
        if not raw_ctx:
            logger.warning("Native open failed", extra={"path": str(resolved), "status": status.value})
            return False

        self._ctx = raw_ctx
        self._sealed = False
        self._file_path = resolved
        return True

    def register_layer(self, layer_idx: int, tensors: List[Any]) -> bool:
        """Validate and register tensor descriptors for a layer in the native lookup table.

        Args:
            layer_idx: Layer index in [0, 256).
            tensors: Tensor metadata records (at most 64 per layer).

        Returns:
            bool: True on successful registration, False if no file is open.

        Raises:
            NativeBoundsError: If counts, names, dimensions, integers, or file extents are out of bounds.
            NativeLayerError: If the table is already sealed or the native call fails otherwise.
        """
        if not self.is_open or self._dll is None:
            return False

        c_layer = _build_c_layer(layer_idx, tensors)
        status = int(self._dll.arc_mmap_register_layer(self._ctx, ctypes.byref(c_layer)))
        if status != 0:
            self._raise_for_status(status, f"register_layer({layer_idx})")
        return True

    def seal(self) -> None:
        """Freeze the native layer table; further registrations raise NativeLayerError.

        Raises:
            NativeLayerError: If the native seal call fails.
        """
        if not self.is_open or self._dll is None or self._sealed:
            return
        status = int(self._dll.arc_mmap_seal(self._ctx))
        if status != 0:
            self._raise_for_status(status, "seal")
        self._sealed = True

    def get_tensor_buffer(
        self,
        layer_idx: int,
        tensor_name: str,
        dims: Optional[Tuple[int, ...]] = None,
        dtype: Any = np.float32,
    ) -> Optional[np.ndarray]:
        """Retrieve a zero-copy NumPy array view wrapped around the mapped memory.

        The first call seals the layer table so later lookups need no locking.

        Args:
            layer_idx: Layer index identifier.
            tensor_name: Short name of tensor.
            dims: Optional dimension tuple to reshape.
            dtype: Target NumPy datatype.

        Returns:
            Optional[np.ndarray]: Zero-copy array view, or None if the tensor is missing,
            empty, or its byte size is not a multiple of the dtype size.
        """
        if not self.is_open or self._dll is None:
            return None

        self.seal()
        out_size = ctypes.c_uint64(0)
        ptr = self._dll.arc_mmap_get_tensor_ptr(
            self._ctx,
            layer_idx,
            tensor_name.encode("utf-8"),
            ctypes.byref(out_size),
        )

        if not ptr or out_size.value == 0:
            return None

        itemsize = np.dtype(dtype).itemsize
        if out_size.value % itemsize != 0:
            logger.warning(
                "Tensor byte size is not a multiple of the dtype size",
                extra={"tensor": tensor_name, "size_bytes": out_size.value, "itemsize": itemsize},
            )
            return None

        raw_array = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))),
            shape=(out_size.value // itemsize,),
        )

        if dims:
            try:
                return raw_array.reshape(dims)
            except ValueError:
                return raw_array

        return raw_array

    def lock_range(self, offset: int, size: int) -> bool:
        """Best-effort pin of a mapped range into physical RAM.

        Args:
            offset: Byte offset into the file.
            size: Number of bytes to pin; must be positive.

        Returns:
            bool: True if pinned; False if no file is open or the OS refused (quota, privileges).

        Raises:
            NativeBoundsError: If offset or size is not an unsigned 64-bit integer.
        """
        if not self.is_open or self._dll is None:
            return False
        status = int(self._dll.arc_mmap_lock_range(self._ctx, _validate_u64(offset, "offset"), _validate_u64(size, "size")))
        if status != 0:
            _, os_error = self.last_error()
            logger.info("Memory lock refused", extra={"status": status, "os_error": os_error})
        return status == 0

    def unlock_range(self, offset: int, size: int) -> bool:
        """Unpin a previously locked mapped range.

        Args:
            offset: Byte offset into the file.
            size: Number of bytes to unpin; must be positive.

        Returns:
            bool: True on success, False if no file is open or the OS refused.

        Raises:
            NativeBoundsError: If offset or size is not an unsigned 64-bit integer.
        """
        if not self.is_open or self._dll is None:
            return False
        status = int(self._dll.arc_mmap_unlock_range(self._ctx, _validate_u64(offset, "offset"), _validate_u64(size, "size")))
        return status == 0

    def close(self) -> None:
        """Unmap views and release native OS handles; safe to call repeatedly."""
        if self._ctx and self._dll:
            self._dll.arc_mmap_close(self._ctx)
        self._ctx = None
        self._sealed = False


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
