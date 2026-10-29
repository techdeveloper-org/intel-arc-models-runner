"""Python foreign function interface bridging native C acceleration library.

Wraps libarc_native_sys.dll for high-performance zero-copy memory mapping,
working set physical RAM pinning, and Level Zero USM buffer allocations.
"""

import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from arc_runner.core.exceptions import NativeBoundsError, NativeLayerError
from arc_runner.core.logging import get_logger
from arc_runner.quant.models import GGMLType

logger = get_logger("arc_runner.native.bridge")

ARC_MAX_NAME_LEN = 128
ARC_MAX_TENSORS_PER_LAYER = 64
ARC_MAX_LAYERS = 256

_U64_MAX = (1 << 64) - 1
_MAX_NAME_BYTES = ARC_MAX_NAME_LEN - 1
_MAX_DIMS = 4
_EXPECTED_ABI_VERSION = 3
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


_U32_MAX = (1 << 32) - 1
_MISSING = object()


def _validate_u32(value: Any, field: str) -> int:
    """Validate that a value is an integer representable as an unsigned 32-bit field.

    Args:
        value: Candidate value; bool and non-int types are rejected.
        field: Field name used in the error message.

    Returns:
        int: The validated value.

    Raises:
        NativeBoundsError: If value is not an int in [0, 2**32 - 1].
    """
    value = _validate_u64(value, field)
    if value > _U32_MAX:
        raise NativeBoundsError(f"{field}={value} is outside the unsigned 32-bit range")
    return value


def _locked(method: Any) -> Any:
    """Run a NativeMemoryEngine method under the engine's re-entrant lock.

    Pointer lookup, view registration and close must not interleave: close() could otherwise unmap
    between the C lookup and add_view.

    Args:
        method: Instance method to wrap.

    Returns:
        Any: The wrapped method.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


_RAW_DTYPES: Dict[int, Any] = {
    int(GGMLType.F32): np.dtype(np.float32),
    int(GGMLType.F16): np.dtype(np.float16),
    int(GGMLType.BF16): np.dtype(np.uint16),
    int(GGMLType.F64): np.dtype(np.float64),
    int(GGMLType.I8): np.dtype(np.int8),
    int(GGMLType.I16): np.dtype(np.int16),
    int(GGMLType.I32): np.dtype(np.int32),
    int(GGMLType.I64): np.dtype(np.int64),
}


def dtype_for_tensor_type(tensor_type: int) -> np.dtype:
    """Return the NumPy dtype used to view a GGML tensor payload without conversion.

    Plain numeric types map to their own dtype; bfloat16 is exposed as its raw uint16 bit
    pattern (NumPy has no bfloat16); every block-quantized type is exposed as raw uint8 bytes.

    Args:
        tensor_type: GGML tensor type id.

    Returns:
        np.dtype: dtype of the zero-copy view.
    """
    return _RAW_DTYPES.get(int(tensor_type), np.dtype(np.uint8))


def view_shape_for_dims(dims: Any, element_count: int) -> Optional[Tuple[int, ...]]:
    """Return the row-major NumPy shape for GGUF dimensions, or None if they do not fit.

    GGUF stores ne[0] as the fastest-varying dimension, so the C-order shape is the reversed list.

    Args:
        dims: Dimension list from the GGUF tensor record.
        element_count: Number of elements actually present in the payload.

    Returns:
        Optional[Tuple[int, ...]]: Reversed dimensions when their product equals element_count, else None.
    """
    if not dims:
        return None
    shape = tuple(int(d) for d in reversed(list(dims)))
    return shape if int(np.prod(shape, dtype=object)) == element_count else None


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

    dll.arc_ze_init_ex.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    dll.arc_ze_init_ex.restype = ctypes.c_void_p

    for accessor, result_type in (
        ("arc_ze_is_dead", ctypes.c_int),
        ("arc_ze_last_status", ctypes.c_int32),
        ("arc_ze_last_result", ctypes.c_uint32),
        ("arc_ze_device_type", ctypes.c_uint32),
        ("arc_ze_alloc_count", ctypes.c_uint32),
        ("arc_ze_sync_timeout_ms", ctypes.c_uint32),
    ):
        function = getattr(dll, accessor)
        function.argtypes = [ctypes.c_void_p]
        function.restype = result_type

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
    c_tensor.tensor_type = _validate_u32(_field(record, "tensor_type", "tensor_type", 0), "tensor_type")
    c_tensor.n_dims = len(dims)
    for index, dim in enumerate(dims):
        c_tensor.dims[index] = _validate_u64(dim, "dimension")
    offset = _field(record, "offset", "offset", _MISSING)
    raw_size = _field(record, "size_bytes", "size_bytes", _MISSING)
    if offset is _MISSING or raw_size is _MISSING:
        raise NativeBoundsError(f"tensor {raw_name!r} needs both 'offset' and 'size_bytes'")
    c_tensor.file_offset = _validate_u64(offset, "offset")
    size_bytes = _validate_u64(raw_size, "size_bytes")
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


class _MappingGuard:
    """Keeps a native mapping alive until every NumPy view created from it is garbage collected.

    The engine asks for the mapping to be closed; the unmap happens immediately if no view
    exists, otherwise when the last view dies. This prevents use-after-unmap of zero-copy views.
    """

    def __init__(self, dll: ctypes.CDLL, ctx: int) -> None:
        """Create a guard for an open native mapping.

        Args:
            dll: Loaded native library.
            ctx: Native context pointer returned by arc_mmap_open.
        """
        self._dll = dll
        self._ctx: Optional[int] = ctx
        self._views = 0
        self._close_requested = False
        self._lock = threading.Lock()

    @property
    def live_views(self) -> int:
        """Return the number of NumPy views still referencing the mapping."""
        with self._lock:
            return self._views

    def add_view(self, owner: Any) -> None:
        """Register a buffer owner; the mapping stays valid until the owner is collected.

        Args:
            owner: ctypes array object that NumPy keeps as the base of the view.
        """
        with self._lock:
            self._views += 1
        owner._arc_mapping_guard = self
        weakref.finalize(owner, self._drop_view)

    def _drop_view(self) -> None:
        """Release one view and close the mapping if it was already requested."""
        with self._lock:
            self._views -= 1
            should_close = self._close_requested and self._views == 0
        if should_close:
            self._close_native()

    def request_close(self) -> None:
        """Close the mapping now if no views exist, otherwise when the last view is released."""
        with self._lock:
            self._close_requested = True
            should_close = self._views == 0
        if should_close:
            self._close_native()

    def _close_native(self) -> None:
        """Unmap the file exactly once."""
        with self._lock:
            ctx, self._ctx = self._ctx, None
        if ctx:
            self._dll.arc_mmap_close(ctx)


class NativeMemoryEngine:
    """Bounds-checked memory-mapped model streaming manager over the native library."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the native memory engine.

        Args:
            file_path: Optional filesystem path to open immediately.
        """
        self._lock = threading.RLock()
        self._dll: Optional[ctypes.CDLL] = _load_native_library()
        self._ctx: Optional[int] = None
        self._guard: Optional[_MappingGuard] = None
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

    @_locked
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
        self._guard = _MappingGuard(self._dll, raw_ctx)
        self._sealed = False
        self._file_path = resolved
        return True

    @_locked
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

    @_locked
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

    @_locked
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
            dims: Optional NumPy shape (row-major, so GGUF dimensions must be reversed by the caller).
            dtype: Target NumPy datatype.

        Returns:
            Optional[np.ndarray]: Zero-copy array view, or None if the tensor is missing,
            empty, or its byte size is not a multiple of the dtype size.

        Raises:
            NativeBoundsError: If layer_idx is not an int in [0, ARC_MAX_LAYERS).
        """
        if not self.is_open or self._dll is None:
            return None

        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or not 0 <= layer_idx < ARC_MAX_LAYERS:
            raise NativeBoundsError(f"layer_idx must be an int in [0, {ARC_MAX_LAYERS}), got {layer_idx!r}")
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

        address = ctypes.cast(ptr, ctypes.c_void_p).value
        if not address or self._guard is None:
            return None
        owner = (ctypes.c_uint8 * out_size.value).from_address(address)
        self._guard.add_view(owner)
        raw_array = np.frombuffer(owner, dtype=dtype)
        raw_array.flags.writeable = False

        if dims:
            try:
                return raw_array.reshape(dims)
            except ValueError:
                return raw_array

        return raw_array

    @_locked
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

    @_locked
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

    @_locked
    def close(self) -> None:
        """Release the mapping; safe to call repeatedly.

        If NumPy views obtained from get_tensor_buffer are still alive, the actual unmap is
        deferred until the last of them is garbage collected, so no view ever points at unmapped memory.
        """
        guard, self._guard = self._guard, None
        self._ctx = None
        self._sealed = False
        if guard is not None:
            guard.request_close()

    def __del__(self) -> None:
        """Release the mapping if the engine is garbage collected without an explicit close."""
        try:
            self.close()
        except Exception:
            return


_USM_TYPES = {"host": 0, "device": 1, "shared": 2}
_MAX_USM_REQUEST_BYTES = (1 << 63) - 1


class NativeLevelZeroBridge:
    """Level Zero Unified Shared Memory and DMA coordinator (GPU only, bounded waits)."""

    def __init__(self) -> None:
        """Initialize the Level Zero driver interface context and pick the first GPU device."""
        self._dll: Optional[ctypes.CDLL] = _load_native_library()
        self._ctx: Optional[int] = None
        self.init_status: int = 0
        if self._dll:
            from arc_runner.devices.sysman import zes_preinit

            zes_preinit()
            status = ctypes.c_int32(0)
            raw_ctx = self._dll.arc_ze_init_ex(ctypes.byref(status))
            self.init_status = int(status.value)
            if raw_ctx:
                self._ctx = raw_ctx

    @property
    def is_available(self) -> bool:
        """Check whether a GPU Level Zero context is open.

        Returns:
            bool: True if the loader initialized and a GPU context exists.
        """
        return self._ctx is not None and self._dll is not None

    @property
    def is_dead(self) -> bool:
        """Report whether a timeout or device loss disabled the context.

        Returns:
            bool: True once the context must be discarded and recreated.
        """
        return bool(self.is_available and self._dll.arc_ze_is_dead(self._ctx))

    @property
    def device_type(self) -> int:
        """Return the Level Zero device type of the selected device (1 is GPU).

        Returns:
            int: ze_device_type_t value, or 0 if unavailable.
        """
        return int(self._dll.arc_ze_device_type(self._ctx)) if self.is_available else 0

    def last_status(self) -> int:
        """Return the last native status code (0 is success, negative is an error).

        Returns:
            int: ArcZeStatus value; ARC_ZE_ERR_NOT_INITIALIZED (-10) when unavailable.
        """
        return int(self._dll.arc_ze_last_status(self._ctx)) if self.is_available else -10

    def allocate_usm(self, size_bytes: int, usm_type: str = "host") -> Optional[int]:
        """Allocate a Unified Shared Memory buffer.

        Args:
            size_bytes: Size in bytes; must be positive and below 2**63.
            usm_type: 'host' (default; best for streamed weights on a shared-memory iGPU), 'device', or 'shared'.

        Returns:
            Optional[int]: Address of the buffer, or None if unavailable, out of range, or refused by the driver.

        Raises:
            ValueError: If usm_type is not one of host, device, shared.
        """
        residency = _USM_TYPES.get(usm_type.lower())
        if residency is None:
            raise ValueError(f"usm_type must be one of {sorted(_USM_TYPES)}, got {usm_type!r}")
        if not self.is_available or self._dll is None:
            return None
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or not 0 < size_bytes <= _MAX_USM_REQUEST_BYTES:
            return None
        return self._dll.arc_ze_alloc_usm(self._ctx, size_bytes, residency)

    def dma_copy_async(self, dst_ptr: int, src_ptr: int, size_bytes: int) -> int:
        """Record a copy on the GPU command list.

        Args:
            dst_ptr: Destination memory address.
            src_ptr: Source memory address.
            size_bytes: Byte count to copy.

        Returns:
            int: 0 on success, 1 if the host memcpy fallback was used, negative on error.
        """
        if not self.is_available or self._dll is None:
            return -10
        for value in (dst_ptr, src_ptr, size_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _U64_MAX:
                return -1
        if size_bytes <= 0 or dst_ptr == 0 or src_ptr == 0:
            return -1
        return int(self._dll.arc_ze_dma_copy_async(self._ctx, dst_ptr, src_ptr, size_bytes))

    def synchronize(self) -> int:
        """Execute recorded commands and wait for completion within the bounded timeout.

        Returns:
            int: 0 on success, negative on error; -9 means the wait timed out and the context is now dead.
        """
        if not self.is_available or self._dll is None:
            return -10
        return int(self._dll.arc_ze_synchronize(self._ctx))

    def free_usm(self, ptr: int) -> None:
        """Free a buffer returned by allocate_usm; unknown pointers are ignored.

        Args:
            ptr: Buffer address.
        """
        if self.is_available and self._dll is not None and ptr:
            self._dll.arc_ze_free_usm(self._ctx, ptr)

    def destroy(self) -> None:
        """Release the Level Zero context and all outstanding allocations; safe to call repeatedly."""
        if self._ctx and self._dll:
            self._dll.arc_ze_destroy(self._ctx)
        self._ctx = None

    def __del__(self) -> None:
        """Release the context if the bridge is garbage collected without destroy()."""
        try:
            self.destroy()
        except Exception:
            return
