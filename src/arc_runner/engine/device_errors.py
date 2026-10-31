"""Device error classification for Intel Arc GPU and NPU fallback decision gates.

Provides a single pure function that maps an exception (or its cause chain) to one of
four categories.  Callers use the category to decide whether a CPU or GPU fallback is
safe or whether the error is unrecoverable and must be re-raised.
"""


_DEVICE_LOST_MARKERS = (
    "CL_OUT_OF_RESOURCES",
    "DEVICE_LOST",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_HOST_MEMORY",
)

_DRIVER_ERROR_MARKERS = (
    "is not registered",
    "not supported",
    "DeviceNotFoundError",
)


def classify_device_error(exc: BaseException) -> str:
    """Classify an exception raised during OpenVINO pipeline construction.

    Walks the full cause chain (__cause__ then __context__) looking for the
    highest-priority category.  DEVICE_LOST beats OOM: if both appear in the
    chain the function returns "DEVICE_LOST".

    Categories:
        - ``"DEVICE_LOST"`` -- OpenCL context loss or out-of-device-memory
          (``CL_OUT_OF_RESOURCES``, ``DEVICE_LOST``, ``CL_MEM_OBJECT_ALLOCATION_FAILURE``,
          ``CL_OUT_OF_HOST_MEMORY``).
        - ``"OOM"`` -- host-side out-of-memory (``MemoryError``).
        - ``"DRIVER_ERROR"`` -- device not registered or operation not supported.
        - ``"UNKNOWN"`` -- everything else.

    Args:
        exc: The exception to classify.  Must not be None.

    Returns:
        One of ``"DEVICE_LOST"``, ``"OOM"``, ``"DRIVER_ERROR"``, or ``"UNKNOWN"``.
    """
    best = "UNKNOWN"
    node: BaseException | None = exc

    while node is not None:
        category = _classify_single(node)
        if category == "DEVICE_LOST":
            return "DEVICE_LOST"
        if category == "OOM" and best != "DEVICE_LOST":
            best = "OOM"
        elif category == "DRIVER_ERROR" and best not in ("DEVICE_LOST", "OOM"):
            best = "DRIVER_ERROR"

        cause = node.__cause__
        context = node.__context__
        node = cause if cause is not None else (context if context is not exc else None)

    return best


def _classify_single(exc: BaseException) -> str:
    """Return the category for a single exception node without walking the chain.

    Args:
        exc: A single exception node.

    Returns:
        One of the four category strings.
    """
    if isinstance(exc, MemoryError):
        return "OOM"

    text = str(exc)

    for marker in _DEVICE_LOST_MARKERS:
        if marker in text:
            return "DEVICE_LOST"

    for marker in _DRIVER_ERROR_MARKERS:
        if marker in text:
            return "DRIVER_ERROR"

    return "UNKNOWN"
