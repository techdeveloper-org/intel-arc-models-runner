"""Pure numerical routines for removing asymmetric zero-points from compressed weights.

Weight compression stores w = s * (q - z) with unsigned q and an unsigned zero-point z.
The Intel NPU compiler needs signed weights without the Subtract, i.e. w' = s' * q'.

Exact case
    If every z equals the symmetric value (8 for u4, 128 for u8), then q - z equals the
    two's complement reading of q XOR z. Flipping the sign bit of each element is a
    bijection and no value changes.

Lossy case
    Otherwise q - z spans more than the signed range and cannot be stored unchanged. Each
    group (the last axis) is requantised with the minimax scale

        s' = max(a_plus / (Qp + 1/2), a_minus / (Qn + 1/2))

    where a_plus = max(max w, 0), a_minus = max(-min w, 0), Qp = 2^(b-1) - 1 and
    Qn = 2^(b-1). With q' = clip(rint(w / s'), -Qn, Qp) the per-element error is at most
    max(s'/2, a_plus - Qp s', a_minus - Qn s'), evaluated with the stored f16 scale. The
    bound holds per element. It says nothing about the quality of the whole network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError

SYMMETRIC_ZERO_POINT = {"u4": 8, "u8": 128}
SIGN_FLIP_BYTE = {"u4": 0x88, "u8": 0x80}
ELEMENT_BITS = {"u4": 4, "u8": 8}
F16_MIN_NORMAL = float(np.finfo(np.float16).tiny)


class ZeroPointMathError(ArcRunnerError):
    """Raised when a zero-point conversion cannot be done without changing the model silently."""

    def __init__(self, message: str) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the violated precondition.
        """
        super().__init__(message=message, error_code="ZERO_POINT_MATH_ERROR", status_code=500)


@dataclass(frozen=True)
class RequantizeResult:
    """Outcome of one requantisation call.

    Attributes:
        q: Signed quantised values with the same shape as the input.
        scale: New float16 scales, one per group (last axis reduced to 1).
        max_abs_error: Largest absolute difference between dequantised input and output.
        max_rel_error: Largest per-group error relative to the group's largest magnitude.
    """

    q: np.ndarray
    scale: np.ndarray
    max_abs_error: float
    max_rel_error: float


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Split packed bytes into 4-bit values, low nibble first.

    Args:
        packed: uint8 array of packed pairs.

    Returns:
        uint8 array of length 2 * len(packed) with values in 0..15.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = packed >> 4
    return out


def pack_nibbles(values: np.ndarray) -> np.ndarray:
    """Pack 4-bit values into bytes, low nibble first.

    Args:
        values: Array of integers in 0..15 with an even element count.

    Returns:
        uint8 array of half the length.

    Raises:
        ZeroPointMathError: If the count is odd or a value does not fit in 4 bits.
    """
    values = np.asarray(values)
    if values.size % 2:
        raise ZeroPointMathError(f"Cannot pack an odd number of nibbles: {values.size}")
    if values.size and (int(values.min()) < 0 or int(values.max()) > 15):
        raise ZeroPointMathError("Nibble value outside 0..15")
    nibbles = values.astype(np.uint8)
    return (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)


def decode_unsigned(raw: bytes, element_type: str, count: int) -> np.ndarray:
    """Decode stored unsigned elements into one uint8 value per element.

    Args:
        raw: Bytes of one Const as stored in the weights binary.
        element_type: Either u4 or u8.
        count: Expected element count.

    Returns:
        uint8 array with ``count`` elements.

    Raises:
        ZeroPointMathError: If the type is unknown or the byte length does not match.
    """
    if element_type not in ELEMENT_BITS:
        raise ZeroPointMathError(f"Unsupported element type: {element_type}")
    expected = count // 2 if element_type == "u4" else count
    if element_type == "u4" and count % 2:
        raise ZeroPointMathError(f"u4 element count must be even: {count}")
    if len(raw) != expected:
        raise ZeroPointMathError(
            f"Byte length {len(raw)} does not match {count} {element_type} elements"
        )
    data = np.frombuffer(raw, dtype=np.uint8)
    return unpack_nibbles(data) if element_type == "u4" else data.copy()


def is_symmetric_zero_point(raw: np.ndarray, element_type: str) -> bool:
    """Tell whether every stored zero-point equals the symmetric value.

    Args:
        raw: Stored bytes of the zero-point Const (packed for u4).
        element_type: Either u4 or u8.

    Returns:
        True when removing the Subtract and flipping the sign bit is exact.

    Raises:
        ZeroPointMathError: If the element type is unknown.
    """
    if element_type not in SYMMETRIC_ZERO_POINT:
        raise ZeroPointMathError(f"Unsupported element type: {element_type}")
    raw = np.asarray(raw, dtype=np.uint8)
    values = unpack_nibbles(raw) if element_type == "u4" else raw
    return bool(np.all(values == SYMMETRIC_ZERO_POINT[element_type]))


def unsigned_to_signed_bytes(raw: bytes, element_type: str) -> bytes:
    """Reinterpret unsigned storage as q - symmetric_zero_point in signed storage.

    Args:
        raw: Stored bytes of the weight Const.
        element_type: Either u4 or u8.

    Returns:
        Bytes with the sign bit of every element flipped.

    Raises:
        ZeroPointMathError: If the element type is unknown.
    """
    if element_type not in SIGN_FLIP_BYTE:
        raise ZeroPointMathError(f"Unsupported element type: {element_type}")
    flipped = np.frombuffer(raw, dtype=np.uint8) ^ np.uint8(SIGN_FLIP_BYTE[element_type])
    return flipped.tobytes()


def encode_signed(q: np.ndarray, bits: int) -> bytes:
    """Encode signed values as two's complement storage.

    Args:
        q: Signed integer array.
        bits: 4 for packed nibbles (low first) or 8 for plain bytes.

    Returns:
        Bytes ready to be written into the weights binary.

    Raises:
        ZeroPointMathError: If bits is unsupported or a value is out of range.
    """
    if bits not in (4, 8):
        raise ZeroPointMathError(f"Unsupported width: {bits}")
    q = np.asarray(q)
    low, high = (-8, 7) if bits == 4 else (-128, 127)
    if q.size and (int(q.min()) < low or int(q.max()) > high):
        raise ZeroPointMathError(f"Signed value outside {low}..{high}")
    if bits == 8:
        return q.astype(np.int8).tobytes()
    return pack_nibbles((q.astype(np.int16) & 0x0F).ravel()).tobytes()


def requantize_symmetric(
    q: np.ndarray, zero_point: np.ndarray, scale: np.ndarray, bits: int
) -> RequantizeResult:
    """Requantise asymmetric groups to signed symmetric values with a new f16 scale.

    Args:
        q: Unsigned quantised values; the last axis is the group axis.
        zero_point: Unsigned zero-points, shape equal to ``scale`` or a single element.
        scale: float16 scales with shape ``q.shape[:-1] + (1,)``.
        bits: 4 or 8.

    Returns:
        The new signed values, the new float16 scales and the measured errors.

    Raises:
        ZeroPointMathError: On an unsupported width, a shape mismatch, a non-finite or
            negative scale, or a new scale that does not fit in float16.
    """
    if bits not in (4, 8):
        raise ZeroPointMathError(f"Unsupported width: {bits}")
    q = np.asarray(q)
    zero_point = np.asarray(zero_point)
    scale = np.asarray(scale, dtype=np.float16)
    if q.ndim < 1 or scale.shape != q.shape[:-1] + (1,):
        raise ZeroPointMathError(
            f"Scale shape {scale.shape} does not match weight shape {q.shape}"
        )
    if zero_point.shape != scale.shape and zero_point.size != 1:
        raise ZeroPointMathError(
            f"Zero-point shape {zero_point.shape} does not match scale shape {scale.shape}"
        )
    if not np.all(np.isfinite(scale)) or np.any(scale < 0):
        raise ZeroPointMathError("Scale contains a non-finite or negative value")
    qp, qn = (7, 8) if bits == 4 else (127, 128)
    weights = scale.astype(np.float64) * (q.astype(np.int64) - zero_point.astype(np.int64))
    a_plus = np.maximum(weights.max(axis=-1, keepdims=True), 0.0)
    a_minus = np.maximum(-weights.min(axis=-1, keepdims=True), 0.0)
    wanted = np.maximum(a_plus / (qp + 0.5), a_minus / (qn + 0.5))
    wanted = np.maximum(wanted, F16_MIN_NORMAL)
    with np.errstate(over="ignore"):
        new_scale = wanted.astype(np.float16)
    if not np.all(np.isfinite(new_scale)):
        raise ZeroPointMathError("Requantised scale does not fit in float16")
    scale64 = new_scale.astype(np.float64)
    new_q = np.clip(np.rint(weights / scale64), -qn, qp).astype(np.int8)
    error = np.abs(weights - new_q.astype(np.float64) * scale64)
    group_max = np.abs(weights).max(axis=-1, keepdims=True)
    relative = np.divide(error, group_max, out=np.zeros_like(error), where=group_max > 0)
    return RequantizeResult(
        q=new_q,
        scale=new_scale,
        max_abs_error=float(error.max()) if error.size else 0.0,
        max_rel_error=float(relative.max()) if relative.size else 0.0,
    )
