"""Silicon-native mathematical transformation engine for Intel Arc GPU XMX and NPU.

Dequantizes non-native GGUF k-quants into single-precision floating point and repacks
weights into Intel Xe Matrix Extensions (XMX) and NPU native Symmetric INT4 Group 128
and INT8 formats with 32-byte cacheline zero-padding alignment and SQNR validation.
"""

from typing import Tuple
import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.quant.math_transforms")


class QuantizationMathError(ArcRunnerError):
    """Base exception for quantization arithmetic and tensor layout failures."""

    def __init__(self, message: str, error_code: str = "QUANT_MATH_ERROR", status_code: int = 422) -> None:
        """Initialize the quantization math error.

        Args:
            message: Descriptive failure explanation.
            error_code: Machine-readable error code.
            status_code: Corresponding HTTP status code.
        """
        super().__init__(message, error_code, status_code)


class UnsupportedGroupSizeError(QuantizationMathError):
    """Raised when group size is not a positive multiple of 32 bytes."""

    def __init__(self, group_size: int) -> None:
        """Initialize the unsupported group size error.

        Args:
            group_size: Invalid group size specified.
        """
        super().__init__(
            f"Group size {group_size} is invalid; must be a positive multiple of 32.",
            "UNSUPPORTED_GROUP_SIZE",
            422,
        )
        self.group_size = group_size


class DequantizationError(QuantizationMathError):
    """Raised when raw byte buffer length or structure violates GGML quantization blocks."""

    def __init__(self, message: str) -> None:
        """Initialize dequantization error.

        Args:
            message: Structural failure description.
        """
        super().__init__(message, "DEQUANTIZATION_ERROR", 422)


class SQNRThresholdViolationError(QuantizationMathError):
    """Raised when repacked tensor SQNR falls below minimum hardware fidelity threshold."""

    def __init__(self, sqnr_db: float, min_threshold_db: float) -> None:
        """Initialize SQNR violation error.

        Args:
            sqnr_db: Measured Signal-to-Quantization-Noise Ratio in decibels.
            min_threshold_db: Minimum required threshold in decibels.
        """
        super().__init__(
            f"Repacked tensor SQNR {sqnr_db:.2f} dB violated minimum fidelity threshold {min_threshold_db:.2f} dB.",
            "SQNR_THRESHOLD_VIOLATION",
            422,
        )
        self.sqnr_db = sqnr_db
        self.min_threshold_db = min_threshold_db


class CachelineAlignmentError(QuantizationMathError):
    """Raised when memory buffer pointer or stride violates 32-byte cacheline alignment."""

    def __init__(self, address_or_len: int) -> None:
        """Initialize cacheline alignment error.

        Args:
            address_or_len: Memory address or length failing 32-byte alignment.
        """
        super().__init__(
            f"Memory buffer offset or length {address_or_len} is not 32-byte cacheline aligned.",
            "CACHELINE_ALIGNMENT_ERROR",
            422,
        )


class QuantizationMathEngine:
    """Mathematical transformation engine for Intel Arc GPU XMX and NPU."""

    INT4_MIN: int = -8
    INT4_MAX: int = 7
    INT8_MIN: int = -128
    INT8_MAX: int = 127
    CACHE_LINE_BYTES: int = 32
    DEFAULT_GROUP_SIZE: int = 128
    MIN_INT4_SQNR_DB: float = 20.0
    MIN_INT8_SQNR_DB: float = 34.0
    MIN_SQNR_DB: float = 20.0

    @classmethod
    def dequantize_q4_0(cls, data_bytes: bytes, num_elements: int) -> np.ndarray:
        """Dequantize Q4_0 blocks to single-precision FP32 array.

        Each Q4_0 block contains 32 values represented in 18 bytes: 2 bytes of
        FP16 scale d followed by 16 bytes containing 32 4-bit nibbles.

        Args:
            data_bytes: Raw binary bytes containing Q4_0 blocks.
            num_elements: Total number of weights expected.

        Returns:
            np.ndarray: Reconstructed FP32 weights of shape (num_elements,).

        Raises:
            DequantizationError: If byte slice is truncated or does not match block size.
        """
        block_size_bytes = 18
        elements_per_block = 32
        expected_blocks = (num_elements + elements_per_block - 1) // elements_per_block
        expected_bytes = expected_blocks * block_size_bytes

        if len(data_bytes) < expected_bytes:
            raise DequantizationError(
                f"Q4_0 buffer truncated: expected at least {expected_bytes} bytes for "
                f"{num_elements} elements, received {len(data_bytes)} bytes."
            )

        output = np.empty(expected_blocks * elements_per_block, dtype=np.float32)

        for b in range(expected_blocks):
            offset = b * block_size_bytes
            scale_bytes = data_bytes[offset : offset + 2]
            scale_f16 = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            scale = float(scale_f16)

            qs_bytes = data_bytes[offset + 2 : offset + 18]
            qs_arr = np.frombuffer(qs_bytes, dtype=np.uint8)

            low_nibbles = (qs_arr & 0x0F).astype(np.int8) - 8
            high_nibbles = ((qs_arr >> 4) & 0x0F).astype(np.int8) - 8

            unpacked = np.empty(32, dtype=np.float32)
            unpacked[0:16] = low_nibbles.astype(np.float32) * scale
            unpacked[16:32] = high_nibbles.astype(np.float32) * scale

            dest_offset = b * elements_per_block
            output[dest_offset : dest_offset + 32] = unpacked

        return output[:num_elements]

    @classmethod
    def dequantize_q4_k(cls, data_bytes: bytes, num_elements: int) -> np.ndarray:
        """Dequantize Q4_K super-blocks to single-precision FP32 array.

        Each Q4_K super-block encodes 256 weights in 144 bytes:
        d (fp16, 2B), dmin (fp16, 2B), scales (12B), and qs (128B).

        Args:
            data_bytes: Raw binary bytes containing Q4_K super-blocks.
            num_elements: Total number of weights expected.

        Returns:
            np.ndarray: Reconstructed FP32 weights of shape (num_elements,).

        Raises:
            DequantizationError: If byte slice is truncated or invalid.
        """
        super_block_size = 144
        elements_per_block = 256
        expected_blocks = (num_elements + elements_per_block - 1) // elements_per_block
        expected_bytes = expected_blocks * super_block_size

        if len(data_bytes) < expected_bytes:
            raise DequantizationError(
                f"Q4_K buffer truncated: expected at least {expected_bytes} bytes for "
                f"{num_elements} elements, received {len(data_bytes)} bytes."
            )

        output = np.empty(expected_blocks * elements_per_block, dtype=np.float32)

        for b in range(expected_blocks):
            offset = b * super_block_size
            d_val = float(np.frombuffer(data_bytes[offset : offset + 2], dtype=np.float16)[0])
            dmin_val = float(np.frombuffer(data_bytes[offset + 2 : offset + 4], dtype=np.float16)[0])

            scales_raw = np.frombuffer(data_bytes[offset + 4 : offset + 16], dtype=np.uint8)
            scales = np.zeros(8, dtype=np.float32)
            mins = np.zeros(8, dtype=np.float32)

            for j in range(4):
                scales[j] = float(scales_raw[j] & 63)
                mins[j] = float(scales_raw[j + 4] & 63)

            for j in range(4, 8):
                scales[j] = float((scales_raw[j + 4] & 0x0F) | ((scales_raw[j - 4] >> 6) << 4))
                mins[j] = float((scales_raw[j + 4] >> 4) | ((scales_raw[j] >> 6) << 4))

            qs = np.frombuffer(data_bytes[offset + 16 : offset + 144], dtype=np.uint8)

            dest_offset = b * elements_per_block
            for j in range(8):
                sc = d_val * scales[j]
                m = dmin_val * mins[j]
                sub_qs = qs[j * 16 : (j + 1) * 16]
                low = (sub_qs & 0x0F).astype(np.float32) * sc - m
                high = ((sub_qs >> 4) & 0x0F).astype(np.float32) * sc - m
                block_weights = np.empty(32, dtype=np.float32)
                block_weights[0:16] = low
                block_weights[16:32] = high
                sub_dest = dest_offset + j * 32
                output[sub_dest : sub_dest + 32] = block_weights

        return output[:num_elements]

    @classmethod
    def dequantize_q8_0(cls, data_bytes: bytes, num_elements: int) -> np.ndarray:
        """Dequantize Q8_0 blocks to single-precision FP32 array.

        Each Q8_0 block contains 32 values represented in 34 bytes: 2 bytes of
        FP16 scale d followed by 32 signed int8 values.

        Args:
            data_bytes: Raw binary bytes containing Q8_0 blocks.
            num_elements: Total number of weights expected.

        Returns:
            np.ndarray: Reconstructed FP32 weights of shape (num_elements,).

        Raises:
            DequantizationError: If byte slice is truncated.
        """
        block_size_bytes = 34
        elements_per_block = 32
        expected_blocks = (num_elements + elements_per_block - 1) // elements_per_block
        expected_bytes = expected_blocks * block_size_bytes

        if len(data_bytes) < expected_bytes:
            raise DequantizationError(
                f"Q8_0 buffer truncated: expected at least {expected_bytes} bytes for "
                f"{num_elements} elements, received {len(data_bytes)} bytes."
            )

        output = np.empty(expected_blocks * elements_per_block, dtype=np.float32)

        for b in range(expected_blocks):
            offset = b * block_size_bytes
            scale_bytes = data_bytes[offset : offset + 2]
            scale_f16 = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            scale = float(scale_f16)

            qs_bytes = data_bytes[offset + 2 : offset + 34]
            qs_arr = np.frombuffer(qs_bytes, dtype=np.int8)

            dest_offset = b * elements_per_block
            output[dest_offset : dest_offset + 32] = qs_arr.astype(np.float32) * scale

        return output[:num_elements]

    @classmethod
    def repack_to_intel_symmetric_int4(
        cls,
        weights_f32: np.ndarray,
        group_size: int = DEFAULT_GROUP_SIZE,
        min_sqnr_db: float = MIN_INT4_SQNR_DB,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Quantize FP32 weights into Intel Arc native symmetric INT4 Group 128.

        Applies per-group scaling Delta_g = max(|w|) / 7, clamps to [-8, 7], packs
        two nibbles per uint8 byte, applies 32-byte cacheline zero-padding, and
        validates that the reconstructed Signal-to-Quantization-Noise Ratio (SQNR)
        meets hardware fidelity standards.

        Args:
            weights_f32: Flat or multi-dimensional FP32 weights array.
            group_size: Quantization block group size, default 128 (must be multiple of 32).
            min_sqnr_db: Minimum acceptable SQNR threshold in decibels.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - Packed uint8 array padded to 32-byte cacheline.
                - Scales array as float16 per group.
                - Calculated SQNR in decibels.

        Raises:
            QuantizationMathError: If weights contain NaN or Inf values.
            UnsupportedGroupSizeError: If group size is not a positive multiple of 32.
            SQNRThresholdViolationError: If measured SQNR is strictly below min_sqnr_db.
        """
        if not np.all(np.isfinite(weights_f32)):
            raise QuantizationMathError(
                "Weights tensor contains non-finite values (NaN or Inf).",
                "INVALID_TENSOR_VALUES",
                422,
            )

        if group_size <= 0 or group_size % 32 != 0:
            raise UnsupportedGroupSizeError(group_size)

        flat_w = weights_f32.astype(np.float32).ravel()
        orig_length = flat_w.shape[0]

        if orig_length == 0:
            empty_packed = np.zeros(0, dtype=np.uint8)
            empty_scales = np.zeros(0, dtype=np.float16)
            return empty_packed, empty_scales, 100.0

        is_all_zero = bool(np.all(flat_w == 0.0))

        remainder = orig_length % group_size
        if remainder != 0:
            pad_amount = group_size - remainder
            padded_w = np.pad(flat_w, (0, pad_amount), mode="constant", constant_values=0.0)
        else:
            padded_w = flat_w

        num_groups = padded_w.shape[0] // group_size
        groups = padded_w.reshape((num_groups, group_size))

        max_abs = np.max(np.abs(groups), axis=1)
        scales_f32 = np.where(max_abs > 0.0, max_abs / 7.0, 1.0).astype(np.float32)
        scales_f16 = scales_f32.astype(np.float16)
        effective_scales = np.where(scales_f16.astype(np.float32) > 0.0, scales_f16.astype(np.float32), 1.0)

        quantized_int8 = np.clip(
            np.round(groups / effective_scales[:, np.newaxis]),
            cls.INT4_MIN,
            cls.INT4_MAX,
        ).astype(np.int8)

        reconstructed_groups = quantized_int8.astype(np.float32) * effective_scales[:, np.newaxis]
        reconstructed_flat = reconstructed_groups.ravel()[:orig_length]

        if is_all_zero:
            sqnr_db = 100.0
        else:
            sqnr_db = cls.calculate_sqnr(flat_w, reconstructed_flat)

        if not is_all_zero and sqnr_db < min_sqnr_db:
            raise SQNRThresholdViolationError(sqnr_db, min_sqnr_db)

        flat_q = quantized_int8.ravel()
        num_nibbles = flat_q.shape[0]
        if num_nibbles % 2 != 0:
            flat_q = np.pad(flat_q, (0, 1), mode="constant", constant_values=0)
            num_nibbles += 1

        even_nibbles = flat_q[0::2] & 0x0F
        odd_nibbles = flat_q[1::2] & 0x0F
        packed_bytes = (even_nibbles | (odd_nibbles << 4)).astype(np.uint8)

        aligned_packed = cls.align_to_cacheline(packed_bytes, cls.CACHE_LINE_BYTES)

        return aligned_packed, scales_f16, sqnr_db

    @classmethod
    def repack_to_intel_symmetric_int8(
        cls,
        weights_f32: np.ndarray,
        group_size: int = DEFAULT_GROUP_SIZE,
        min_sqnr_db: float = MIN_INT8_SQNR_DB,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Quantize FP32 weights into Intel Arc native symmetric INT8 Group 128.

        Applies per-group scaling Delta_g = max(|w|) / 127, clamps to [-128, 127],
        applies 32-byte cacheline zero-padding, and validates SQNR >= 34.0 dB.

        Args:
            weights_f32: Flat or multi-dimensional FP32 weights array.
            group_size: Quantization block group size, default 128 (must be multiple of 32).
            min_sqnr_db: Minimum acceptable SQNR threshold in decibels.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - Packed int8 array padded to 32-byte cacheline.
                - Scales array as float16 per group.
                - Calculated SQNR in decibels.

        Raises:
            QuantizationMathError: If weights contain NaN or Inf values.
            UnsupportedGroupSizeError: If group size is not a positive multiple of 32.
            SQNRThresholdViolationError: If measured SQNR is strictly below min_sqnr_db.
        """
        if not np.all(np.isfinite(weights_f32)):
            raise QuantizationMathError(
                "Weights tensor contains non-finite values (NaN or Inf).",
                "INVALID_TENSOR_VALUES",
                422,
            )

        if group_size <= 0 or group_size % 32 != 0:
            raise UnsupportedGroupSizeError(group_size)

        flat_w = weights_f32.astype(np.float32).ravel()
        orig_length = flat_w.shape[0]

        if orig_length == 0:
            return np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float16), 100.0

        is_all_zero = bool(np.all(flat_w == 0.0))

        remainder = orig_length % group_size
        if remainder != 0:
            pad_amount = group_size - remainder
            padded_w = np.pad(flat_w, (0, pad_amount), mode="constant", constant_values=0.0)
        else:
            padded_w = flat_w

        num_groups = padded_w.shape[0] // group_size
        groups = padded_w.reshape((num_groups, group_size))

        max_abs = np.max(np.abs(groups), axis=1)
        scales_f32 = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
        scales_f16 = scales_f32.astype(np.float16)
        effective_scales = np.where(scales_f16.astype(np.float32) > 0.0, scales_f16.astype(np.float32), 1.0)

        quantized_int8 = np.clip(
            np.round(groups / effective_scales[:, np.newaxis]),
            cls.INT8_MIN,
            cls.INT8_MAX,
        ).astype(np.int8)

        reconstructed_groups = quantized_int8.astype(np.float32) * effective_scales[:, np.newaxis]
        reconstructed_flat = reconstructed_groups.ravel()[:orig_length]

        if is_all_zero:
            sqnr_db = 100.0
        else:
            sqnr_db = cls.calculate_sqnr(flat_w, reconstructed_flat)

        if not is_all_zero and sqnr_db < min_sqnr_db:
            raise SQNRThresholdViolationError(sqnr_db, min_sqnr_db)

        flat_q = quantized_int8.ravel()
        aligned_packed = cls.align_to_cacheline(flat_q.view(np.uint8), cls.CACHE_LINE_BYTES).view(np.int8)

        return aligned_packed, scales_f16, sqnr_db

    @classmethod
    def align_to_cacheline(cls, data: np.ndarray, alignment: int = CACHE_LINE_BYTES) -> np.ndarray:
        """Pad a 1D uint8 array to a multiple of alignment bytes for Direct Memory Access.

        Args:
            data: Input 1D numpy array.
            alignment: Byte boundary requirement (default 32 bytes for Intel Xe cacheline).

        Returns:
            np.ndarray: Padded array aligned to specified byte multiple.
        """
        curr_len = data.shape[0]
        remainder = curr_len % alignment
        if remainder == 0:
            return data
        pad_needed = alignment - remainder
        return np.pad(data, (0, pad_needed), mode="constant", constant_values=0)

    @staticmethod
    def calculate_sqnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute exact Signal-to-Quantization-Noise Ratio in decibels.

        Args:
            original: Ground truth FP32 weights.
            reconstructed: Dequantized FP32 weights.

        Returns:
            float: SQNR in decibels.
        """
        eps = 1e-12
        orig_flat = original.ravel().astype(np.float64)
        recon_flat = reconstructed.ravel().astype(np.float64)

        signal_power = np.sum(orig_flat**2)
        noise_power = np.sum((orig_flat - recon_flat) ** 2)

        if noise_power < eps:
            return 100.0

        if signal_power < eps:
            return 0.0

        return float(10.0 * np.log10(signal_power / noise_power))
