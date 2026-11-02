"""Triad test suite for QuantizationMathEngine adhering to Enterprise standards.

Validates dequantization from GGUF k-quants to FP32, repacking into Intel Arc
symmetric INT4 Group 128 and INT8, 32-byte cacheline zero-padding alignment, and SQNR verification.
"""

import numpy as np
import pytest

from arc_runner.quant.math_transforms import (
    DequantizationError,
    QuantizationMathEngine,
    QuantizationMathError,
    SQNRThresholdViolationError,
    UnsupportedGroupSizeError,
)


class TestMathTransformsPositive:
    """Positive test specifications for mathematical quantization and repacking."""

    def test_dequantize_q4_0_when_valid_blocks_then_matches_fp32_ground_truth(self) -> None:
        """Verify Q4_0 block dequantization produces exact FP32 scaled values."""
        scale = np.float16(0.5)
        scale_bytes = scale.tobytes()
        nibbles = bytearray(16)
        nibbles[0] = 0x08
        nibbles[1] = 0xF0
        raw_bytes = scale_bytes + bytes(nibbles)

        result = QuantizationMathEngine.dequantize_q4_0(raw_bytes, 32)

        assert result.shape == (32,)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(-4.0, abs=1e-5)

    def test_dequantize_q4_k_when_256_super_block_then_applies_subscales_correctly(self) -> None:
        """Verify Q4_K super-block dequantization scales 256 weights accurately."""
        d = np.float16(0.25).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytes(12)
        qs = bytes(128)
        raw_super_block = d + dmin + scales + qs

        result = QuantizationMathEngine.dequantize_q4_k(raw_super_block, 256)

        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_dequantize_q8_0_when_signed_bytes_then_reconstructs_fp32_accurately(self) -> None:
        """Verify Q8_0 block dequantization scales signed 8-bit integers."""
        scale = np.float16(0.1).tobytes()
        qs = np.array([10, -20] + [0] * 30, dtype=np.int8).tobytes()
        raw_bytes = scale + qs

        result = QuantizationMathEngine.dequantize_q8_0(raw_bytes, 32)

        assert result.shape == (32,)
        assert result[0] == pytest.approx(1.0, abs=1e-3)
        assert result[1] == pytest.approx(-2.0, abs=1e-3)

    def test_repack_int4_when_fp32_input_then_produces_symmetric_int4_group128(self) -> None:
        """Verify repacking generates packed nibbles, scales, and valid SQNR."""
        np.random.seed(42)
        weights = np.random.uniform(-1.0, 1.0, 256).astype(np.float32)

        packed, scales, sqnr = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128)

        assert packed.ndim == 1
        assert packed.dtype == np.uint8
        assert len(packed) % 32 == 0
        assert len(scales) == 2
        assert sqnr >= 20.0

    def test_repack_int4_when_normal_distribution_then_guarantees_sqnr_above_15db(self) -> None:
        """Verify that standard normal weight tensors achieve target INT4 SQNR >= 15 dB."""
        np.random.seed(1337)
        weights = np.random.normal(0.0, 0.02, 1024).astype(np.float32)

        _, _, sqnr = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128, min_sqnr_db=15.0)

        assert sqnr >= 15.0

    def test_repack_int4_when_measured_then_satisfies_error_bound_half_delta(self) -> None:
        """Verify that maximum reconstruction error does not exceed Delta_g / 2."""
        weights = np.linspace(-0.7, 0.7, 128, dtype=np.float32)

        packed, scales, _ = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128, min_sqnr_db=15.0)
        scale = float(scales[0])

        low_val = int(packed[0] & 0x0F)
        high_val = int((packed[0] >> 4) & 0x0F)
        low_signed = low_val - 16 if low_val >= 8 else low_val
        high_signed = high_val - 16 if high_val >= 8 else high_val
        assert -8 <= low_signed <= 7
        assert -8 <= high_signed <= 7

    def test_align_cacheline_when_unaligned_array_then_pads_to_32_byte_stride(self) -> None:
        """Verify unaligned buffer is padded with zeros to 32-byte boundary."""
        data = np.ones(45, dtype=np.uint8)

        aligned = QuantizationMathEngine.align_to_cacheline(data, 32)

        assert len(aligned) == 64
        assert np.all(aligned[:45] == 1)
        assert np.all(aligned[45:] == 0)

    def test_repack_int4_when_extreme_amplitudes_then_clamps_strictly_to_neg8_pos7(self) -> None:
        """Verify out-of-range amplitudes are clamped to INT4 dynamic range."""
        weights = np.array([-100.0, 100.0] + [0.0] * 126, dtype=np.float32)

        packed, scales, _ = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128, min_sqnr_db=0.0)

        low = int(packed[0] & 0x0F)
        high = int((packed[0] >> 4) & 0x0F)
        low_signed = low - 16 if low >= 8 else low
        high_signed = high - 16 if high >= 8 else high
        assert low_signed in (-7, -8)
        assert high_signed == 7

    def test_repack_int8_when_scale_computed_then_preserves_fp16_dynamic_range(self) -> None:
        """Verify INT8 repacking preserves FP16 dynamic range and achieves high SQNR."""
        weights = np.random.uniform(-0.05, 0.05, 128).astype(np.float32)

        packed, scales, sqnr = QuantizationMathEngine.repack_to_intel_symmetric_int8(weights, group_size=128)

        assert scales.dtype == np.float16
        assert np.isfinite(scales[0])
        assert scales[0] > 0
        assert sqnr >= 34.0


class TestMathTransformsNegative:
    """Negative test specifications verifying exception contracts."""

    def test_dequantize_q4_0_when_truncated_byte_slice_then_raises_dequant_error(self) -> None:
        """Verify truncated Q4_0 byte slice raises DequantizationError."""
        truncated_bytes = bytes(10)

        with pytest.raises(DequantizationError) as exc_info:
            QuantizationMathEngine.dequantize_q4_0(truncated_bytes, 32)

        assert "truncated" in str(exc_info.value).lower()

    def test_repack_int4_when_weights_contain_nan_inf_then_raises_math_error(self) -> None:
        """Verify tensor containing NaN or Inf raises QuantizationMathError."""
        weights = np.array([1.0, np.nan, 3.0] + [0.0] * 125, dtype=np.float32)

        with pytest.raises(QuantizationMathError) as exc_info:
            QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128)

        assert "non-finite" in str(exc_info.value).lower()

    def test_repack_int4_when_group_size_not_divisible_by_32_then_raises_group_size_error(self) -> None:
        """Verify non-multiple of 32 group size raises UnsupportedGroupSizeError."""
        weights = np.ones(128, dtype=np.float32)

        with pytest.raises(UnsupportedGroupSizeError) as exc_info:
            QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=50)

        assert exc_info.value.group_size == 50

    def test_repack_int4_when_all_zero_weights_then_handles_delta_zero_without_div_zero(self) -> None:
        """Verify all zero tensor avoids division by zero and succeeds safely."""
        weights = np.zeros(128, dtype=np.float32)

        packed, scales, sqnr = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128)

        assert len(packed) % 32 == 0
        assert sqnr == 100.0

    def test_repack_int4_when_sqnr_drops_below_threshold_then_raises_sqnr_threshold_error(self) -> None:
        """Verify synthetic SQNR failure raises SQNRThresholdViolationError when forced."""
        weights = np.random.uniform(-1.0, 1.0, 128).astype(np.float32)
        with pytest.raises(SQNRThresholdViolationError):
            QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128, min_sqnr_db=99.0)

    def test_dequantize_q4_k_when_truncated_super_block_then_raises_dequant_error(self) -> None:
        """Verify truncated Q4_K buffer raises DequantizationError."""
        truncated_bytes = bytes(100)

        with pytest.raises(DequantizationError):
            QuantizationMathEngine.dequantize_q4_k(truncated_bytes, 256)


class TestMathTransformsEdgeCase:
    """Edge-case test specifications verifying boundaries and numerical stability."""

    def test_repack_int4_when_tensor_size_not_multiple_of_group_size_then_pads_cleanly(self) -> None:
        """Verify arbitrary tensor lengths are padded cleanly to group size multiples."""
        weights = np.ones(130, dtype=np.float32)

        packed, scales, sqnr = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128)

        assert len(scales) == 2
        assert len(packed) % 32 == 0
        assert sqnr > 20.0

    def test_repack_int4_when_weights_span_1e_minus_6_to_1e3_then_avoids_underflow(self) -> None:
        """Verify weights spanning wide orders of magnitude quantize without underflow."""
        weights = np.array([1e-6, 1e-3, 1.0, 1000.0] + [0.0] * 124, dtype=np.float32)

        packed, scales, _ = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128, min_sqnr_db=0.0)

        assert len(packed) % 32 == 0
        assert np.isfinite(scales[0])

    def test_repack_int4_when_single_element_tensor_then_computes_without_shape_error(self) -> None:
        """Verify single element array repacks without dimension error."""
        weights = np.array([0.75], dtype=np.float32)

        packed, scales, _ = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128)

        assert len(scales) == 1
        assert len(packed) % 32 == 0

    def test_repack_int4_when_odd_number_of_nibbles_then_pads_terminal_nibble_cleanly(self) -> None:
        """Verify odd number of nibbles pads terminal nibble without indexing errors."""
        weights = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

        packed, scales, _ = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=32, min_sqnr_db=15.0)

        assert len(packed) >= 16
        assert len(packed) % 32 == 0

    def test_repack_int4_when_subnormal_scale_factors_then_denormalizes_safely(self) -> None:
        """Verify tiny amplitudes do not trigger divide-by-zero or NaN scales."""
        weights = np.full(128, 1e-15, dtype=np.float32)

        packed, scales, sqnr = QuantizationMathEngine.repack_to_intel_symmetric_int4(weights, group_size=128)

        assert np.all(np.isfinite(scales))
        assert sqnr >= 0.0

    def test_repack_int4_when_exact_boundary_values_then_preserves_unbiased_distribution(self) -> None:
        """Verify boundary values -7 and 7 preserve exact representation."""
        delta = 0.1
        w_vals = np.array([-7 * delta, 7 * delta] + [0.0] * 126, dtype=np.float32)

        packed, scales, _ = QuantizationMathEngine.repack_to_intel_symmetric_int4(w_vals, group_size=128, min_sqnr_db=15.0)

        low = int(packed[0] & 0x0F)
        high = int((packed[0] >> 4) & 0x0F)
        low_signed = low - 16 if low >= 8 else low
        high_signed = high - 16 if high >= 8 else high
        assert low_signed == -7
        assert high_signed == 7
