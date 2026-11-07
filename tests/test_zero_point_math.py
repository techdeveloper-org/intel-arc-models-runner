"""Tests for the pure zero-point math used by the NPU graph rewriter.

The oracle in this file is written independently of the implementation: it dequantises
with float64 arithmetic and checks the proven per-element error bound.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from arc_runner.quant.zero_point_math import (
    F16_MIN_NORMAL,
    ZeroPointMathError,
    decode_unsigned,
    encode_signed,
    is_symmetric_zero_point,
    pack_nibbles,
    requantize_symmetric,
    unpack_nibbles,
    unsigned_to_signed_bytes,
)

PROFILE = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _signed_nibble(value: int) -> int:
    """Return the two's complement reading of a 4-bit pattern."""
    return value if value < 8 else value - 16


def _bound_per_group(weights: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    """Independent oracle for the per-element error bound of one group."""
    qp, qn = (7, 8) if bits == 4 else (127, 128)
    a_plus = np.maximum(weights.max(axis=-1, keepdims=True), 0.0)
    a_minus = np.maximum(-weights.min(axis=-1, keepdims=True), 0.0)
    return np.maximum.reduce([scale / 2.0, a_plus - qp * scale, a_minus - qn * scale])


class TestNibbleCodec:
    """Low nibble first packing, as measured on the OpenVINO IR of Qwen3-0.6B."""

    def test_unpack_nibbles_low_nibble_first_returns_expected_values(self):
        values = unpack_nibbles(np.array([0x21, 0xF0], dtype=np.uint8))
        assert values.tolist() == [1, 2, 0, 15]

    def test_pack_nibbles_odd_count_raises(self):
        with pytest.raises(ZeroPointMathError):
            pack_nibbles(np.array([1, 2, 3], dtype=np.uint8))

    def test_pack_nibbles_value_above_fifteen_raises(self):
        with pytest.raises(ZeroPointMathError):
            pack_nibbles(np.array([1, 16], dtype=np.uint8))

    @PROFILE
    @given(st.lists(st.integers(0, 15), min_size=0, max_size=64).filter(lambda v: len(v) % 2 == 0))
    def test_pack_unpack_nibbles_roundtrip_returns_input(self, values):
        arr = np.array(values, dtype=np.uint8)
        assert unpack_nibbles(pack_nibbles(arr)).tolist() == values


class TestSymmetricZeroPoint:
    """The exactness precondition for deleting a Subtract."""

    @pytest.mark.parametrize(
        "element_type,raw,expected",
        [
            ("u4", bytes([0x88, 0x88]), True),
            ("u4", bytes([0x88, 0x87]), False),
            ("u4", bytes([0x78, 0x88]), False),
            ("u8", bytes([128, 128, 128]), True),
            ("u8", bytes([128, 127, 128]), False),
        ],
    )
    def test_is_symmetric_zero_point_detects_every_deviation(self, element_type, raw, expected):
        assert is_symmetric_zero_point(np.frombuffer(raw, dtype=np.uint8), element_type) is expected

    def test_is_symmetric_zero_point_unknown_type_raises(self):
        with pytest.raises(ZeroPointMathError):
            is_symmetric_zero_point(np.zeros(2, dtype=np.uint8), "u16")


class TestExactConversion:
    """q XOR 8 equals q - 8 in 4-bit two's complement; q XOR 0x80 equals q - 128 as int8."""

    def test_unsigned_to_signed_bytes_u4_equals_subtraction_of_eight_for_all_values(self):
        for low in range(16):
            for high in range(16):
                out = unsigned_to_signed_bytes(bytes([low | (high << 4)]), "u4")[0]
                assert _signed_nibble(out & 0x0F) == low - 8
                assert _signed_nibble(out >> 4) == high - 8

    def test_unsigned_to_signed_bytes_u8_equals_subtraction_of_128_for_all_values(self):
        raw = bytes(range(256))
        out = np.frombuffer(unsigned_to_signed_bytes(raw, "u8"), dtype=np.int8)
        assert out.tolist() == [v - 128 for v in range(256)]

    def test_unsigned_to_signed_bytes_is_an_involution(self):
        raw = bytes(range(256))
        assert unsigned_to_signed_bytes(unsigned_to_signed_bytes(raw, "u4"), "u4") == raw


class TestRequantizeSymmetric:
    """Per-group symmetric requantisation with the proven bound."""

    @PROFILE
    @given(
        data=st.data(),
        bits=st.sampled_from([4, 8]),
        shape=st.tuples(st.integers(1, 3), st.integers(1, 3), st.sampled_from([1, 2, 32, 128])),
    )
    def test_requantize_symmetric_error_within_bound_for_any_group(self, data, bits, shape):
        top = (1 << bits) - 1
        q = data.draw(hnp.arrays(np.uint8, shape, elements=st.integers(0, top)))
        zp = data.draw(hnp.arrays(np.uint8, shape[:-1] + (1,), elements=st.integers(0, top)))
        scale = data.draw(
            hnp.arrays(
                np.float16,
                shape[:-1] + (1,),
                elements=st.floats(min_value=0.0, max_value=50.0, width=16, allow_nan=False),
            )
        )
        result = requantize_symmetric(q, zp, scale, bits)
        weights = scale.astype(np.float64) * (q.astype(np.int64) - zp.astype(np.int64))
        new_scale = result.scale.astype(np.float64)
        recon = result.q.astype(np.float64) * new_scale
        bound = _bound_per_group(weights, new_scale, bits)
        assert np.all(np.abs(weights - recon) <= bound + 1e-9 * (1.0 + np.abs(weights).max()))
        assert result.q.dtype == np.int8
        assert result.scale.dtype == np.float16
        assert np.all(result.scale.astype(np.float64) >= F16_MIN_NORMAL)

    @pytest.mark.parametrize("bits,low,high", [(4, -8, 7), (8, -128, 127)])
    def test_requantize_symmetric_output_stays_in_signed_range(self, bits, low, high):
        rng = np.random.default_rng(0)
        top = (1 << bits) - 1
        q = rng.integers(0, top + 1, size=(4, 3, 64)).astype(np.uint8)
        zp = rng.integers(0, top + 1, size=(4, 3, 1)).astype(np.uint8)
        scale = rng.uniform(0.001, 1.0, size=(4, 3, 1)).astype(np.float16)
        result = requantize_symmetric(q, zp, scale, bits)
        assert int(result.q.min()) >= low
        assert int(result.q.max()) <= high

    def test_requantize_symmetric_zero_group_stays_zero_with_floor_scale(self):
        q = np.full((1, 1, 4), 5, dtype=np.uint8)
        zp = np.full((1, 1, 1), 5, dtype=np.uint8)
        scale = np.full((1, 1, 1), 0.5, dtype=np.float16)
        result = requantize_symmetric(q, zp, scale, 4)
        assert result.q.tolist() == [[[0, 0, 0, 0]]]
        assert float(result.scale[0, 0, 0]) >= F16_MIN_NORMAL
        assert result.max_abs_error == 0.0

    def test_requantize_symmetric_tiny_group_scale_is_floored_at_f16_normal(self):
        q = np.array([[[0, 15]]], dtype=np.uint8)
        zp = np.array([[[8]]], dtype=np.uint8)
        scale = np.array([[[1e-6]]], dtype=np.float16)
        result = requantize_symmetric(q, zp, scale, 4)
        assert float(result.scale[0, 0, 0]) >= F16_MIN_NORMAL

    def test_requantize_symmetric_nonfinite_scale_raises(self):
        q = np.zeros((1, 1, 4), dtype=np.uint8)
        zp = np.zeros((1, 1, 1), dtype=np.uint8)
        scale = np.array([[[np.inf]]], dtype=np.float16)
        with pytest.raises(ZeroPointMathError):
            requantize_symmetric(q, zp, scale, 4)

    def test_requantize_symmetric_scale_overflow_raises(self):
        q = np.array([[[0, 15]]], dtype=np.uint8)
        zp = np.array([[[0]]], dtype=np.uint8)
        scale = np.array([[[60000.0]]], dtype=np.float16)
        with pytest.raises(ZeroPointMathError):
            requantize_symmetric(q, zp, scale, 4)

    def test_requantize_symmetric_shape_mismatch_raises(self):
        q = np.zeros((2, 3, 4), dtype=np.uint8)
        zp = np.zeros((2, 3, 1), dtype=np.uint8)
        scale = np.ones((2, 2, 1), dtype=np.float16)
        with pytest.raises(ZeroPointMathError):
            requantize_symmetric(q, zp, scale, 4)

    def test_requantize_symmetric_unsupported_bits_raises(self):
        q = np.zeros((1, 1, 2), dtype=np.uint8)
        with pytest.raises(ZeroPointMathError):
            requantize_symmetric(q, np.zeros((1, 1, 1), np.uint8), np.ones((1, 1, 1), np.float16), 5)

    def test_requantize_symmetric_reports_measured_errors_not_bounds(self):
        rng = np.random.default_rng(1)
        q = rng.integers(0, 16, size=(8, 4, 128)).astype(np.uint8)
        zp = rng.integers(0, 16, size=(8, 4, 1)).astype(np.uint8)
        scale = rng.uniform(0.01, 0.1, size=(8, 4, 1)).astype(np.float16)
        result = requantize_symmetric(q, zp, scale, 4)
        weights = scale.astype(np.float64) * (q.astype(np.int64) - zp.astype(np.int64))
        recon = result.q.astype(np.float64) * result.scale.astype(np.float64)
        err = np.abs(weights - recon)
        group_max = np.abs(weights).max(axis=-1, keepdims=True)
        assert result.max_abs_error == pytest.approx(float(err.max()), rel=1e-12)
        assert result.max_rel_error == pytest.approx(float((err / group_max).max()), rel=1e-12)
        assert result.max_rel_error <= 1.0 / 15.0 + 2.0 ** -11 + 1e-9


class TestSignedCodec:
    """Signed storage written back into the weights binary."""

    def test_encode_signed_int4_packs_twos_complement_low_nibble_first(self):
        q = np.array([-8, 7, -1, 0], dtype=np.int8)
        assert encode_signed(q, 4) == bytes([0x78, 0x0F])

    def test_encode_signed_int8_writes_raw_bytes(self):
        q = np.array([-128, 127, -1], dtype=np.int8)
        assert encode_signed(q, 8) == bytes([0x80, 0x7F, 0xFF])

    def test_encode_signed_out_of_range_raises(self):
        with pytest.raises(ZeroPointMathError):
            encode_signed(np.array([8, 0], dtype=np.int8), 4)

    def test_decode_unsigned_u4_returns_requested_count(self):
        out = decode_unsigned(bytes([0x21, 0x43]), "u4", 4)
        assert out.tolist() == [1, 2, 3, 4]

    def test_decode_unsigned_wrong_size_raises(self):
        with pytest.raises(ZeroPointMathError):
            decode_unsigned(bytes([0x21]), "u4", 4)
