"""Triad test suite for Local GGUF to OpenVINO IR Transpiler and Proofs M7, M8, M9, M20.

Verifies symmetric INT4 dequantization arithmetic, XMX cacheline bit-packing,
SQNR signal-to-noise bounds, and end-to-end OpenVINO IR container generation.
"""

import math
from pathlib import Path
import struct
import numpy as np
import pytest

from arc_runner.quant.models import GGMLType, GGUFValueType
from arc_runner.quant.transpiler import (
    DequantizationError,
    GGUFTranspiler,
    TranspilationError,
    UnsupportedBitWidthError,
)


def create_minimal_gguf(path: Path) -> None:
    """Helper to write a valid minimal GGUF v3 file containing one F32 tensor."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 1))

        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", GGUFValueType.STRING.value))
        val = b"llama"
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        t_name = b"token_embd.weight"
        f.write(struct.pack("<Q", len(t_name)))
        f.write(t_name)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<I", GGMLType.F32.value))
        f.write(struct.pack("<Q", 0))

        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)

        f.write(struct.pack("<16f", *([1.0] * 16)))


class TestDequantizationM7:
    """Mathematical validation of Proof M7: Symmetric INT4 Dequantization."""

    def test_dequantize_q4_0_block_positive(self) -> None:
        """Verify 18-byte Q4_0 block dequantizes to float16 scale and 32 signed integers."""
        scale_bytes = struct.pack("<e", 0.5)
        raw_nibbles = bytes([0x88] * 16)
        block = scale_bytes + raw_nibbles

        scale, nibbles = GGUFTranspiler.dequantize_q4_0_block(block)
        assert math.isclose(scale, 0.5, rel_tol=1e-3)
        assert len(nibbles) == 32
        assert all(n == 0 for n in nibbles)

    def test_dequantize_q4_0_block_negative_corrupt_size(self) -> None:
        """Verify non-18 byte buffer raises DequantizationError."""
        with pytest.raises(DequantizationError, match="must be exactly 18 bytes"):
            GGUFTranspiler.dequantize_q4_0_block(b"\x00" * 17)

    def test_dequantize_q4_0_block_negative_invalid_scale(self) -> None:
        """Verify zero or NaN scale factor raises DequantizationError."""
        zero_scale = struct.pack("<e", 0.0) + bytes([0x00] * 16)
        with pytest.raises(DequantizationError, match="Invalid Q4_0 scale"):
            GGUFTranspiler.dequantize_q4_0_block(zero_scale)

    def test_dequantize_q4_0_block_edge_case_extremes(self) -> None:
        """Verify extreme 4-bit nibbles (low=0x0 -> -8, high=0xF -> 7) dequantize cleanly."""
        scale_bytes = struct.pack("<e", 1.0)
        extreme_bytes = bytes([(0xF << 4) | 0x0] * 16)
        block = scale_bytes + extreme_bytes

        scale, nibbles = GGUFTranspiler.dequantize_q4_0_block(block)
        assert math.isclose(scale, 1.0, rel_tol=1e-3)
        assert nibbles[0] == -8
        assert nibbles[1] == 7


class TestXMXBitPackingM9:
    """Mathematical validation of Proof M9: Intel XMX Matrix Packing Layout."""

    def test_pack_xmx_nibbles_positive_alignment(self) -> None:
        """Verify packing 64 nibbles yields 32 bytes strictly aligned to 32-byte cacheline."""
        nibbles = [i % 16 for i in range(64)]
        packed = GGUFTranspiler.pack_xmx_nibbles(nibbles)

        assert len(packed) == 32
        assert len(packed) % 32 == 0

    def test_pack_xmx_nibbles_edge_case_odd_count(self) -> None:
        """Verify odd number of nibbles pads cleanly to 32-byte boundary."""
        nibbles = [1, 2, 3]
        packed = GGUFTranspiler.pack_xmx_nibbles(nibbles)
        assert len(packed) == 32
        assert len(packed) % 32 == 0


class TestSQNRMetricM8:
    """Mathematical validation of Proof M8: SQNR Error Bounds."""

    def test_compute_sqnr_positive_identical(self) -> None:
        """Verify identical original and quantized matrices yield perfect 99.0 dB SQNR."""
        orig = np.array([1.0, 2.0, 3.0, 4.0])
        sqnr = GGUFTranspiler.compute_sqnr(orig, orig)
        assert sqnr == 99.0

    def test_compute_sqnr_with_noise(self) -> None:
        """Verify noisy quantization yields measurable finite SQNR in dB."""
        orig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        quant = orig + np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        sqnr = GGUFTranspiler.compute_sqnr(orig, quant)
        assert sqnr > 20.0
        assert not math.isinf(sqnr)


class TestTranspilerExecution:
    """End-to-end integration tests for GGUFTranspiler."""

    def test_transpile_negative_missing_input_file(self, tmp_path: Path) -> None:
        """Verify non-existent input path raises TranspilationError."""
        missing = tmp_path / "does_not_exist.gguf"
        out_dir = tmp_path / "out"
        with pytest.raises(TranspilationError, match="does not exist"):
            GGUFTranspiler(str(missing), str(out_dir))

    def test_transpile_negative_unsupported_precision(self, tmp_path: Path) -> None:
        """Verify unsupported bit width raises UnsupportedBitWidthError."""
        gguf_path = tmp_path / "test.gguf"
        create_minimal_gguf(gguf_path)
        out_dir = tmp_path / "out"

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        with pytest.raises(UnsupportedBitWidthError):
            transpiler.transpile(precision="int2")

    def test_transpile_positive_minimal_model(self, tmp_path: Path) -> None:
        """Verify successful transpilation generates valid openvino_model.xml and .bin."""
        gguf_path = tmp_path / "minimal.gguf"
        create_minimal_gguf(gguf_path)
        out_dir = tmp_path / "ov_ir_export"

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        result = transpiler.transpile(precision="int4", target_device="GPU")

        assert result.total_tensors == 1
        assert result.xml_path.exists()
        assert result.bin_path.exists()
        assert (out_dir / "config.json").exists()

        xml_text = result.xml_path.read_text(encoding="utf-8")
        assert '<net name="llama" version="11">' in xml_text
        assert 'token_embd.weight' in xml_text
        assert result.total_bytes > 0
        assert result.duration_ms > 0.0
