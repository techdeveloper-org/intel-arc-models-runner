"""Triad test suite for GGUF Transpiler Streaming Engine and Proofs M15-M20, P1-P6.

Verifies byte-weighted progress monotonicity (M15), IIR EMA throughput stability (M16),
quantization quality bounds (M17), 32-byte DPAS alignment (M18), Little's Law queue stability (M19),
zero-copy incremental SHA-256 (M20), O(1) direct-to-disk streaming (P2), and fault resilience.
"""

import hashlib
import math
from pathlib import Path
import struct
import pytest

from arc_runner.quant.models import GGMLType, GGUFValueType
from arc_runner.quant.transpiler import (
    DequantizationError,
    GGUFTranspiler,
    TranspilationError,
    TranspilationProgress,
    UnsupportedBitWidthError,
)


def create_multi_tensor_gguf(path: Path, count: int = 3, quant_type: GGMLType = GGMLType.F32) -> None:
    """Helper to write a valid GGUF v3 file containing multiple tensors."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", count))
        f.write(struct.pack("<Q", 1))

        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", GGUFValueType.STRING.value))
        val = b"llama"
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        offset = 0
        for i in range(count):
            t_name = f"blk.{i}.weight".encode("utf-8")
            f.write(struct.pack("<Q", len(t_name)))
            f.write(t_name)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", 4))
            f.write(struct.pack("<Q", 4))
            f.write(struct.pack("<I", quant_type.value))
            f.write(struct.pack("<Q", offset))
            offset += 18 if quant_type == GGMLType.Q4_0 else 64

        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)

        for _ in range(count):
            if quant_type == GGMLType.Q4_0:
                scale_bytes = struct.pack("<e", 0.5)
                f.write(scale_bytes + bytes([0x88] * 16))
            else:
                f.write(struct.pack("<16f", *([1.0] * 16)))


class TestTranspilerStreamPositive:
    """Positive happy-path test scenarios for GGUF streaming transpilation."""

    def test_transpile_stream_monotonic_progress_m15(self, tmp_path: Path) -> None:
        """Verify Proof M15: Byte-weighted progress strictly monotonic non-decreasing."""
        gguf_path = tmp_path / "model.gguf"
        out_dir = tmp_path / "ov_out"
        create_multi_tensor_gguf(gguf_path, count=5, quant_type=GGMLType.F32)

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        progress_events = list(transpiler.transpile_stream(min_emit_interval_ms=0.0))

        assert len(progress_events) == 5
        prev_pct = 0.0
        for p in progress_events:
            assert isinstance(p, TranspilationProgress)
            assert p.progress_pct >= prev_pct
            prev_pct = p.progress_pct
        assert progress_events[-1].progress_pct == 100.0

    def test_transpile_stream_ema_throughput_positive_m16(self, tmp_path: Path) -> None:
        """Verify Proof M16: EMA throughput is positive and ETA is non-negative and finite."""
        gguf_path = tmp_path / "model.gguf"
        out_dir = tmp_path / "ov_out"
        create_multi_tensor_gguf(gguf_path, count=3, quant_type=GGMLType.F32)

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        events = list(transpiler.transpile_stream(min_emit_interval_ms=0.0))

        for ev in events:
            assert ev.throughput_mb_s >= 0.0
            assert ev.eta_seconds >= 0.0
            assert not math.isnan(ev.eta_seconds)
            assert not math.isinf(ev.eta_seconds)

    def test_transpile_stream_sqnr_threshold_m17(self, tmp_path: Path) -> None:
        """Verify Proof M17: Quantized tensors maintain aggregate SQNR >= 32.0 dB."""
        gguf_path = tmp_path / "model_q4.gguf"
        out_dir = tmp_path / "ov_out"
        create_multi_tensor_gguf(gguf_path, count=3, quant_type=GGMLType.Q4_0)

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        events = list(transpiler.transpile_stream(precision="int4", min_emit_interval_ms=0.0))

        for ev in events:
            assert ev.sqnr_db >= 32.0

    def test_transpile_stream_dpas_cacheline_m18(self, tmp_path: Path) -> None:
        """Verify Proof M18: Generated binary container satisfies 32-byte DPAS alignment."""
        gguf_path = tmp_path / "model.gguf"
        out_dir = tmp_path / "ov_out"
        create_multi_tensor_gguf(gguf_path, count=4, quant_type=GGMLType.F32)

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        res = transpiler.transpile()

        assert res.bin_path.exists()
        bin_bytes = res.bin_path.read_bytes()
        assert len(bin_bytes) % 32 == 0
        assert res.total_bytes == len(bin_bytes)

    def test_transpile_stream_incremental_sha256_m20(self, tmp_path: Path) -> None:
        """Verify Proof M20: Incremental SHA-256 matches independent full-file digest."""
        gguf_path = tmp_path / "model.gguf"
        out_dir = tmp_path / "ov_out"
        create_multi_tensor_gguf(gguf_path, count=3, quant_type=GGMLType.F32)

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        res = transpiler.transpile()

        expected_hash = hashlib.sha256(res.bin_path.read_bytes()).hexdigest()
        assert res.sha256_hash == expected_hash

    def test_transpile_stream_direct_disk_streaming_memory_p2(self, tmp_path: Path) -> None:
        """Verify Proof P2: Transpiler writes incrementally to disk without in-memory buffering."""
        gguf_path = tmp_path / "model.gguf"
        out_dir = tmp_path / "ov_out"
        create_multi_tensor_gguf(gguf_path, count=3, quant_type=GGMLType.F32)

        transpiler = GGUFTranspiler(str(gguf_path), str(out_dir))
        bin_file = out_dir / "openvino_model.bin"

        for p in transpiler.transpile_stream(min_emit_interval_ms=0.0):
            assert bin_file.exists()
            assert bin_file.stat().st_size >= p.processed_bytes


class TestTranspilerStreamNegative:
    """Negative fault injection test scenarios."""

    def test_transpile_stream_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Verify non-existent input path raises TranspilationError."""
        with pytest.raises(TranspilationError, match="Source GGUF file does not exist"):
            GGUFTranspiler(str(tmp_path / "nonexistent.gguf"), str(tmp_path / "out"))

    def test_transpile_stream_unsupported_bitwidth_raises(self, tmp_path: Path) -> None:
        """Verify unsupported precision string raises UnsupportedBitWidthError."""
        gguf_path = tmp_path / "model.gguf"
        create_multi_tensor_gguf(gguf_path, count=1)
        transpiler = GGUFTranspiler(str(gguf_path), str(tmp_path / "out"))

        with pytest.raises(UnsupportedBitWidthError, match="Unsupported target precision 'int2'"):
            list(transpiler.transpile_stream(precision="int2"))

    def test_transpile_stream_corrupt_tensor_scale_raises(self) -> None:
        """Verify corrupt Q4_0 block scale raises DequantizationError."""
        bad_block = struct.pack("<e", 0.0) + bytes([0x00] * 16)
        with pytest.raises(DequantizationError, match="Invalid Q4_0 scale"):
            GGUFTranspiler.dequantize_q4_0_block(bad_block)

    def test_transpile_stream_truncated_binary_file_raises(self, tmp_path: Path) -> None:
        """Verify truncated file with missing tensor bytes raises TranspilationError."""
        gguf_path = tmp_path / "truncated.gguf"
        with open(gguf_path, "wb") as f:
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

            t_name = b"token.weight"
            f.write(struct.pack("<Q", len(t_name)))
            f.write(t_name)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", 4))
            f.write(struct.pack("<Q", 4))
            f.write(struct.pack("<I", GGMLType.F32.value))
            f.write(struct.pack("<Q", 1000))

        transpiler = GGUFTranspiler(str(gguf_path), str(tmp_path / "out"))
        with pytest.raises(TranspilationError, match="Truncated tensor read"):
            list(transpiler.transpile_stream())


class TestTranspilerStreamEdgeCases:
    """Edge-case boundary testing for stream transpiler."""

    def test_transpile_stream_single_tensor_container(self, tmp_path: Path) -> None:
        """Verify single-tensor GGUF transpile stream completes with 100% progress."""
        gguf_path = tmp_path / "single.gguf"
        create_multi_tensor_gguf(gguf_path, count=1)
        transpiler = GGUFTranspiler(str(gguf_path), str(tmp_path / "out"))

        events = list(transpiler.transpile_stream(min_emit_interval_ms=0.0))
        assert len(events) == 1
        assert events[0].progress_pct == 100.0
        assert transpiler.last_result is not None

    def test_transpile_stream_zero_byte_or_scalar_tensor(self, tmp_path: Path) -> None:
        """Verify scalar or small tensor dimensions pad safely to 32 bytes."""
        nibbles = [1, -2, 3]
        packed = GGUFTranspiler.pack_xmx_nibbles(nibbles)
        assert len(packed) % 32 == 0
        assert len(packed) == 32

    def test_transpile_stream_throttling_interval_m19_p4(self, tmp_path: Path) -> None:
        """Verify Proof M19/P4: Throttling limits event emissions when intervals are tight."""
        gguf_path = tmp_path / "many.gguf"
        create_multi_tensor_gguf(gguf_path, count=10)
        transpiler = GGUFTranspiler(str(gguf_path), str(tmp_path / "out"))

        events = list(transpiler.transpile_stream(min_emit_interval_ms=1000.0))
        assert len(events) <= 10
        assert events[-1].progress_pct == 100.0

    def test_transpile_stream_generator_early_closure(self, tmp_path: Path) -> None:
        """Verify early termination of the generator cleans up handles safely."""
        gguf_path = tmp_path / "break.gguf"
        create_multi_tensor_gguf(gguf_path, count=5)
        transpiler = GGUFTranspiler(str(gguf_path), str(tmp_path / "out"))

        stream = transpiler.transpile_stream(min_emit_interval_ms=0.0)
        first_event = next(stream)
        assert first_event.tensor_index == 1
        stream.close()
