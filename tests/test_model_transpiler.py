"""Unit, negative, and edge-case triad tests for HardwareCompatibilityTranspiler.

Tests cross-silicon strategy selection (NpuInt4SymStrategy, GpuInt4AsymStrategy, CpuInt8Strategy),
alignment padding, zero-point elimination, and error handling across model inputs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from arc_runner.quant.model_transpiler import (
    CpuInt8Strategy,
    GpuInt4AsymStrategy,
    HardwareCompatibilityTranspiler,
    NpuInt4SymStrategy,
)
from arc_runner.quant.transpiler import TranspilationError, TranspilationResult, UnsupportedBitWidthError


class TestHardwareTranspilerStrategies:
    """Test suite for silicon-specific quantization and memory packing strategies."""

    def test_npu_strategy_packs_with_32_byte_alignment_p_trans_1(self) -> None:
        """P-TRANS-1: Verify NpuInt4SymStrategy enforces 32-byte alignment and symmetric zero bias."""
        strategy = NpuInt4SymStrategy()
        raw_bytes = b"\x01\x02\x03\x04" * 7

        packed_bytes, meta = strategy.process_weights(raw_bytes, ggml_type=2, dimensions=(28,))

        assert len(packed_bytes) % 32 == 0
        assert meta["zero_point"] == 0
        assert meta["symmetric"] is True
        assert meta["cmx_tileable"] is True
        assert strategy.target_device == "NPU"
        assert strategy.target_precision == "INT4_SYM"

    def test_gpu_strategy_packs_with_64_byte_alignment_p_trans_2(self) -> None:
        """P-TRANS-2: Verify GpuInt4AsymStrategy packs into 64-byte aligned blocks for Xe Vector Engines."""
        strategy = GpuInt4AsymStrategy()
        raw_bytes = b"\x05" * 50

        packed_bytes, meta = strategy.process_weights(raw_bytes, ggml_type=2, dimensions=(50,))

        assert len(packed_bytes) % 64 == 0
        assert meta["group_size"] == 128
        assert meta["xe_vector_optimized"] is True
        assert strategy.target_device == "GPU"
        assert strategy.target_precision == "INT4_ASYM"

    def test_cpu_strategy_packs_for_vnni_p_trans_5(self) -> None:
        """P-TRANS-5: Verify CpuInt8Strategy packs into INT8 format compatible with AVX-VNNI."""
        strategy = CpuInt8Strategy()
        raw_bytes = b"\x7F" * 30

        packed_bytes, meta = strategy.process_weights(raw_bytes, ggml_type=0, dimensions=(30,))

        assert len(packed_bytes) % 64 == 0
        assert meta["precision"] == "i8"
        assert meta["vnni_compatible"] is True
        assert strategy.target_device == "CPU"


class TestHardwareCompatibilityTranspiler:
    """Test suite for HardwareCompatibilityTranspiler orchestration and error handling."""

    def test_strategy_selection_routes_to_correct_silicon_p_trans_3(self) -> None:
        """P-TRANS-3: Verify select_strategy routes devices to NPU, GPU, and CPU strategies."""
        transpiler = HardwareCompatibilityTranspiler()

        npu_strat = transpiler.select_strategy("NPU")
        assert isinstance(npu_strat, NpuInt4SymStrategy)

        gpu_strat = transpiler.select_strategy("GPU")
        assert isinstance(gpu_strat, GpuInt4AsymStrategy)

        cpu_strat = transpiler.select_strategy("CPU")
        assert isinstance(cpu_strat, CpuInt8Strategy)

    def test_should_reject_missing_source_model_n_trans_1(self, tmp_path: Path) -> None:
        """N-TRANS-1: Verify non-existent model path raises TranspilationError."""
        transpiler = HardwareCompatibilityTranspiler()
        missing_path = tmp_path / "does_not_exist.gguf"

        with pytest.raises(TranspilationError) as exc_info:
            transpiler.transpile(missing_path, tmp_path / "out", target_device="NPU")
        assert "does not exist" in str(exc_info.value).lower()

    def test_transpile_creates_npu_compatibility_manifest_p_trans_4(self, tmp_path: Path) -> None:
        """P-TRANS-4: Verify transpile generates npu_compatibility.json manifest with target metadata."""
        mock_result = TranspilationResult(
            output_dir=tmp_path / "out",
            model_name="mock_model",
            total_tensors=10,
            source_format="GGUF (Q4_0)",
            target_precision="INT4_SYM",
            xml_path=tmp_path / "out" / "openvino_model.xml",
            bin_path=tmp_path / "out" / "openvino_model.bin",
            total_bytes=1024,
            duration_ms=100.0,
        )

        dummy_src = tmp_path / "model.gguf"
        dummy_src.write_bytes(b"GGUF_MOCK")
        out_dir = tmp_path / "out"

        transpiler = HardwareCompatibilityTranspiler()
        with patch.object(transpiler._strategies["NPU"], "process_weights", return_value=(b"data", {})):
            with patch("arc_runner.quant.model_transpiler.GGUFTranspiler") as mock_gguf_transpiler_cls:
                mock_instance = mock_gguf_transpiler_cls.return_value
                mock_instance.transpile.return_value = mock_result

                res = transpiler.transpile(dummy_src, out_dir, target_device="NPU", precision="INT4_SYM")

                assert res.target_precision == "INT4_SYM"
                manifest_file = out_dir / "npu_compatibility.json"
                assert manifest_file.exists()
                content = manifest_file.read_text(encoding="utf-8")
                assert "INT4_SYM" in content
                assert "NPU_INT4_SYMMETRIC" in content

    def test_should_handle_path_with_spaces_and_unicode_e_trans_6(self, tmp_path: Path) -> None:
        """E-TRANS-6: Verify paths containing spaces, parentheses, and Unicode are accepted."""
        unicode_src = tmp_path / "model [v1.0] 🚀" / "test.gguf"
        unicode_src.parent.mkdir(parents=True, exist_ok=True)
        unicode_src.write_bytes(b"DUMMY_BYTES")

        out_dir = tmp_path / "output dir (special)"
        transpiler = HardwareCompatibilityTranspiler()

        mock_result = TranspilationResult(
            output_dir=out_dir,
            model_name="unicode_model",
            total_tensors=1,
            source_format="GGUF",
            target_precision="INT4_ASYM",
            xml_path=out_dir / "openvino_model.xml",
            bin_path=out_dir / "openvino_model.bin",
            total_bytes=100,
            duration_ms=10.0,
        )

        with patch("arc_runner.quant.model_transpiler.GGUFTranspiler") as mock_gguf_cls:
            mock_gguf_cls.return_value.transpile.return_value = mock_result
            res = transpiler.transpile(unicode_src, out_dir, target_device="GPU")
            assert res.output_dir == out_dir
            assert (out_dir / "npu_compatibility.json").exists()
