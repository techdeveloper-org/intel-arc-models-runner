"""Unit, negative, and edge-case triad tests for Heterogeneous NPU-GPU Speculative Execution.

Tests heterogeneous device routing, NPU compatibility validation, CPU fallback on compiler
failure, low-alpha circuit breaker tripping, and telemetry generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from arc_runner.core.exceptions import ModelNotFoundError
from arc_runner.engine.models import GenerationConfig, SpeculativeConfig
from arc_runner.engine.npu_adapter import NpuCompatibilityReport
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine


class TestSpeculativeNpuRouting:
    """Test suite for heterogeneous NPU draft and GPU verifier co-execution."""

    def test_should_route_draft_to_npu_when_model_compatible_p_spec_1(self, tmp_path: Path) -> None:
        """P-SPEC-1: Verify draft model compiles on NPU when NpuModelAdapter reports compatible."""
        target_dir = tmp_path / "target_model"
        target_dir.mkdir()
        (target_dir / "openvino_tokenizer.xml").write_text("<xml/>", encoding="utf-8")

        draft_dir = tmp_path / "draft_model"
        draft_dir.mkdir()
        (draft_dir / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")

        spec_config = SpeculativeConfig(
            draft_model_path=str(draft_dir),
            draft_device="NPU",
            num_assistant_tokens=4,
        )

        mock_report = NpuCompatibilityReport(
            is_compatible=True,
            model_path=draft_dir / "openvino_model.xml",
            detected_architecture="3720",
        )

        mock_draft_handle = MagicMock()
        mock_pipeline = MagicMock()

        engine = HybridSpeculativeEngine()
        with patch("arc_runner.engine.npu_adapter.NpuModelAdapter.inspect_model_compatibility", return_value=mock_report):
            with patch("openvino_genai.draft_model", return_value=mock_draft_handle) as mock_ov_draft:
                with patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
                    engine.load_model(str(target_dir), speculative_config=spec_config, device="GPU")

                    assert engine.draft_device == "NPU"
                    assert engine.is_loaded is True
                    mock_ov_draft.assert_called_once_with(str(draft_dir), "NPU")

    def test_should_fallback_to_cpu_when_draft_incompatible_n_spec_1(self, tmp_path: Path) -> None:
        """N-SPEC-1: Verify incompatible draft model (e.g. u4 ops) falls back to CPU draft execution."""
        target_dir = tmp_path / "target_model"
        target_dir.mkdir()
        (target_dir / "openvino_tokenizer.xml").write_text("<xml/>", encoding="utf-8")

        draft_dir = tmp_path / "draft_model"
        draft_dir.mkdir()
        (draft_dir / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")

        spec_config = SpeculativeConfig(
            draft_model_path=str(draft_dir),
            draft_device="NPU",
            num_assistant_tokens=4,
        )

        mock_report = NpuCompatibilityReport(
            is_compatible=False,
            model_path=draft_dir / "openvino_model.xml",
            detected_architecture="3720",
            rejection_reason="Found asymmetric u4 dequantize ops",
        )

        mock_draft_handle = MagicMock()
        mock_pipeline = MagicMock()

        engine = HybridSpeculativeEngine()
        with patch("arc_runner.engine.npu_adapter.NpuModelAdapter.inspect_model_compatibility", return_value=mock_report):
            with patch("openvino_genai.draft_model", return_value=mock_draft_handle) as mock_ov_draft:
                with patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
                    engine.load_model(str(target_dir), speculative_config=spec_config, device="GPU")

                    assert engine._draft_device == "CPU"
                    mock_ov_draft.assert_called_once_with(str(draft_dir), "CPU")

    def test_should_reject_missing_target_model_n_spec_3(self, tmp_path: Path) -> None:
        """N-SPEC-3: Verify non-existent target model path raises ModelNotFoundError immediately."""
        engine = HybridSpeculativeEngine()
        missing_target = tmp_path / "non_existent_target"

        with pytest.raises(ModelNotFoundError):
            engine.load_model(str(missing_target), device="GPU")

    def test_should_record_npu_active_in_telemetry_p_spec_2(self, tmp_path: Path) -> None:
        """P-SPEC-2: Verify generate() populates npu_active and fused_silicon_tier in EngineTelemetry."""
        target_dir = tmp_path / "target_model"
        target_dir.mkdir()
        (target_dir / "openvino_tokenizer.xml").write_text("<xml/>", encoding="utf-8")

        draft_dir = tmp_path / "draft_model"
        draft_dir.mkdir()
        (draft_dir / "openvino_model.xml").write_text("<xml/>", encoding="utf-8")

        spec_config = SpeculativeConfig(
            draft_model_path=str(draft_dir),
            draft_device="NPU",
            num_assistant_tokens=4,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.get_tokenizer.return_value.encode.return_value.input_ids.data.flatten.return_value.tolist.return_value = [1, 2, 3]

        engine = HybridSpeculativeEngine()
        with patch("arc_runner.engine.npu_adapter.NpuModelAdapter.inspect_model_compatibility", return_value=NpuCompatibilityReport(True, draft_dir, "3720")):
            with patch("openvino_genai.draft_model", return_value=MagicMock()):
                with patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
                    engine.load_model(str(target_dir), speculative_config=spec_config, device="GPU")

        def fake_generate(prompts, ov_config=None):
            res = MagicMock()
            res.texts = ["Hello World"]
            return res

        mock_pipeline.generate.side_effect = fake_generate

        output_text = engine.generate("Test prompt", GenerationConfig(max_new_tokens=10))
        assert output_text == "Hello World"

        telemetry = engine.get_telemetry()
        assert telemetry.npu_active is True
        assert telemetry.draft_device == "NPU"
        assert telemetry.fused_silicon_tier == "LAPTOP_UMA_FUSED"
        assert telemetry.circuit_breaker_tripped is False

    def test_circuit_breaker_trips_when_alpha_below_threshold_e_spec_1(self) -> None:
        """E-SPEC-1: Verify Low-Alpha Circuit Breaker trips when rolling acceptance falls below threshold."""
        engine = HybridSpeculativeEngine()
        engine._spec_config = SpeculativeConfig(draft_model_path="dummy", num_assistant_tokens=4, circuit_breaker_threshold=0.55)
        engine._circuit_breaker_threshold = 0.55
        engine._pipeline = MagicMock()

        for _ in range(10):
            engine._adapt_gamma(accepted_count=0, draft_count=4)

        assert engine._circuit_breaker_tripped is True
        assert engine._current_gamma == 1
