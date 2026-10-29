"""Unit tests for the heterogeneous dual-silicon speculative decoding engine.

Implements the 20-test Triad (Positive, Negative, Edge-Case & Chaos) verifying
engine lifecycle, dynamic lookahead horizon adaptation, hardware fallbacks,
and streaming backpressure in accordance with Rule 41 and Rule 12 standards.
"""

import time
import unittest.mock as mock
from pathlib import Path
from typing import Iterator

import pytest
from pydantic import ValidationError

from arc_runner.core.exceptions import (
    ArcRunnerError,
    ModelLoadError,
    ModelNotFoundError,
)
from arc_runner.engine.models import (
    EngineTelemetry,
    GenerationConfig,
    SpeculativeConfig,
    TokenChunk,
)
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine


def test_speculative_config_valid_defaults() -> None:
    """Verify that SpeculativeConfig initializes with correct default hyperparameters."""
    config = SpeculativeConfig(draft_model_path="valid_path")
    assert config.draft_model_path == "valid_path"
    assert config.draft_device == "NPU"
    assert config.num_assistant_tokens == 4
    assert config.assistant_confidence_threshold == 0.5
    assert config.dynamic_gamma is True
    assert config.min_gamma == 1
    assert config.max_gamma == 8


def test_speculative_config_custom_bounds() -> None:
    """Verify custom parameter initialization within valid numerical bounds."""
    config = SpeculativeConfig(
        draft_model_path="/custom/path",
        draft_device="GPU",
        num_assistant_tokens=6,
        assistant_confidence_threshold=0.7,
        dynamic_gamma=False,
        min_gamma=2,
        max_gamma=12,
    )
    assert config.draft_device == "GPU"
    assert config.num_assistant_tokens == 6
    assert config.assistant_confidence_threshold == 0.7
    assert config.dynamic_gamma is False
    assert config.min_gamma == 2
    assert config.max_gamma == 12


def test_hybrid_speculative_engine_init() -> None:
    """Assert that HybridSpeculativeEngine instantiates in a clean unloaded state."""
    engine = HybridSpeculativeEngine()
    assert engine.is_loaded is False
    assert engine.model_path == ""
    assert engine.draft_model_path == ""
    assert engine.device == "GPU"
    assert engine.draft_device == "NPU"
    assert engine.current_gamma == 4


def test_hybrid_speculative_load_model_npu_success(tmp_path: Path) -> None:
    """Verify successful dual-silicon model load passing draft model to LLMPipeline."""
    target_dir = tmp_path / "target_model"
    draft_dir = tmp_path / "draft_model"
    target_dir.mkdir()
    draft_dir.mkdir()

    mock_draft_handle = mock.MagicMock()
    mock_pipeline = mock.MagicMock()

    with mock.patch("openvino_genai.draft_model", return_value=mock_draft_handle) as mock_draft_fn:
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline) as mock_pipe_cls:
            engine = HybridSpeculativeEngine()
            spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir), draft_device="NPU")
            engine.load_model(str(target_dir), speculative_config=spec_cfg, device="GPU")

            assert engine.is_loaded is True
            assert engine.draft_model_path == str(draft_dir.resolve())
            mock_draft_fn.assert_called_once_with(str(draft_dir.resolve()), "NPU")
            mock_pipe_cls.assert_called_once_with(
                str(target_dir.resolve()),
                "GPU",
                draft_model=mock_draft_handle,
            )


def test_speculative_generate_synchronous_success(tmp_path: Path) -> None:
    """Verify that synchronous speculative generation returns expected text and telemetry."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = "Dual silicon execution verified."

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir))
            engine.load_model(str(target_dir), speculative_config=spec_cfg)

            output = engine.generate("Prompt query")
            assert output == "Dual silicon execution verified."
            telemetry = engine.get_telemetry()
            assert telemetry.is_speculative is True
            assert telemetry.draft_device == "NPU"
            assert telemetry.metrics_source == "unavailable"
            assert telemetry.is_tps_valid is False


def test_speculative_generate_stream_tokens_emitted(tmp_path: Path) -> None:
    """Verify that streaming generation yields incremental chunks and finishes cleanly."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    def fake_generate(prompt: str, config: object = None, streamer: object = None) -> None:
        if callable(streamer):
            streamer("Speculative ")
            streamer("Token ")
            streamer("Stream")

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.side_effect = fake_generate

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            engine.load_model(str(target_dir), SpeculativeConfig(draft_model_path=str(draft_dir)))

            chunks = list(engine.generate_stream("Stream prompt"))
            assert len(chunks) == 4
            text_result = "".join(c.text for c in chunks)
            assert text_result == "Speculative Token Stream"
            assert chunks[-1].is_finished is True
            assert chunks[-1].finish_reason == "stop"


def test_speculative_telemetry_metrics_computed(tmp_path: Path) -> None:
    """Assert that speculative telemetry never reports invented acceptance or speedup values."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = "One two three four five six seven eight nine ten"

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            engine.load_model(str(target_dir), SpeculativeConfig(draft_model_path=str(draft_dir)))
            engine.generate("Test prompt")

            telem = engine.get_telemetry()
            assert telem.is_speculative is True
            assert telem.metrics_source == "unavailable"
            assert telem.draft_tokens_total == 0
            assert telem.accepted_tokens_total == 0
            assert telem.acceptance_rate == 0.0
            assert telem.speculative_speedup == 1.0


def test_dynamic_gamma_adaptation_step(tmp_path: Path) -> None:
    """Verify that dynamic gamma updates based on simulated acceptance rates."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock.MagicMock()):
            engine = HybridSpeculativeEngine()
            engine.load_model(
                str(target_dir),
                SpeculativeConfig(
                    draft_model_path=str(draft_dir),
                    num_assistant_tokens=4,
                    min_gamma=1,
                    max_gamma=6,
                    dynamic_gamma=True,
                ),
            )
            assert engine.current_gamma == 4

            engine._adapt_gamma(accepted_count=4, draft_count=4)
            assert engine.current_gamma == 5

            engine._adapt_gamma(accepted_count=1, draft_count=5)
            assert engine.current_gamma == 4


def test_speculative_config_invalid_gamma_zero_raises() -> None:
    """Assert that zero assistant tokens raises a Pydantic validation error."""
    with pytest.raises(ValidationError):
        SpeculativeConfig(draft_model_path="path", num_assistant_tokens=0)


def test_speculative_config_negative_confidence_raises() -> None:
    """Assert that negative confidence threshold raises a Pydantic validation error."""
    with pytest.raises(ValidationError):
        SpeculativeConfig(draft_model_path="path", assistant_confidence_threshold=-0.2)


def test_speculative_generate_without_load_raises() -> None:
    """Assert that calling generate() prior to load_model() raises ArcRunnerError."""
    engine = HybridSpeculativeEngine()
    with pytest.raises(ArcRunnerError) as exc_info:
        engine.generate("Query")
    assert exc_info.value.error_code == "MODEL_NOT_LOADED"


def test_speculative_load_missing_draft_path_raises(tmp_path: Path) -> None:
    """Assert that nonexistent draft directory raises ModelNotFoundError."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    missing_draft = tmp_path / "nonexistent_draft"

    engine = HybridSpeculativeEngine()
    spec_cfg = SpeculativeConfig(draft_model_path=str(missing_draft))

    with pytest.raises(ModelNotFoundError):
        engine.load_model(str(target_dir), speculative_config=spec_cfg)


def test_speculative_load_missing_target_path_raises(tmp_path: Path) -> None:
    """Assert that nonexistent target directory raises ModelNotFoundError."""
    missing_target = tmp_path / "nonexistent_target"
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()

    engine = HybridSpeculativeEngine()
    spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir))

    with pytest.raises(ModelNotFoundError):
        engine.load_model(str(missing_target), speculative_config=spec_cfg)


def test_speculative_draft_model_runtime_error_wraps(tmp_path: Path) -> None:
    """Assert that OpenVINO backend exceptions wrap cleanly into ModelLoadError."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", side_effect=RuntimeError("MCDM driver fault")):
        engine = HybridSpeculativeEngine()
        spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir))

        with pytest.raises(ModelLoadError) as exc_info:
            engine.load_model(str(target_dir), speculative_config=spec_cfg)
        assert "MCDM driver fault" in str(exc_info.value.message) or "Failed to initialize draft model" in str(exc_info.value.message)


def test_npu_unavailable_graceful_fallback_to_cpu(tmp_path: Path) -> None:
    """Verify that when NPU draft fails, engine logs warning and compiles on CPU."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    def draft_side_effect(path: str, device: str) -> mock.MagicMock:
        if device == "NPU":
            raise RuntimeError("NPU device not found")
        return mock.MagicMock()

    mock_pipeline = mock.MagicMock()

    with mock.patch("openvino_genai.draft_model", side_effect=draft_side_effect):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir), draft_device="NPU")
            engine.load_model(str(target_dir), speculative_config=spec_cfg)

            assert engine.is_loaded is True
            assert engine.draft_device == "CPU"


def test_npu_and_cpu_unavailable_fallback_to_baseline(tmp_path: Path) -> None:
    """Verify that if all draft candidates fail, ModelLoadError is raised indicating failure."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", side_effect=RuntimeError("Device compilation failure")):
        engine = HybridSpeculativeEngine()
        spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir), draft_device="NPU")

        with pytest.raises(ModelLoadError):
            engine.load_model(str(target_dir), speculative_config=spec_cfg)


def test_dynamic_gamma_clamping_at_min_and_max(tmp_path: Path) -> None:
    """Verify that gamma never drops below min_gamma or exceeds max_gamma under extreme inputs."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock.MagicMock()):
            engine = HybridSpeculativeEngine()
            engine.load_model(
                str(target_dir),
                SpeculativeConfig(
                    draft_model_path=str(draft_dir),
                    num_assistant_tokens=4,
                    min_gamma=2,
                    max_gamma=5,
                ),
            )

            for _ in range(10):
                engine._adapt_gamma(accepted_count=10, draft_count=10)
            assert engine.current_gamma == 5

            for _ in range(10):
                engine._adapt_gamma(accepted_count=0, draft_count=10)
            assert engine.current_gamma == 2


def test_speculative_streaming_early_cancellation(tmp_path: Path) -> None:
    """Verify that breaking out of the generator safely signals the streamer callback to abort."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    tokens_yielded = 0

    def fake_generate(prompt: str, config: object = None, streamer: object = None) -> None:
        nonlocal tokens_yielded
        if callable(streamer):
            for i in range(50):
                tokens_yielded += 1
                time.sleep(0.01)
                should_abort = streamer(f"Token_{i}")
                if should_abort:
                    break

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.side_effect = fake_generate

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            engine.load_model(str(target_dir), SpeculativeConfig(draft_model_path=str(draft_dir)))

            for idx, chunk in enumerate(engine.generate_stream("Prompt")):
                if idx == 1:
                    break

            assert tokens_yielded < 50


def test_zero_acceptance_rate_handling(tmp_path: Path) -> None:
    """Verify that engine maintains numerical stability when zero draft tokens are accepted."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = "One two three"

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
            engine = HybridSpeculativeEngine()
            engine.load_model(str(target_dir), SpeculativeConfig(draft_model_path=str(draft_dir)))

            engine._adapt_gamma(accepted_count=0, draft_count=0)
            assert engine.current_gamma == 4

            engine._adapt_gamma(accepted_count=0, draft_count=5)
            assert engine.current_gamma == 3


def test_repeated_speculative_load_unload_memory_clean(tmp_path: Path) -> None:
    """Verify that repeated speculative load and unload cycles maintain clean state."""
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()

    with mock.patch("openvino_genai.draft_model", return_value=mock.MagicMock()):
        with mock.patch("openvino_genai.LLMPipeline", return_value=mock.MagicMock()):
            engine = HybridSpeculativeEngine()
            spec_cfg = SpeculativeConfig(draft_model_path=str(draft_dir))

            for _ in range(5):
                engine.load_model(str(target_dir), speculative_config=spec_cfg)
                assert engine.is_loaded is True
                engine.unload()
                assert engine.is_loaded is False
                assert engine.model_path == ""
                assert engine.draft_model_path == ""
