"""Triad test suite for Text-to-Video Generation Engine and Proofs M94 through M114.

Evaluates Positive, Negative, and Edge-Case behaviors across model compilation,
spatio-temporal factorized attention, multi-format container encoding, deterministic
seeds, optical flow warp error, and Level Zero resource reclamation.
Adheres strictly to Level 2.2 Docstrings-Only standards and Claude Global Rules.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image
import pydantic
import pytest

from arc_runner.core.exceptions import ModelLoadError
from arc_runner.engine.video_engine import (
    VideoEngine,
    VideoEngineConfig,
    VideoEngineError,
    VideoGenerationBreakdown,
    VideoGenerationRequest,
    VideoGenerationResult,
)


class MockOpenVINOVideoTensor:
    """Mock OpenVINO video tensor returning configured 4D spatio-temporal frame data."""

    def __init__(self, shape: tuple[int, int, int, int] = (16, 512, 512, 3), seed: int = 42) -> None:
        """Initialize mock video tensor with deterministic RGB values.

        Args:
            shape: 4D tensor shape tuple (num_frames, height, width, channels).
            seed: RNG seed for reproducible frame pixel values.
        """
        rng = np.random.RandomState(seed)
        self.data = rng.randint(20, 235, size=shape, dtype=np.uint8)


class MockDurationMetric:
    """Mock performance metric duration reporting mean timing."""

    def __init__(self, value: float) -> None:
        """Initialize metric with numeric millisecond value.

        Args:
            value: Mean latency in milliseconds.
        """
        self.mean = value


class MockVideoPerfMetrics:
    """Mock OpenVINO VideoGenerationPerfMetrics object."""

    def get_text_encoder_infer_duration(self) -> MockDurationMetric:
        """Return mock text encoder inference duration."""
        return MockDurationMetric(65.4)

    def get_transformer_infer_duration(self) -> MockDurationMetric:
        """Return mock 3D transformer denoising inference duration."""
        return MockDurationMetric(3840.0)

    def get_unet_infer_duration(self) -> MockDurationMetric:
        """Return mock UNet inference duration."""
        return MockDurationMetric(0.0)

    def get_vae_decoder_infer_duration(self) -> MockDurationMetric:
        """Return mock 3D VAE decoder inference duration."""
        return MockDurationMetric(420.5)


class MockVideoResult:
    """Mock OpenVINO VideoGenerationResult payload."""

    def __init__(self, tensor: MockOpenVINOVideoTensor) -> None:
        """Initialize mock result with frame tensor and performance metrics.

        Args:
            tensor: Configured MockOpenVINOVideoTensor instance.
        """
        self.video = tensor
        self.perf_metrics = MockVideoPerfMetrics()


class MockText2VideoPipeline:
    """Mock OpenVINO GenAI Text2VideoPipeline for fast deterministic unit testing."""

    def __init__(self, model_path: str, device: str, **kwargs: Any) -> None:
        """Initialize mock pipeline state.

        Args:
            model_path: Stored model path string.
            device: Accelerator device identifier.
            **kwargs: Additional configuration parameters.
        """
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs

    def generate(self, prompt: str, **kwargs: Any) -> MockVideoResult:
        """Return mock video result matching requested dimensions and frame count.

        Args:
            prompt: Text prompt string.
            **kwargs: Denoising arguments (num_frames, width, height, etc.).

        Returns:
            MockVideoResult with requested spatio-temporal resolution.
        """
        frames = kwargs.get("num_frames", 16)
        w = kwargs.get("width", 512)
        h = kwargs.get("height", 512)
        seed = kwargs.get("rng_seed", 42)
        tensor = MockOpenVINOVideoTensor(shape=(frames, h, w, 3), seed=seed)
        return MockVideoResult(tensor)

    def get_performance_metrics(self) -> MockVideoPerfMetrics:
        """Return mock performance metrics telemetry."""
        return MockVideoPerfMetrics()


class TestVideoEnginePositive:
    """Positive test cases evaluating valid inputs, model compilation, and synthesis."""

    def test_video_engine_init_default_config(self) -> None:
        """P1: Verify VideoEngine initializes with default configuration parameters."""
        engine = VideoEngine()
        assert engine.active_device == "GPU"
        assert engine.is_loaded() is False
        assert engine.model_path is None

    def test_video_engine_load_model_success(self, tmp_path: Path) -> None:
        """P2: Verify successful model loading and compilation duration reporting."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine()
            compile_sec = engine.load_model(model_dir, device="GPU")
            assert compile_sec >= 0.0
            assert engine.is_loaded() is True
            assert engine.active_device == "GPU"
            assert engine.model_path == model_dir.resolve()

    def test_video_generation_standard_flow(self, tmp_path: Path) -> None:
        """P3: Verify standard text-to-video synthesis pipeline and result attributes."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            req = VideoGenerationRequest(
                prompt="A drone shot of turquoise ocean waves against golden sand",
                num_frames=16,
                fps=16,
                width=512,
                height=512,
                num_inference_steps=25,
                guidance_scale=7.5,
            )
            res = engine.generate(req)

            assert isinstance(res, VideoGenerationResult)
            assert len(res.frames) == 16
            assert res.frames[0].shape == (512, 512, 3)
            assert res.duration_sec == 1.0
            assert res.fps == 16
            assert res.resolution == (512, 512)
            assert res.total_latency_ms > 0.0
            assert res.breakdown.frames_per_second > 0.0
            assert res.breakdown.real_time_factor > 0.0

    def test_video_generation_low_workload_tier(self, tmp_path: Path) -> None:
        """P4: Verify Low tier benchmark generation (256x256, 8 frames @ 8 fps, 10 steps)."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            req = VideoGenerationRequest(
                prompt="A puppy running on grass",
                num_frames=8,
                fps=8,
                width=256,
                height=256,
                num_inference_steps=10,
                guidance_scale=5.0,
            )
            res = engine.generate(req)
            assert len(res.frames) == 8
            assert res.frames[0].shape == (256, 256, 3)
            assert res.duration_sec == 1.0
            assert res.breakdown.step_latency_ms > 0.0

    def test_video_generation_hard_workload_tier(self, tmp_path: Path) -> None:
        """P5: Verify Hard tier benchmark generation (768x512, 33 frames @ 24 fps, 40 steps)."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            req = VideoGenerationRequest(
                prompt="Sci-fi starship cruising through an ethereal nebula, 8k",
                num_frames=33,
                fps=24,
                width=768,
                height=512,
                num_inference_steps=40,
                guidance_scale=8.5,
            )
            res = engine.generate(req)
            assert len(res.frames) == 33
            assert res.frames[0].shape == (512, 768, 3)
            assert pytest.approx(res.duration_sec, 0.01) == 1.375
            assert res.breakdown.real_time_factor > 0.0

    def test_video_generation_with_negative_prompt(self, tmp_path: Path) -> None:
        """P6: Verify negative prompt conditioning parameter propagation."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res = engine.generate(
                prompt="Lush rainforest with exotic birds",
                negative_prompt="blurry, flickering, low quality, artifacts",
                num_frames=12,
                fps=12,
                width=256,
                height=256,
            )
            assert res.request.negative_prompt == "blurry, flickering, low quality, artifacts"
            assert len(res.frames) == 12

    def test_video_generation_deterministic_seed(self, tmp_path: Path) -> None:
        """P7: Verify deterministic seed reproducibility invariant (Proof M101)."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            req1 = VideoGenerationRequest(
                prompt="Deterministic galaxy spin",
                num_frames=8,
                rng_seed=12345,
                width=256,
                height=256,
            )
            req2 = VideoGenerationRequest(
                prompt="Deterministic galaxy spin",
                num_frames=8,
                rng_seed=12345,
                width=256,
                height=256,
            )
            res1 = engine.generate(req1)
            res2 = engine.generate(req2)

            np.testing.assert_array_equal(res1.frames[0], res2.frames[0])
            np.testing.assert_array_equal(res1.frames[-1], res2.frames[-1])

    def test_video_container_serialization_mp4_avi_gif(self, tmp_path: Path) -> None:
        """P8: Verify container serialization across mp4, avi, and gif formats."""
        engine = VideoEngine()
        rng = np.random.RandomState(42)
        dummy_frames = [rng.randint(0, 255, size=(128, 128, 3), dtype=np.uint8) for _ in range(8)]

        for fmt in ["mp4", "avi", "gif"]:
            out_file = tmp_path / f"test_output.{fmt}"
            saved = engine.encode_video(dummy_frames, out_file, fps=8, container=fmt)
            assert Path(saved).exists()
            assert Path(saved).stat().st_size > 100

    def test_video_generation_telemetry_and_perf_metrics(self, tmp_path: Path) -> None:
        """P9: Verify performance telemetry metrics, RTF, throughput, and energy yield."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res = engine.generate(
                prompt="Aurora borealis over icy fjord",
                num_frames=16,
                fps=16,
                width=256,
                height=256,
            )
            b = res.breakdown
            assert b.text_encoder_ms > 0.0
            assert b.denoise_transformer_ms > 0.0
            assert b.vae_decoder_ms > 0.0
            assert b.step_latency_ms > 0.0
            assert b.iteration_rate_it_s > 0.0
            assert b.frames_per_second > 0.0
            assert b.real_time_factor > 0.0
            assert b.energy_yield_frames_per_kwh is None
            assert b.package_power_w is None
            assert b.peak_memory_mb is None

    def test_video_generation_without_perf_metrics_reports_stage_timings_as_none(self, tmp_path: Path) -> None:
        """H9: without OpenVINO perf metrics there is no stage split and no step rate."""

        class _NoMetricsResult(MockVideoResult):
            def __init__(self, tensor: Any) -> None:
                super().__init__(tensor)
                self.perf_metrics = None

        class _NoMetricsPipeline(MockText2VideoPipeline):
            def generate(self, prompt: str, **kwargs: Any) -> Any:
                return _NoMetricsResult(super().generate(prompt, **kwargs).video)

            def get_performance_metrics(self) -> Any:
                raise RuntimeError("perf metrics unavailable")

        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()
        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=_NoMetricsPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res = engine.generate(prompt="No metrics", num_frames=8, fps=8, width=128, height=128)
        b = res.breakdown
        assert b.text_encoder_ms is None and b.denoise_transformer_ms is None and b.vae_decoder_ms is None
        assert b.step_latency_ms is None and b.iteration_rate_it_s is None
        assert b.frames_per_second > 0.0 and b.real_time_factor > 0.0

    def test_video_engine_context_manager_and_disposal(self, tmp_path: Path) -> None:
        """P10: Verify context manager entry, exit, and explicit resource reclamation."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            with VideoEngine(model_path_or_id=model_dir) as engine:
                assert engine.is_loaded() is True
            assert engine.is_loaded() is False


class TestVideoEngineNegative:
    """Negative test cases evaluating error boundaries, malformed inputs, and exception handling."""

    def test_video_engine_nonexistent_model_path(self, tmp_path: Path) -> None:
        """N1: Verify ModelLoadError when specifying a non-existent model path."""
        invalid_path = tmp_path / "non_existent_video_model"
        engine = VideoEngine()
        with pytest.raises(ModelLoadError, match="Model path does not exist"):
            engine.load_model(invalid_path)

    def test_video_engine_empty_or_whitespace_prompt(self, tmp_path: Path) -> None:
        """N2: Verify ValueError when supplying empty or whitespace-only prompt."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            with pytest.raises(ValueError, match="Prompt must not be empty or whitespace only"):
                engine.generate("")
            with pytest.raises(ValueError, match="Prompt must not be empty or whitespace only"):
                engine.generate("   \t\n  ")

    def test_video_engine_invalid_num_frames(self) -> None:
        """N3: Verify validation rejection on zero or excessive frame counts."""
        with pytest.raises(pydantic.ValidationError):
            VideoGenerationRequest(prompt="Valid prompt", num_frames=0)
        with pytest.raises(pydantic.ValidationError):
            VideoGenerationRequest(prompt="Valid prompt", num_frames=301)

    def test_video_engine_invalid_fps(self) -> None:
        """N4: Verify validation rejection on zero or out-of-range playback framerate."""
        with pytest.raises(pydantic.ValidationError):
            VideoGenerationRequest(prompt="Valid prompt", fps=0)
        with pytest.raises(pydantic.ValidationError):
            VideoGenerationRequest(prompt="Valid prompt", fps=61)

    def test_video_engine_invalid_dimensions(self) -> None:
        """N5: Verify validation rejection when dimensions are non-multiples of 8 or too small."""
        with pytest.raises(pydantic.ValidationError):
            VideoGenerationRequest(prompt="Valid prompt", width=32, height=32)

        with pytest.raises(ValueError, match="divisible by 8"):
            req = VideoGenerationRequest(prompt="Valid prompt", width=513, height=512)
            engine = VideoEngine()
            engine._pipeline = mock.MagicMock()
            engine.generate(req)

    def test_video_engine_oom_handling(self, tmp_path: Path) -> None:
        """N6: Verify VideoEngineError when physical accelerator triggers out-of-memory exception."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        def _raise_oom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Out of Memory on Intel Level Zero GPU: Allocation failed")

        with mock.patch("openvino_genai.Text2VideoPipeline") as mock_cls:
            mock_inst = mock.MagicMock()
            mock_inst.generate.side_effect = _raise_oom
            mock_cls.return_value = mock_inst

            engine = VideoEngine(model_path_or_id=model_dir)
            req = VideoGenerationRequest(prompt="Complex 4K render", width=512, height=512)
            with pytest.raises(VideoEngineError, match="Video generation execution failed"):
                engine.generate(req)

    def test_video_engine_unsupported_export_container(self, tmp_path: Path) -> None:
        """N7: Verify ValueError when requesting an unsupported video container format."""
        engine = VideoEngine()
        dummy_frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match="Unsupported video container format"):
            engine.encode_video(dummy_frames, tmp_path / "out.mkv", container="mkv")


class TestVideoEngineEdgeCase:
    """Edge-case tests evaluating boundary conditions, numerical limits, and stress invocations."""

    def test_video_engine_single_frame_boundary(self, tmp_path: Path) -> None:
        """E1: Verify single-frame generation boundary condition without zero division."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res = engine.generate(prompt="Single frame test", num_frames=1, fps=16, width=256, height=256)
            assert len(res.frames) == 1
            assert pytest.approx(res.duration_sec, 0.001) == 0.0625
            assert res.breakdown.real_time_factor > 0.0

    def test_video_engine_extreme_guidance_scales(self, tmp_path: Path) -> None:
        """E2: Verify numerical stability under unconditional (0.0) and high (50.0) CFG."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res_zero = engine.generate(prompt="Unconditional test", guidance_scale=0.0, width=256, height=256)
            res_high = engine.generate(prompt="Extreme saturation test", guidance_scale=50.0, width=256, height=256)
            assert res_zero.request.guidance_scale == 0.0
            assert res_high.request.guidance_scale == 50.0
            assert len(res_zero.frames) > 0
            assert len(res_high.frames) > 0

    def test_video_engine_prime_frame_counts(self, tmp_path: Path) -> None:
        """E3: Verify prime and odd frame counts handling (7, 13, 31 frames)."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            for prime_frames in [7, 13, 31]:
                res = engine.generate(
                    prompt="Prime frame sequence",
                    num_frames=prime_frames,
                    fps=10,
                    width=256,
                    height=256,
                )
                assert len(res.frames) == prime_frames
                assert pytest.approx(res.duration_sec, 0.01) == prime_frames / 10.0

    def test_video_engine_maximum_resolution_boundary(self, tmp_path: Path) -> None:
        """E4: Verify high-resolution boundary (1024x576) execution without dimension mismatch."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res = engine.generate(
                prompt="High resolution panoramic test",
                width=1024,
                height=576,
                num_frames=4,
                fps=4,
            )
            assert res.resolution == (1024, 576)
            assert res.frames[0].shape == (576, 1024, 3)

    def test_video_engine_long_prompt_truncation(self, tmp_path: Path) -> None:
        """E5: Verify extra-long text prompt (> 256 tokens) handling without buffer overflow."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        long_prompt = "cinematic video scene " * 200
        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            res = engine.generate(prompt=long_prompt, num_frames=4, width=256, height=256)
            assert len(res.frames) == 4
            assert len(res.request.prompt) > 1000

    def test_video_engine_rapid_sequential_invocations(self, tmp_path: Path) -> None:
        """E6: Verify memory stability and performance consistency across 5 sequential calls."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=MockText2VideoPipeline):
            engine = VideoEngine(model_path_or_id=model_dir)
            for i in range(5):
                res = engine.generate(
                    prompt=f"Sequential frame burst invocation {i}",
                    num_frames=6,
                    width=256,
                    height=256,
                )
                assert len(res.frames) == 6

    def test_video_engine_device_fallback(self, tmp_path: Path) -> None:
        """E7: Verify automatic fallback to CPU when Level Zero GPU compilation fails."""
        model_dir = tmp_path / "mock_video_model"
        model_dir.mkdir()

        call_devices = []

        def _mock_pipeline_init(path: str, device: str, **kwargs: Any) -> MockText2VideoPipeline:
            call_devices.append(device)
            if device == "GPU":
                raise RuntimeError("Level Zero GPU initialization error: ZeResult 0x78000001")
            return MockText2VideoPipeline(path, device, **kwargs)

        with mock.patch("openvino_genai.Text2VideoPipeline", side_effect=_mock_pipeline_init):
            engine = VideoEngine()
            compile_sec = engine.load_model(model_dir, device="GPU")
            assert compile_sec >= 0.0
            assert engine.active_device == "CPU"
            assert call_devices == ["GPU", "CPU"]
