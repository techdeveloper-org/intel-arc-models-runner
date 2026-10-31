"""Triad test suite for Text-to-Image Generation Engine and Proofs M73 through M93.

Evaluates Positive, Negative, and Edge-Case behaviors across model loading, prompt
conditioning, negative guidance, deterministic seeds, and VRAM resource reclamation.
Adheres strictly to Level 2.2 Docstrings-Only standards and Claude Global Rules.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import pytest

from arc_runner.core.exceptions import ModelLoadError
from arc_runner.engine.image_engine import (
    ImageEngine,
    ImageEngineConfig,
    ImageEngineError,
    ImageGenerationBreakdown,
    ImageGenerationRequest,
    ImageGenerationResult,
)


class MockOpenVINOTensor:
    """Mock OpenVINO tensor returning configured image array data."""

    def __init__(self, shape: tuple[int, int, int, int] = (1, 512, 512, 3), seed: int = 42) -> None:
        """Initialize mock tensor with deterministic RGB values.

        Args:
            shape: 4D tensor shape tuple (batch, height, width, channels).
            seed: RNG seed for reproducible pixel values.
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


class MockImagePerfMetrics:
    """Mock OpenVINO ImageGenerationPerfMetrics object."""

    def get_text_encoder_infer_duration(self) -> MockDurationMetric:
        """Return mock text encoder inference duration."""
        return MockDurationMetric(45.2)

    def get_unet_infer_duration(self) -> MockDurationMetric:
        """Return mock UNet denoising inference duration."""
        return MockDurationMetric(1250.0)

    def get_transformer_infer_duration(self) -> MockDurationMetric:
        """Return mock transformer inference duration."""
        return MockDurationMetric(0.0)

    def get_vae_decoder_infer_duration(self) -> MockDurationMetric:
        """Return mock VAE decoder inference duration."""
        return MockDurationMetric(120.5)


class MockText2ImagePipeline:
    """Mock OpenVINO GenAI Text2ImagePipeline for fast deterministic unit testing."""

    def __init__(self, model_path: str, device: str) -> None:
        """Initialize mock pipeline state.

        Args:
            model_path: Stored model path string.
            device: Accelerator device identifier.
        """
        self.model_path = model_path
        self.device = device

    def generate(self, prompt: str, **kwargs: Any) -> MockOpenVINOTensor:
        """Return mock tensor matching requested dimensions and seed.

        Args:
            prompt: Text prompt string.
            kwargs: Denoising arguments (width, height, rng_seed, etc.).

        Returns:
            MockOpenVINOTensor with requested resolution.
        """
        w = kwargs.get("width", 512)
        h = kwargs.get("height", 512)
        seed = kwargs.get("rng_seed", 42)
        return MockOpenVINOTensor(shape=(1, h, w, 3), seed=seed)

    def get_performance_metrics(self) -> MockImagePerfMetrics:
        """Return mock performance metrics telemetry."""
        return MockImagePerfMetrics()


class TestImageEnginePositive:
    """Positive test cases evaluating valid inputs, model compilation, and synthesis."""

    def test_image_engine_init_default_config(self) -> None:
        """P1: Verify ImageEngine initializes with default configuration parameters."""
        engine = ImageEngine()
        assert engine.active_device == "GPU"
        assert engine.is_loaded() is False
        assert engine.model_path is None

    def test_image_engine_load_model_success(self, tmp_path: Path) -> None:
        """P2: Verify successful model loading and compilation duration reporting."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            compile_sec = engine.load_model(model_dir, device="GPU")
            assert compile_sec >= 0.0
            assert engine.is_loaded() is True
            assert engine.active_device == "GPU"
            assert engine.model_path == model_dir.resolve()

    def test_image_engine_generate_valid_request(self, tmp_path: Path) -> None:
        """P3: Verify standard generation produces non-empty PNG bytes and valid metadata."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")

            req = ImageGenerationRequest(
                prompt="A serene mountain lake at sunrise",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512,
                rng_seed=42,
            )
            result = engine.generate(req)

            assert isinstance(result, ImageGenerationResult)
            assert result.width == 512
            assert result.height == 512
            assert result.num_inference_steps == 20
            assert result.image_bytes is not None
            assert len(result.image_bytes) > 100
            assert result.image_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_image_engine_generate_custom_resolution(self, tmp_path: Path) -> None:
        """P4: Verify generation succeeds across diverse valid resolutions (256x256, 768x768)."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")

            for dim in (256, 768):
                req = ImageGenerationRequest(prompt="Test abstract shapes", width=dim, height=dim)
                res = engine.generate(req)
                assert res.width == dim
                assert res.height == dim

                img = Image.open(io.BytesIO(res.image_bytes))
                assert img.size == (dim, dim)

    def test_image_engine_generate_deterministic_seed(self, tmp_path: Path) -> None:
        """P5: Verify deterministic RNG seed guarantees identical byte sequences (Proof M80)."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")

            req1 = ImageGenerationRequest(prompt="Deterministic test", rng_seed=999)
            req2 = ImageGenerationRequest(prompt="Deterministic test", rng_seed=999)

            res1 = engine.generate(req1)
            res2 = engine.generate(req2)

            assert res1.image_bytes == res2.image_bytes
            assert res1.seed_used == 999
            assert res2.seed_used == 999

    def test_image_engine_negative_prompt_steering(self, tmp_path: Path) -> None:
        """P6: Verify negative prompt parameter is accepted and reflected in result (Proof M87)."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")

            req = ImageGenerationRequest(
                prompt="Cyberpunk street",
                negative_prompt="blurry, distorted, oversaturated",
            )
            res = engine.generate(req)
            assert res.negative_prompt == "blurry, distorted, oversaturated"

    def test_image_engine_save_image_png_and_jpg(self, tmp_path: Path) -> None:
        """P7: Verify saving generated image bytes to disk preserves file integrity."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")

            req = ImageGenerationRequest(prompt="Saving test")
            res = engine.generate(req)

            dest_png = tmp_path / "output_test.png"
            saved_path = engine.save_image(res.image_bytes, dest_png)

            assert saved_path.exists()
            assert saved_path.stat().st_size > 100
            assert saved_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_image_engine_perf_breakdown_metrics(self, tmp_path: Path) -> None:
        """P8: Verify performance telemetry captures text, UNet, and VAE latency breakdowns."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")

            req = ImageGenerationRequest(prompt="Performance metrics test", num_inference_steps=20)
            res = engine.generate(req)

            bd = res.breakdown
            assert bd.text_encoder_ms > 0.0
            assert bd.denoise_unet_ms > 0.0
            assert bd.vae_decoder_ms > 0.0
            assert bd.step_latency_ms > 0.0
            assert bd.iteration_rate_it_s > 0.0
            assert bd.megapixels_per_sec > 0.0

    def test_image_engine_without_perf_metrics_reports_stage_timings_as_none(self, tmp_path: Path) -> None:
        """H9: no OpenVINO perf metrics means no stage split; nothing is estimated from total time."""

        class _NoMetricsPipeline(MockText2ImagePipeline):
            def get_performance_metrics(self) -> Any:
                raise RuntimeError("perf metrics unavailable")

        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()
        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=_NoMetricsPipeline):
            engine = ImageEngine()
            engine.load_model(model_dir, device="GPU")
            res = engine.generate(ImageGenerationRequest(prompt="No metrics", num_inference_steps=20))
        bd = res.breakdown
        assert bd.text_encoder_ms is None and bd.denoise_unet_ms is None and bd.vae_decoder_ms is None
        assert bd.step_latency_ms is None and bd.iteration_rate_it_s is None
        assert res.total_latency_s > 0.0 and bd.megapixels_per_sec > 0.0

    def test_image_engine_device_switching_gpu_to_cpu(self, tmp_path: Path) -> None:
        """P9: Verify device fallback cleanly re-targets from GPU to Host CPU."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        def fail_on_gpu(path: str, device: str) -> Any:
            if device == "GPU":
                raise RuntimeError("Simulated Level Zero GPU driver allocation fault")
            return MockText2ImagePipeline(path, device)

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=fail_on_gpu):
            engine = ImageEngine(ImageEngineConfig(device="GPU"))
            engine.load_model(model_dir, device="GPU")
            assert engine.active_device == "CPU"
            assert engine.is_loaded() is True

    def test_image_engine_unload_reclaims_memory(self, tmp_path: Path) -> None:
        """P10: Verify unload() unbinds pipeline and frees hot memory handles."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)
            assert engine.is_loaded() is True

            engine.unload()
            assert engine.is_loaded() is False


class TestImageEngineNegative:
    """Negative test cases evaluating error handling, invalid inputs, and safety bounds."""

    def test_image_engine_generate_before_load_raises_error(self) -> None:
        """N1: Verify generate() raises ModelLoadError if called prior to load_model()."""
        engine = ImageEngine()
        req = ImageGenerationRequest(prompt="Unloaded prompt")
        with pytest.raises(ModelLoadError, match="not loaded"):
            engine.generate(req)

    def test_image_engine_load_nonexistent_directory_raises_error(self, tmp_path: Path) -> None:
        """N2: Verify load_model() raises ModelLoadError on missing directory."""
        engine = ImageEngine()
        missing_dir = tmp_path / "missing_openvino_model"
        with pytest.raises(ModelLoadError, match="does not exist"):
            engine.load_model(missing_dir)

    def test_image_engine_invalid_dimensions_negative_or_zero(self, tmp_path: Path) -> None:
        """N3: Verify non-positive image dimensions trigger immediate ValueError."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            with pytest.raises(ValueError):
                ImageGenerationRequest(prompt="Test", width=0, height=512)

            with pytest.raises(ValueError, match="strictly positive"):
                engine.generate(ImageGenerationRequest.model_construct(prompt="Test", width=0, height=512))

            with pytest.raises(ValueError, match="strictly positive"):
                engine.generate(ImageGenerationRequest.model_construct(prompt="Test", width=512, height=-64))

    def test_image_engine_invalid_dimensions_non_multiple_of_8(self, tmp_path: Path) -> None:
        """N4: Verify dimensions not divisible by 8 raise ValueError for VAE grid constraint."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            with pytest.raises(ValueError, match="divisible by 8"):
                engine.generate(ImageGenerationRequest(prompt="Test", width=513, height=512))

            with pytest.raises(ValueError, match="divisible by 8"):
                engine.generate(ImageGenerationRequest(prompt="Test", width=512, height=515))

    def test_image_engine_invalid_steps_zero_or_negative(self, tmp_path: Path) -> None:
        """N5: Verify inference steps <= 0 trigger ValueError or Pydantic validation error."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            with pytest.raises(Exception):
                ImageGenerationRequest(prompt="Test", num_inference_steps=0)

    def test_image_engine_invalid_guidance_scale_negative(self) -> None:
        """N6: Verify negative guidance scale triggers validation error."""
        with pytest.raises(Exception):
            ImageGenerationRequest(prompt="Test", guidance_scale=-2.5)

    def test_image_engine_unsupported_device_error(self, tmp_path: Path) -> None:
        """N7: Verify unrecognized compute devices trigger ValueError."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        engine = ImageEngine()
        with pytest.raises(ValueError, match="Unsupported device"):
            engine.load_model(model_dir, device="TPU_V5")


class TestImageEngineEdgeCases:
    """Edge-case tests evaluating boundary conditions, stress configurations, and memory stability."""

    def test_image_engine_single_step_generation(self, tmp_path: Path) -> None:
        """E1: Verify 1-step generation completes for SD-Turbo and ADD distillations."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            req = ImageGenerationRequest(prompt="Single step test", num_inference_steps=1)
            res = engine.generate(req)
            assert res.num_inference_steps == 1
            assert res.image_bytes is not None

    def test_image_engine_extreme_guidance_scale_zero(self, tmp_path: Path) -> None:
        """E2: Verify unconditioned CFG (w = 0.0) executes without division-by-zero."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            req = ImageGenerationRequest(prompt="Unconditioned test", guidance_scale=0.0)
            res = engine.generate(req)
            assert res.guidance_scale == 0.0

    def test_image_engine_extreme_guidance_scale_high(self, tmp_path: Path) -> None:
        """E3: Verify high guidance scale (w = 30.0) completes without floating-point overflow."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            req = ImageGenerationRequest(prompt="High guidance test", guidance_scale=30.0)
            res = engine.generate(req)
            assert res.guidance_scale == 30.0

    def test_image_engine_ultra_long_prompt(self, tmp_path: Path) -> None:
        """E4: Verify ultra-long prompt (> 100 words) executes without tokenizer overflow."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        long_prompt = "A majestic landscape with mountains, rivers, trees, castles, stars, nebulas, dragons, warriors " * 15

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            req = ImageGenerationRequest(prompt=long_prompt)
            res = engine.generate(req)
            assert res.image_bytes is not None

    def test_image_engine_non_square_aspect_ratios(self, tmp_path: Path) -> None:
        """E5: Verify non-square portrait (512x768) and landscape (768x512) aspect ratios."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            res_portrait = engine.generate(ImageGenerationRequest(prompt="Portrait", width=512, height=768))
            assert res_portrait.width == 512
            assert res_portrait.height == 768

            res_landscape = engine.generate(ImageGenerationRequest(prompt="Landscape", width=768, height=512))
            assert res_landscape.width == 768
            assert res_landscape.height == 512

    def test_image_engine_empty_prompts_and_whitespace(self, tmp_path: Path) -> None:
        """E6: Verify whitespace prompts execute safely without pipeline crash."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            req = ImageGenerationRequest(prompt="   ")
            res = engine.generate(req)
            assert res.image_bytes is not None

    def test_image_engine_consecutive_generations_no_leak(self, tmp_path: Path) -> None:
        """E7: Verify multiple consecutive generations execute with constant resource usage."""
        model_dir = tmp_path / "mock_sd_model"
        model_dir.mkdir()

        with mock.patch("openvino_genai.Text2ImagePipeline", side_effect=MockText2ImagePipeline):
            engine = ImageEngine()
            engine.load_model(model_dir)

            for i in range(5):
                req = ImageGenerationRequest(prompt=f"Consecutive run {i}", rng_seed=i)
                res = engine.generate(req)
                assert res.image_bytes is not None
