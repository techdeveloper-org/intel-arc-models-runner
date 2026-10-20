"""Triad test suite for Multimodal Vision-Language Model Engine and Proofs M5, M6, M18.

Verifies image patch token expansion mathematics, base64/tensor decoding pipelines,
and VLM streaming execution.
"""

import base64
import io
import math
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import pytest

from arc_runner.core.exceptions import ModelLoadError
from arc_runner.engine.models import GenerationConfig, TokenChunk
from arc_runner.engine.vision_engine import (
    InvalidImageFormatError,
    VisionEngine,
    VisionEngineError,
)


class TestVisionPatchExpansionM5:
    """Mathematical validation of Proof M5: Spatial Patch Expansion."""

    def test_calculate_visual_tokens_positive_standard_resolutions(self) -> None:
        """Verify standard 336x336 image yields exactly 577 visual tokens with patch size 14."""
        with mock.patch("openvino_genai.VLMPipeline"):
            engine = VisionEngine.__new__(VisionEngine)
            engine._patch_size = 14

            assert engine.calculate_visual_tokens(336, 336) == 577
            assert engine.calculate_visual_tokens(224, 224) == 257

    def test_calculate_visual_tokens_negative_invalid_dimensions(self) -> None:
        """Verify non-positive dimensions raise ValueError."""
        engine = VisionEngine.__new__(VisionEngine)
        engine._patch_size = 14

        with pytest.raises(ValueError, match="strictly positive"):
            engine.calculate_visual_tokens(0, 336)

        with pytest.raises(ValueError, match="strictly positive"):
            engine.calculate_visual_tokens(336, -10)

    def test_calculate_visual_tokens_edge_case_panoramic_aspect_ratio(self) -> None:
        """Verify extreme 16:1 panoramic aspect ratio calculates tokens without overflow."""
        engine = VisionEngine.__new__(VisionEngine)
        engine._patch_size = 14

        tokens = engine.calculate_visual_tokens(1920, 108)
        expected_w = math.ceil(1920 / 14)
        expected_h = math.ceil(108 / 14)
        assert tokens == (expected_w * expected_h + 1)


@pytest.fixture
def sample_png_base64() -> str:
    """Generate a valid 64x64 test PNG image encoded as base64."""
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestImageDecoding:
    """Validation of image decoding to OpenVINO RGB Tensors."""

    def test_decode_image_tensor_positive(self, sample_png_base64: str) -> None:
        """Verify base64 PNG image decodes to valid 64x64x3 OpenVINO Tensor."""
        tensor = VisionEngine.decode_image_tensor(sample_png_base64)
        assert list(tensor.shape) == [64, 64, 3]

    def test_decode_image_tensor_data_uri_prefix(self, sample_png_base64: str) -> None:
        """Verify data URI prefix is stripped cleanly prior to decoding."""
        data_uri = f"data:image/png;base64,{sample_png_base64}"
        tensor = VisionEngine.decode_image_tensor(data_uri)
        assert list(tensor.shape) == [64, 64, 3]

    def test_decode_image_tensor_negative_corrupted_base64(self) -> None:
        """Verify malformed base64 strings raise InvalidImageFormatError."""
        with pytest.raises(InvalidImageFormatError):
            VisionEngine.decode_image_tensor("not_a_valid_base64_payload!@#$%^")

    def test_decode_image_tensor_negative_missing_file_path(self, tmp_path: Path) -> None:
        """Verify missing image file path raises InvalidImageFormatError."""
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(InvalidImageFormatError, match="does not exist"):
            VisionEngine.decode_image_tensor(missing)


class TestVisionEngineExecution:
    """Execution lifecycle and error handling tests for VisionEngine."""

    def test_vision_engine_negative_missing_model_dir(self, tmp_path: Path) -> None:
        """Verify non-existent model directory raises ModelLoadError."""
        non_existent = tmp_path / "non_existent_vlm"
        with pytest.raises(ModelLoadError, match="does not exist"):
            VisionEngine(str(non_existent))

    def test_vision_engine_streaming_generation(self, sample_png_base64: str, tmp_path: Path) -> None:
        """Verify VLM token generation yields TokenChunk items."""
        model_dir = tmp_path / "mock-vlm"
        model_dir.mkdir()

        mock_vlm_pipeline = mock.MagicMock()

        def fake_generate(*args, **kwargs):
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("The ")
                streamer("image ")
                streamer("shows ")
                streamer("red.")
            return mock.MagicMock()

        mock_vlm_pipeline.generate.side_effect = fake_generate

        with mock.patch("openvino_genai.VLMPipeline", return_value=mock_vlm_pipeline):
            engine = VisionEngine(str(model_dir), device="GPU")
            chunks = list(engine.generate("Describe this image", [sample_png_base64]))

            assert len(chunks) == 4
            assert "".join(c.text for c in chunks) == "The image shows red."
            assert chunks[-1].is_finished is True
            assert chunks[-1].finish_reason == "stop"
