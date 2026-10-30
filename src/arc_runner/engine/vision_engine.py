"""Multimodal Vision-Language Model (VLM) execution engine for Intel Arc GPUs.

Integrates OpenVINO GenAI VLMPipeline for visual question answering, document OCR,
and zero-copy image patch tensorization on Intel Arc Xe-LPG matrix engines.
"""

import base64
import io
import math
from pathlib import Path
import time
from typing import Any, Generator, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from arc_runner.core.exceptions import ArcRunnerError, ModelLoadError
from arc_runner.core.logging import get_logger
from arc_runner.engine.models import GenerationConfig, TokenChunk

logger = get_logger("arc_runner.engine.vision")


class VisionEngineError(ArcRunnerError):
    """Raised when multimodal vision inference fails."""


class InvalidImageFormatError(VisionEngineError):
    """Raised when input image data is corrupt or unparseable."""


class VisionEngine:
    """Multimodal vision-language model inference engine wrapping openvino_genai.VLMPipeline."""

    def __init__(
        self,
        model_path: str,
        device: str = "GPU",
        patch_size: int = 14,
    ) -> None:
        """Initialize the multimodal vision pipeline on the requested Intel device.

        Args:
            model_path: Directory path containing exported OpenVINO VLM model components.
            device: Accelerator device identifier ('GPU', 'CPU', 'NPU').
            patch_size: Visual spatial patch dimension in pixels for token expansion math.

        Raises:
            ModelLoadError: If VLM model files are missing or pipeline fails to load.
        """
        self._model_path = Path(model_path)
        self._device = device.upper()
        self._patch_size = patch_size
        self._pipeline: Any = None

        self._initialize_pipeline()

    @property
    def model_path(self) -> Path:
        """Return the active model directory path."""
        return self._model_path

    @property
    def device(self) -> str:
        """Return the target accelerator compute device."""
        return self._device

    @property
    def patch_size(self) -> int:
        """Return the spatial patch dimension."""
        return self._patch_size

    def _initialize_pipeline(self) -> None:
        """Construct the OpenVINO VLMPipeline instance.

        Raises:
            ModelLoadError: If loading fails on the selected accelerator.
        """
        if not self._model_path.exists():
            raise ModelLoadError(f"VLM model directory does not exist: {self._model_path}")

        try:
            import openvino_genai as genai

            try:
                self._pipeline = genai.VLMPipeline(str(self._model_path), self._device)
                logger.info(
                    "Initialized VLMPipeline",
                    extra={"device": self._device, "model_path": str(self._model_path)},
                )
            except Exception as device_exc:
                if self._device == "GPU":
                    logger.warning(
                        "Arc GPU VLM initialization failed, falling back to CPU",
                        extra={"error": str(device_exc)},
                    )
                    self._device = "CPU"
                    self._pipeline = genai.VLMPipeline(str(self._model_path), self._device)
                else:
                    raise device_exc

        except Exception as exc:
            logger.error(
                "Failed to construct OpenVINO VLMPipeline",
                extra={"model_path": str(self._model_path), "device": self._device, "error": str(exc)},
            )
            raise ModelLoadError(
                f"Failed to load VLM model at '{self._model_path}' on device '{self._device}': {exc}"
            ) from exc

    def calculate_visual_tokens(self, width: int, height: int) -> int:
        """Compute the expanded visual token count for an image resolution (Proof M5).

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Number of spatial patch tokens plus the CLS token.

        Raises:
            ValueError: If width or height is non-positive.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be strictly positive integers.")

        patches_w = math.ceil(width / self._patch_size)
        patches_h = math.ceil(height / self._patch_size)
        return int(patches_w * patches_h + 1)

    @classmethod
    def decode_image_tensor(cls, image_data: Union[str, bytes, Path]) -> Any:
        """Decode a base64 string, raw bytes, or image path to an OpenVINO Tensor.

        Args:
            image_data: Base64 data URI, raw base64 string, binary bytes, or Path.

        Returns:
            openvino.Tensor object containing RGB image data with shape (H, W, 3).

        Raises:
            InvalidImageFormatError: If image bytes cannot be decoded or image is corrupted.
        """
        raw_bytes: bytes

        if isinstance(image_data, Path):
            if not image_data.exists():
                raise InvalidImageFormatError(f"Image file does not exist: {image_data}")
            raw_bytes = image_data.read_bytes()
        elif isinstance(image_data, str):
            clean_str = image_data.strip()
            if clean_str.startswith("data:image/") and ";base64," in clean_str:
                clean_str = clean_str.split(";base64,")[1]
            try:
                raw_bytes = base64.b64decode(clean_str)
            except Exception as exc:
                raise InvalidImageFormatError(f"Failed to decode base64 image data: {exc}") from exc
        elif isinstance(image_data, bytes):
            raw_bytes = image_data
        else:
            raise InvalidImageFormatError(f"Unsupported image input type: {type(image_data).__name__}")

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)

            import openvino as ov
            return ov.Tensor(arr)
        except Exception as exc:
            raise InvalidImageFormatError(f"Failed to decode image pixels into tensor: {exc}") from exc

    def generate(
        self,
        prompt: str,
        images: Sequence[Union[str, bytes, Path]],
        config: Optional[GenerationConfig] = None,
    ) -> Generator[TokenChunk, None, None]:
        """Stream visual reasoning response tokens for a multimodal prompt and image array.

        Args:
            prompt: Text instruction or conversation query.
            images: Sequence of base64 strings, file paths, or image bytes.
            config: Generation sampling parameters (temperature, max_new_tokens, top_p).

        Yields:
            TokenChunk items containing incremental generated text and finish status.

        Raises:
            VisionEngineError: If multimodal generation fails.
        """
        gen_cfg = config or GenerationConfig()
        image_tensors = [self.decode_image_tensor(img) for img in images]

        import openvino_genai as genai

        ov_cfg = genai.GenerationConfig()
        ov_cfg.max_new_tokens = gen_cfg.max_new_tokens
        ov_cfg.temperature = gen_cfg.temperature
        ov_cfg.top_p = gen_cfg.top_p

        accumulated_tokens: List[str] = []
        token_index = 0

        def streamer(chunk_text: str) -> bool:
            nonlocal token_index
            accumulated_tokens.append(chunk_text)
            token_index += 1
            return False

        try:
            res = self._pipeline.generate(
                prompt=prompt,
                images=image_tensors,
                generation_config=ov_cfg,
                streamer=streamer,
            )

            for idx, text_token in enumerate(accumulated_tokens):
                is_last = idx == (len(accumulated_tokens) - 1)
                yield TokenChunk(
                    text=text_token,
                    is_finished=is_last,
                    finish_reason="stop" if is_last else None,
                )

        except Exception as exc:
            logger.error("VLM generation failed", extra={"error": str(exc)})
            raise VisionEngineError(f"VLM generation execution failed: {exc}") from exc
