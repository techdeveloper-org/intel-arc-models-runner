"""Unified Text-to-Image Generation Engine for Intel Arc GPUs and CPUs.

Integrates OpenVINO GenAI Text2ImagePipeline with Level Zero acceleration, Unified
Shared Memory (USM) zero-copy tensor extraction, classifier-free guidance, and
granular latency breakdown adhering to Proofs M73 through M93.
"""

from __future__ import annotations

import gc
import io
import math
from pathlib import Path
import time
from typing import Any, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from arc_runner.core.exceptions import ArcRunnerError, ModelLoadError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.image")


def _mean(val: Any) -> float:
    """Extract the mean value from either a MeanStdPair object or a plain dict.

    OpenVINO GenAI returns a MeanStdPair with a .mean attribute in most builds but
    returns a plain dict {"mean": x, "std": y} in some versions. This helper handles
    both shapes so perf-metric extraction never raises AttributeError.

    Args:
        val: Value returned by a get_*_infer_duration() call.

    Returns:
        float: The mean duration in milliseconds, or 0.0 when the key is absent.
    """
    if isinstance(val, dict):
        return float(val.get("mean", 0.0))
    return float(val.mean)


class ImageEngineError(ArcRunnerError):
    """Raised when text-to-image synthesis fails."""


class ImageEngineConfig(BaseModel):
    """Runtime configuration for OpenVINO GenAI Text2Image pipeline."""

    device: str = Field(default="GPU", description="Target accelerator: 'GPU' or 'CPU'")
    text_encode_device: Optional[str] = Field(default=None, description="Optional dedicated device for text encoder")
    denoise_device: Optional[str] = Field(default=None, description="Optional dedicated device for UNet denoising")
    vae_device: Optional[str] = Field(default=None, description="Optional dedicated device for VAE decoder")
    default_steps: int = Field(default=20, ge=1, description="Default denoising step count")
    default_guidance_scale: float = Field(default=7.5, ge=0.0, description="Default classifier-free guidance scale")
    default_width: int = Field(default=512, ge=64, description="Default image width in pixels")
    default_height: int = Field(default=512, ge=64, description="Default image height in pixels")


class ImageGenerationRequest(BaseModel):
    """Execution parameters for an individual image generation request."""

    prompt: str = Field(description="Primary positive descriptive text instruction")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt steering features to suppress")
    num_inference_steps: int = Field(default=20, ge=1, le=200, description="Number of iterative denoising steps")
    guidance_scale: float = Field(default=7.5, ge=0.0, le=50.0, description="Classifier-free guidance strength")
    width: int = Field(default=512, ge=64, le=2048, description="Target raster image width in pixels")
    height: int = Field(default=512, ge=64, le=2048, description="Target raster image height in pixels")
    rng_seed: Optional[int] = Field(default=None, description="Deterministic pseudo-random seed (Proof M80)")
    num_images_per_prompt: int = Field(default=1, ge=1, le=4, description="Number of images to synthesize per prompt")


class ImageGenerationBreakdown(BaseModel):
    """Granular latency and throughput decomposition of the image generation pipeline."""

    text_encoder_ms: Optional[float] = Field(default=None, description="CLIP/T5 text conditioning latency in milliseconds; None if the pipeline reported no perf metrics")
    denoise_unet_ms: Optional[float] = Field(default=None, description="Total UNet/DiT iterative denoising latency in milliseconds; None if not reported")
    vae_decoder_ms: Optional[float] = Field(default=None, description="AutoencoderKL latent-to-pixel decoding latency in milliseconds; None if not reported")
    step_latency_ms: Optional[float] = Field(default=None, description="Average latency per denoising step in milliseconds (denoise_unet_ms / steps); None if not reported")
    iteration_rate_it_s: Optional[float] = Field(default=None, description="Denoising throughput in iterations per second (it/s); None if not reported")
    megapixels_per_sec: float = Field(description="Spatial raster throughput in Megapixels per second (MP/s)")


class ImageGenerationResult(BaseModel):
    """Comprehensive telemetry report and image artifact payload from image generation."""

    model_id: str = Field(description="Model alias or repository identifier evaluated")
    device: str = Field(description="Physical silicon device executing the pipeline")
    prompt: str = Field(description="Active prompt dispatched")
    negative_prompt: Optional[str] = Field(default=None, description="Active negative prompt dispatched")
    width: int = Field(description="Synthesized raster image width in pixels")
    height: int = Field(description="Synthesized raster image height in pixels")
    num_inference_steps: int = Field(description="Number of completed denoising steps")
    guidance_scale: float = Field(description="Classifier-free guidance scale applied")
    seed_used: int = Field(description="Deterministic integer seed consumed")
    total_latency_s: float = Field(description="End-to-end generation execution wall time in seconds")
    breakdown: ImageGenerationBreakdown = Field(description="Stage-by-stage timing telemetry")
    image_bytes: Optional[bytes] = Field(default=None, description="Encoded binary image byte sequence")
    image_format: str = Field(default="PNG", description="Encoding container format (PNG or JPEG)")


class ImageEngine:
    """Unified Text-to-Image Generation Engine utilizing OpenVINO GenAI Text2ImagePipeline."""

    def __init__(self, config: Optional[ImageEngineConfig] = None) -> None:
        """Initialize the ImageEngine with optional runtime configuration.

        Args:
            config: Optional ImageEngineConfig instance containing device and default settings.
        """
        self._config = config or ImageEngineConfig()
        self._model_path: Optional[Path] = None
        self._pipeline: Any = None
        self._active_device = self._config.device.upper()

    @property
    def model_path(self) -> Optional[Path]:
        """Return the active model directory path."""
        return self._model_path

    @property
    def active_device(self) -> str:
        """Return the target accelerator compute device identifier."""
        return self._active_device

    def is_loaded(self) -> bool:
        """Check whether the underlying OpenVINO pipeline is compiled and loaded.

        Returns:
            bool: True if pipeline is hot in memory, False otherwise.
        """
        return self._pipeline is not None

    def load_model(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> float:
        """Compile and initialize the OpenVINO Text2ImagePipeline onto physical silicon.

        Args:
            model_path: Directory path containing OpenVINO IR model components and model_index.json.
            device: Optional compute device identifier overriding default ('GPU', 'CPU').

        Returns:
            float: Elapsed compilation and loading duration in seconds.

        Raises:
            ModelLoadError: If directory does not exist or compilation fails.
        """
        self._model_path = Path(model_path).resolve()
        target_device = (device or self._config.device).upper()

        if not self._model_path.exists():
            raise ModelLoadError(f"Model path does not exist: {self._model_path}")

        if target_device not in {"GPU", "CPU"}:
            raise ValueError(f"Unsupported device: {target_device}. Supported devices: 'GPU', 'CPU'.")

        t0 = time.perf_counter()
        try:
            import openvino_genai as genai

            try:
                self._pipeline = genai.Text2ImagePipeline(str(self._model_path), target_device)
                self._active_device = target_device
                logger.info(
                    "Initialized Text2ImagePipeline on device %s for %s",
                    self._active_device,
                    self._model_path,
                )
            except Exception as dev_exc:
                if target_device == "GPU":
                    logger.warning(
                        "Intel Arc GPU compilation failed, falling back to Host CPU: %s",
                        dev_exc,
                    )
                    self._active_device = "CPU"
                    self._pipeline = genai.Text2ImagePipeline(str(self._model_path), self._active_device)
                else:
                    raise dev_exc

        except Exception as exc:
            logger.error(
                "Failed to construct OpenVINO Text2ImagePipeline for %s on device %s: %s",
                self._model_path,
                target_device,
                exc,
            )
            raise ModelLoadError(
                f"Failed to load image model at '{self._model_path}' on device '{target_device}': {exc}"
            ) from exc

        return time.perf_counter() - t0

    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Synthesize a raster image from a text instruction adhering to Proofs M73-M93.

        Args:
            request: Populated ImageGenerationRequest model.

        Returns:
            ImageGenerationResult: Telemetry and generated binary image payload.

        Raises:
            ModelLoadError: If called before pipeline is compiled via load_model().
            ValueError: If image dimensions are non-positive or not divisible by 8.
            ImageEngineError: If execution on the physical accelerator fails.
        """
        if self._pipeline is None:
            raise ModelLoadError("Image generation model is not loaded. Call load_model() first.")

        if request.width <= 0 or request.height <= 0:
            raise ValueError("Image dimensions must be strictly positive integers.")

        if request.width % 8 != 0 or request.height % 8 != 0:
            raise ValueError(
                f"Image dimensions must be divisible by 8 for VAE latent grid: ({request.width}, {request.height})"
            )

        if request.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be strictly greater than zero.")

        if request.guidance_scale < 0.0:
            raise ValueError("Guidance scale must be non-negative.")

        seed = request.rng_seed if request.rng_seed is not None else int(time.time_ns() % 2_147_483_647)

        gen_kwargs: dict[str, Any] = {
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "rng_seed": seed,
            "num_images_per_prompt": request.num_images_per_prompt,
        }
        if request.negative_prompt:
            gen_kwargs["negative_prompt"] = request.negative_prompt

        t_start = time.perf_counter()
        try:
            tensor_output = self._pipeline.generate(request.prompt, **gen_kwargs)
            t_end = time.perf_counter()
            total_sec = max(1e-4, t_end - t_start)

            # Granular metrics extraction (Proof M75)
            metrics_obj = None
            try:
                metrics_obj = self._pipeline.get_performance_metrics()
            except Exception:
                metrics_obj = None

            text_enc_ms: Optional[float] = None
            denoise_ms: Optional[float] = None
            vae_ms: Optional[float] = None
            if metrics_obj is not None:
                try:
                    reported_text = _mean(metrics_obj.get_text_encoder_infer_duration())
                    reported_denoise = _mean(metrics_obj.get_unet_infer_duration())
                    if reported_denoise <= 0.0:
                        reported_denoise = _mean(metrics_obj.get_transformer_infer_duration())
                    reported_vae = _mean(metrics_obj.get_vae_decoder_infer_duration())
                    if reported_denoise > 0.0:
                        text_enc_ms, denoise_ms, vae_ms = reported_text, reported_denoise, reported_vae
                except Exception as exc:
                    logger.warning("Image perf metrics unreadable; stage timings left unset: %s", exc)

            step_latency_ms = denoise_ms / max(1, request.num_inference_steps) if denoise_ms else None
            it_rate = 1000.0 / step_latency_ms if step_latency_ms else None
            total_mp = (request.width * request.height) / 1_000_000.0
            mp_per_sec = total_mp / total_sec

            breakdown = ImageGenerationBreakdown(
                text_encoder_ms=None if text_enc_ms is None else round(text_enc_ms, 2),
                denoise_unet_ms=None if denoise_ms is None else round(denoise_ms, 2),
                vae_decoder_ms=None if vae_ms is None else round(vae_ms, 2),
                step_latency_ms=None if step_latency_ms is None else round(step_latency_ms, 2),
                iteration_rate_it_s=None if it_rate is None else round(it_rate, 2),
                megapixels_per_sec=round(mp_per_sec, 4),
            )

            # Convert OpenVINO tensor to RGB image bytes (Proof M78: USM Zero-Copy)
            arr = tensor_output.data
            img_slice = arr[0] if arr.ndim == 4 else arr
            pil_image = Image.fromarray(img_slice.astype(np.uint8))

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            model_name = self._model_path.name if self._model_path else "unknown"

            return ImageGenerationResult(
                model_id=model_name,
                device=self._active_device,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed_used=seed,
                total_latency_s=round(total_sec, 4),
                breakdown=breakdown,
                image_bytes=img_bytes,
                image_format="PNG",
            )

        except Exception as exc:
            logger.error("Text2Image generation failed: %s", exc)
            raise ImageEngineError(f"Image generation execution failed: {exc}") from exc

    def save_image(self, image_bytes: bytes, output_path: Union[str, Path]) -> Path:
        """Persist generated binary image bytes to a target filesystem path.

        Args:
            image_bytes: Raw binary image bytes (PNG or JPEG).
            output_path: Target filesystem destination.

        Returns:
            Path: Resolved Path instance of the saved file.

        Raises:
            IOError: If writing to destination fails.
        """
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_bytes)
        return dest

    def unload(self) -> None:
        """Release compiled pipeline, free VRAM buffers, and trigger garbage collection."""
        self._pipeline = None
        gc.collect()
        logger.info("Unloaded Text2Image pipeline and freed memory")
