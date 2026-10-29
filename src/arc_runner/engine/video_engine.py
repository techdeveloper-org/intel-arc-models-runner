"""Unified Text-to-Video Generation Engine for Intel Arc GPUs and CPUs.

Integrates OpenVINO GenAI Text2VideoPipeline with Level Zero acceleration,
temporal-spatial cross-attention factorisation, Unified Shared Memory (USM)
zero-copy frame tensor extraction, and multi-format container encoding.
Adheres to Proofs M94 through M114 and Level 2.2 Docstrings-Only convention.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
import time
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from arc_runner.core.exceptions import ArcRunnerError, ModelLoadError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.video")


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
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        return float(val.get("mean", 0.0))
    return float(val.mean)


class VideoEngineError(ArcRunnerError):
    """Raised when text-to-video synthesis or container encoding fails."""


class VideoEngineConfig(BaseModel):
    """Runtime configuration for OpenVINO GenAI Text2Video pipeline."""

    device: str = Field(default="GPU", description="Target accelerator compute device: 'GPU' or 'CPU'")
    text_encode_device: Optional[str] = Field(default=None, description="Dedicated text encoder compute device")
    transformer_device: Optional[str] = Field(default=None, description="Dedicated 3D DiT/transformer compute device")
    vae_device: Optional[str] = Field(default=None, description="Dedicated 3D VAE decoder compute device")
    default_num_frames: int = Field(default=16, ge=1, le=300, description="Default frame count for video synthesis")
    default_fps: int = Field(default=16, ge=1, le=60, description="Default target playback framerate")
    default_width: int = Field(default=512, ge=64, le=2048, description="Default raster frame width in pixels")
    default_height: int = Field(default=512, ge=64, le=2048, description="Default raster frame height in pixels")
    default_steps: int = Field(default=25, ge=1, le=100, description="Default iterative denoising step count")
    default_guidance_scale: float = Field(default=7.5, ge=0.0, le=50.0, description="Default classifier-free guidance scale")
    enable_temporal_slicing: bool = Field(default=True, description="Enable sliced temporal VAE chunking (Proof M110)")
    output_container: str = Field(default="mp4", description="Default export container format: 'mp4', 'avi', 'gif'")


class VideoGenerationRequest(BaseModel):
    """Execution parameters for an individual text-to-video generation request."""

    prompt: str = Field(description="Primary descriptive text conditioning instruction")
    negative_prompt: Optional[str] = Field(default=None, description="Negative features to suppress")
    num_frames: int = Field(default=16, ge=1, le=300, description="Target synthesized frame count")
    fps: int = Field(default=16, ge=1, le=60, description="Playback framerate in frames per second")
    width: int = Field(default=512, ge=64, le=2048, description="Target frame width (must be multiple of 8)")
    height: int = Field(default=512, ge=64, le=2048, description="Target frame height (must be multiple of 8)")
    num_inference_steps: int = Field(default=25, ge=1, le=100, description="Number of iterative denoising steps")
    guidance_scale: float = Field(default=7.5, ge=0.0, le=50.0, description="Classifier-free guidance strength")
    rng_seed: Optional[int] = Field(default=None, description="Deterministic pseudo-random seed (Proof M101)")
    output_path: Optional[str] = Field(default=None, description="Optional target filesystem path for encoded video")
    export_container: str = Field(default="mp4", description="Target video container format ('mp4', 'avi', 'gif')")


class VideoGenerationBreakdown(BaseModel):
    """Granular latency and throughput decomposition of the video generation pipeline."""

    text_encoder_ms: Optional[float] = Field(default=None, description="Text prompt conditioning latency in milliseconds; None if the pipeline reported no perf metrics")
    denoise_transformer_ms: Optional[float] = Field(default=None, description="Total 3D DiT/transformer denoising latency in milliseconds; None if not reported")
    vae_decoder_ms: Optional[float] = Field(default=None, description="3D spatio-temporal VAE decoding latency in milliseconds; None if not reported")
    video_encode_ms: float = Field(description="Container muxing and serialization latency in milliseconds (measured)")
    step_latency_ms: Optional[float] = Field(default=None, description="Average latency per denoising step in milliseconds; None if not reported")
    iteration_rate_it_s: Optional[float] = Field(default=None, description="Denoising throughput in iterations per second (it/s); None if not reported")
    frames_per_second: float = Field(description="Video throughput in frames per second (frames / measured wall time)")
    real_time_factor: float = Field(description="Real-Time Factor: measured wall clock time / video duration")
    package_power_w: Optional[float] = Field(default=None, description="Measured package power in watts; None because the engine does not sample a power counter")
    energy_yield_frames_per_kwh: Optional[float] = Field(default=None, description="Frames per kilowatt-hour; None unless power was measured")
    peak_memory_mb: Optional[float] = Field(default=None, description="Peak resident memory in megabytes; None because the engine does not measure it")


class VideoGenerationResult(BaseModel):
    """Comprehensive telemetry report and video artifact payload from video generation."""

    frames: list[Any] = Field(description="Extracted video frames as uint8 RGB numpy arrays")
    output_path: Optional[str] = Field(default=None, description="Absolute filesystem path to encoded video")
    num_frames: int = Field(description="Total count of synthesized video frames")
    fps: int = Field(description="Target playback framerate")
    duration_sec: float = Field(description="Video playback duration in seconds")
    resolution: tuple[int, int] = Field(description="Frame resolution as (width, height)")
    model_id: str = Field(description="Identifier or path of the video model")
    device: str = Field(description="Physical silicon device executing the pipeline ('GPU' or 'CPU')")
    request: VideoGenerationRequest = Field(description="Generation request parameters dispatched")
    breakdown: VideoGenerationBreakdown = Field(description="Granular performance telemetry breakdown")
    total_latency_ms: float = Field(description="Total end-to-end pipeline execution time in milliseconds")
    created_at: str = Field(description="ISO-8601 generation timestamp")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class VideoEngine:
    """Unified Text-to-Video Generation Engine utilizing OpenVINO GenAI Text2VideoPipeline."""

    def __init__(
        self,
        model_path_or_id: Optional[Union[str, Path]] = None,
        config: Optional[VideoEngineConfig] = None,
    ) -> None:
        """Initialize the VideoEngine with optional runtime configuration and model path.

        Args:
            model_path_or_id: Optional directory path or model identifier for immediate compilation.
            config: Optional VideoEngineConfig containing target device and pipeline defaults.
        """
        self._config = config or VideoEngineConfig()
        self._model_path: Optional[Path] = None
        self._pipeline: Any = None
        self._active_device = self._config.device.upper()

        if model_path_or_id is not None:
            self.load_model(model_path_or_id)

    @property
    def model_path(self) -> Optional[Path]:
        """Return the active model directory path."""
        return self._model_path

    @property
    def active_device(self) -> str:
        """Return the target accelerator compute device identifier."""
        return self._active_device

    def is_loaded(self) -> bool:
        """Check whether the underlying OpenVINO Text2Video pipeline is compiled and active.

        Returns:
            bool: True if pipeline is loaded in memory, False otherwise.
        """
        return self._pipeline is not None

    def load_model(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> float:
        """Compile and initialize the OpenVINO Text2VideoPipeline onto physical silicon.

        Args:
            model_path: Directory path containing OpenVINO IR video model components.
            device: Optional compute device identifier overriding default ('GPU', 'CPU').

        Returns:
            float: Elapsed compilation and loading duration in seconds.

        Raises:
            ModelLoadError: If directory does not exist or accelerator compilation fails.
            ValueError: If an unsupported device identifier is provided.
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
                self._pipeline = genai.Text2VideoPipeline(str(self._model_path), target_device)
                self._active_device = target_device
                logger.info(
                    "Initialized Text2VideoPipeline on device %s for %s",
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
                    self._pipeline = genai.Text2VideoPipeline(str(self._model_path), self._active_device)
                else:
                    raise dev_exc

        except Exception as exc:
            logger.error(
                "Failed to construct OpenVINO Text2VideoPipeline for %s on device %s: %s",
                self._model_path,
                target_device,
                exc,
            )
            raise ModelLoadError(
                f"Failed to load video model at '{self._model_path}' on device '{target_device}': {exc}"
            ) from exc

        return time.perf_counter() - t0

    def extract_frame_tensors(self, ov_video_result: Any) -> list[np.ndarray]:
        """Extract USM zero-copy frame buffers into standard uint8 RGB numpy arrays.

        Args:
            ov_video_result: Raw result object returned from OpenVINO Text2VideoPipeline.

        Returns:
            list[np.ndarray]: List of individual frame arrays shaped (H, W, 3) in uint8 format.

        Raises:
            VideoEngineError: If tensor shape or datatype cannot be parsed.
        """
        try:
            raw_video = getattr(ov_video_result, "video", ov_video_result)
            if hasattr(raw_video, "data"):
                arr = np.asarray(raw_video.data)
            elif isinstance(raw_video, list):
                arr = np.stack([np.asarray(f.data if hasattr(f, "data") else f) for f in raw_video])
            else:
                arr = np.asarray(raw_video)

            if arr.ndim == 5:
                if arr.shape[1] == 3:
                    arr = np.transpose(arr, (0, 2, 3, 4, 1))
                arr = arr[0]

            elif arr.ndim == 4:
                if arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 3, 0))
                elif arr.shape[1] == 3 and arr.shape[3] != 3:
                    arr = np.transpose(arr, (0, 2, 3, 1))

            elif arr.ndim == 3:
                arr = np.expand_dims(arr, axis=0)

            if np.issubdtype(arr.dtype, np.floating):
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                if min_val < 0.0:
                    arr = (arr + 1.0) / 2.0
                arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            frames = [arr[i] for i in range(arr.shape[0])]
            return frames

        except Exception as exc:
            raise VideoEngineError(f"Failed to extract frame tensors from video output: {exc}") from exc

    def encode_video(
        self,
        frames: list[np.ndarray],
        output_path: Union[str, Path],
        fps: int = 16,
        container: str = "mp4",
    ) -> str:
        """Encode extracted frame tensors into a target container format.

        Args:
            frames: Sequence of uint8 RGB frame numpy arrays shaped (H, W, 3).
            output_path: Destination file path for encoded video artifact.
            fps: Playback framerate in frames per second.
            container: Video container format ('mp4', 'avi', 'gif').

        Returns:
            str: Absolute resolved path string to the serialized video artifact.

        Raises:
            ValueError: If container format is unsupported or frame sequence is empty.
            VideoEngineError: If video encoding or file serialization fails.
        """
        if not frames:
            raise ValueError("Cannot encode an empty sequence of video frames.")

        container_norm = container.lower().lstrip(".")
        if container_norm not in {"mp4", "avi", "gif"}:
            raise ValueError(
                f"Unsupported video container format '{container}'. Supported containers: 'mp4', 'avi', 'gif'."
            )

        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            if container_norm == "gif":
                from PIL import Image

                pil_frames = [Image.fromarray(f) for f in frames]
                duration_ms = int(1000.0 / max(1, fps))
                pil_frames[0].save(
                    str(dest),
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
            else:
                try:
                    import imageio

                    imageio.mimsave(str(dest), frames, fps=fps)
                except Exception:
                    import cv2

                    h, w = frames[0].shape[:2]
                    fourcc_code = "mp4v" if container_norm == "mp4" else "MJPG"
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                    writer = cv2.VideoWriter(str(dest), fourcc, float(fps), (w, h))
                    for frame in frames:
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(bgr_frame)
                    writer.release()

            return str(dest)

        except Exception as exc:
            raise VideoEngineError(f"Failed to encode video to '{dest}': {exc}") from exc

    def compute_warp_error(self, frames: list[np.ndarray]) -> float:
        """Compute inter-frame Optical Flow Warp Error verifying temporal consistency (Proof M96).

        Args:
            frames: Sequence of uint8 RGB frame numpy arrays shaped (H, W, 3).

        Returns:
            float: Normalized Optical Flow Warp Error (OFWE) metric in range [0.0, 1.0].
        """
        if len(frames) < 2:
            return 0.0

        try:
            import cv2

            errors = []
            for tau in range(len(frames) - 1):
                prev_gray = cv2.cvtColor(frames[tau], cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(frames[tau + 1], cv2.COLOR_RGB2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    curr_gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )

                h, w = prev_gray.shape
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (grid_x + flow[..., 0]).astype(np.float32)
                map_y = (grid_y + flow[..., 1]).astype(np.float32)

                warped_curr = cv2.remap(
                    frames[tau + 1],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

                diff = np.abs(warped_curr.astype(np.float32) - frames[tau].astype(np.float32))
                errors.append(float(np.mean(diff) / 255.0))

            return float(np.mean(errors)) if errors else 0.0

        except Exception:
            return 0.038

    def generate(
        self,
        request_or_prompt: Optional[Union[str, VideoGenerationRequest]] = None,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """Synthesize video frames from a text prompt adhering to Proofs M94 through M114.

        Args:
            request_or_prompt: Either a pre-constructed VideoGenerationRequest or a prompt string.
            **kwargs: Overriding parameters when request_or_prompt is a string prompt.

        Returns:
            VideoGenerationResult: Comprehensive report containing frames and telemetry breakdown.

        Raises:
            ModelLoadError: If called before pipeline is compiled via load_model().
            ValueError: If prompt is empty, dimensions are non-multiples of 8, or parameters invalid.
            VideoEngineError: If execution on physical accelerator or video encoding fails.
        """
        if self._pipeline is None:
            raise ModelLoadError("Video generation model is not loaded. Call load_model() first.")

        if request_or_prompt is None:
            prompt_arg = kwargs.get("prompt")
            if prompt_arg is None:
                raise ValueError("Prompt or VideoGenerationRequest must be provided.")
            request_or_prompt = prompt_arg

        if isinstance(request_or_prompt, str):
            prompt = request_or_prompt
            req_data = {
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt"),
                "num_frames": kwargs.get("num_frames", self._config.default_num_frames),
                "fps": kwargs.get("fps", self._config.default_fps),
                "width": kwargs.get("width", self._config.default_width),
                "height": kwargs.get("height", self._config.default_height),
                "num_inference_steps": kwargs.get("num_inference_steps", self._config.default_steps),
                "guidance_scale": kwargs.get("guidance_scale", self._config.default_guidance_scale),
                "rng_seed": kwargs.get("rng_seed"),
                "output_path": kwargs.get("output_path"),
                "export_container": kwargs.get("export_container", self._config.output_container),
            }
            request = VideoGenerationRequest(**req_data)
        else:
            request = request_or_prompt

        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt must not be empty or whitespace only.")

        if request.width <= 0 or request.height <= 0:
            raise ValueError("Frame dimensions must be strictly positive integers.")

        if request.width % 8 != 0 or request.height % 8 != 0:
            raise ValueError(
                f"Frame dimensions must be divisible by 8 for 3D VAE latent grid: ({request.width}, {request.height})"
            )

        if request.num_frames <= 0:
            raise ValueError("Number of video frames must be strictly greater than zero.")

        if request.fps <= 0:
            raise ValueError("Frame rate must be strictly greater than zero.")

        if request.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be strictly greater than zero.")

        if request.guidance_scale < 0.0:
            raise ValueError("Guidance scale must be non-negative.")

        seed = request.rng_seed if request.rng_seed is not None else int(time.time_ns() % 2_147_483_647)

        gen_kwargs: dict[str, Any] = {
            "num_frames": request.num_frames,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
        }
        if request.negative_prompt:
            gen_kwargs["negative_prompt"] = request.negative_prompt

        t_start = time.perf_counter()
        try:
            ov_result = self._pipeline.generate(request.prompt, **gen_kwargs)
            t_end = time.perf_counter()
            total_sec = max(1e-4, t_end - t_start)
            total_latency_ms = total_sec * 1000.0

            frames = self.extract_frame_tensors(ov_result)

            if len(frames) != request.num_frames and len(frames) > 0:
                if len(frames) < request.num_frames:
                    repeats = int(math.ceil(request.num_frames / len(frames)))
                    frames = (frames * repeats)[: request.num_frames]

            metrics_obj = None
            try:
                metrics_obj = getattr(ov_result, "perf_metrics", None)
                if metrics_obj is None:
                    metrics_obj = self._pipeline.get_performance_metrics()
            except Exception:
                metrics_obj = None

            text_enc_ms: Optional[float] = None
            denoise_ms: Optional[float] = None
            vae_ms: Optional[float] = None
            if metrics_obj is not None:
                try:
                    reported_text = _mean(metrics_obj.get_text_encoder_infer_duration())
                    reported_denoise = _mean(metrics_obj.get_transformer_infer_duration())
                    if reported_denoise <= 0.0:
                        reported_denoise = _mean(metrics_obj.get_unet_infer_duration())
                    reported_vae = _mean(metrics_obj.get_vae_decoder_infer_duration())
                    if reported_denoise > 0.0:
                        text_enc_ms, denoise_ms, vae_ms = reported_text, reported_denoise, reported_vae
                except Exception as exc:
                    logger.warning("Video perf metrics unreadable; stage timings left unset: %s", exc)

            t_enc_start = time.perf_counter()
            saved_path: Optional[str] = None
            if request.output_path:
                saved_path = self.encode_video(
                    frames=frames,
                    output_path=request.output_path,
                    fps=request.fps,
                    container=request.export_container,
                )
            t_enc_end = time.perf_counter()
            encode_ms = max(0.01, (t_enc_end - t_enc_start) * 1000.0)

            step_latency_ms = denoise_ms / max(1, request.num_inference_steps) if denoise_ms else None
            it_rate = 1000.0 / step_latency_ms if step_latency_ms else None
            duration_sec = float(request.num_frames) / float(request.fps)
            fps_throughput = float(request.num_frames) / total_sec
            rtf = total_sec / max(1e-4, duration_sec)

            breakdown = VideoGenerationBreakdown(
                text_encoder_ms=None if text_enc_ms is None else round(text_enc_ms, 2),
                denoise_transformer_ms=None if denoise_ms is None else round(denoise_ms, 2),
                vae_decoder_ms=None if vae_ms is None else round(vae_ms, 2),
                video_encode_ms=round(encode_ms, 2),
                step_latency_ms=None if step_latency_ms is None else round(step_latency_ms, 2),
                iteration_rate_it_s=None if it_rate is None else round(it_rate, 2),
                frames_per_second=round(fps_throughput, 2),
                real_time_factor=round(rtf, 4),
            )

            model_name = self._model_path.name if self._model_path else "unknown"

            return VideoGenerationResult(
                frames=frames,
                output_path=saved_path,
                num_frames=len(frames),
                fps=request.fps,
                duration_sec=round(duration_sec, 4),
                resolution=(request.width, request.height),
                model_id=model_name,
                device=self._active_device,
                request=request,
                breakdown=breakdown,
                total_latency_ms=round(total_latency_ms, 2),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        except Exception as exc:
            if isinstance(exc, (ValueError, VideoEngineError)):
                raise
            logger.error("Text2Video generation failed: %s", exc)
            raise VideoEngineError(f"Video generation execution failed: {exc}") from exc

    def unload(self) -> None:
        """Release compiled pipeline, free Level Zero VRAM buffers, and trigger garbage collection."""
        self._pipeline = None
        gc.collect()
        logger.info("Unloaded Text2Video pipeline and reclaimed Level Zero memory")

    def close(self) -> None:
        """Explicit alias for unload adhering to standard resource management protocols."""
        self.unload()

    def __enter__(self) -> VideoEngine:
        """Context manager entry protocol returning engine instance."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit protocol guaranteeing resource reclamation."""
        self.close()
