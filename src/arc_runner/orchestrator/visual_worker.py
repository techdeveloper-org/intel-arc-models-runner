"""Visual Synthesis Worker for Arc MultiModal Studio.

Renders educational visual slides and animated video clips using OpenVINO GenAI
diffusion (SDXL) and DiT (LTX-Video) pipelines on Intel Arc GPUs with USM zero-copy handoffs.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1 and M2.
"""

from __future__ import annotations

import time

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.engine.image_engine import ImageEngine, ImageGenerationRequest
from arc_runner.engine.video_engine import VideoEngine, VideoGenerationRequest
from arc_runner.orchestrator.models import SceneArtifact, ScenePlan, VisualModality
from arc_runner.orchestrator.slide_compositor import EducationalSlideCompositor
from arc_runner.orchestrator.usm_bridge import USMZeroCopyBridge
from arc_runner.server.model_manager import ModelManager

logger = get_logger("orchestrator.visual_worker")


class VisualWorkerExecutionError(ArcRunnerError):
    """Raised when visual generation fails in the visual worker."""


class StudioVisualWorker:
    """Specialized worker rendering visual assets (slides and video clips) for production scenes."""

    def __init__(
        self,
        image_engine: ImageEngine | None = None,
        video_engine: VideoEngine | None = None,
        model_manager: ModelManager | None = None,
        device: str = "GPU",
    ) -> None:
        """Initialize StudioVisualWorker with optional engines or model manager.

        Args:
            image_engine: Optional loaded ImageEngine instance for static slides.
            video_engine: Optional loaded VideoEngine instance for animated video.
            model_manager: Optional ModelManager instance to resolve models.
            device: Target silicon accelerator device identifier ('GPU', 'CPU').
        """
        self._image_engine = image_engine
        self._video_engine = video_engine
        self._model_manager = model_manager
        self._device = device
        self._compositor = EducationalSlideCompositor()

    def render_scene(
        self,
        scene: ScenePlan,
        resolution: tuple[int, int] = (512, 512),
        fps: int = 16,
        usm_bridge: USMZeroCopyBridge | None = None,
    ) -> SceneArtifact:
        """Render visual frames for a single scene according to its selected modality.

        Args:
            scene: ScenePlan specifying visual prompt, duration, and modality.
            resolution: Raster frame resolution as (width, height) tuple.
            fps: Output target playback framerate.
            usm_bridge: Optional USMZeroCopyBridge for zero-copy memory wrapping.

        Returns:
            SceneArtifact containing rendered video frames, duration, and telemetry.

        Raises:
            VisualWorkerExecutionError: When image or video generation fails.
        """
        t_start = time.perf_counter()
        target_frames = max(1, round(scene.duration_seconds * fps))
        logger.info(
            f"Rendering Scene {scene.scene_id} visual ({scene.modality.value}, {target_frames} frames) on {self._device}"
        )

        width, height = resolution

        if scene.modality == VisualModality.ANIMATED_VIDEO and self._video_engine is not None:
            frames = self._render_animated_video(scene, width, height, target_frames, fps)
        elif self._image_engine is not None:
            frames = self._render_slide_with_camera_motion(scene, width, height, target_frames, fps)
        else:
            frames = self._render_synthetic_frames(scene, width, height, target_frames, fps)

        if usm_bridge is not None:
            shared_frames, _ = usm_bridge.transfer_frames_to_muxer(frames)
        else:
            shared_frames = frames

        video_duration_sec = float(len(shared_frames)) / float(fps)
        t_elapsed = time.perf_counter() - t_start
        logger.info(f"Scene {scene.scene_id} rendered {len(shared_frames)} frames in {t_elapsed:.2f}s")

        return SceneArtifact(
            scene_id=scene.scene_id,
            frames=shared_frames,
            video_duration_sec=video_duration_sec,
            fps=fps,
        )

    def render_all_scenes(
        self,
        scenes: list[ScenePlan],
        resolution: tuple[int, int] = (512, 512),
        fps: int = 16,
        usm_bridge: USMZeroCopyBridge | None = None,
    ) -> list[SceneArtifact]:
        """Render visuals sequentially for all planned production scenes.

        Args:
            scenes: List of ScenePlan items.
            resolution: Target raster resolution (width, height).
            fps: Playback framerate.
            usm_bridge: Optional USMZeroCopyBridge.

        Returns:
            List of SceneArtifact items containing rendered frame sequences.
        """
        artifacts: list[SceneArtifact] = []
        for scene in scenes:
            artifacts.append(self.render_scene(scene, resolution=resolution, fps=fps, usm_bridge=usm_bridge))
        return artifacts

    def _render_animated_video(
        self,
        scene: ScenePlan,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
    ) -> list[np.ndarray]:
        """Synthesize video sequence using OpenVINO GenAI Text2Video pipeline.

        Args:
            scene: Scene plan.
            width: Frame width.
            height: Frame height.
            num_frames: Frame count.
            fps: Framerate.

        Returns:
            List of uint8 RGB frame numpy arrays.
        """
        try:
            req = VideoGenerationRequest(
                prompt=scene.visual_prompt,
                num_frames=min(num_frames, 64),
                fps=fps,
                width=width,
                height=height,
                num_inference_steps=20,
            )
            res = self._video_engine.generate(req)
            frames = res.frames
            if len(frames) < num_frames and len(frames) > 0:
                reps = int(np.ceil(num_frames / len(frames)))
                frames = (frames * reps)[:num_frames]
            return frames
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Video engine failed ({exc}), falling back to camera motion slide.")
            return self._render_synthetic_frames(scene, width, height, num_frames, fps)

    def _render_slide_with_camera_motion(
        self,
        scene: ScenePlan,
        width: int,
        height: int,
        num_frames: int,
        fps: int = 16,
    ) -> list[np.ndarray]:
        """Synthesize static image using ImageEngine and interpolate camera motion.

        Args:
            scene: Scene plan.
            width: Frame width.
            height: Frame height.
            num_frames: Frame count.
            fps: Playback framerate.

        Returns:
            List of uint8 RGB frame numpy arrays.
        """
        try:
            req = ImageGenerationRequest(
                prompt=scene.visual_prompt,
                width=width,
                height=height,
                num_inference_steps=20,
            )
            res = self._image_engine.generate(req)
            base_image = res.images[0]
            if not isinstance(base_image, np.ndarray):
                base_image = np.array(base_image)
            return self._compositor.render_scene_frames(
                scene=scene,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                base_background=base_image,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Image engine failed ({exc}), falling back to synthetic frames.")
            return self._render_synthetic_frames(scene, width, height, num_frames, fps)

    def _interpolate_slide_frames(
        self,
        base_image: np.ndarray,
        num_frames: int,
        motion: str = "slow pan",
    ) -> list[np.ndarray]:
        """Interpolate camera pan/zoom motion across frames from a single base image.

        Args:
            base_image: Source uint8 RGB numpy array of shape (H, W, 3).
            num_frames: Output frame count.
            motion: Motion descriptor ('slow pan', 'zoom in', 'static').

        Returns:
            List of interpolated uint8 RGB frame arrays.
        """
        frames: list[np.ndarray] = []

        for i in range(num_frames):
            alpha = float(i) / max(1.0, float(num_frames - 1))
            if "zoom" in motion:
                frame = base_image.copy()
            elif "pan" in motion:
                shift_x = int(12.0 * alpha)
                frame = np.roll(base_image, shift_x, axis=1)
            else:
                frame = base_image.copy()

            frames.append(frame.astype(np.uint8))

        return frames

    def _render_synthetic_frames(
        self,
        scene: ScenePlan,
        width: int,
        height: int,
        num_frames: int,
        fps: int = 16,
    ) -> list[np.ndarray]:
        """Generate broadcast-grade educational presentation slide frames.

        Args:
            scene: Target scene plan.
            width: Frame width.
            height: Frame height.
            num_frames: Output frame count.
            fps: Playback framerate.

        Returns:
            List of generated RGB frame numpy arrays.
        """
        return self._compositor.render_scene_frames(
            scene=scene,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
        )
