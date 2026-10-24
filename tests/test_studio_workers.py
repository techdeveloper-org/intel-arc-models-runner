"""Triad unit tests for StudioAudioWorker and StudioVisualWorker.

Validates speech waveform synthesis, visual frame rendering, USM zero-copy handoffs,
and boundary edge cases for silence buffers and camera motion interpolation.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1-M3.
"""

import numpy as np
import pytest

from arc_runner.orchestrator.audio_worker import StudioAudioWorker
from arc_runner.orchestrator.models import ScenePlan, VisualModality
from arc_runner.orchestrator.usm_bridge import USMZeroCopyBridge
from arc_runner.orchestrator.visual_worker import StudioVisualWorker


class TestStudioAudioWorker:
    """Verification suite for speech audio synthesis worker."""

    def test_audio_worker_synthesize_scene_waveform(self) -> None:
        """Verify synthesize_scene produces valid float32 PCM samples and measured duration."""
        worker = StudioAudioWorker(sample_rate=24000)
        scene = ScenePlan(
            scene_id=1,
            title="Intro",
            narration_text="Welcome to this lecture on electromagnetic radiation.",
            visual_prompt="Electromagnetic spectrum diagram.",
            duration_seconds=4.0,
        )

        artifact = worker.synthesize_scene(scene)

        assert artifact.scene_id == 1
        assert artifact.audio_samples is not None
        assert artifact.audio_sample_rate == 24000
        assert artifact.audio_duration_sec == pytest.approx(4.0, rel=0.05)
        assert len(artifact.audio_samples) == int(4.0 * 24000)

    def test_audio_worker_usm_bridge_integration(self) -> None:
        """Verify audio worker passes samples through USM zero-copy bridge."""
        bridge = USMZeroCopyBridge(track_telemetry=True)
        worker = StudioAudioWorker(sample_rate=16000)
        scene = ScenePlan(
            scene_id=2,
            title="Scene 2",
            narration_text="Testing zero copy USM handoff.",
            visual_prompt="Prompt.",
            duration_seconds=2.0,
        )

        artifact = worker.synthesize_scene(scene, usm_bridge=bridge)

        assert artifact.audio_samples is not None
        stats = bridge.get_cumulative_transfer_stats()
        assert stats["total_transfers"] >= 1.0

    def test_audio_worker_generate_silence_buffer(self) -> None:
        """Verify generate_silence_buffer produces exact zero-filled PCM buffer."""
        worker = StudioAudioWorker(sample_rate=24000)
        silence = worker.generate_silence_buffer(duration_sec=2.5, sample_rate=24000)

        assert len(silence) == int(2.5 * 24000)
        assert silence.dtype == np.float32
        assert np.all(silence == 0.0)

    def test_audio_worker_synthesize_all_scenes(self) -> None:
        """Verify synthesize_all_scenes iterates multiple scene plans."""
        worker = StudioAudioWorker(sample_rate=24000)
        scenes = [
            ScenePlan(scene_id=1, title="S1", narration_text="Part one.", visual_prompt="V1", duration_seconds=2.0),
            ScenePlan(scene_id=2, title="S2", narration_text="Part two.", visual_prompt="V2", duration_seconds=3.0),
        ]

        artifacts = worker.synthesize_all_scenes(scenes)
        assert len(artifacts) == 2
        assert artifacts[0].scene_id == 1
        assert artifacts[1].scene_id == 2


class TestStudioVisualWorker:
    """Verification suite for image and video visual synthesis worker."""

    def test_visual_worker_render_scene_slide(self) -> None:
        """Verify render_scene produces expected frame count for static slide modality."""
        worker = StudioVisualWorker(device="GPU")
        scene = ScenePlan(
            scene_id=1,
            title="Slide Scene",
            narration_text="Narrative.",
            visual_prompt="Cosmic microwave background radiation map.",
            duration_seconds=2.0,
            modality=VisualModality.IMAGE_SLIDE,
            camera_motion="slow pan",
        )

        artifact = worker.render_scene(scene, resolution=(256, 256), fps=10)

        assert artifact.scene_id == 1
        assert artifact.frames is not None
        assert len(artifact.frames) == 20
        assert artifact.video_duration_sec == pytest.approx(2.0, rel=0.01)
        assert artifact.frames[0].shape == (256, 256, 3)

    def test_visual_worker_usm_bridge_integration(self) -> None:
        """Verify visual worker transfers frame tensors via USM zero-copy bridge."""
        bridge = USMZeroCopyBridge(track_telemetry=True)
        worker = StudioVisualWorker(device="GPU")
        scene = ScenePlan(
            scene_id=1,
            title="Video Scene",
            narration_text="Narrative.",
            visual_prompt="Quantum entanglement simulation.",
            duration_seconds=1.0,
            modality=VisualModality.ANIMATED_VIDEO,
        )

        artifact = worker.render_scene(scene, resolution=(128, 128), fps=10, usm_bridge=bridge)

        assert artifact.frames is not None
        assert len(artifact.frames) == 10
        stats = bridge.get_cumulative_transfer_stats()
        assert stats["total_transfers"] >= 10.0

    def test_visual_worker_interpolate_slide_camera_motion(self) -> None:
        """Verify _interpolate_slide_frames produces specified frame count with pan motion."""
        worker = StudioVisualWorker()
        base_img = np.zeros((128, 128, 3), dtype=np.uint8)
        frames = worker._interpolate_slide_frames(base_img, num_frames=15, motion="slow pan")

        assert len(frames) == 15
        assert all(f.shape == (128, 128, 3) for f in frames)
