"""Triad unit tests for StudioTimelineCompositor, Muxer, and Pacing Convergence Loop.

Validates multi-scene audio-video concatenation, closed-loop pacing convergence,
error-path rejections on empty streams, and extreme duration mismatch edge cases.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proof M3.
"""

from pathlib import Path
import numpy as np
import pytest

from arc_runner.orchestrator.models import SceneArtifact, ScenePlan
from arc_runner.orchestrator.muxer import MuxerError, StudioTimelineCompositor


class TestStudioMuxerPositive:
    """Happy-path verification tests for timeline composition and muxing."""

    def test_muxer_compose_and_mux_two_scenes(self, tmp_path: Path) -> None:
        """Verify compose_and_mux outputs valid MP4 file and matching total duration."""
        compositor = StudioTimelineCompositor(output_dir=tmp_path)
        output_file = tmp_path / "final_output.mp4"

        frames_s1 = [np.full((128, 128, 3), 50, dtype=np.uint8) for _ in range(16)]
        audio_s1 = np.sin(np.linspace(0, 440, 24000)).astype(np.float32)

        frames_s2 = [np.full((128, 128, 3), 150, dtype=np.uint8) for _ in range(16)]
        audio_s2 = np.sin(np.linspace(0, 880, 24000)).astype(np.float32)

        scenes = [
            ScenePlan(scene_id=1, title="S1", narration_text="Intro speech.", visual_prompt="V1", duration_seconds=1.0),
            ScenePlan(scene_id=2, title="S2", narration_text="Second speech.", visual_prompt="V2", duration_seconds=1.0),
        ]

        artifacts = [
            SceneArtifact(
                scene_id=1,
                frames=frames_s1,
                audio_samples=audio_s1,
                audio_sample_rate=24000,
                audio_duration_sec=1.0,
                video_duration_sec=1.0,
                fps=16,
            ),
            SceneArtifact(
                scene_id=2,
                frames=frames_s2,
                audio_samples=audio_s2,
                audio_sample_rate=24000,
                audio_duration_sec=1.0,
                video_duration_sec=1.0,
                fps=16,
            ),
        ]

        final_path, total_dur, iters = compositor.compose_and_mux(scenes, artifacts, output_file, fps=16)

        assert final_path.exists()
        assert final_path.stat().st_size > 0
        assert total_dur == pytest.approx(2.0, rel=0.05)


class TestStudioMuxerNegative:
    """Negative error-boundary verification tests for Muxer."""

    def test_muxer_empty_artifacts_list_raises_error(self, tmp_path: Path) -> None:
        """Verify compose_and_mux raises MuxerError on empty artifacts input."""
        compositor = StudioTimelineCompositor(output_dir=tmp_path)
        dest = tmp_path / "empty.mp4"

        with pytest.raises(MuxerError) as exc_info:
            compositor.compose_and_mux(scenes=[], artifacts=[], output_path=dest)
        assert "empty scene artifacts" in str(exc_info.value)

    def test_muxer_no_frames_raises_error(self, tmp_path: Path) -> None:
        """Verify compose_and_mux raises MuxerError if artifacts have zero frames."""
        compositor = StudioTimelineCompositor(output_dir=tmp_path)
        dest = tmp_path / "no_frames.mp4"
        scenes = [ScenePlan(scene_id=1, title="S1", narration_text="Text", visual_prompt="V", duration_seconds=1.0)]
        artifacts = [
            SceneArtifact(scene_id=1, frames=[], audio_samples=np.zeros(24000, dtype=np.float32), audio_duration_sec=1.0)
        ]

        with pytest.raises(MuxerError) as exc_info:
            compositor.compose_and_mux(scenes=scenes, artifacts=artifacts, output_path=dest)
        assert "No visual frames available" in str(exc_info.value)


class TestStudioMuxerPacingConvergenceEdgeCases:
    """Edge-case tests for the closed-loop audio-video pacing convergence algorithm."""

    def test_pacing_convergence_pads_video_when_audio_longer(self) -> None:
        """Verify pacing convergence pads video frames to match longer audio without drift."""
        compositor = StudioTimelineCompositor(convergence_tolerance_sec=0.05)
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(16)]
        audio_dur = 2.0
        audio_samples = np.zeros(int(audio_dur * 24000), dtype=np.float32)

        raw_artifact = SceneArtifact(
            scene_id=1,
            frames=frames,
            audio_samples=audio_samples,
            audio_sample_rate=24000,
            audio_duration_sec=audio_dur,
            video_duration_sec=1.0,
            fps=16,
        )

        synced, iters = compositor._align_scene_duration(raw_artifact, target_fps=16)

        assert iters >= 1
        assert len(synced.frames) == 32
        assert synced.video_duration_sec == pytest.approx(2.0, abs=0.05)
        assert synced.temporal_drift_sec <= 0.05

    def test_pacing_convergence_pads_audio_silence_when_video_longer(self) -> None:
        """Verify pacing convergence appends silence to audio when video frames exceed speech."""
        compositor = StudioTimelineCompositor(convergence_tolerance_sec=0.05)
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(32)]
        audio_dur = 1.0
        audio_samples = np.zeros(int(audio_dur * 24000), dtype=np.float32)

        raw_artifact = SceneArtifact(
            scene_id=1,
            frames=frames,
            audio_samples=audio_samples,
            audio_sample_rate=24000,
            audio_duration_sec=audio_dur,
            video_duration_sec=2.0,
            fps=16,
        )

        synced, iters = compositor._align_scene_duration(raw_artifact, target_fps=16)

        assert iters >= 1
        assert synced.temporal_drift_sec <= 0.05
        assert len(synced.audio_samples) == pytest.approx(int(2.0 * 24000), abs=100)
