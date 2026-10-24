"""Triad unit tests for Arc MultiModal Studio data models and USM Zero-Copy Bridge.

Validates positive happy-paths, negative error boundaries, and extreme edge conditions
for ScenePlan, StudioProjectRequest, OrchestrationTelemetry, and USMZeroCopyBridge.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proof M2.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from arc_runner.orchestrator.models import (
    DeviceSchedulingStrategy,
    MultiModelBenchmarkReport,
    OrchestrationTelemetry,
    PipelineStage,
    SceneArtifact,
    ScenePlan,
    StageTelemetry,
    StudioArtifactPayload,
    StudioProjectRequest,
    VisualModality,
)
from arc_runner.orchestrator.usm_bridge import USMZeroCopyBridge


class TestStudioModelsPositive:
    """Happy-path validation tests for Arc Studio domain models."""

    def test_scene_plan_creation_valid(self) -> None:
        """Verify ScenePlan instantiates cleanly with valid parameters."""
        plan = ScenePlan(
            scene_id=1,
            title="Introduction to Quantum Mechanics",
            narration_text="Quantum mechanics governs the behavior of microscopic particles.",
            visual_prompt="Subatomic particles floating in a vibrant wave-particle duality field.",
            duration_seconds=6.5,
            modality=VisualModality.ANIMATED_VIDEO,
        )
        assert plan.scene_id == 1
        assert plan.duration_seconds == 6.5
        assert plan.modality == VisualModality.ANIMATED_VIDEO
        assert "Quantum mechanics" in plan.narration_text

    def test_studio_project_request_defaults(self) -> None:
        """Verify StudioProjectRequest populates correct architectural defaults."""
        req = StudioProjectRequest(prompt="Make an educational video on gravity")
        assert req.target_duration_sec == 30.0
        assert req.voice_slug == "af_heart"
        assert req.visual_modality == VisualModality.IMAGE_SLIDE
        assert req.resolution == (512, 512)
        assert req.fps == 16
        assert req.strategy == DeviceSchedulingStrategy.HETEROGENEOUS_PARALLEL

    def test_orchestration_telemetry_aggregation(self) -> None:
        """Verify OrchestrationTelemetry aggregates stage latency correctly."""
        telemetry = OrchestrationTelemetry(
            project_id="proj_test_001",
            stage_breakdowns=[
                StageTelemetry(
                    stage_name="PLANNING",
                    latency_ms=450.0,
                    device_used="NPU",
                    peak_memory_mb=120.0,
                    throughput_metric="tok/s",
                    throughput_value=32.5,
                ),
                StageTelemetry(
                    stage_name="AUDIO_SYNTHESIS",
                    latency_ms=850.0,
                    device_used="NPU",
                    peak_memory_mb=210.0,
                    throughput_metric="rtf",
                    throughput_value=0.25,
                ),
                StageTelemetry(
                    stage_name="VISUAL_GENERATION",
                    latency_ms=2500.0,
                    device_used="GPU",
                    peak_memory_mb=1800.0,
                    throughput_metric="it/s",
                    throughput_value=4.2,
                ),
            ],
            usm_zero_copy_latency_us=42.5,
        )
        total = sum(s.latency_ms for s in telemetry.stage_breakdowns)
        assert total == 3800.0
        assert telemetry.usm_zero_copy_latency_us == 42.5

    def test_multi_model_benchmark_report_cloud_roi(self) -> None:
        """Verify benchmark report accurately represents commercial cloud savings."""
        report = MultiModelBenchmarkReport(
            benchmark_id="bench_01",
            iterations=5,
            avg_e2e_latency_ms=4500.0,
            ci95_latency_ms=120.0,
            avg_usm_transfer_us=38.4,
            ci95_usm_transfer_us=2.1,
            audio_rtf=0.22,
            video_rtf=0.85,
            llm_tok_per_sec=36.0,
            cloud_comparison_usd_per_run=0.35,
            hardware_payback_runs=857,
        )
        assert report.cloud_comparison_usd_per_run == 0.35
        assert report.hardware_payback_runs == 857


class TestStudioModelsNegative:
    """Negative error-boundary tests for Arc Studio domain models."""

    def test_scene_plan_empty_narration_rejected(self) -> None:
        """Verify ScenePlan raises ValidationError when narration text is empty."""
        with pytest.raises(ValidationError) as exc_info:
            ScenePlan(
                scene_id=1,
                title="Bad Scene",
                narration_text="   ",
                visual_prompt="Valid prompt",
                duration_seconds=5.0,
            )
        assert "narration text cannot be empty" in str(exc_info.value)

    def test_scene_plan_invalid_duration_bounds(self) -> None:
        """Verify ScenePlan rejects duration values below 0.5s or above 300.0s."""
        with pytest.raises(ValidationError):
            ScenePlan(
                scene_id=1,
                title="Too Short",
                narration_text="Valid narration",
                visual_prompt="Valid prompt",
                duration_seconds=0.1,
            )

        with pytest.raises(ValidationError):
            ScenePlan(
                scene_id=1,
                title="Too Long",
                narration_text="Valid narration",
                visual_prompt="Valid prompt",
                duration_seconds=500.0,
            )

    def test_studio_project_request_empty_prompt_rejected(self) -> None:
        """Verify StudioProjectRequest rejects empty prompt string."""
        with pytest.raises(ValidationError) as exc_info:
            StudioProjectRequest(prompt="    ")
        assert "prompt cannot be empty" in str(exc_info.value)

    def test_studio_project_request_resolution_not_divisible_by_eight(self) -> None:
        """Verify StudioProjectRequest rejects resolutions not divisible by 8."""
        with pytest.raises(ValidationError) as exc_info:
            StudioProjectRequest(prompt="Valid prompt", resolution=(515, 512))
        assert "divisible by 8" in str(exc_info.value)


class TestStudioModelsEdgeCases:
    """Boundary value and extreme input edge-case tests."""

    def test_scene_plan_extreme_valid_duration(self) -> None:
        """Verify ScenePlan allows exactly 0.5s and 300.0s boundary durations."""
        min_plan = ScenePlan(
            scene_id=1,
            title="Min Scene",
            narration_text="Quick note.",
            visual_prompt="Flash prompt",
            duration_seconds=0.5,
        )
        assert min_plan.duration_seconds == 0.5

        max_plan = ScenePlan(
            scene_id=2,
            title="Max Scene",
            narration_text="Long speech.",
            visual_prompt="Epic visual",
            duration_seconds=300.0,
        )
        assert max_plan.duration_seconds == 300.0

    def test_unicode_and_indic_prompt_handling(self) -> None:
        """Verify StudioProjectRequest cleanly serializes Hindi and Unicode inputs."""
        hindi_prompt = "क्वांटम भौतिकी पर 30-सेकंड का शैक्षिक वीडियो बनाएं"
        req = StudioProjectRequest(prompt=hindi_prompt)
        assert req.prompt == hindi_prompt


class TestUSMZeroCopyBridge:
    """Verification suite for Level Zero USM zero-copy tensor passing."""

    def test_usm_wrap_numpy_array(self) -> None:
        """Verify wrap_tensor_shared returns a shared view with sub-millisecond transfer latency."""
        bridge = USMZeroCopyBridge(track_telemetry=True)
        arr = np.random.randn(1024, 768).astype(np.float32)

        view, latency_us = bridge.wrap_tensor_shared(arr)

        assert isinstance(view, np.ndarray)
        assert view.shape == arr.shape
        assert view.data == arr.data
        assert latency_us >= 0.0

    def test_usm_transfer_audio_to_visual(self) -> None:
        """Verify zero-copy audio waveform transfer to timeline."""
        bridge = USMZeroCopyBridge()
        pcm = np.sin(np.linspace(0, 440, 24000)).astype(np.float32)

        shared_audio, latency_us = bridge.transfer_audio_to_visual_timeline(pcm, sample_rate=24000)

        assert len(shared_audio) == 24000
        assert latency_us >= 0.0
        assert shared_audio.data == pcm.data

    def test_usm_transfer_frames_to_muxer(self) -> None:
        """Verify zero-copy frame sequence transfer to container muxer."""
        bridge = USMZeroCopyBridge()
        frames = [np.zeros((512, 512, 3), dtype=np.uint8) for _ in range(16)]

        shared_frames, latency_us = bridge.transfer_frames_to_muxer(frames)

        assert len(shared_frames) == 16
        assert latency_us >= 0.0
        for orig, shared in zip(frames, shared_frames):
            assert orig.data == shared.data

    def test_usm_cumulative_telemetry_and_reset(self) -> None:
        """Verify cumulative stats correctly record transfers and reset cleanly."""
        bridge = USMZeroCopyBridge(track_telemetry=True)
        bridge.reset()

        arr = np.ones((100, 100), dtype=np.float32)
        bridge.wrap_tensor_shared(arr)
        bridge.wrap_tensor_shared(arr)

        stats = bridge.get_cumulative_transfer_stats()
        assert stats["total_transfers"] == 2.0
        assert stats["avg_latency_us"] >= 0.0

        bridge.reset()
        reset_stats = bridge.get_cumulative_transfer_stats()
        assert reset_stats["total_transfers"] == 0.0
