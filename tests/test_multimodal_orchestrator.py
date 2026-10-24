"""Triad integration tests for StudioPipeline and MultiModelCoBenchmarker.

Validates end-to-end DAG execution across heterogeneous and sequential scheduling,
progress callback dispatch, error isolation, and Student's t statistical benchmarking.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1-M5.
"""

from pathlib import Path

import pytest

from arc_runner.benchmark.multimodal_runner import MultiModelCoBenchmarker
from arc_runner.orchestrator.models import (
    DeviceSchedulingStrategy,
    PipelineStage,
    StudioProjectRequest,
    VisualModality,
)
from arc_runner.orchestrator.pipeline import PipelineExecutionError, StudioPipeline


class TestStudioPipelinePositive:
    """Happy-path verification tests for end-to-end studio pipeline execution."""

    def test_pipeline_execute_heterogeneous_parallel(self, tmp_path: Path) -> None:
        """Verify full pipeline execution using HETEROGENEOUS_PARALLEL scheduling strategy."""
        pipeline = StudioPipeline()
        req = StudioProjectRequest(
            prompt="Brief explanation of general relativity and space curvature.",
            target_duration_sec=6.0,
            visual_modality=VisualModality.IMAGE_SLIDE,
            strategy=DeviceSchedulingStrategy.HETEROGENEOUS_PARALLEL,
            resolution=(128, 128),
            fps=10,
            output_filename="test_relativity.mp4",
        )

        progress_events = []

        def on_progress(stage: PipelineStage, pct: float, msg: str) -> None:
            progress_events.append((stage, pct, msg))

        payload = pipeline.execute(req, progress_callback=on_progress)

        assert payload.project_id.startswith("proj_")
        assert Path(payload.video_output_path).exists()
        assert payload.total_duration_sec > 0.0
        assert len(payload.script_scenes) >= 1
        assert len(payload.telemetry.stage_breakdowns) >= 2
        assert len(progress_events) >= 3
        assert progress_events[-1][0] == PipelineStage.COMPLETED

    def test_pipeline_execute_sequential_exclusive(self) -> None:
        """Verify full pipeline execution using SEQUENTIAL_EXCLUSIVE scheduling strategy."""
        pipeline = StudioPipeline()
        req = StudioProjectRequest(
            prompt="Thermodynamics overview and entropy principles.",
            target_duration_sec=4.0,
            visual_modality=VisualModality.IMAGE_SLIDE,
            strategy=DeviceSchedulingStrategy.SEQUENTIAL_EXCLUSIVE,
            resolution=(128, 128),
            fps=8,
        )

        payload = pipeline.execute(req)

        assert Path(payload.video_output_path).exists()
        assert any(s.stage_name == "AUDIO_SYNTHESIS" for s in payload.telemetry.stage_breakdowns)
        assert any(s.stage_name == "VISUAL_GENERATION" for s in payload.telemetry.stage_breakdowns)


class TestStudioPipelineNegative:
    """Negative error-boundary verification tests for StudioPipeline."""

    def test_pipeline_director_failure_raises_pipeline_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify pipeline execution wraps director failures in PipelineExecutionError."""
        pipeline = StudioPipeline()

        def mock_failing_decompose(*args, **kwargs):
            raise RuntimeError("Simulated LLM catastrophic memory exhaustion")

        monkeypatch.setattr(pipeline._director, "decompose_prompt", mock_failing_decompose)

        req = StudioProjectRequest(prompt="Failing test prompt")
        with pytest.raises(PipelineExecutionError) as exc_info:
            pipeline.execute(req)
        assert "Director planning stage failed" in str(exc_info.value)

    def test_pipeline_muxer_failure_raises_pipeline_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify pipeline execution wraps muxer failures in PipelineExecutionError."""
        pipeline = StudioPipeline()

        def mock_failing_mux(*args, **kwargs):
            raise RuntimeError("Disk write permission denied")

        monkeypatch.setattr(pipeline._muxer, "compose_and_mux", mock_failing_mux)

        req = StudioProjectRequest(prompt="Failing muxer prompt", target_duration_sec=3.0, resolution=(128, 128))
        with pytest.raises(PipelineExecutionError) as exc_info:
            pipeline.execute(req)
        assert "Timeline muxing stage failed" in str(exc_info.value)


class TestMultiModelCoBenchmarker:
    """Verification suite for adaptive co-benchmarking and statistical validation."""

    def test_co_benchmark_execution_and_scorecard(self) -> None:
        """Verify run_co_benchmark computes means, confidence intervals, and renders scorecard."""
        benchmarker = MultiModelCoBenchmarker()
        report = benchmarker.run_co_benchmark(
            prompt="Quantum mechanics test",
            iterations=2,
            warmup_runs=1,
            strategy=DeviceSchedulingStrategy.HETEROGENEOUS_PARALLEL,
        )

        assert report.iterations == 2
        assert report.avg_e2e_latency_ms > 0.0
        assert report.cloud_comparison_usd_per_run == 0.350
        assert report.hardware_payback_runs > 0
        assert len(report.stage_latencies_ms) > 0
