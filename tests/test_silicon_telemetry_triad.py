"""Triad test suite for SiliconTelemetryLogger and SiliconBottleneckAnalyzer.

Validates hardware metric logging, persistent JSON/MD reports, Roofline bandwidth analysis,
automated bottleneck detection rules, and actionable tuning recommendations.
"""

from pathlib import Path
import json
import pytest

from arc_runner.benchmarks.bottleneck_analyzer import (
    BottleneckAnalysisError,
    BottleneckDiagnosisDto,
    SiliconBottleneckAnalyzer,
)
from arc_runner.benchmarks.silicon_telemetry import (
    SiliconTelemetryLogger,
    SiliconTelemetryMetricsDto,
    TelemetryLoggingError,
)


class TestSiliconTelemetryPositive:
    """Positive test specifications for telemetry collection and bottleneck diagnosis."""

    def test_record_benchmark_when_normal_run_then_computes_accurate_mean_and_stddev(self) -> None:
        """Verify telemetry logger aggregates tokens, latency, and throughput stddev."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("qwen2.5:3b", "GPU", prompt_tokens=32, weight_size_gb=1.85)
        logger.record_prefill(duration_ms=45.0)

        for _ in range(10):
            logger.record_token_step(latency_ms=25.0, p_core_active=True)

        dto = logger.finish_benchmark()

        assert dto.model_name == "qwen2.5:3b"
        assert dto.generated_tokens == 10
        assert dto.tok_per_sec_mean == pytest.approx(40.0, abs=0.1)
        assert dto.tok_per_sec_stddev == pytest.approx(0.0, abs=0.1)
        assert dto.p_core_residency_pct == 100.0

    def test_persist_reports_when_benchmark_finished_then_writes_json_md_and_log(self, tmp_path: Path) -> None:
        """Verify persistent JSON, Markdown, and event log files are created on disk."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("test-model", "GPU")
        logger.record_prefill(30.0)
        logger.record_token_step(20.0)

        json_p, md_p, log_p = logger.persist_reports(output_dir=tmp_path, file_prefix="test_report")

        assert json_p.exists()
        assert md_p.exists()
        assert log_p.exists()

        data = json.loads(json_p.read_text(encoding="utf-8"))
        assert data["model_name"] == "test-model"
        assert "Silicon Benchmark Report" in md_p.read_text(encoding="utf-8")
        assert "START benchmark" in log_p.read_text(encoding="utf-8")

    def test_analyze_metrics_when_optimal_telemetry_then_returns_optimal_status(self) -> None:
        """Verify clean telemetry with flat tok/s yields OPTIMAL status."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("qwen2.5:3b", "GPU")
        for _ in range(20):
            logger.record_token_step(latency_ms=25.0, p_core_active=True)

        metrics = logger.finish_benchmark()
        diagnosis = SiliconBottleneckAnalyzer.analyze_metrics(metrics)

        assert diagnosis.performance_score >= 90.0
        assert isinstance(diagnosis, BottleneckDiagnosisDto)

    def test_analyze_file_when_valid_json_report_then_parses_and_diagnoses_cleanly(self, tmp_path: Path) -> None:
        """Verify SiliconBottleneckAnalyzer parses report file directly from disk."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("qwen2.5:3b", "GPU")
        logger.record_token_step(25.0)
        json_p, _, _ = logger.persist_reports(output_dir=tmp_path)

        diagnosis = SiliconBottleneckAnalyzer.analyze_file(json_p)

        assert diagnosis.model_name == "qwen2.5:3b"

    def test_format_diagnostic_report_when_diagnosed_then_outputs_executive_summary(self) -> None:
        """Verify formatting outputs readable CLI executive summary."""
        diagnosis = BottleneckDiagnosisDto(
            model_name="qwen2.5:3b",
            target_device="GPU",
            status="OPTIMAL",
            performance_score=95.0,
            roofline_classification="BALANCED",
            identified_bottlenecks=[],
            actionable_recommendations=[],
        )

        text = SiliconBottleneckAnalyzer.format_diagnostic_report(diagnosis)
        assert "SILICON BOTTLENECK DIAGNOSIS" in text
        assert "OPTIMAL" in text


class TestSiliconTelemetryNegative:
    """Negative test specifications verifying detection of performance bottlenecks."""

    def test_analyze_file_when_file_not_found_then_raises_bottleneck_analysis_error(self) -> None:
        """Verify missing report path raises BottleneckAnalysisError."""
        with pytest.raises(BottleneckAnalysisError):
            SiliconBottleneckAnalyzer.analyze_file(Path("non_existent_report.json"))

    def test_analyze_file_when_corrupted_json_then_raises_bottleneck_analysis_error(self, tmp_path: Path) -> None:
        """Verify corrupted JSON raises BottleneckAnalysisError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{broken json", encoding="utf-8")

        with pytest.raises(BottleneckAnalysisError):
            SiliconBottleneckAnalyzer.analyze_file(bad_file)

    def test_analyze_metrics_when_severe_jitter_then_flags_core_director_bottleneck(self) -> None:
        """Verify high throughput standard deviation triggers jitter bottleneck detection."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("jittery-model", "GPU")
        logger.record_token_step(15.0)
        logger.record_token_step(80.0)
        logger.record_token_step(15.0)
        logger.record_token_step(90.0)

        metrics = logger.finish_benchmark()
        diagnosis = SiliconBottleneckAnalyzer.analyze_metrics(metrics)

        assert any("jitter" in b.lower() for b in diagnosis.identified_bottlenecks)
        assert any("threadaffinityguard" in r.lower() for r in diagnosis.actionable_recommendations)

    def test_analyze_metrics_when_gc_pauses_occurred_then_flags_gc_stall(self) -> None:
        """Verify recorded GC pauses trigger GC burst gate recommendation."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("gc-stalled-model", "GPU")
        logger.record_token_step(20.0, gc_occurred=True, gc_freeze_ms=120.0)

        metrics = logger.finish_benchmark()
        diagnosis = SiliconBottleneckAnalyzer.analyze_metrics(metrics)

        assert any("garbage collection" in b.lower() for b in diagnosis.identified_bottlenecks)
        assert any("gcburstgate" in r.lower() for r in diagnosis.actionable_recommendations)

    def test_analyze_metrics_when_thermal_downclock_then_flags_power_throttling(self) -> None:
        """Verify low GPU frequency triggers Sysman frequency lock recommendation."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("throttled-model", "GPU", gpu_clock_mhz=1300.0)
        logger.record_token_step(25.0)

        metrics = logger.finish_benchmark()
        diagnosis = SiliconBottleneckAnalyzer.analyze_metrics(metrics)

        assert any("throttled" in b.lower() for b in diagnosis.identified_bottlenecks)
        assert any("sysman" in r.lower() for r in diagnosis.actionable_recommendations)


class TestSiliconTelemetryEdgeCase:
    """Edge-case test specifications verifying boundaries and stress conditions."""

    def test_finish_benchmark_when_zero_tokens_recorded_then_returns_zero_metrics(self) -> None:
        """Verify finish_benchmark handles empty token stream without division by zero."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("empty-model", "GPU")
        metrics = logger.finish_benchmark()

        assert metrics.generated_tokens == 0
        assert metrics.tok_per_sec_mean == 0.0
        assert metrics.effective_memory_bandwidth_gbps == 0.0

    def test_persist_reports_when_custom_prefix_then_uses_specified_prefix(self, tmp_path: Path) -> None:
        """Verify custom prefix applies to all generated reports."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("custom-model", "GPU")
        json_p, md_p, log_p = logger.persist_reports(output_dir=tmp_path, file_prefix="custom_run_01")

        assert json_p.name == "custom_run_01.json"
        assert md_p.name == "custom_run_01.md"
        assert log_p.name == "custom_run_01.log"

    def test_effective_bandwidth_when_high_tokens_then_saturates_bus_correctly(self) -> None:
        """Verify roofline bandwidth calculation on fast token decoding."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("fast-model", "GPU", weight_size_gb=2.0)
        for _ in range(50):
            logger.record_token_step(latency_ms=25.0)

        metrics = logger.finish_benchmark()
        assert metrics.effective_memory_bandwidth_gbps > 50.0
        assert metrics.memory_bus_saturation_pct > 50.0

    def test_analyze_metrics_when_p_core_residency_drops_then_deducts_health_score(self) -> None:
        """Verify core hopping deducts from performance score."""
        logger = SiliconTelemetryLogger()
        logger.start_benchmark("core-hop-model", "GPU")
        logger.record_token_step(25.0, p_core_active=True)
        logger.record_token_step(25.0, p_core_active=False)

        metrics = logger.finish_benchmark()
        diagnosis = SiliconBottleneckAnalyzer.analyze_metrics(metrics)

        assert metrics.p_core_residency_pct == 50.0
        assert diagnosis.performance_score < 100.0
