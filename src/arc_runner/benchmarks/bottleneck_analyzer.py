"""Automated bottleneck diagnosis engine for Intel Arc LLM inference.

Parses persistent silicon telemetry benchmark reports, diagnoses performance cliffs against
hardware heuristics, and prescribes prioritized, actionable system tuning recommendations.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from arc_runner.benchmarks.silicon_telemetry import SiliconTelemetryMetricsDto
from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.benchmarks.bottleneck_analyzer")


class BottleneckAnalysisError(ArcRunnerError):
    """Raised when parsing or diagnosing telemetry reports fails."""

    def __init__(self, message: str) -> None:
        """Initialize bottleneck analysis error.

        Args:
            message: Failure reason details.
        """
        super().__init__(message, "BOTTLENECK_ANALYSIS_ERROR", 422)


@dataclass(frozen=True)
class BottleneckDiagnosisDto:
    """Descriptor DTO summarizing automated bottleneck findings and tuning actions."""

    model_name: str
    target_device: str
    status: str
    performance_score: float
    roofline_classification: str
    identified_bottlenecks: List[str]
    actionable_recommendations: List[str]


class SiliconBottleneckAnalyzer:
    """Diagnoses performance limiters from telemetry metrics and suggests remediations."""

    MAX_ACCEPTABLE_JITTER_STDDEV: float = 1.5
    MIN_ACCEPTABLE_P_CORE_RESIDENCY: float = 98.0
    HIGH_BUS_SATURATION_THRESHOLD: float = 70.0

    @classmethod
    def analyze_metrics(cls, m: SiliconTelemetryMetricsDto) -> BottleneckDiagnosisDto:
        """Analyze structured telemetry metrics DTO and output diagnosis.

        Args:
            m: Telemetry metrics descriptor.

        Returns:
            BottleneckDiagnosisDto: Comprehensive diagnostic findings and prescriptions.
        """
        bottlenecks: List[str] = []
        recommendations: List[str] = []
        score = 100.0

        if m.memory_bus_saturation_pct >= cls.HIGH_BUS_SATURATION_THRESHOLD:
            roofline = "MEMORY_BANDWIDTH_BOUND"
            bottlenecks.append(
                f"Memory bus saturation reached {m.memory_bus_saturation_pct:.1f}% "
                f"({m.effective_memory_bandwidth_gbps:.1f} GB/s of 112 GB/s peak)."
            )
            recommendations.append(
                "Weights are memory-bus bound. Apply Symmetric INT4 Group 128 quantization "
                "or enable static Level Zero USM page-locked buffers to maximize DMA bus saturation."
            )
        else:
            roofline = "COMPUTE_BOUND"

        if m.tok_per_sec_stddev > cls.MAX_ACCEPTABLE_JITTER_STDDEV:
            score -= 15.0
            bottlenecks.append(
                f"Token speed jitter standard deviation ({m.tok_per_sec_stddev:.2f} tok/s) "
                f"exceeds stability ceiling ({cls.MAX_ACCEPTABLE_JITTER_STDDEV:.2f} tok/s)."
            )
            recommendations.append(
                "Activate ThreadAffinityGuard with Win32 P-core mask 0x00FF and elevation "
                "to THREAD_PRIORITY_HIGHEST to stop Thread Director migration."
            )

        if m.p_core_residency_pct < cls.MIN_ACCEPTABLE_P_CORE_RESIDENCY:
            score -= 20.0
            bottlenecks.append(
                f"Inference thread P-Core residency dropped to {m.p_core_residency_pct:.1f}% "
                "(OS scheduler routed feeder thread to E-cores or LP-E cores)."
            )
            recommendations.append(
                "Enforce strict Win32 SetThreadAffinityMask pinning to dedicate physical P-cores 0-7."
            )

        if m.gc_pause_count > 0:
            score -= 15.0
            bottlenecks.append(
                f"Python cyclic garbage collection halted decode loop {m.gc_pause_count} times "
                f"causing {m.gc_total_freeze_ms:.1f} ms freeze frames."
            )
            recommendations.append(
                "Wrap inference loop with GCBurstGate (gc.disable() during decode, single bounded sweep on finish)."
            )

        if m.gpu_clock_mhz < 1600.0:
            score -= 10.0
            bottlenecks.append(
                f"GPU clock throttled down to {m.gpu_clock_mhz:.0f} MHz (thermal or PL1 budget dropoff)."
            )
            recommendations.append(
                "Lock sustained GPU equilibrium frequency via Level Zero Sysman to 1.85 GHz."
            )

        if m.sqnr_average_db < 20.0:
            score -= 25.0
            bottlenecks.append(
                f"Average layer SQNR ({m.sqnr_average_db:.1f} dB) falls below fidelity threshold."
            )
            recommendations.append(
                "Quantization noise detected. Retain sensitive projection layers in FP16 or use INT8 repacking."
            )

        status = "OPTIMAL" if not bottlenecks else ("BOTTLENECK_DETECTED" if score >= 60.0 else "DEGRADED")
        score = max(0.0, min(100.0, score))

        return BottleneckDiagnosisDto(
            model_name=m.model_name,
            target_device=m.target_device,
            status=status,
            performance_score=score,
            roofline_classification=roofline,
            identified_bottlenecks=bottlenecks,
            actionable_recommendations=recommendations,
        )

    @classmethod
    def analyze_file(cls, report_json_path: Path) -> BottleneckDiagnosisDto:
        """Parse on-disk JSON telemetry report and perform automated bottleneck diagnosis.

        Args:
            report_json_path: Path to silicon benchmark JSON file.

        Returns:
            BottleneckDiagnosisDto: Diagnosis results.

        Raises:
            BottleneckAnalysisError: If file is absent or invalid JSON.
        """
        p = Path(report_json_path)
        if not p.is_file():
            raise BottleneckAnalysisError(f"Telemetry report file not found at '{p}'.")

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = SiliconTelemetryMetricsDto(**data)
        except Exception as exc:
            raise BottleneckAnalysisError(f"Failed to parse telemetry JSON from '{p}': {exc}")

        return cls.analyze_metrics(metrics)

    @classmethod
    def format_diagnostic_report(cls, d: BottleneckDiagnosisDto) -> str:
        """Format diagnosis into clean executive report.

        Args:
            d: Diagnosis DTO.

        Returns:
            str: Formatted report text.
        """
        lines = [
            f"================================================================================",
            f"SILICON BOTTLENECK DIAGNOSIS: {d.model_name} on {d.target_device}",
            f"Status: {d.status} | Performance Health Score: {d.performance_score:.1f} / 100",
            f"Roofline Classification: {d.roofline_classification}",
            f"================================================================================",
        ]

        if not d.identified_bottlenecks:
            lines.append("✔ No hardware bottlenecks detected. Engine operating at peak silicon efficiency.")
        else:
            lines.append(f"Detected {len(d.identified_bottlenecks)} Bottlenecks:")
            for idx, b in enumerate(d.identified_bottlenecks, 1):
                lines.append(f"  {idx}. [!] {b}")

            lines.append("\nActionable System Tuning Prescriptions:")
            for idx, r in enumerate(d.actionable_recommendations, 1):
                lines.append(f"  {idx}. [->] {r}")

        lines.append(f"================================================================================")
        return "\n".join(lines)
