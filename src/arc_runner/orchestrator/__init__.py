"""Autonomous Multi-Modal Agentic Studio and Adaptive Co-Benchmarking Package.

Engineered specifically for Intel Arc GPUs, Intel AI Boost NPUs, and Intel Core Ultra CPUs.
Provides multi-agent DAG orchestration, Level Zero USM zero-copy tensor passing,
closed-loop pacing convergence, and comprehensive hardware co-benchmarking.
Strictly adheres to Level 2.2 Docstrings-Only standards.
"""

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

__all__ = [
    "DeviceSchedulingStrategy",
    "MultiModelBenchmarkReport",
    "OrchestrationTelemetry",
    "PipelineStage",
    "SceneArtifact",
    "ScenePlan",
    "StageTelemetry",
    "StudioArtifactPayload",
    "StudioProjectRequest",
    "USMZeroCopyBridge",
    "VisualModality",
]
