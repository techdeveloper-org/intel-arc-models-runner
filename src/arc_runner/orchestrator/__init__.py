"""Autonomous Multi-Modal Agentic Studio and Adaptive Co-Benchmarking Package.

Engineered specifically for Intel Arc GPUs, Intel AI Boost NPUs, and Intel Core Ultra CPUs.
Provides multi-agent DAG orchestration, Level Zero USM zero-copy tensor passing,
closed-loop pacing convergence, dynamic word-level karaoke subtitles, Intel QSV hardware encoding,
smooth alpha cross-dissolves, ambient audio ducking, and autonomous VLM inspection.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1 through M12.
"""

from arc_runner.orchestrator.audio_ducking import AmbientAudioDuckingEngine
from arc_runner.orchestrator.audio_worker import StudioAudioWorker
from arc_runner.orchestrator.director import StudioDirector
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
from arc_runner.orchestrator.muxer import StudioTimelineCompositor
from arc_runner.orchestrator.pipeline import StudioPipeline
from arc_runner.orchestrator.slide_compositor import EducationalSlideCompositor
from arc_runner.orchestrator.usm_bridge import USMZeroCopyBridge
from arc_runner.orchestrator.visual_worker import StudioVisualWorker
from arc_runner.orchestrator.vlm_gate import ClosedLoopVLMInspectionGate

__all__ = [
    "AmbientAudioDuckingEngine",
    "ClosedLoopVLMInspectionGate",
    "DeviceSchedulingStrategy",
    "EducationalSlideCompositor",
    "MultiModelBenchmarkReport",
    "OrchestrationTelemetry",
    "PipelineStage",
    "SceneArtifact",
    "ScenePlan",
    "StageTelemetry",
    "StudioArtifactPayload",
    "StudioAudioWorker",
    "StudioDirector",
    "StudioPipeline",
    "StudioProjectRequest",
    "StudioTimelineCompositor",
    "StudioVisualWorker",
    "USMZeroCopyBridge",
    "VisualModality",
]
