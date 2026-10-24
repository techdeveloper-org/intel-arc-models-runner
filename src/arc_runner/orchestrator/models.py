"""Data contracts and validation schemas for Arc MultiModal Studio.

Defines domain primitives for scene planning, project requests, hardware scheduling,
timeline artifacts, and multi-model co-benchmarking telemetry.
Strictly adheres to Level 2.2 Docstrings-Only standards and Pydantic v2 conventions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VisualModality(str, Enum):
    """Visual generation modality selection for scene rendering."""

    IMAGE_SLIDE = "IMAGE_SLIDE"
    ANIMATED_VIDEO = "ANIMATED_VIDEO"


class DeviceSchedulingStrategy(str, Enum):
    """Silicon compute device scheduling strategy for multi-agent DAG execution."""

    HETEROGENEOUS_PARALLEL = "HETEROGENEOUS_PARALLEL"
    SEQUENTIAL_EXCLUSIVE = "SEQUENTIAL_EXCLUSIVE"


class PipelineStage(str, Enum):
    """Discrete execution phases of the multi-modal studio lifecycle."""

    PLANNING = "PLANNING"
    AUDIO_SYNTHESIS = "AUDIO_SYNTHESIS"
    VISUAL_GENERATION = "VISUAL_GENERATION"
    TIMELINE_MUXING = "TIMELINE_MUXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ScenePlan(BaseModel):
    """Structured architectural plan for an individual scene segment in a production."""

    scene_id: int = Field(ge=1, description="Sequential 1-based scene index.")
    title: str = Field(description="Descriptive title or label of the scene.")
    narration_text: str = Field(description="Exact spoken dialogue or speech script for voiceover.")
    visual_prompt: str = Field(description="Detailed visual conditioning prompt for diffusion/DiT.")
    duration_seconds: float = Field(default=5.0, ge=0.5, le=300.0, description="Target duration in seconds.")
    modality: VisualModality = Field(default=VisualModality.IMAGE_SLIDE, description="Target visual rendering mode.")
    voice_slug: str | None = Field(default=None, description="Optional override speaker voice slug.")
    camera_motion: str = Field(default="slow pan", description="Camera motion styling cue for visual rendering.")

    @field_validator("narration_text")
    @classmethod
    def validate_narration_text(cls, value: str) -> str:
        """Validate that narration text contains non-whitespace content.

        Args:
            value: Raw speech text input.

        Returns:
            Cleaned and stripped text string.

        Raises:
            ValueError: When speech text is empty or only whitespace.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Scene narration text cannot be empty or purely whitespace.")
        return cleaned

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        """Validate and sanitize scene title.

        Args:
            value: Raw scene title string.

        Returns:
            Sanitized scene title.
        """
        return value.strip() or "Untitled Scene"


class StudioProjectRequest(BaseModel):
    """User input configuration dispatched to the multi-modal studio pipeline."""

    prompt: str = Field(description="Natural language prompt specifying topic, style, and requirements.")
    target_duration_sec: float = Field(default=30.0, ge=2.0, le=600.0, description="Target total video duration.")
    voice_slug: str = Field(default="af_heart", description="Default speaker voice style slug for Kokoro TTS.")
    visual_modality: VisualModality = Field(
        default=VisualModality.IMAGE_SLIDE,
        description="Default visual generation modality across scenes.",
    )
    resolution: tuple[int, int] = Field(
        default=(512, 512),
        description="Target video raster frame resolution as (width, height).",
    )
    fps: int = Field(default=16, ge=1, le=60, description="Playback framerate in frames per second.")
    strategy: DeviceSchedulingStrategy = Field(
        default=DeviceSchedulingStrategy.HETEROGENEOUS_PARALLEL,
        description="Silicon scheduling policy across Intel NPU, GPU, and CPU.",
    )
    export_format: str = Field(default="mp4", description="Output video container format ('mp4', 'mkv').")
    output_filename: str | None = Field(default=None, description="Optional custom destination filename.")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Validate that project prompt is non-empty.

        Args:
            value: User input prompt string.

        Returns:
            Cleaned prompt string.

        Raises:
            ValueError: If input prompt is empty.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Studio project prompt cannot be empty or whitespace.")
        return cleaned

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Validate that resolution dimensions are positive multiples of 8.

        Args:
            value: Tuple of (width, height) integers.

        Returns:
            Validated resolution tuple.

        Raises:
            ValueError: If dimensions are out of bounds or not multiples of 8.
        """
        w, h = value
        if w < 64 or h < 64 or w > 3840 or h > 2160:
            raise ValueError(f"Resolution dimensions {value} outside allowed bounds [64..3840, 64..2160].")
        if w % 8 != 0 or h % 8 != 0:
            raise ValueError(f"Resolution dimensions {value} must be divisible by 8 for video encoder compatibility.")
        return value


class SceneArtifact(BaseModel):
    """Intermediate generated binary payload and metadata for an individual scene."""

    scene_id: int = Field(description="Scene identifier matching ScenePlan.")
    audio_path: str | None = Field(default=None, description="Filesystem path to synthesized scene audio WAV.")
    audio_samples: Any | None = Field(default=None, description="Synthesized PCM audio samples as 1D numpy array.")
    audio_sample_rate: int = Field(default=24000, description="Sampling rate of audio samples in Hz.")
    audio_duration_sec: float = Field(default=0.0, ge=0.0, description="Measured audio duration in seconds.")
    video_path: str | None = Field(default=None, description="Filesystem path to rendered scene video clip.")
    frames: list[Any] | None = Field(default=None, description="Rendered visual frames as list of uint8 arrays.")
    video_duration_sec: float = Field(default=0.0, ge=0.0, description="Calculated video duration in seconds.")
    fps: int = Field(default=16, description="Framerate at which video frames were rendered.")
    temporal_drift_sec: float = Field(default=0.0, description="Duration discrepancy |audio - video| before sync.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StageTelemetry(BaseModel):
    """Detailed performance and hardware utilization telemetry for a pipeline stage."""

    stage_name: str = Field(description="Name of the measured pipeline phase.")
    latency_ms: float = Field(description="Execution wall-clock latency in milliseconds.")
    device_used: str = Field(description="Identifier of physical silicon executing the stage.")
    peak_memory_mb: float = Field(description="Peak resident memory consumption during stage execution.")
    throughput_metric: str = Field(description="Primary throughput metric unit.")
    throughput_value: float = Field(description="Measured throughput numeric quantity.")


class OrchestrationTelemetry(BaseModel):
    """Aggregated end-to-end telemetry report for multi-agent studio execution."""

    project_id: str = Field(description="Unique production project run identifier.")
    total_latency_ms: float = Field(default=0.0, description="Total wall-clock execution time in milliseconds.")
    stage_breakdowns: list[StageTelemetry] = Field(
        default_factory=list,
        description="Per-stage performance breakdown entries.",
    )
    usm_zero_copy_latency_us: float = Field(
        default=0.0,
        description="Cumulative inter-engine USM pointer handoff latency in microseconds.",
    )
    audio_video_drift_ms: float = Field(
        default=0.0,
        description="Final post-convergence audio-video desync drift in milliseconds.",
    )
    pacing_convergence_iterations: int = 0
    devices_utilized: list[str] = Field(default_factory=list, description="Silicon accelerators engaged.")
    peak_vram_mb: float = Field(default=0.0, description="Highest observed VRAM allocation during pipeline.")


class StudioArtifactPayload(BaseModel):
    """Final packaged media production artifact and verification metadata."""

    project_id: str = Field(description="Unique project run identifier.")
    video_output_path: str = Field(description="Absolute filesystem path to final synchronized MP4 video.")
    script_scenes: list[ScenePlan] = Field(description="List of structured scene plans generated by Director.")
    total_duration_sec: float = Field(description="Final video playback duration in seconds.")
    resolution: tuple[int, int] = Field(description="Video raster resolution as (width, height).")
    telemetry: OrchestrationTelemetry = Field(description="End-to-end hardware and stage performance telemetry.")
    cloud_cost_saved_usd: float = Field(description="Estimated cloud API replacement cost saved in USD.")
    created_at: str = Field(description="ISO-8601 production timestamp.")


class MultiModelBenchmarkReport(BaseModel):
    """Statistical hardware co-benchmarking report comparing multi-modal execution and cloud ROI."""

    benchmark_id: str = Field(description="Benchmark run sequence identifier.")
    iterations: int = Field(description="Number of measured benchmark execution iterations.")
    avg_e2e_latency_ms: float = Field(description="Sample mean end-to-end pipeline latency in milliseconds.")
    ci95_latency_ms: float = Field(description="Student's t 95% confidence interval half-width for latency.")
    avg_usm_transfer_us: float = Field(description="Sample mean USM zero-copy tensor transfer latency in microseconds.")
    ci95_usm_transfer_us: float = Field(description="Student's t 95% confidence interval for USM transfer.")
    audio_rtf: float = Field(description="Speech synthesis Real-Time Factor (synthesis time / audio duration).")
    video_rtf: float = Field(description="Video synthesis Real-Time Factor (render time / video duration).")
    llm_tok_per_sec: float = Field(description="Director LLM planning throughput in tokens per second.")
    cloud_comparison_usd_per_run: float = Field(description="Commercial cloud API cost equivalent in USD ($0.35/run).")
    hardware_payback_runs: int = Field(description="Number of generation runs required to amortize Intel Arc GPU cost.")
    stage_latencies_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Average execution latency mapped by pipeline stage name.",
    )


class BenchmarkRequest(BaseModel):
    """Execution parameters for triggering an automated multi-model co-benchmark."""

    prompt: str = Field(default="Quantum computing overview", description="Benchmark test prompt.")
    iterations: int = Field(default=3, ge=1, le=50, description="Number of measured benchmark iterations.")
    warmup_runs: int = Field(default=1, ge=0, le=5, description="Number of discarded warmup passes.")
    strategy: DeviceSchedulingStrategy = Field(
        default=DeviceSchedulingStrategy.HETEROGENEOUS_PARALLEL,
        description="Target silicon scheduling policy.",
    )
