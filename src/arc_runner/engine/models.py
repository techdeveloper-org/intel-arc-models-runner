"""Data transfer objects and schemas for model execution and token streaming.

Defines strongly-typed Pydantic schemas for generation parameters, token chunks,
hardware telemetry, and key-value memory allocations.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from arc_runner.core.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


class ModelPrecision(str, Enum):
    """Weight and activation quantization precision formats."""

    INT4 = "INT4"
    INT8 = "INT8"
    FP16 = "FP16"
    FP32 = "FP32"


class LoRAAdapterSpec(BaseModel):
    """User-requested adapter binding with scaling weight multiplier."""

    name: str = Field(..., description="Name or identifier of the LoRA adapter")
    scale: float = Field(default=1.0, ge=0.0, le=2.0, description="Linear blending scale factor lambda")
    path: Optional[str] = Field(default=None, description="Optional explicit filesystem path to adapter directory")


class GenerationConfig(BaseModel):
    """Execution parameters controlling token sampling and decoding."""

    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=32768, description="Maximum tokens to generate")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0, description="Sampling randomness modulation")
    top_p: float = Field(default=DEFAULT_TOP_P, gt=0.0, le=1.0, description="Nucleus sampling probability mass threshold")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, description="Top-k filtering vocabulary cutoff")
    repetition_penalty: float = Field(default=DEFAULT_REPETITION_PENALTY, ge=1.0, le=2.0, description="Logit penalty for repeated tokens")
    stop_strings: List[str] = Field(default_factory=list, description="Sequences that immediately halt generation")
    stop_token_ids: List[int] = Field(default_factory=list, description="Token IDs that halt generation")
    stream: bool = Field(default=False, description="Whether to stream tokens incrementally")
    adapters: Optional[List[LoRAAdapterSpec]] = Field(default=None, description="Active LoRA adapters to compose during generation")


class TokenChunk(BaseModel):
    """Incremental token emission payload yielded during streaming generation."""

    text: str = Field(default="", description="Decoded string delta for this token")
    token_id: Optional[int] = Field(default=None, description="Vocabulary token integer index")
    seq_id: Optional[str] = Field(default=None, description="Sequence identifier for continuous batch routing")
    is_finished: bool = Field(default=False, description="True if generation has concluded")
    finish_reason: Optional[str] = Field(default=None, description="Completion reason (stop, length, cancelled)")
    prompt_tokens: Optional[int] = Field(default=None, description="Total input prompt token count")
    completion_tokens: Optional[int] = Field(default=None, description="Tokens generated so far")


class EngineTelemetry(BaseModel):
    """Inference execution metrics and throughput telemetry."""

    model_id: str = Field(default="", description="Identifier of the executed model")
    device: str = Field(default="", description="Target silicon device utilized")
    ttft_ms: float = Field(default=0.0, description="Time to first token in milliseconds")
    tokens_per_second: float = Field(default=0.0, description="Decoding rate excluding prefill")
    total_tokens: int = Field(default=0, description="Total tokens generated in this request")
    total_latency_ms: float = Field(default=0.0, description="Total end-to-end wall-clock duration in milliseconds")
    is_speculative: bool = Field(default=False, description="True if speculative decoding was active")
    draft_device: str = Field(default="", description="Silicon device executing draft model")
    draft_tokens_total: int = Field(default=0, description="Total draft tokens proposed across generation")
    accepted_tokens_total: int = Field(default=0, description="Total draft tokens accepted by target model")
    acceptance_rate: float = Field(default=0.0, description="Empirical acceptance rate alpha in [0.0, 1.0]")
    speculative_speedup: float = Field(default=1.0, description="Effective throughput speedup factor over baseline")


class SpeculativeConfig(BaseModel):
    """Configuration parameters for heterogeneous dual-silicon speculative decoding."""

    draft_model_path: str = Field(..., description="Filesystem directory containing OpenVINO IR draft model")
    draft_device: str = Field(default="NPU", description="Silicon target for draft generation (NPU, GPU, CPU)")
    num_assistant_tokens: int = Field(default=4, ge=1, le=16, description="Lookahead gamma count of draft tokens to propose per step")
    assistant_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Early stopping probability threshold for draft predictions")
    dynamic_gamma: bool = Field(default=True, description="Whether to dynamically adapt gamma based on acceptance rate")
    min_gamma: int = Field(default=1, ge=1, le=8, description="Minimum allowable lookahead horizon under dynamic adaptation")
    max_gamma: int = Field(default=8, ge=2, le=16, description="Maximum allowable lookahead horizon under dynamic adaptation")


class KVAllocation(BaseModel):
    """Memory reservation tracking for active KV-cache sequences."""

    allocation_id: str = Field(..., description="Unique UUID for this reservation")
    prompt_tokens: int = Field(..., description="Prompt token count allocated")
    reserved_new_tokens: int = Field(..., description="Headroom reserved for generated tokens")
    total_tokens: int = Field(..., description="Combined sequence length reserved")
    memory_footprint_mb: float = Field(..., description="Estimated memory footprint in megabytes")
    is_active: bool = Field(default=True, description="True if buffer is currently allocated")
