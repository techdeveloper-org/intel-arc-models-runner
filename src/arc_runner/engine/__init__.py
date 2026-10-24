"""Inference execution engines and key-value memory management modules."""

from typing import List
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.engine.audio_engine import (
    AudioEngine,
    AudioEngineConfig,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.continuous_batcher import (
    AdmissionDeniedError,
    BatchSequence,
    ContinuousBatcher,
    ContinuousBatcherError,
    SequenceState,
)
from arc_runner.engine.embedding_engine import (
    DimensionMismatchError,
    EmbeddingEngine,
    EmbeddingEngineError,
    EmptyMaskError,
)
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.models import (
    EngineTelemetry,
    GenerationConfig,
    KVAllocation,
    ModelPrecision,
    SpeculativeConfig,
    TokenChunk,
)
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
from arc_runner.engine.paged_attention import (
    BlockTableLookupError,
    PagedAttentionError,
    PagedBlockTable,
    RadixPrefixCache,
    VRAMExhaustedError,
)
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine
from arc_runner.engine.vision_engine import (
    InvalidImageFormatError,
    VisionEngine,
    VisionEngineError,
)

__all__: List[str] = [
    "AdmissionDeniedError",
    "AOTCompilationManager",
    "AudioEngine",
    "AudioEngineConfig",
    "BaseEngine",
    "BatchSequence",
    "BlockTableLookupError",
    "ContinuousBatcher",
    "ContinuousBatcherError",
    "DimensionMismatchError",
    "EmbeddingEngine",
    "EmbeddingEngineError",
    "EmptyMaskError",
    "InvalidImageFormatError",
    "OpenVINOGenAIEngine",
    "HybridSpeculativeEngine",
    "KVCacheManager",
    "GenerationConfig",
    "PagedAttentionError",
    "PagedBlockTable",
    "RadixPrefixCache",
    "SequenceState",
    "SpeculativeConfig",
    "SynthesisResult",
    "TokenChunk",
    "TranscriptionResult",
    "TranscriptionSegment",
    "EngineTelemetry",
    "KVAllocation",
    "ModelPrecision",
    "VisionEngine",
    "VisionEngineError",
    "VRAMExhaustedError",
]
