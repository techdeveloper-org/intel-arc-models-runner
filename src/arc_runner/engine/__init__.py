"""Inference execution engines and key-value memory management modules."""

from typing import List
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.engine.base import BaseEngine
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
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine

__all__: List[str] = [
    "AOTCompilationManager",
    "BaseEngine",
    "OpenVINOGenAIEngine",
    "HybridSpeculativeEngine",
    "KVCacheManager",
    "GenerationConfig",
    "SpeculativeConfig",
    "TokenChunk",
    "EngineTelemetry",
    "KVAllocation",
    "ModelPrecision",
]
