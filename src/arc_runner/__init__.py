"""Intel Arc Models Runner.

Native local model execution engine and OpenAI/Ollama compatible API server
optimized specifically for Intel Arc GPUs (discrete and integrated) and
Intel AI Boost NPUs.
"""

from typing import List
from arc_runner.core.exceptions import (
    ArcRunnerError,
    ContextWindowExceededError,
    DeviceNotFoundError,
    InferenceTimeoutError,
    ModelLoadError,
    ModelNotFoundError,
    VRAMExhaustedError,
)
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import (
    DeviceArchitecture,
    DeviceInfo,
    DeviceType,
    SystemTopology,
)
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
from arc_runner.utils.dll_loader import initialize_windows_dll_environment

initialize_windows_dll_environment()

__version__: str = "0.1.0"
__all__: List[str] = [
    "__version__",
    "initialize_windows_dll_environment",
    "DeviceDetector",
    "DeviceInfo",
    "DeviceType",
    "DeviceArchitecture",
    "SystemTopology",
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
    "ArcRunnerError",
    "ModelLoadError",
    "ModelNotFoundError",
    "DeviceNotFoundError",
    "ContextWindowExceededError",
    "VRAMExhaustedError",
    "InferenceTimeoutError",
]
