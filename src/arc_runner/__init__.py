"""Intel Arc Models Runner.

Native local model execution engine and OpenAI/Ollama compatible API server
optimized specifically for Intel Arc GPUs (discrete and integrated) and
Intel AI Boost NPUs.
"""

from typing import Any, List

from arc_runner._version import __version__
from arc_runner.core.exceptions import (
    ArcRunnerError,
    ContextWindowExceededError,
    DeviceNotFoundError,
    InferenceTimeoutError,
    ModelLoadError,
    ModelNotFoundError,
    VRAMExhaustedError,
)
from arc_runner.core.logging import (
    clear_request_context,
    get_logger,
    get_request_id,
    set_request_context,
    setup_logging,
)
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import (
    DeviceArchitecture,
    DeviceInfo,
    DeviceType,
    SystemTopology,
)
from arc_runner.utils.dll_loader import initialize_windows_dll_environment

initialize_windows_dll_environment()

_LAZY: dict = {
    "BaseEngine": ("arc_runner.engine.base", "BaseEngine"),
    "GenerationConfig": ("arc_runner.engine.models", "GenerationConfig"),
    "SpeculativeConfig": ("arc_runner.engine.models", "SpeculativeConfig"),
    "TokenChunk": ("arc_runner.engine.models", "TokenChunk"),
    "EngineTelemetry": ("arc_runner.engine.models", "EngineTelemetry"),
    "KVAllocation": ("arc_runner.engine.models", "KVAllocation"),
    "ModelPrecision": ("arc_runner.engine.models", "ModelPrecision"),
    "OpenVINOGenAIEngine": ("arc_runner.engine.openvino_engine", "OpenVINOGenAIEngine"),
    "HybridSpeculativeEngine": ("arc_runner.engine.speculative_engine", "HybridSpeculativeEngine"),
    "KVCacheManager": ("arc_runner.engine.kv_cache", "KVCacheManager"),
    "AOTCompilationManager": ("arc_runner.engine.aot_cache", "AOTCompilationManager"),
    "create_app": ("arc_runner.server.app", "create_app"),
    "ModelManager": ("arc_runner.server.model_manager", "ModelManager"),
    "HubManager": ("arc_runner.hub.manager", "HubManager"),
    "ModelRegistry": ("arc_runner.hub.registry", "ModelRegistry"),
    "CuratedModelInfo": ("arc_runner.hub.registry", "CuratedModelInfo"),
    "DownloadProgress": ("arc_runner.hub.downloader", "DownloadProgress"),
    "ModelDownloader": ("arc_runner.hub.downloader", "ModelDownloader"),
    "OllamaImporter": ("arc_runner.hub.ollama_importer", "OllamaImporter"),
    "BenchmarkRunner": ("arc_runner.benchmark.runner", "BenchmarkRunner"),
    "BenchmarkRunResult": ("arc_runner.benchmark.runner", "BenchmarkRunResult"),
    "BenchmarkSummary": ("arc_runner.benchmark.runner", "BenchmarkSummary"),
    "GGUFParser": ("arc_runner.quant.gguf_parser", "GGUFParser"),
    "AOTCompilationResult": ("arc_runner.quant.models", "AOTCompilationResult"),
    "CacheEntryInfo": ("arc_runner.quant.models", "CacheEntryInfo"),
    "GGMLType": ("arc_runner.quant.models", "GGMLType"),
    "GGUFHeader": ("arc_runner.quant.models", "GGUFHeader"),
    "GGUFSummary": ("arc_runner.quant.models", "GGUFSummary"),
    "GGUFTensorInfo": ("arc_runner.quant.models", "GGUFTensorInfo"),
    "GGUFValueType": ("arc_runner.quant.models", "GGUFValueType"),
    "OllamaModelManifest": ("arc_runner.quant.models", "OllamaModelManifest"),
}


def __getattr__(name: str) -> Any:
    """Defer heavy imports until first attribute access (PEP 562).

    Args:
        name: Attribute name requested on the arc_runner package.

    Returns:
        The requested object, imported on first access.

    Raises:
        AttributeError: When name is not a known public name in this package.
    """
    if name in _LAZY:
        import importlib
        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'arc_runner' has no attribute {name!r}")


def __dir__() -> List[str]:
    """Return all public names so dir(arc_runner) is complete before any lazy load.

    Returns:
        Sorted list of all public names in this package.
    """
    return sorted(__all__)


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
    "create_app",
    "ModelManager",
    "HubManager",
    "ModelRegistry",
    "CuratedModelInfo",
    "DownloadProgress",
    "ModelDownloader",
    "BenchmarkRunner",
    "BenchmarkRunResult",
    "BenchmarkSummary",
    "GGUFParser",
    "OllamaImporter",
    "AOTCompilationManager",
    "GGUFHeader",
    "GGUFTensorInfo",
    "GGUFSummary",
    "GGMLType",
    "GGUFValueType",
    "OllamaModelManifest",
    "AOTCompilationResult",
    "CacheEntryInfo",
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "get_request_id",
]
