"""WP-15: lazy-import gate -- arc_runner package must not load heavy modules on bare import.

The --version cold-start path does nothing but import arc_runner and read __version__.
Loading OpenVINO, FastAPI, the benchmark stack, the hub, and the GGUF parser on that
path wastes ~664 ms.  After WP-15 those modules are deferred behind a PEP 562
__getattr__ and only pulled in when the caller first accesses the name.
"""

import sys
import types
from importlib import import_module
from typing import Generator

import pytest


_HEAVY_MODULES = [
    "arc_runner.engine.openvino_engine",
    "arc_runner.engine.speculative_engine",
    "arc_runner.engine.aot_cache",
    "arc_runner.engine.kv_cache",
    "arc_runner.server.app",
    "arc_runner.server.model_manager",
    "arc_runner.benchmark.runner",
    "arc_runner.hub.manager",
    "arc_runner.hub.downloader",
    "arc_runner.hub.registry",
    "arc_runner.hub.ollama_importer",
    "arc_runner.quant.gguf_parser",
    "arc_runner.quant.models",
]

_ARC_PREFIX = "arc_runner."

_EXPECTED_ALL = {
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
}


@pytest.fixture()
def fresh_arc_runner() -> Generator[types.ModuleType, None, None]:
    """Yield a freshly imported arc_runner with a clean sys.modules slate.

    Saves every arc_runner.* entry, removes them, imports arc_runner from
    scratch, yields it, then restores the saved entries so the rest of the
    test session is not affected.
    """
    saved = {k: v for k, v in sys.modules.items()
             if k == "arc_runner" or k.startswith(_ARC_PREFIX)}
    for k in saved:
        del sys.modules[k]
    try:
        mod = import_module("arc_runner")
        yield mod
    finally:
        fresh_keys = [k for k in sys.modules
                      if k == "arc_runner" or k.startswith(_ARC_PREFIX)]
        for k in fresh_keys:
            del sys.modules[k]
        sys.modules.update(saved)


class TestHeavyModulesNotLoadedOnImport:
    """Heavy modules must stay out of sys.modules after a bare import."""

    def test_openvino_engine_not_imported_on_package_import(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.engine.openvino_engine must not load until accessed."""
        assert "arc_runner.engine.openvino_engine" not in sys.modules

    def test_server_app_not_imported_on_package_import(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.server.app must not load until accessed."""
        assert "arc_runner.server.app" not in sys.modules

    def test_benchmark_runner_not_imported_on_package_import(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.benchmark.runner must not load until accessed."""
        assert "arc_runner.benchmark.runner" not in sys.modules

    def test_gguf_parser_not_imported_on_package_import(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.quant.gguf_parser must not load until accessed."""
        assert "arc_runner.quant.gguf_parser" not in sys.modules

    def test_hub_manager_not_imported_on_package_import(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.hub.manager must not load until accessed."""
        assert "arc_runner.hub.manager" not in sys.modules

    def test_speculative_engine_not_imported_on_package_import(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.engine.speculative_engine must not load until accessed."""
        assert "arc_runner.engine.speculative_engine" not in sys.modules


class TestVersionAccessDoesNotTriggerHeavyImports:
    """Accessing __version__ on its own must not load any heavy module."""

    def test_version_access_leaves_heavy_modules_absent(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """Reading __version__ must not cause any heavy module to load."""
        _ = fresh_arc_runner.__version__
        for mod_name in _HEAVY_MODULES:
            assert mod_name not in sys.modules, (
                f"Accessing __version__ must not load {mod_name}"
            )

    def test_version_value_is_a_string(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """__version__ must be a non-empty string even after lazy-import refactor."""
        ver = fresh_arc_runner.__version__
        assert isinstance(ver, str) and ver, (
            "__version__ must be a non-empty string"
        )


class TestLazyAccessTriggersImport:
    """Accessing a lazy name must load its module and return the correct object."""

    def test_access_openvino_engine_loads_module(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """Accessing arc_runner.OpenVINOGenAIEngine triggers the engine import."""
        cls = fresh_arc_runner.OpenVINOGenAIEngine
        assert "arc_runner.engine.openvino_engine" in sys.modules
        from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
        assert cls is OpenVINOGenAIEngine

    def test_access_create_app_loads_server(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """Accessing arc_runner.create_app triggers arc_runner.server.app import."""
        fn = fresh_arc_runner.create_app
        assert "arc_runner.server.app" in sys.modules
        assert callable(fn)

    def test_access_benchmark_runner_loads_module(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """Accessing arc_runner.BenchmarkRunner triggers the benchmark import."""
        cls = fresh_arc_runner.BenchmarkRunner
        assert "arc_runner.benchmark.runner" in sys.modules
        from arc_runner.benchmark.runner import BenchmarkRunner
        assert cls is BenchmarkRunner

    def test_access_gguf_parser_loads_module(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """Accessing arc_runner.GGUFParser triggers the quant import."""
        cls = fresh_arc_runner.GGUFParser
        assert "arc_runner.quant.gguf_parser" in sys.modules
        from arc_runner.quant.gguf_parser import GGUFParser
        assert cls is GGUFParser

    def test_second_access_returns_same_object(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """Repeated access to the same lazy name returns the identical object."""
        first = fresh_arc_runner.OpenVINOGenAIEngine
        second = fresh_arc_runner.OpenVINOGenAIEngine
        assert first is second


class TestAllAndDir:
    """__all__ and dir() contract must be preserved after the lazy-import refactor."""

    def test_all_contains_all_expected_names(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """arc_runner.__all__ must contain every name that was there before WP-15."""
        missing = _EXPECTED_ALL - set(fresh_arc_runner.__all__)
        assert not missing, f"Names missing from __all__: {sorted(missing)}"

    def test_dir_includes_all_names(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """dir(arc_runner) must list every name in __all__."""
        listed = set(dir(fresh_arc_runner))
        missing = _EXPECTED_ALL - listed
        assert not missing, f"Names missing from dir(arc_runner): {sorted(missing)}"

    def test_unknown_attribute_raises_attribute_error(
        self, fresh_arc_runner: types.ModuleType
    ) -> None:
        """__getattr__ must raise AttributeError for names not in __all__."""
        with pytest.raises(AttributeError):
            _ = fresh_arc_runner.NoSuchThingInArcRunnerXYZ
