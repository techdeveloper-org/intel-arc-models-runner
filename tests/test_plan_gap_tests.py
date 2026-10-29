"""Tests that close gaps found by auditing the master plan's test list against the repository."""

import os
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from arc_runner.core.exceptions import NativeBoundsError
from arc_runner.native import bridge
from arc_runner.native.bridge import NativeMemoryEngine

requires_native = pytest.mark.skipif(not bridge._load_native_library(), reason="native library not built or not loadable")

PAGE_4K = 4096
BLOCK_64K = 65536


def _blob(tmp_path: Path, size: int) -> Path:
    """Create a patterned file whose byte i is (i * 7 + i // 251) & 0xFF."""
    data = np.arange(size, dtype=np.uint64)
    path = tmp_path / "pattern.bin"
    path.write_bytes(((data * 7 + data // 251) & 0xFF).astype(np.uint8).tobytes())
    return path


@requires_native
@pytest.mark.parametrize("offset,size", [(PAGE_4K - 8, 16), (BLOCK_64K - 4, 8), (BLOCK_64K - 1, 2), (PAGE_4K * 3 + 1, PAGE_4K)])
def test_get_tensor_buffer_crossing_page_and_allocation_boundaries_reads_exact_bytes(tmp_path: Path, offset: int, size: int) -> None:
    """E3: a tensor straddling a 4 KiB page or a 64 KiB allocation-granularity boundary maps to the exact file bytes."""
    path = _blob(tmp_path, 2 * BLOCK_64K)
    expected = path.read_bytes()[offset:offset + size]
    engine = NativeMemoryEngine(path)
    try:
        engine.register_layer(0, [{"name": "w", "tensor_type": 0, "dimensions": [size // 4], "offset": offset, "size_bytes": size}])
        view = engine.get_tensor_buffer(0, "w", dtype=np.dtype(np.uint8))
        assert view is not None and view.tobytes() == expected
        del view
    finally:
        engine.close()


@requires_native
def test_register_80_layers_ok_and_layer_index_at_or_above_limit_rejected(tmp_path: Path) -> None:
    """E5: 80 layers register and read back; an index at the table limit raises instead of corrupting memory."""
    path = _blob(tmp_path, PAGE_4K * 4)
    engine = NativeMemoryEngine(path)
    try:
        for layer in range(80):
            assert engine.register_layer(layer, [{"name": "w", "tensor_type": 0, "dimensions": [4], "offset": layer * 16, "size_bytes": 16}]) is True
        view = engine.get_tensor_buffer(79, "w", dtype=np.dtype(np.uint8))
        assert view is not None and view.tobytes() == path.read_bytes()[79 * 16:79 * 16 + 16]
        del view
        with pytest.raises(NativeBoundsError):
            engine.register_layer(1 << 20, [{"name": "w", "tensor_type": 0, "dimensions": [4], "offset": 0, "size_bytes": 16}])
    finally:
        engine.close()


@requires_native
def test_lock_range_quota_failure_returns_false_and_logs_without_raising(tmp_path: Path) -> None:
    """C7/E8: when the OS refuses the lock (VirtualLock 1453 / mlock ENOMEM) the call returns False and the engine stays usable."""
    path = _blob(tmp_path, PAGE_4K)
    engine = NativeMemoryEngine(path)
    try:
        real_dll = engine._dll
        refusing = mock.MagicMock(wraps=real_dll)
        refusing.arc_mmap_lock_range.return_value = -7
        engine._dll = refusing
        assert engine.lock_range(0, 64) is False
        engine._dll = real_dll
        assert engine.is_open
    finally:
        engine.close()


@requires_native
def test_disabled_native_layer_streams_byte_identical_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N5: with ARC_NATIVE_DISABLE=1 the weight manager returns exactly the bytes the native path returns."""
    from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager
    from tests.test_flash_stream_triad import build_synthetic_flash_gguf

    model = build_synthetic_flash_gguf(tmp_path / "m.gguf", num_layers=2, hidden_dim=32)
    monkeypatch.delenv("ARC_NATIVE_DISABLE", raising=False)
    with FlashStreamWeightManager(model, ring_buffer_slots=2) as native:
        native_bytes = {k: v.tobytes() for k, v in native.acquire_layer_weights(1).items()}
    monkeypatch.setenv("ARC_NATIVE_DISABLE", "1")
    with FlashStreamWeightManager(model, ring_buffer_slots=2) as python_only:
        python_bytes = {k: v.tobytes() for k, v in python_only.acquire_layer_weights(1).items()}
    assert native_bytes == python_bytes and native_bytes


MODEL_DIR_ENV = "ARC_TEST_MODEL_DIR"


def _model_dir() -> Path:
    """Return the OpenVINO LLM directory for hardware tests or skip."""
    value = os.environ.get(MODEL_DIR_ENV, "")
    if not value or not Path(value).is_dir():
        pytest.skip(f"set {MODEL_DIR_ENV} to an OpenVINO LLM directory to run this hardware test")
    return Path(value)


@pytest.mark.hardware
def test_generate_ttft_uses_perf_metrics_and_is_reported_valid(require_intel_gpu: Any) -> None:
    """P8: on real silicon the engine reports perf-metric TTFT and decode rate (no threshold is asserted; the value goes to the benchmark files)."""
    from arc_runner.engine.models import GenerationConfig
    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    engine = OpenVINOGenAIEngine()
    engine.load_model(_model_dir(), device="GPU")
    engine.generate("Say hello.", GenerationConfig(max_new_tokens=16, temperature=0.0))
    telemetry = engine.get_telemetry()
    assert telemetry.metrics_source == "ov_perf_metrics" and telemetry.is_ttft_valid and telemetry.ttft_ms > 0.0
    assert telemetry.total_tokens > 0 and telemetry.prompt_tokens > 0


@pytest.mark.hardware
def test_kv_cache_u8_greedy_prefix_matches_default_precision(require_intel_gpu: Any) -> None:
    """P9: the u8 KV-cache quality gate passes on the registered prompt set (evidence for a human decision; the default is not changed)."""
    import importlib.util

    from arc_runner.benchmark.kv_quality import compare_kv_precision
    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bench_kv_quality", root / "scripts" / "bench_kv_quality.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    report = compare_kv_precision(script.build_factory(OpenVINOGenAIEngine, _model_dir(), "GPU"), script.DEFAULT_PROMPTS, 64)
    assert report.passed, f"{report.pass_count}/{len(report.prompts)} prompts agreed"
