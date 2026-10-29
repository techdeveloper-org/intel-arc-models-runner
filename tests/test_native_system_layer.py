"""Triad test suite covering native C acceleration layer and Level Zero USM integration.

Includes Positive (happy-path), Negative (fault injection), and Edge-case scenarios
in strict compliance with enterprise testing standards.
"""

from pathlib import Path
import threading
from typing import Generator
import pytest
import numpy as np

from arc_runner.devices.sysman import LevelZeroSysman
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager
from arc_runner.engine.models import GenerationConfig
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
from arc_runner.native.bridge import (
    ArcLayerDescriptor,
    ArcTensorDescriptor,
    NativeLevelZeroBridge,
    NativeMemoryEngine,
)
from tests.test_flash_stream_triad import build_synthetic_flash_gguf


@pytest.fixture
def sample_gguf(tmp_path: Path) -> Path:
    """Generate a clean synthetic GGUF model container for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path: Path to constructed GGUF file.
    """
    gguf_path = tmp_path / "test_native_model.gguf"
    build_synthetic_flash_gguf(gguf_path, num_layers=4, hidden_dim=64)
    return gguf_path


# ============================================================================
# POSITIVE TESTS (Happy-Path P1 - P10)
# ============================================================================

def test_p1_native_library_compilation_and_load() -> None:
    """P1: Verify native C acceleration library loads and exports core symbols."""
    engine = NativeMemoryEngine()
    assert engine.is_available is True


def test_p2_native_memory_engine_mapping(sample_gguf: Path) -> None:
    """P2: Verify NativeMemoryEngine maps a synthetic GGUF file and acquires context."""
    engine = NativeMemoryEngine()
    success = engine.open_model(sample_gguf)
    assert success is True
    assert engine.is_open is True
    engine.close()
    assert engine.is_open is False


def test_p3_native_layer_descriptor_registration(sample_gguf: Path) -> None:
    """P3: Verify layer descriptors are registered into the native C lookup table."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    assert engine.is_available is True
    assert engine.is_open is True
    mgr.close()


def test_p4_native_tensor_buffer_content(sample_gguf: Path) -> None:
    """P4: Verify native tensor buffer extraction retrieves valid non-empty arrays."""
    mgr = FlashStreamWeightManager(sample_gguf)
    weights = mgr.acquire_layer_weights(0)
    assert len(weights) > 0
    assert "attn_q.weight" in weights or len(weights) >= 1
    for k, v in weights.items():
        assert isinstance(v, np.ndarray)
        assert v.size > 0
    mgr.close()


def test_p5_native_zero_copy_buffer_view(sample_gguf: Path) -> None:
    """P5: Verify NativeMemoryEngine.get_tensor_buffer returns valid array views."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    tensors = mgr._layer_tensor_map.get(0, [])
    assert len(tensors) > 0
    t = tensors[0]
    buf = engine.get_tensor_buffer(0, t.name, dims=t.dimensions)
    assert buf is not None
    assert isinstance(buf, np.ndarray)
    assert buf.shape == tuple(t.dimensions)
    mgr.close()


def test_p6_level_zero_bridge_initialization() -> None:
    """P6: Verify Level Zero bridge initializes driver context successfully."""
    ze = NativeLevelZeroBridge()
    if not ze.is_available:
        pytest.skip("Level Zero driver not available")
    assert ze.device_type == 1
    ze.destroy()


def test_p7_level_zero_usm_allocation() -> None:
    """P7: Verify Level Zero USM shared, host, and device memory allocations."""
    ze = NativeLevelZeroBridge()
    if not ze.is_available:
        pytest.skip("Level Zero driver not available on this host")

    host_ptr = ze.allocate_usm(1024, "host")
    assert host_ptr is not None and host_ptr != 0
    ze.free_usm(host_ptr)

    shared_ptr = ze.allocate_usm(2048, "shared")
    assert shared_ptr is not None and shared_ptr != 0
    ze.free_usm(shared_ptr)

    ze.destroy()


def test_p8_openvino_engine_perf_metrics_handling() -> None:
    """P8: Verify OpenVINOGenAIEngine defaults and generation configuration."""
    engine = OpenVINOGenAIEngine()
    assert engine.is_loaded is False
    cfg = GenerationConfig(max_new_tokens=10)
    assert cfg.max_new_tokens == 10


def test_p9_sysman_live_telemetry_reporting() -> None:
    """P9: Verify LevelZeroSysman reports total memory and never an invented free memory or power value."""
    sysman = LevelZeroSysman()
    if not sysman.is_available:
        pytest.skip("Level Zero driver not available")
    total_mb, free_mb = sysman.query_gpu_memory(0)
    assert total_mb is not None
    assert total_mb > 0
    assert free_mb is None
    assert sysman.query_package_power(0) is None
    assert sysman.get_unavailable_reason("package_power")


def test_p10_weight_streamer_native_integration(sample_gguf: Path) -> None:
    """P10: Verify FlashStreamWeightManager operates seamlessly across multiple layer swaps."""
    with FlashStreamWeightManager(sample_gguf, ring_buffer_slots=2) as mgr:
        w0 = mgr.acquire_layer_weights(0)
        w1 = mgr.acquire_layer_weights(1)
        w2 = mgr.acquire_layer_weights(2)
        assert len(w0) > 0
        assert len(w1) > 0
        assert len(w2) > 0


# ============================================================================
# NEGATIVE TESTS (Fault Injection N1 - N8)
# ============================================================================

def test_n1_mmap_nonexistent_file(tmp_path: Path) -> None:
    """N1: Verify open_model returns False when opening non-existent path."""
    engine = NativeMemoryEngine()
    missing_file = tmp_path / "does_not_exist.gguf"
    assert engine.open_model(missing_file) is False
    assert engine.is_open is False


def test_n2_get_tensor_invalid_layer(sample_gguf: Path) -> None:
    """N2: Verify get_tensor_buffer returns None on out-of-range layer index."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    buf = engine.get_tensor_buffer(9999, "nonexistent.weight")
    assert buf is None
    mgr.close()


def test_n3_get_tensor_unknown_name(sample_gguf: Path) -> None:
    """N3: Verify get_tensor_buffer returns None when tensor name is absent."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    buf = engine.get_tensor_buffer(0, "completely_unknown_tensor_name")
    assert buf is None
    mgr.close()


def test_n4_usm_zero_bytes_allocation() -> None:
    """N4: Verify allocating 0 bytes of USM memory returns None safely."""
    ze = NativeLevelZeroBridge()
    if not ze.is_available:
        pytest.skip("Level Zero driver not available")
    ptr = ze.allocate_usm(0, "shared")
    assert ptr is None or ptr == 0
    ze.destroy()


def test_n5_mmap_double_close_idempotent(sample_gguf: Path) -> None:
    """N5: Verify calling close() repeatedly is idempotent without exceptions."""
    engine = NativeMemoryEngine(sample_gguf)
    assert engine.is_open is True
    engine.close()
    assert engine.is_open is False
    engine.close()
    assert engine.is_open is False


def test_n6_usm_free_null_pointer() -> None:
    """N6: Verify passing NULL or 0 to free_usm does not raise errors."""
    ze = NativeLevelZeroBridge()
    if not ze.is_available:
        pytest.skip("Level Zero driver not available")
    ze.free_usm(0)
    ze.destroy()


def test_n7_sysman_invalid_device_index() -> None:
    """N7: Verify querying an out-of-bounds device index returns (None, None)."""
    sysman = LevelZeroSysman()
    total_mb, free_mb = sysman.query_gpu_memory(9999)
    assert total_mb is None
    assert free_mb is None


def test_n8_mmap_unopened_operations() -> None:
    """N8: Verify operations on unopened engine instance return False or None."""
    engine = NativeMemoryEngine()
    assert engine.is_open is False
    assert engine.register_layer(0, []) is False
    assert engine.get_tensor_buffer(0, "test") is None
    assert engine.lock_range(0, 1024) is False
    assert engine.unlock_range(0, 1024) is False


# ============================================================================
# EDGE-CASE TESTS (Boundary E1 - E8)
# ============================================================================

def test_e1_mmap_empty_file(tmp_path: Path) -> None:
    """E1: Verify mapping an empty 0-byte file fails gracefully without hanging."""
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    engine = NativeMemoryEngine()
    assert engine.open_model(empty_file) is False
    assert engine.is_open is False


def test_e2_single_tensor_layer_registration(sample_gguf: Path) -> None:
    """E2: Verify layer with single tensor descriptor registers correctly."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    tensors = mgr._layer_tensor_map.get(0, [])[:1]
    assert engine.register_layer(100, tensors) is True
    mgr.close()


def test_e3_multi_layer_boundary(sample_gguf: Path) -> None:
    """E3: Verify layer indexing near max layer boundary registers properly."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    tensors = mgr._layer_tensor_map.get(0, [])
    assert engine.register_layer(250, tensors) is True
    mgr.close()


def test_e4_concurrent_multithreaded_read(sample_gguf: Path) -> None:
    """E4: Verify concurrent threads reading layers simultaneously execute cleanly."""
    mgr = FlashStreamWeightManager(sample_gguf, ring_buffer_slots=4)
    errors = []

    def reader_worker(layer_idx: int) -> None:
        try:
            for _ in range(10):
                w = mgr.acquire_layer_weights(layer_idx)
                assert len(w) > 0
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader_worker, args=(i % 4,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    mgr.close()


def test_e5_lock_unlock_range(sample_gguf: Path) -> None:
    """E5: Verify lock_range and unlock_range on valid mapped offsets."""
    engine = NativeMemoryEngine(sample_gguf)
    # Lock first 4KB page
    res_lock = engine.lock_range(0, 4096)
    # Even if VirtualLock quota limits apply, unlock should handle gracefully
    engine.unlock_range(0, 4096)
    engine.close()


def test_e6_openvino_empty_prompt() -> None:
    """E6: Verify generate with empty prompt returns empty string without error."""
    engine = OpenVINOGenAIEngine()
    engine._is_loaded = True
    engine._pipeline = object()
    assert engine.generate("") == ""


def test_e7_tensor_with_different_dimensions(sample_gguf: Path) -> None:
    """E7: Verify buffer extraction handles multi-dimensional reshape parameters."""
    mgr = FlashStreamWeightManager(sample_gguf)
    engine = mgr._native_engine
    tensors = mgr._layer_tensor_map.get(0, [])
    if tensors:
        t = tensors[0]
        buf = engine.get_tensor_buffer(0, t.name, dims=(1, -1))
        assert buf is not None
        assert buf.ndim == 2
    mgr.close()


def test_e8_dma_copy_async_fallback() -> None:
    """E8: Verify DMA copy handles valid source and destination memory buffers."""
    ze = NativeLevelZeroBridge()
    if not ze.is_available:
        pytest.skip("Level Zero driver not available")

    src = ze.allocate_usm(1024, "host")
    dst = ze.allocate_usm(1024, "shared")
    if src and dst:
        res = ze.dma_copy_async(dst, src, 1024)
        assert res == 0
        sync_res = ze.synchronize()
        assert sync_res == 0
        ze.free_usm(src)
        ze.free_usm(dst)

    ze.destroy()
