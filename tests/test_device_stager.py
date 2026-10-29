"""Layer staging into Level Zero USM: fake-bridge logic tests and a fake-loader integration test."""

import ctypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from arc_runner.core.exceptions import NativeLayerError
from arc_runner.engine.flash_stream.device_stager import DeviceLayerStager
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager
from arc_runner.native.bridge import NativeLevelZeroBridge
from tests.test_flash_stream_triad import build_synthetic_flash_gguf
from tests.test_level_zero_bridge import fake_env, fake_loader, requires_native  # noqa: F401


class _FakeBridge:
    """Records USM allocations, copies and frees; hands out fake addresses from a host buffer."""

    def __init__(self, copy_status: int = 0, alloc_ok: bool = True, sync_status: int = 0) -> None:
        self.is_available = True
        self.copy_status = copy_status
        self.alloc_ok = alloc_ok
        self.sync_status = sync_status
        self.buffers: Dict[int, Any] = {}
        self.copies: List[tuple] = []
        self.freed: List[int] = []

    def allocate_usm(self, size_bytes: int, usm_type: str = "host") -> Optional[int]:
        if not self.alloc_ok:
            return None
        buf = (ctypes.c_ubyte * size_bytes)()
        addr = ctypes.addressof(buf)
        self.buffers[addr] = buf
        return addr

    def dma_copy_async(self, dst: int, src: int, size: int) -> int:
        self.copies.append((dst, src, size))
        if self.copy_status < 0:
            return self.copy_status
        ctypes.memmove(dst, src, size)
        return self.copy_status

    def synchronize(self) -> int:
        return self.sync_status

    def free_usm(self, ptr: int) -> None:
        self.freed.append(ptr)


def _arrays() -> Dict[str, np.ndarray]:
    return {
        "a": np.arange(10, dtype=np.float32),
        "b": np.arange(7, dtype=np.uint8),
        "c": np.ones((3, 3), dtype=np.float16),
    }


def test_stage_layer_copies_every_tensor_into_one_aligned_buffer() -> None:
    """The staged buffer holds each tensor's bytes at its recorded 64-byte aligned offset."""
    bridge = _FakeBridge()
    stager = DeviceLayerStager(bridge)
    staged = stager.stage_layer(4, _arrays())
    assert staged.layer_idx == 4
    assert all(off % 64 == 0 for off, _ in staged.extents.values())
    for name, array in _arrays().items():
        off, size = staged.extents[name]
        assert size == array.nbytes
        assert ctypes.string_at(staged.ptr + off, size) == array.tobytes()
    assert len(bridge.buffers) == 1


def test_stage_layer_counts_bytes_copies_and_measured_time() -> None:
    """Statistics come from what was copied, not from estimates."""
    stager = DeviceLayerStager(_FakeBridge())
    stager.stage_layer(0, _arrays())
    stats = stager.stats()
    assert stats.layers_staged == 1
    assert stats.bytes_staged == sum(a.nbytes for a in _arrays().values())
    assert stats.seconds_total > 0.0
    assert stats.fallback_copies == 0


def test_stage_layer_reports_cpu_fallback_copies() -> None:
    """A copy status of 1 (host memcpy fallback) is counted, never presented as DMA."""
    stager = DeviceLayerStager(_FakeBridge(copy_status=1))
    stager.stage_layer(0, _arrays())
    assert stager.stats().fallback_copies == 3


def test_stage_layer_allocation_refused_raises_native_layer_error() -> None:
    """No silent skip: a refused allocation is an error."""
    stager = DeviceLayerStager(_FakeBridge(alloc_ok=False))
    with pytest.raises(NativeLayerError):
        stager.stage_layer(0, _arrays())


@pytest.mark.parametrize("kwargs", [{"copy_status": -5}, {"sync_status": -9}])
def test_stage_layer_failed_copy_or_sync_frees_buffer_and_raises(kwargs: Dict[str, int]) -> None:
    """A failed copy or bounded sync releases the USM buffer and surfaces the native status."""
    bridge = _FakeBridge(**kwargs)
    stager = DeviceLayerStager(bridge)
    with pytest.raises(NativeLayerError) as excinfo:
        stager.stage_layer(0, _arrays())
    assert excinfo.value.native_status < 0
    assert bridge.freed and stager.stats().layers_staged == 0


def test_release_layer_frees_buffer_once_and_ignores_unknown_layers() -> None:
    """Release is idempotent."""
    bridge = _FakeBridge()
    stager = DeviceLayerStager(bridge)
    staged = stager.stage_layer(2, _arrays())
    stager.release_layer(2)
    stager.release_layer(2)
    stager.release_layer(99)
    assert bridge.freed == [staged.ptr]
    assert stager.get(2) is None


def test_close_frees_every_staged_layer() -> None:
    """close() leaves no USM allocation behind."""
    bridge = _FakeBridge()
    stager = DeviceLayerStager(bridge)
    ptrs = [stager.stage_layer(i, _arrays()).ptr for i in range(3)]
    stager.close()
    assert sorted(bridge.freed) == sorted(ptrs)


def test_stage_layer_with_unavailable_bridge_raises() -> None:
    """An unavailable bridge is an explicit error, not a no-op."""
    bridge = _FakeBridge()
    bridge.is_available = False
    with pytest.raises(NativeLayerError):
        DeviceLayerStager(bridge).stage_layer(0, _arrays())


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Six-layer synthetic container."""
    return build_synthetic_flash_gguf(tmp_path / "stage.gguf", num_layers=6, hidden_dim=32)


def test_manager_stages_loaded_layer_and_frees_it_on_eviction(model_path: Path) -> None:
    """The ring stages a layer after loading it and frees the USM buffer when the slot is recycled."""
    bridge = _FakeBridge()
    stager = DeviceLayerStager(bridge)
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3, device_stager=stager)
    manager.acquire_layer_weights(0)
    assert stager.get(0) is not None
    manager.acquire_layer_weights(3)
    assert stager.get(0) is None and stager.get(3) is not None
    manager.release_layer_slot(0)
    manager.release_layer_slot(3)
    assert stager.get(3) is None
    manager.close()
    assert len(bridge.freed) == len(bridge.buffers)


def test_manager_staging_failure_keeps_weights_usable_and_disables_staging(model_path: Path) -> None:
    """A staging error must not break streaming; staging stops after the first failure."""
    bridge = _FakeBridge(alloc_ok=False)
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3, device_stager=DeviceLayerStager(bridge))
    assert manager.acquire_layer_weights(0)
    assert manager.acquire_layer_weights(1)
    assert manager.get_staging_error() is not None
    manager.close()


@requires_native
def test_stager_with_fake_level_zero_loader_stages_through_the_gpu_command_list(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """Integration: real bridge + fake loader accept every copy on the command list (no CPU fallback)."""
    ze = NativeLevelZeroBridge()
    try:
        stager = DeviceLayerStager(ze)
        staged = stager.stage_layer(1, _arrays())
        assert staged.ptr and set(staged.extents) == {"a", "b", "c"}
        stats = stager.stats()
        assert stats.layers_staged == 1 and stats.fallback_copies == 0
        stager.close()
        assert stager.get(1) is None
    finally:
        ze.destroy()


@requires_native
def test_stager_refuses_cpu_copy_from_untracked_source_when_driver_rejects(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """Safety: the native layer never memcpy's from an untracked pointer, so a driver rejection is an error."""
    fake_env.setenv("FAKE_ZE_COPY_FAIL", "1")
    ze = NativeLevelZeroBridge()
    try:
        with pytest.raises(NativeLayerError):
            DeviceLayerStager(ze).stage_layer(1, _arrays())
    finally:
        ze.destroy()
