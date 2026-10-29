"""Regression tests for defects found by the independent C and driver reviews (R01, R09)."""

import ctypes
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from pathlib import Path
from typing import Any, List
from types import SimpleNamespace

import numpy as np
import pytest

from arc_runner.core.exceptions import NativeBoundsError, NativeLayerError
from arc_runner.native import bridge
from arc_runner.native.bridge import NativeLevelZeroBridge, NativeMemoryEngine
from tests.test_level_zero_bridge import fake_env, fake_loader  # noqa: F401

requires_native = pytest.mark.skipif(not bridge._load_native_library(), reason="native library not built or not loadable")
REPO_ROOT = Path(__file__).resolve().parents[1]


@requires_native
def test_dma_fallback_refuses_a_copy_larger_than_the_allocations(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """R01 D1: the CPU fallback checks the whole range in both buffers, not only the start pointer."""
    fake_env.setenv("FAKE_ZE_COPY_FAIL", "1")
    ze = NativeLevelZeroBridge()
    try:
        src, dst = ze.allocate_usm(64, "host"), ze.allocate_usm(64, "host")
        assert ze.dma_copy_async(dst, src, 4096) < 0
        assert ze.dma_copy_async(dst + 40, src, 32) < 0
        assert ze.dma_copy_async(dst, src + 40, 32) < 0
        ctypes.memset(src, 0x5A, 64)
        assert ze.dma_copy_async(dst, src, 64) == 1
        assert ctypes.string_at(dst, 64) == b"\x5a" * 64
    finally:
        ze.destroy()


@requires_native
def test_dma_fallback_is_refused_while_gpu_commands_are_queued(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """R01 D2: a synchronous CPU copy would reorder against queued GPU copies, so it is refused until synchronize."""
    ze = NativeLevelZeroBridge()
    try:
        src, dst = ze.allocate_usm(64, "host"), ze.allocate_usm(64, "host")
        assert ze.dma_copy_async(dst, src, 16) == 0
        fake_env.setenv("FAKE_ZE_COPY_FAIL", "1")
        assert ze.dma_copy_async(dst, src, 16) < 0
        fake_env.delenv("FAKE_ZE_COPY_FAIL")
        assert ze.synchronize() == 0
        fake_env.setenv("FAKE_ZE_COPY_FAIL", "1")
        assert ze.dma_copy_async(dst, src, 16) == 1
    finally:
        ze.destroy()


@requires_native
def test_synchronize_never_overruns_the_configured_timeout(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """R01 D3 / R09: the last polling slice is shortened so the wait ends at the deadline, not a slice later."""
    fake_env.setenv("FAKE_ZE_SYNC", "never")
    fake_env.setenv("ARC_ZE_SYNC_TIMEOUT_MS", "400")
    ze = NativeLevelZeroBridge()
    try:
        started = time.monotonic()
        assert ze.synchronize() == -9
        assert time.monotonic() - started < 0.47
    finally:
        ze.destroy()


@requires_native
def test_register_layer_rejects_values_ctypes_would_narrow(tmp_path: Path) -> None:
    """R01 D8: tensor_type above uint32, layer_idx above int32 and missing offset/size are errors, not silent truncation."""
    path = tmp_path / "b.bin"
    path.write_bytes(bytes(4096))
    engine = NativeMemoryEngine(path)
    try:
        good = {"name": "w", "tensor_type": 0, "dimensions": [4], "offset": 0, "size_bytes": 16}
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [{**good, "tensor_type": 2**32 + 5}])
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [{k: v for k, v in good.items() if k != "size_bytes"}])
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [{k: v for k, v in good.items() if k != "offset"}])
        assert engine.register_layer(0, [good]) is True
        with pytest.raises(NativeBoundsError):
            engine.get_tensor_buffer(2**32, "w", dtype=np.dtype(np.uint8))
    finally:
        engine.close()


@requires_native
def test_dma_copy_async_rejects_unrepresentable_size_and_pointers(fake_env: pytest.MonkeyPatch) -> None:  # noqa: F811
    """R01 D8: a size or pointer that would wrap in ctypes is rejected with INVALID_ARG."""
    ze = NativeLevelZeroBridge()
    try:
        src, dst = ze.allocate_usm(64, "host"), ze.allocate_usm(64, "host")
        assert ze.dma_copy_async(dst, src, 2**64 + 16) == -1
        assert ze.dma_copy_async(-5, src, 16) == -1
        assert ze.dma_copy_async(dst, 2**64, 16) == -1
        assert ze.dma_copy_async(dst, src, True) == -1
    finally:
        ze.destroy()


@requires_native
def test_get_tensor_buffer_racing_close_never_crashes(tmp_path: Path) -> None:
    """R01 D5: pointer lookup, view registration and close are serialised; run in a subprocess so a crash is visible."""
    code = textwrap.dedent(
        f"""
        import sys, threading, numpy as np
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        from arc_runner.native.bridge import NativeMemoryEngine
        path = {str(tmp_path / "r.bin")!r}
        open(path, "wb").write(bytes(65536))
        for _ in range(60):
            engine = NativeMemoryEngine(path)
            engine.register_layer(0, [{{"name": "w", "tensor_type": 0, "dimensions": [64], "offset": 0, "size_bytes": 256}}])
            stop = threading.Event()
            def reader():
                while not stop.is_set():
                    try:
                        v = engine.get_tensor_buffer(0, "w", dtype=np.dtype(np.uint8))
                        if v is not None:
                            int(v.sum())
                    except Exception:
                        pass
            threads = [threading.Thread(target=reader) for _ in range(4)]
            [t.start() for t in threads]
            engine.close()
            stop.set()
            [t.join() for t in threads]
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-400:]


@requires_native
@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_closing_the_engine_closes_file_descriptor_zero(tmp_path: Path) -> None:
    """R01 D7: a process without stdin gets fd 0 for the model file; close must still release it."""
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        from arc_runner.native.bridge import NativeMemoryEngine
        path = {str(tmp_path / "z.bin")!r}
        open(path, "wb").write(bytes(4096))
        os.close(0)
        engine = NativeMemoryEngine(path)
        assert os.path.exists("/proc/self/fd/0")
        engine.close()
        print("leaked" if os.path.exists("/proc/self/fd/0") else "clean")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == "clean", result.stdout + result.stderr


def test_zes_preinit_enables_sysman_unless_the_user_chose_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """R09: ZES_ENABLE_SYSMAN=1 is set before the loader is touched; an explicit value is kept."""
    from arc_runner.devices import sysman

    monkeypatch.delenv("ZES_ENABLE_SYSMAN", raising=False)
    monkeypatch.setenv("ARC_ZE_LOADER_PATH", "/nonexistent/libze.so")
    sysman.zes_preinit()
    import os

    assert os.environ["ZES_ENABLE_SYSMAN"] == "1"
    monkeypatch.setenv("ZES_ENABLE_SYSMAN", "0")
    sysman.zes_preinit()
    assert os.environ["ZES_ENABLE_SYSMAN"] == "0"


class _StagerBridge:
    """Bridge double counting concurrent use and recording frees."""

    is_available = True

    def __init__(self, sync_status: int = 0) -> None:
        self.sync_status = sync_status
        self.freed: List[int] = []
        self._buffers: List[Any] = []
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def allocate_usm(self, size: int, usm_type: str = "host") -> int:
        buf = (ctypes.c_ubyte * size)()
        self._buffers.append(buf)
        return ctypes.addressof(buf)

    def _enter(self) -> None:
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.01)
        with self._guard:
            self._active -= 1

    def dma_copy_async(self, dst: int, src: int, size: int) -> int:
        self._enter()
        return 0

    def synchronize(self) -> int:
        self._enter()
        return self.sync_status

    def free_usm(self, ptr: int) -> None:
        self.freed.append(ptr)


def test_stager_serialises_use_of_the_shared_command_list() -> None:
    """R01 D4 / R09: concurrent stage_layer calls must not interleave on the single command list."""
    from arc_runner.engine.flash_stream.device_stager import DeviceLayerStager

    bridge_double = _StagerBridge()
    stager = DeviceLayerStager(bridge_double)
    arrays = {"a": np.arange(8, dtype=np.float32)}
    threads = [threading.Thread(target=stager.stage_layer, args=(i, arrays)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert bridge_double.max_active == 1


def test_stager_does_not_free_memory_the_gpu_may_still_use_after_a_timeout() -> None:
    """R01 D4 / R09: after a sync timeout (-9) or dead context (-8) the buffer is retained, never freed."""
    from arc_runner.engine.flash_stream.device_stager import DeviceLayerStager

    for status in (-9, -8):
        bridge_double = _StagerBridge(sync_status=status)
        stager = DeviceLayerStager(bridge_double)
        with pytest.raises(NativeLayerError):
            stager.stage_layer(0, {"a": np.arange(8, dtype=np.float32)})
        assert bridge_double.freed == []
        assert stager.retained_after_failure() == 1


def test_stager_still_frees_after_an_ordinary_failure() -> None:
    """Other sync errors (device alive) still release the buffer."""
    from arc_runner.engine.flash_stream.device_stager import DeviceLayerStager

    bridge_double = _StagerBridge(sync_status=-5)
    stager = DeviceLayerStager(bridge_double)
    with pytest.raises(NativeLayerError):
        stager.stage_layer(0, {"a": np.arange(8, dtype=np.float32)})
    assert len(bridge_double.freed) == 1 and stager.retained_after_failure() == 0
