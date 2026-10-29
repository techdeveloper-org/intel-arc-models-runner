"""Slice S-D tests: Level Zero bridge and Sysman against a fake Level Zero loader (no hardware needed)."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterator

import pytest

from arc_runner.devices.sysman import LevelZeroSysman
from arc_runner.native import bridge
from arc_runner.native.bridge import NativeLevelZeroBridge

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "src" / "arc_runner" / "native"
GIB = 1024**3

requires_native = pytest.mark.skipif(not bridge._load_native_library(), reason="native library not built or not loadable")
requires_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")


@pytest.fixture(scope="module")
def fake_loader(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the fake Level Zero loader once per module."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    output = tmp_path_factory.mktemp("fakeze") / "libze_fake.so"
    subprocess.run(
        ["gcc", "-std=c17", "-O1", "-fPIC", "-shared", str(REPO_ROOT / "tests" / "native_c" / "fake_ze_loader.c"), "-o", str(output)],
        check=True,
        capture_output=True,
    )
    return output


@pytest.fixture
def fake_env(fake_loader: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Point the bridge and Sysman at the fake loader and clear its controls."""
    monkeypatch.setenv("ARC_ZE_LOADER_PATH", str(fake_loader))
    for name in ("FAKE_ZE_LAYOUT", "FAKE_ZE_FAIL", "FAKE_ZE_SYNC", "FAKE_ZE_COPY_FAIL", "ARC_ZE_SYNC_TIMEOUT_MS"):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch


@requires_native
def test_arc_ze_init_selects_gpu_not_npu_when_npu_driver_comes_first(fake_env: pytest.MonkeyPatch) -> None:
    """P6/L4: with the NPU driver listed first the bridge still picks the GPU (device type 1)."""
    ze = NativeLevelZeroBridge()
    try:
        assert ze.is_available is True
        assert ze.device_type == 1
        assert ze.last_status() == 0
    finally:
        ze.destroy()


@requires_native
def test_arc_ze_init_without_gpu_reports_unavailable(fake_env: pytest.MonkeyPatch) -> None:
    """L4: a machine that only has an NPU yields no context and the NO_GPU status (-4)."""
    fake_env.setenv("FAKE_ZE_LAYOUT", "5")
    ze = NativeLevelZeroBridge()
    assert ze.is_available is False
    assert ze.init_status == -4
    assert ze.allocate_usm(64) is None
    assert ze.synchronize() == -10


@requires_native
def test_arc_ze_init_failing_driver_call_reports_api_error(fake_env: pytest.MonkeyPatch) -> None:
    """L5: a failing driver call is reported, not ignored, and nothing leaks (init returns None)."""
    fake_env.setenv("FAKE_ZE_FAIL", "zeCommandQueueCreate")
    ze = NativeLevelZeroBridge()
    assert ze.is_available is False
    assert ze.init_status == -5


@requires_native
def test_allocate_usm_oversize_and_invalid_sizes_return_none(fake_env: pytest.MonkeyPatch) -> None:
    """N4/L10: zero, negative, 2**62 and 2**63 sizes are refused for every residency type."""
    ze = NativeLevelZeroBridge()
    try:
        for usm_type in ("host", "device", "shared"):
            for size in (0, -1, 2**62, 2**63, 2**64):
                assert ze.allocate_usm(size, usm_type) is None
        with pytest.raises(ValueError):
            ze.allocate_usm(64, "bogus")
        assert ze.allocate_usm(True, "host") is None
    finally:
        ze.destroy()


@requires_native
def test_allocate_usm_defaults_to_host_and_tracks_allocations(fake_env: pytest.MonkeyPatch) -> None:
    """DL-005: host USM is the default; freed or destroyed allocations are not leaked."""
    ze = NativeLevelZeroBridge()
    pointer = ze.allocate_usm(4096)
    assert pointer
    assert ze._dll.arc_ze_alloc_count(ze._ctx) == 1
    ze.free_usm(pointer)
    ze.free_usm(pointer)
    assert ze._dll.arc_ze_alloc_count(ze._ctx) == 0
    assert ze.allocate_usm(4096, "shared") and ze.allocate_usm(4096, "device")
    ze.destroy()
    ze.destroy()
    assert ze.is_available is False


@requires_native
def test_dma_copy_falls_back_to_memcpy_for_host_and_shared_pointers(fake_env: pytest.MonkeyPatch) -> None:
    """L7: when the driver rejects a copy between host/shared buffers the bridge copies on the CPU and says so."""
    ze = NativeLevelZeroBridge()
    try:
        src = ze.allocate_usm(64, "host")
        dst = ze.allocate_usm(64, "shared")
        ctypes.memset(src, 0x7E, 64)
        assert ze.dma_copy_async(dst, src, 64) == 0
        fake_env.setenv("FAKE_ZE_COPY_FAIL", "1")
        assert ze.dma_copy_async(dst, src, 64) == 1
        assert ctypes.string_at(dst, 64) == b"\x7e" * 64
        device = ze.allocate_usm(64, "device")
        assert ze.dma_copy_async(device, src, 64) < 0
        assert ze.dma_copy_async(dst, src, 0) < 0
    finally:
        ze.destroy()


@requires_native
def test_synchronize_timeout_marks_context_dead_within_tdr_limit(fake_env: pytest.MonkeyPatch) -> None:
    """L8: a hung queue is abandoned after the configured timeout (< 2 s) and the context is dead."""
    fake_env.setenv("FAKE_ZE_SYNC", "never")
    fake_env.setenv("ARC_ZE_SYNC_TIMEOUT_MS", "300")
    ze = NativeLevelZeroBridge()
    try:
        started = time.monotonic()
        assert ze.synchronize() == -9
        assert 0.28 <= time.monotonic() - started < 1.2
        assert ze.is_dead is True
        assert ze.synchronize() == -8
        assert ze.allocate_usm(64) is None
    finally:
        ze.destroy()


@requires_native
def test_synchronize_device_lost_marks_context_dead(fake_env: pytest.MonkeyPatch) -> None:
    """L8: ZE_RESULT_ERROR_DEVICE_LOST kills the context immediately instead of resetting it."""
    fake_env.setenv("FAKE_ZE_SYNC", "lost")
    ze = NativeLevelZeroBridge()
    try:
        assert ze.synchronize() == -5
        assert ze.is_dead is True
    finally:
        ze.destroy()


@requires_native
def test_synchronize_timeout_is_clamped_below_tdr(fake_env: pytest.MonkeyPatch) -> None:
    """L8: a configured timeout above 1900 ms is clamped."""
    fake_env.setenv("ARC_ZE_SYNC_TIMEOUT_MS", "60000")
    ze = NativeLevelZeroBridge()
    try:
        assert ze._dll.arc_ze_sync_timeout_ms(ze._ctx) == 1900
    finally:
        ze.destroy()


@requires_gcc
def test_c_level_zero_harness_passes_under_asan_and_ubsan(tmp_path: Path) -> None:
    """The C harness (66 checks incl. wrong-stype detection, table limits, bounded waits) is clean under ASan/UBSan."""
    fake = tmp_path / "libze_fake.so"
    harness = tmp_path / "ze_harness"
    sanitizers = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    built_fake = subprocess.run(
        ["gcc", "-std=c17", "-O1", "-g", "-Wall", "-Wextra", "-fPIC", "-shared", *sanitizers,
         str(REPO_ROOT / "tests" / "native_c" / "fake_ze_loader.c"), "-o", str(fake)],
        capture_output=True, text=True,
    )
    if built_fake.returncode != 0 and "sanitize" in built_fake.stderr:
        pytest.skip("sanitizers unavailable on this toolchain")
    assert built_fake.returncode == 0, built_fake.stderr
    built = subprocess.run(
        ["gcc", "-std=gnu17", "-O1", "-g", "-Wall", "-Wextra", "-Werror", *sanitizers,
         str(REPO_ROOT / "tests" / "native_c" / "ze_harness.c"), str(NATIVE_DIR / "arc_level_zero_bridge.c"),
         "-ldl", "-o", str(harness)],
        capture_output=True, text=True,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(harness), str(fake)], capture_output=True, text=True, timeout=120)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "failures=0" in ran.stdout


@pytest.mark.skipif(shutil.which("x86_64-w64-mingw32-gcc") is None, reason="mingw-w64 cross compiler not available")
def test_windows_level_zero_branch_compiles_clean_with_werror(tmp_path: Path) -> None:
    """The Win32 loader path compiles under -Wpedantic -Werror via mingw-w64."""
    result = subprocess.run(
        ["x86_64-w64-mingw32-gcc", "-std=c17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
         "-c", str(NATIVE_DIR / "arc_level_zero_bridge.c"), "-o", str(tmp_path / "ze_win.o")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@requires_gcc
def test_sysman_keeps_only_gpu_devices_and_reports_total_memory_with_correct_stype(fake_env: pytest.MonkeyPatch) -> None:
    """L3/L4: Sysman skips the NPU, uses stype 0x7 for memory properties, and still reports no free memory."""
    sysman = LevelZeroSysman()
    assert sysman.is_available
    assert len(sysman._devices) == 1
    total_mb, free_mb = sysman.query_gpu_memory(0)
    assert total_mb == 8 * 1024
    assert free_mb is None
    assert sysman.query_package_power(0) is None


@requires_gcc
def test_sysman_without_gpu_has_no_devices_and_a_reason(fake_env: pytest.MonkeyPatch) -> None:
    """L4: an NPU-only machine yields no devices, (None, None) memory, and an explanation."""
    fake_env.setenv("FAKE_ZE_LAYOUT", "5")
    sysman = LevelZeroSysman()
    assert sysman._devices == []
    assert sysman.query_gpu_memory(0) == (None, None)
    assert sysman.get_unavailable_reason("devices") == "no GPU device found"
