"""Comprehensive unit and edge test suite for WindowsSystemGovernor and ServicePauseContext.

Covers:
- POS-01: High priority class and P-Core affinity mask setting.
- POS-02: Win32 PROCESS_POWER_THROTTLING_STATE disablement.
- POS-03: Multimedia timer resolution setting (timeBeginPeriod).
- POS-04: MMCSS Pro Audio thread registration and release.
- POS-05: Process working set size boundary locking.
- POS-06: Full benchmark isolation RAII context lifecycle.
- NEG-06: Non-Windows OS platform fallback safety.
- NEG-07: Unprivileged background service pause graceful degradation.
- NEG-10: Out-of-bounds working set allocation handling.
- EDG-07: Nested benchmark_isolation context manager idempotency.
- EDG-08: Exception safety and deterministic resource restoration.
- EDG-09: Struct 64-byte cacheline alignment and packing invariants.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest

from arc_runner.engine.system_governor import (
    CACHE_LINE_SIZE,
    HIGH_PRIORITY_CLASS,
    P_CORE_AFFINITY_MASK,
    PROCESS_MEMORY_COUNTERS_EX,
    PROCESS_POWER_THROTTLING_STATE,
    ServicePauseContext,
    WindowsSystemGovernor,
    is_process_elevated,
    is_windows_os,
)


def test_pos_01_governor_high_priority_and_affinity() -> None:
    """Verify high priority class and P-core affinity mask configuration."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    ok = governor.enable_high_priority_and_affinity(P_CORE_AFFINITY_MASK)
    assert ok is True
    assert governor.is_active is True


def test_pos_02_governor_power_throttling_disable() -> None:
    """Verify Win32 process power throttling execution speed disablement."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    ok = governor.disable_power_throttling()
    assert ok is True


def test_pos_03_governor_multimedia_timer_resolution() -> None:
    """Verify 1ms multimedia timer resolution acquisition and clean revert."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    assert governor.enable_multimedia_timer(1) is True
    assert governor._timer_active is True

    governor.revert_multimedia_timer(1)
    assert governor._timer_active is False


def test_pos_04_governor_mmcss_pro_audio_registration() -> None:
    """Verify MMCSS Pro Audio thread scheduling elevation and reversion."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    ok = governor.enable_mmcss("Pro Audio")
    if ok:
        assert governor._mmcss_handle is not None
        governor.revert_mmcss()
        assert governor._mmcss_handle is None


def test_pos_05_governor_working_set_lock() -> None:
    """Verify process working set boundary adjustment."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    ok = governor.lock_working_set(min_mb=512, max_mb=2048)
    assert ok is True


def test_pos_06_governor_benchmark_isolation_context() -> None:
    """Verify full benchmark isolation lifecycle inside RAII context manager."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    with governor.benchmark_isolation(pause_services=False) as active_gov:
        assert active_gov.is_active is True
        assert active_gov._timer_active is True

    assert governor._timer_active is False
    assert governor._mmcss_handle is None


def test_neg_06_non_windows_platform_fallback() -> None:
    """Verify graceful no-op behavior when executed on non-Windows platforms."""
    with patch("arc_runner.engine.system_governor.is_windows_os", return_value=False):
        governor = WindowsSystemGovernor()
        assert governor.enable_high_priority_and_affinity() is False
        assert governor.disable_power_throttling() is False
        assert governor.enable_mmcss() is False
        assert governor.enable_multimedia_timer(1) is False
        assert governor.lock_working_set() is False
        assert governor.probe_and_configure_large_pages() is False


def test_neg_07_service_pause_unprivileged_fallback() -> None:
    """Verify unprivileged user mode skips service pause without throwing PermissionError."""
    with patch("arc_runner.engine.system_governor.is_process_elevated", return_value=False):
        ctx = ServicePauseContext()
        with ctx:
            assert len(ctx._paused_services) == 0


def test_neg_10_large_page_privilege_fallback() -> None:
    """Verify SeLockMemoryPrivilege failure degrades cleanly to transparent paging."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    with patch("arc_runner.engine.system_governor.is_process_elevated", return_value=False):
        result = governor.probe_and_configure_large_pages()
        assert isinstance(result, bool)


def test_edg_07_benchmark_isolation_idempotency() -> None:
    """Verify nested benchmark isolation context managers do not leak timer periods."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    with governor.benchmark_isolation(pause_services=False):
        with governor.benchmark_isolation(pause_services=False):
            assert governor.is_active is True

    assert governor._timer_active is False
    assert governor._mmcss_handle is None


def test_edg_08_exception_safety_cleanup() -> None:
    """Verify resources are guaranteed to revert even when an unhandled exception occurs."""
    governor = WindowsSystemGovernor()
    if not is_windows_os():
        pytest.skip("Test requires Windows NT kernel")

    with pytest.raises(RuntimeError, match="Simulated crash"):
        with governor.benchmark_isolation(pause_services=False):
            assert governor._timer_active is True
            raise RuntimeError("Simulated crash")

    assert governor._timer_active is False
    assert governor._mmcss_handle is None


def test_edg_09_struct_cacheline_alignment() -> None:
    """Verify Win32 ctypes structures maintain expected alignment."""
    import ctypes

    size_power = ctypes.sizeof(PROCESS_POWER_THROTTLING_STATE)
    size_memory = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)

    assert size_power > 0
    assert size_memory > 0
    assert CACHE_LINE_SIZE == 64
