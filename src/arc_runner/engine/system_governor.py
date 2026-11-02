"""Native Windows OS Kernel, Thread Scheduler, MMCSS, and Memory Governor.

Provides bare-metal execution environment tuning for Intel Core Ultra processors:
1. MMCSS (Multimedia Class Scheduler Service) Pro Audio real-time scheduling.
2. Win32 PROCESS_POWER_THROTTLING_STATE execution speed throttling disablement.
3. P-Core CPU affinity pinning (0x00FF) and HIGH_PRIORITY_CLASS enforcement.
4. Win32 1ms multimedia timer resolution (timeBeginPeriod) locking.
5. SeLockMemoryPrivilege probe with graceful large-page / working-set fallback.
6. Privilege-aware background service isolation (SysMain/WSearch pause with clean rollback).
"""

import contextlib
import ctypes
from ctypes import wintypes
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple


logger = logging.getLogger("arc_runner.engine.system_governor")

PROCESS_POWER_THROTTLING_CURRENT_VERSION: int = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED: int = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION: int = 0x4
ProcessPowerThrottling: int = 4
HIGH_PRIORITY_CLASS: int = 0x00000080
P_CORE_AFFINITY_MASK: int = 0x00FF
CACHE_LINE_SIZE: int = 64
DEFAULT_SERVICES_TO_PAUSE: Tuple[str, ...] = ("SysMain", "WSearch")

SE_PRIVILEGE_ENABLED: int = 0x00000002
TOKEN_ADJUST_PRIVILEGES: int = 0x0020
TOKEN_QUERY: int = 0x0008
SE_LOCK_MEMORY_NAME: str = "SeLockMemoryPrivilege"


class LUID(ctypes.Structure):
    """Win32 Locally Unique Identifier (LUID)."""

    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


class LUID_AND_ATTRIBUTES(ctypes.Structure):
    """Win32 LUID and privilege attributes structure."""

    _fields_ = [
        ("Luid", LUID),
        ("Attributes", wintypes.DWORD),
    ]


class TOKEN_PRIVILEGES(ctypes.Structure):
    """Win32 TOKEN_PRIVILEGES structure containing privilege array."""

    _fields_ = [
        ("PrivilegeCount", wintypes.DWORD),
        ("Privileges", LUID_AND_ATTRIBUTES * 1),
    ]


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    """Win32 PROCESS_POWER_THROTTLING_STATE struct for controlling thread efficiency mode."""

    _fields_ = [
        ("Version", wintypes.DWORD),
        ("ControlMask", wintypes.DWORD),
        ("StateMask", wintypes.DWORD),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    """Win32 PROCESS_MEMORY_COUNTERS_EX struct for working set inspection."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def is_windows_os() -> bool:
    """Check if host platform is Windows NT.

    Returns:
        bool: True if running on Windows.
    """
    return sys.platform.startswith("win")


def is_process_elevated() -> bool:
    """Check if current process has elevated Administrator rights.

    Returns:
        bool: True if running under an elevated security token.
    """
    if not is_windows_os():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception as exc:
        logger.debug("Failed to query Administrator token elevation: %s", exc)
        return False


def _init_kernel32_signatures() -> Any:
    """Initialize 64-bit argument and return types for kernel32 procedures.

    Returns:
        Any: Configured kernel32 WinDLL handle.
    """
    if not is_windows_os():
        return None

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    kernel32.SetProcessWorkingSetSize.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
    kernel32.SetProcessWorkingSetSize.restype = wintypes.BOOL
    return kernel32


class ServicePauseContext:
    """Context manager for pausing non-essential Windows background services."""

    def __init__(self, services: Tuple[str, ...] = DEFAULT_SERVICES_TO_PAUSE) -> None:
        """Initialize service pause manager.

        Args:
            services: Tuple of Windows service names to pause.
        """
        self.services: Tuple[str, ...] = services
        self._paused_services: List[str] = []

    def __enter__(self) -> "ServicePauseContext":
        """Pause candidate background services if administrator privileges are held.

        Returns:
            ServicePauseContext: Active context manager instance.
        """
        if not is_windows_os():
            return self

        if not is_process_elevated():
            logger.info("Non-elevated user mode detected: skipping service pause (running safely as standard user).")
            return self

        for svc in self.services:
            try:
                query_res = subprocess.run(
                    ["sc", "query", svc],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if "RUNNING" in query_res.stdout:
                    pause_res = subprocess.run(
                        ["net", "pause", svc],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if pause_res.returncode == 0:
                        self._paused_services.append(svc)
                        logger.info("Service paused for benchmark isolation: %s", svc)
                    else:
                        stop_res = subprocess.run(
                            ["net", "stop", svc],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if stop_res.returncode == 0:
                            self._paused_services.append(svc)
                            logger.info("Service stopped for benchmark isolation: %s", svc)
            except Exception as exc:
                logger.warning("Could not pause service %s: %s", svc, exc)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore all services that were paused or stopped on enter.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Exception traceback if raised.
        """
        if not is_windows_os() or not self._paused_services:
            return

        for svc in reversed(self._paused_services):
            try:
                subprocess.run(["net", "continue", svc], capture_output=True, check=False)
                subprocess.run(["net", "start", svc], capture_output=True, check=False)
                logger.info("Service restored post-benchmark: %s", svc)
            except Exception as exc:
                logger.warning("Could not restore service %s: %s", svc, exc)
        self._paused_services.clear()


class WindowsSystemGovernor:
    """Bare-metal governor controlling thread scheduling, timers, and memory policies."""

    def __init__(self) -> None:
        """Initialize governor runtime state."""
        self._timer_active: bool = False
        self._mmcss_handle: Optional[wintypes.HANDLE] = None
        self._power_throttling_disabled: bool = False
        self._large_pages_available: bool = False
        self._working_set_locked: bool = False
        self._affinity_set: bool = False
        self._kernel32 = _init_kernel32_signatures()

    @property
    def is_active(self) -> bool:
        """Return True if any governor optimization is actively engaged.

        Returns:
            bool: State flag.
        """
        return (
            self._timer_active
            or (self._mmcss_handle is not None)
            or self._power_throttling_disabled
            or self._affinity_set
        )

    def enable_high_priority_and_affinity(self, affinity_mask: int = P_CORE_AFFINITY_MASK) -> bool:
        """Pin process affinity to P-cores and elevate process priority class.

        Args:
            affinity_mask: Bitmask of CPU cores to pin process threads to.

        Returns:
            bool: True if affinity and priority were successfully set.
        """
        if not is_windows_os() or self._kernel32 is None:
            return False

        try:
            current_proc = self._kernel32.GetCurrentProcess()

            prio_ok = bool(self._kernel32.SetPriorityClass(current_proc, HIGH_PRIORITY_CLASS))
            aff_ok = bool(self._kernel32.SetProcessAffinityMask(current_proc, affinity_mask))
            self._affinity_set = prio_ok and aff_ok
            return self._affinity_set
        except Exception as exc:
            logger.warning("Failed to configure process priority/affinity: %s", exc)
            return False

    def disable_power_throttling(self) -> bool:
        """Disable Windows 11 power throttling and efficiency mode for this process.

        Returns:
            bool: True if power throttling was successfully disabled.
        """
        if not is_windows_os() or self._kernel32 is None:
            return False

        try:
            state = PROCESS_POWER_THROTTLING_STATE()
            state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
            state.ControlMask = (
                PROCESS_POWER_THROTTLING_EXECUTION_SPEED
                | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
            )
            state.StateMask = 0

            res = self._kernel32.SetProcessInformation(
                self._kernel32.GetCurrentProcess(),
                ProcessPowerThrottling,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
            self._power_throttling_disabled = bool(res)
            return self._power_throttling_disabled
        except Exception as exc:
            logger.warning("Failed to disable Win32 power throttling: %s", exc)
            return False

    def enable_mmcss(self, task_name: str = "Pro Audio") -> bool:
        """Register current thread with Windows Multimedia Class Scheduler Service.

        Args:
            task_name: MMCSS task profile name (e.g. 'Pro Audio' or 'Games').

        Returns:
            bool: True if MMCSS profile was registered.
        """
        if not is_windows_os():
            return False

        try:
            avrt = ctypes.windll.avrt
            task_idx = wintypes.DWORD(0)
            handle = avrt.AvSetMmThreadCharacteristicsW(task_name, ctypes.byref(task_idx))
            if handle:
                self._mmcss_handle = handle
                return True
            return False
        except Exception as exc:
            logger.warning("Failed to register MMCSS task %s: %s", task_name, exc)
            return False

    def revert_mmcss(self) -> None:
        """Unregister current thread from MMCSS."""
        if not is_windows_os() or self._mmcss_handle is None:
            return

        try:
            avrt = ctypes.windll.avrt
            avrt.AvRevertMmThreadCharacteristics(self._mmcss_handle)
        except Exception as exc:
            logger.debug("Failed to revert MMCSS handle: %s", exc)
        finally:
            self._mmcss_handle = None

    def enable_multimedia_timer(self, resolution_ms: int = 1) -> bool:
        """Lock system multimedia timer to sub-millisecond dispatch slices.

        Args:
            resolution_ms: Target timer period in milliseconds.

        Returns:
            bool: True if timer resolution was set.
        """
        if not is_windows_os():
            return False

        try:
            winmm = ctypes.windll.winmm
            res = winmm.timeBeginPeriod(resolution_ms)
            self._timer_active = (res == 0)
            return self._timer_active
        except Exception as exc:
            logger.warning("Failed to set timer resolution: %s", exc)
            return False

    def revert_multimedia_timer(self, resolution_ms: int = 1) -> None:
        """Release multimedia timer resolution override.

        Args:
            resolution_ms: Target timer period to release.
        """
        if not is_windows_os() or not self._timer_active:
            return

        try:
            winmm = ctypes.windll.winmm
            winmm.timeEndPeriod(resolution_ms)
        except Exception as exc:
            logger.debug("Failed to end timer period: %s", exc)
        finally:
            self._timer_active = False

    def probe_and_configure_large_pages(self) -> bool:
        """Probe for SeLockMemoryPrivilege and determine if 2 MB large pages are supported.

        Returns:
            bool: True if large pages can be allocated without privilege error.
        """
        if not is_windows_os() or self._kernel32 is None:
            return False

        try:
            advapi32 = ctypes.windll.advapi32
            token_handle = wintypes.HANDLE()

            if not advapi32.OpenProcessToken(
                self._kernel32.GetCurrentProcess(),
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                ctypes.byref(token_handle),
            ):
                return False

            luid = LUID()
            if not advapi32.LookupPrivilegeValueW(None, SE_LOCK_MEMORY_NAME, ctypes.byref(luid)):
                self._kernel32.CloseHandle(token_handle)
                return False

            tp = TOKEN_PRIVILEGES()
            tp.PrivilegeCount = 1
            tp.Privileges[0].Luid = luid
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED

            ok = advapi32.AdjustTokenPrivileges(token_handle, False, ctypes.byref(tp), 0, None, None)
            last_err = self._kernel32.GetLastError()
            self._kernel32.CloseHandle(token_handle)

            # Error 1314 represents ERROR_PRIVILEGE_NOT_HELD
            if ok and last_err != 1314:
                self._large_pages_available = True
                return True
            logger.info("SeLockMemoryPrivilege not granted (code %d); using transparent standard paging.", last_err)
            self._large_pages_available = False
            return False
        except Exception as exc:
            logger.debug("Large page privilege query failed: %s", exc)
            self._large_pages_available = False
            return False

    def lock_working_set(self, min_mb: int = 512, max_mb: int = 2048) -> bool:
        """Set process working set boundaries to reduce memory paging.

        Args:
            min_mb: Minimum working set in megabytes.
            max_mb: Maximum working set in megabytes.

        Returns:
            bool: True if working set boundaries were updated.
        """
        if not is_windows_os() or self._kernel32 is None:
            return False

        try:
            min_bytes = ctypes.c_size_t(min_mb * 1024 * 1024)
            max_bytes = ctypes.c_size_t(max_mb * 1024 * 1024)
            ok = bool(self._kernel32.SetProcessWorkingSetSize(self._kernel32.GetCurrentProcess(), min_bytes, max_bytes))
            self._working_set_locked = ok
            return ok
        except Exception as exc:
            logger.debug("Could not lock process working set size: %s", exc)
            return False

    @contextlib.contextmanager
    def benchmark_isolation(
        self,
        affinity_mask: int = P_CORE_AFFINITY_MASK,
        pause_services: bool = False,
    ) -> Iterator["WindowsSystemGovernor"]:
        """RAII context manager isolating benchmark execution from OS jitter.

        Args:
            affinity_mask: P-Core affinity bitmask.
            pause_services: If True and elevated, pauses background services during run.

        Yields:
            WindowsSystemGovernor: Active governor instance.
        """
        self.enable_high_priority_and_affinity(affinity_mask)
        self.disable_power_throttling()
        self.enable_multimedia_timer(1)
        self.enable_mmcss("Pro Audio")
        self.probe_and_configure_large_pages()
        self.lock_working_set(512, 2048)

        service_ctx: Optional[ServicePauseContext] = ServicePauseContext() if pause_services else None
        if service_ctx is not None:
            service_ctx.__enter__()

        try:
            yield self
        finally:
            if service_ctx is not None:
                service_ctx.__exit__(None, None, None)
            self.revert_mmcss()
            self.revert_multimedia_timer(1)
