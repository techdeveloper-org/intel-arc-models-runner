"""Python wrapper and RAII context manager for native Win32 system governor.

Controls process priority class, CPU thread affinity to Performance cores (P-cores),
and 1ms multimedia timer resolution to prevent clock throttling on Intel Core Ultra platforms.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger(__name__)


class GovernorError(ArcRunnerError):
    """Exception raised when governor fails to adjust OS process parameters."""


class _GovernorState(ctypes.Structure):
    _fields_ = [
        ("original_priority_class", ctypes.c_uint32),
        ("original_affinity_mask", ctypes.c_uint64),
        ("timer_period_active", ctypes.c_uint32),
        ("status_flags", ctypes.c_uint32),
    ]


class ProcessGovernor:
    """System-level execution governor for low-latency LLM inference."""

    _instance: Optional[ProcessGovernor] = None
    _dll: Optional[ctypes.CDLL] = None

    def __init__(self) -> None:
        """Initialize the governor state."""
        self._state = _GovernorState()
        self._is_active: bool = False
        self._init_native()

    def _init_native(self) -> None:
        """Load native DLL and bind C-ABI functions."""
        if ProcessGovernor._dll is not None:
            return

        dll_name = "libarc_native_sys.dll" if sys.platform == "win32" else "libarc_native_sys.so"
        dll_path = Path(__file__).resolve().parent / dll_name

        if not dll_path.exists():
            logger.warning("Native library %s not found at %s", dll_name, dll_path)
            return

        try:
            dll = ctypes.CDLL(str(dll_path))

            dll.arc_governor_init.argtypes = [ctypes.POINTER(_GovernorState)]
            dll.arc_governor_init.restype = ctypes.c_int

            dll.arc_governor_boost_high.argtypes = [ctypes.POINTER(_GovernorState)]
            dll.arc_governor_boost_high.restype = ctypes.c_int

            dll.arc_governor_set_affinity.argtypes = [ctypes.POINTER(_GovernorState), ctypes.c_uint64]
            dll.arc_governor_set_affinity.restype = ctypes.c_int

            dll.arc_governor_enable_timer.argtypes = [ctypes.POINTER(_GovernorState)]
            dll.arc_governor_enable_timer.restype = ctypes.c_int

            dll.arc_governor_restore.argtypes = [ctypes.POINTER(_GovernorState)]
            dll.arc_governor_restore.restype = ctypes.c_int

            ProcessGovernor._dll = dll
        except Exception as exc:
            logger.warning("Failed to bind native governor functions: %s", exc)

    def boost(self, pin_p_cores: bool = True) -> bool:
        """Boost process priority to HIGH and pin to P-cores.

        Args:
            pin_p_cores: When True, sets affinity to P-cores (cores 0-7, mask 0xFF).

        Returns:
            bool: True if priority boost succeeded.
        """
        if sys.platform != "win32" or ProcessGovernor._dll is None:
            return False

        dll = ProcessGovernor._dll
        dll.arc_governor_init(ctypes.byref(self._state))

        ret_boost = dll.arc_governor_boost_high(ctypes.byref(self._state))
        if ret_boost != 0:
            logger.warning("Failed to set HIGH_PRIORITY_CLASS (error=%d)", ret_boost)

        dll.arc_governor_enable_timer(ctypes.byref(self._state))

        if pin_p_cores:
            # Meteor Lake 125H: P-Cores are logical cores 0-7 (mask 0xFF)
            ret_aff = dll.arc_governor_set_affinity(ctypes.byref(self._state), 0xFF)
            if ret_aff != 0:
                logger.debug("Core affinity pinning returned %d (using system default)", ret_aff)

        self._is_active = True
        logger.debug("Process governor active: high priority and multimedia timer enabled")
        return ret_boost == 0

    def restore(self) -> None:
        """Restore default process priority, affinity, and timer resolution."""
        if not self._is_active or ProcessGovernor._dll is None:
            return

        dll = ProcessGovernor._dll
        dll.arc_governor_restore(ctypes.byref(self._state))
        self._is_active = False
        logger.debug("Process governor restored default system scheduling")

    def __enter__(self) -> ProcessGovernor:
        """Context manager entry."""
        self.boost()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.restore()
