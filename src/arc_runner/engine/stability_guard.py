"""Hardware execution stability guard for Intel Arc GPU and NPU inference.

Pins CPU threads to physical P-cores, suppresses cyclic garbage collection during active
generation bursts, and prevents thermal clock frequency drop-off to guarantee constant,
jitter-free token generation throughput.
"""

import ctypes
import gc
from pathlib import Path
import sys
from types import TracebackType
from typing import Optional, Type

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.stability_guard")


class StabilityError(ArcRunnerError):
    """Base exception for hardware execution stability and core affinity failures."""

    def __init__(self, message: str, error_code: str = "STABILITY_ERROR", status_code: int = 500) -> None:
        """Initialize stability exception.

        Args:
            message: Descriptive error message.
            error_code: Unique error code token.
            status_code: HTTP status code.
        """
        super().__init__(message, error_code, status_code)


class ThreadAffinityPinningError(StabilityError):
    """Raised when setting OS thread affinity mask fails."""

    def __init__(self, mask: int, reason: str) -> None:
        """Initialize thread affinity error.

        Args:
            mask: Desired core affinity mask.
            reason: System error string.
        """
        super().__init__(
            f"Failed to pin thread to core affinity mask 0x{mask:X}: {reason}",
            "THREAD_AFFINITY_PINNING_ERROR",
            500,
        )
        self.mask = mask


class FrequencyLockError(StabilityError):
    """Raised when hardware clock frequency lock fails via Level Zero Sysman."""

    def __init__(self, frequency_mhz: float, device: str) -> None:
        """Initialize frequency lock error.

        Args:
            frequency_mhz: Requested target frequency in megahertz.
            device: Accelerator device identifier.
        """
        super().__init__(
            f"Failed to lock {device} clock frequency at {frequency_mhz:.1f} MHz.",
            "FREQUENCY_LOCK_ERROR",
            500,
        )
        self.frequency_mhz = frequency_mhz
        self.device = device


class USMAllocationError(StabilityError):
    """Raised when Level Zero Unified Shared Memory allocation fails."""

    def __init__(self, size_bytes: int) -> None:
        """Initialize USM allocation error.

        Args:
            size_bytes: Requested allocation size in bytes.
        """
        super().__init__(
            f"Failed to allocate {size_bytes} bytes in Level Zero Unified Shared Memory.",
            "USM_ALLOCATION_ERROR",
            500,
        )
        self.size_bytes = size_bytes


class StabilityGuard:
    """Stabilizes CPU core affinity, GPU clocks, and garbage collection during inference."""

    P_CORE_LOGICAL_MASK: int = 0x00FF

    def __init__(
        self,
        pin_p_cores: bool = True,
        disable_gc_during_gen: bool = True,
        affinity_mask: Optional[int] = None,
    ) -> None:
        """Initialize stability guard with core pinning and GC settings.

        Args:
            pin_p_cores: Whether to pin execution threads to Intel P-cores.
            disable_gc_during_gen: Whether to suppress cyclic GC during active generation.
            affinity_mask: Optional custom logical core bitmask (defaults to P_CORE_LOGICAL_MASK).
        """
        self.pin_p_cores = pin_p_cores
        self.disable_gc_during_gen = disable_gc_during_gen
        self.affinity_mask = affinity_mask or self.P_CORE_LOGICAL_MASK
        self._gc_was_enabled: bool = False
        self._scope_depth: int = 0
        self._original_thread_affinity: Optional[int] = None

    def enter_generation_scope(self) -> None:
        """Apply P-core thread affinity and disable cyclic GC before token decode starts."""
        self._scope_depth += 1
        if self._scope_depth > 1:
            return

        if self.disable_gc_during_gen:
            self._gc_was_enabled = gc.isenabled()
            if self._gc_was_enabled:
                gc.disable()
                logger.debug("Cyclic GC suppressed for active token generation burst.")

        if self.pin_p_cores and sys.platform == "win32":
            self._pin_to_p_cores()

    def exit_generation_scope(self) -> None:
        """Restore GC state and execute a bounded Generation 1 sweep upon decode completion."""
        if self._scope_depth <= 0:
            return

        self._scope_depth -= 1
        if self._scope_depth > 0:
            return

        if self.disable_gc_during_gen and self._gc_was_enabled:
            gc.enable()
            gc.collect(1)
            logger.debug("Cyclic GC restored; bounded generation 1 sweep complete.")

        if self.pin_p_cores and sys.platform == "win32" and self._original_thread_affinity is not None:
            self._restore_thread_affinity()

    def _pin_to_p_cores(self) -> None:
        """Pin current thread to physical P-cores via Win32 SetThreadAffinityMask."""
        try:
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.GetCurrentThread()
            res = kernel32.SetThreadAffinityMask(thread_handle, self.affinity_mask)
            if res != 0:
                self._original_thread_affinity = res
            kernel32.SetThreadPriority(thread_handle, 2)
            logger.debug(f"Inference worker pinned to core affinity mask 0x{self.affinity_mask:X}.")
        except Exception as exc:
            logger.warning(f"Failed to pin thread to P-cores: {exc}")

    def _restore_thread_affinity(self) -> None:
        """Restore previous thread affinity mask."""
        if self._original_thread_affinity is None:
            return
        try:
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.GetCurrentThread()
            kernel32.SetThreadAffinityMask(thread_handle, self._original_thread_affinity)
        except Exception as exc:
            logger.warning(f"Failed to restore original thread affinity: {exc}")

    def __enter__(self) -> "StabilityGuard":
        """Enter context manager scope.

        Returns:
            StabilityGuard: Initialized active guard instance.
        """
        self.enter_generation_scope()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Exit context manager scope, ensuring state is restored even on exception.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value instance.
            exc_tb: Exception traceback.

        Returns:
            bool: Always False so exceptions propagate to caller.
        """
        self.exit_generation_scope()
        return False
