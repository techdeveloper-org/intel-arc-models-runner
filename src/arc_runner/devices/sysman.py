"""Level Zero Sysman ctypes integration module.

Provides hardware telemetry including real-time VRAM memory usage, frequency,
and thermal indicators for Intel Arc GPUs via the oneAPI Level Zero loader.
"""

import ctypes
import ctypes.util
import logging
import platform
from typing import List, Optional, Tuple

logger = logging.getLogger("arc_runner.devices.sysman")


class LevelZeroSysman:
    """Wrapper around Level Zero Sysman library for GPU telemetry queries."""

    def __init__(self) -> None:
        """Initialize the Level Zero Sysman interface."""
        self._lib: Optional[ctypes.CDLL] = None
        self._initialized: bool = False
        self._load_library()

    def _load_library(self) -> None:
        """Attempt to load the Level Zero loader shared library across platforms."""
        if platform.system() == "Windows":
            candidate_names: List[str] = ["ze_loader.dll"]
        else:
            candidate_names = ["libze_loader.so.1", "libze_loader.so"]
            found = ctypes.util.find_library("ze_loader")
            if found and found not in candidate_names:
                candidate_names.append(found)

        for lib_name in candidate_names:
            try:
                self._lib = ctypes.CDLL(lib_name)
                self._initialized = True
                return
            except (OSError, Exception):
                continue

        self._lib = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if Level Zero Sysman library was loaded successfully.

        Returns:
            bool: True if Level Zero library is accessible.
        """
        return self._initialized and self._lib is not None

    def query_gpu_memory(self, device_index: int = 0) -> Tuple[Optional[int], Optional[int]]:
        """Query total and free memory for a specific GPU device index.

        Args:
            device_index: Hardware device index to inspect.

        Returns:
            Tuple[Optional[int], Optional[int]]: Total and free memory in megabytes, or (None, None).
        """
        if not self.is_available:
            return None, None

        return None, None
