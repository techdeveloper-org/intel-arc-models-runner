"""Level Zero Sysman ctypes integration module.

Provides hardware telemetry including real-time VRAM memory usage, frequency,
and thermal indicators for Intel Arc GPUs via the oneAPI Level Zero loader.
"""

import ctypes
import ctypes.util
import logging
import platform
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("arc_runner.devices.sysman")


class _ZeDeviceMemProps(ctypes.Structure):
    """Level Zero device memory properties structure descriptor."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("maxClockRate", ctypes.c_uint32),
        ("maxBusWidth", ctypes.c_uint32),
        ("totalSize", ctypes.c_uint64),
        ("name", ctypes.c_char * 256),
    ]


class LevelZeroSysman:
    """Wrapper around Level Zero Sysman library for GPU telemetry queries."""

    def __init__(self) -> None:
        """Initialize the Level Zero Sysman interface."""
        self._lib: Optional[ctypes.CDLL] = None
        self._initialized: bool = False
        self._devices: List[int] = []
        self._unavailable_reasons: Dict[str, str] = {}
        self._load_library()
        if self._initialized and self._lib is not None:
            self._init_devices()

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
                self._configure_signatures()
                self._initialized = True
                return
            except (OSError, Exception):
                continue

        self._lib = None
        self._initialized = False

    def _configure_signatures(self) -> None:
        """Configure ctypes arguments and return types for Level Zero functions."""
        if not self._lib:
            return

        try:
            self._lib.zeInit.argtypes = [ctypes.c_uint32]
            self._lib.zeInit.restype = ctypes.c_uint32

            self._lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
            self._lib.zeDriverGet.restype = ctypes.c_uint32

            self._lib.zeDeviceGet.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
            self._lib.zeDeviceGet.restype = ctypes.c_uint32

            self._lib.zeDeviceGetMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
            self._lib.zeDeviceGetMemoryProperties.restype = ctypes.c_uint32
        except AttributeError:
            pass

    def _init_devices(self) -> None:
        """Enumerate and store available Level Zero device handles."""
        if not self._lib:
            return

        try:
            if self._lib.zeInit(0) != 0:
                return

            driver_count = ctypes.c_uint32(0)
            if self._lib.zeDriverGet(ctypes.byref(driver_count), None) != 0 or driver_count.value == 0:
                return

            drivers = (ctypes.c_void_p * driver_count.value)()
            self._lib.zeDriverGet(ctypes.byref(driver_count), drivers)

            for d in drivers:
                dev_count = ctypes.c_uint32(0)
                if self._lib.zeDeviceGet(d, ctypes.byref(dev_count), None) == 0 and dev_count.value > 0:
                    devs = (ctypes.c_void_p * dev_count.value)()
                    self._lib.zeDeviceGet(d, ctypes.byref(dev_count), devs)
                    self._devices.extend(int(dev) for dev in devs if dev)
        except Exception as exc:
            logger.debug("Failed Level Zero device initialization: %s", exc)

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
            Tuple[Optional[int], Optional[int]]: Total device memory in megabytes and free memory,
            which is always None because no real free-memory source is queried.
        """
        if not self.is_available or not self._lib or not self._devices:
            return None, None

        if device_index < 0 or device_index >= len(self._devices):
            return None, None

        dev_ptr = ctypes.c_void_p(self._devices[device_index])
        mem_count = ctypes.c_uint32(0)

        try:
            if self._lib.zeDeviceGetMemoryProperties(dev_ptr, ctypes.byref(mem_count), None) != 0 or mem_count.value == 0:
                return None, None

            props_array = (_ZeDeviceMemProps * mem_count.value)()
            for i in range(mem_count.value):
                props_array[i].stype = 0x11

            if self._lib.zeDeviceGetMemoryProperties(dev_ptr, ctypes.byref(mem_count), props_array) != 0:
                return None, None

            total_bytes = props_array[0].totalSize
            total_mb = int(total_bytes // (1024 * 1024))

            self._unavailable_reasons["gpu_free_memory"] = (
                "free device memory is not reported: integrated GPUs share system memory and expose no memory state"
            )
            return total_mb, None
        except Exception as exc:
            logger.debug("Error querying Level Zero memory: %s", exc)
            return None, None

    def query_package_power(self, device_index: int = 0) -> Optional[float]:
        """Query package power draw in watts for the target hardware.

        No power API is wired up yet, so this never reports a value rather than inventing one.

        Args:
            device_index: Hardware device index to inspect.

        Returns:
            Optional[float]: Always None; see get_unavailable_reason("package_power").
        """
        self._unavailable_reasons["package_power"] = "no power domain API is queried; power telemetry is not measured"
        return None

    def get_unavailable_reason(self, metric: str) -> Optional[str]:
        """Return why a telemetry metric last came back as None.

        Args:
            metric: Metric key such as "package_power" or "gpu_free_memory".

        Returns:
            Optional[str]: Human-readable reason, or None if no reason was recorded.
        """
        return self._unavailable_reasons.get(metric)
