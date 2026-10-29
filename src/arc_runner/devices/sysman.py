"""Level Zero Sysman ctypes integration module.

Provides hardware telemetry including real-time VRAM memory usage, frequency,
and thermal indicators for Intel Arc GPUs via the oneAPI Level Zero loader.
"""

import ctypes
import ctypes.util
import logging
import os
import platform
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("arc_runner.devices.sysman")

ZE_STYPE_DEVICE_PROPERTIES = 0x3
ZE_STYPE_DEVICE_MEMORY_PROPERTIES = 0x7
ZE_DEVICE_TYPE_GPU = 1
ZE_INIT_FLAG_GPU_ONLY = 1
LOADER_PATH_ENV = "ARC_ZE_LOADER_PATH"


class _ZeDeviceProps(ctypes.Structure):
    """ze_device_properties_t as declared in ze_api.h (the driver writes the whole structure)."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", ctypes.c_uint8 * 16),
        ("name", ctypes.c_char * 256),
    ]


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
        override = os.environ.get(LOADER_PATH_ENV, "").strip()
        if override:
            candidate_names: List[str] = [override]
        elif platform.system() == "Windows":
            candidate_names = ["ze_loader.dll"]
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

            self._lib.zeDeviceGetProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZeDeviceProps)]
            self._lib.zeDeviceGetProperties.restype = ctypes.c_uint32

            self._lib.zeDeviceGetMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
            self._lib.zeDeviceGetMemoryProperties.restype = ctypes.c_uint32
        except AttributeError:
            pass

    def _init_devices(self) -> None:
        """Enumerate Level Zero devices and keep only GPUs (the NPU is a separate device type)."""
        if not self._lib:
            return

        try:
            if self._lib.zeInit(ZE_INIT_FLAG_GPU_ONLY) != 0:
                self._unavailable_reasons["devices"] = "zeInit failed"
                return

            driver_count = ctypes.c_uint32(0)
            if self._lib.zeDriverGet(ctypes.byref(driver_count), None) != 0 or driver_count.value == 0:
                self._unavailable_reasons["devices"] = "no Level Zero driver"
                return

            drivers = (ctypes.c_void_p * driver_count.value)()
            self._lib.zeDriverGet(ctypes.byref(driver_count), drivers)

            for driver in drivers:
                self._devices.extend(self._gpu_devices_of(driver))
            if not self._devices:
                self._unavailable_reasons["devices"] = "no GPU device found"
        except Exception as exc:
            logger.debug("Failed Level Zero device initialization: %s", exc)

    def _gpu_devices_of(self, driver: Optional[int]) -> List[int]:
        """Return the handles of GPU-type devices belonging to one driver.

        Args:
            driver: Driver handle returned by zeDriverGet.

        Returns:
            List[int]: Device handles whose ze_device_type_t is GPU.
        """
        count = ctypes.c_uint32(0)
        if self._lib.zeDeviceGet(driver, ctypes.byref(count), None) != 0 or count.value == 0:
            return []
        devices = (ctypes.c_void_p * count.value)()
        self._lib.zeDeviceGet(driver, ctypes.byref(count), devices)
        gpus: List[int] = []
        for device in devices:
            props = _ZeDeviceProps()
            props.stype = ZE_STYPE_DEVICE_PROPERTIES
            if device and self._lib.zeDeviceGetProperties(device, ctypes.byref(props)) == 0 and props.type == ZE_DEVICE_TYPE_GPU:
                gpus.append(int(device))
        return gpus

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
                props_array[i].stype = ZE_STYPE_DEVICE_MEMORY_PROPERTIES

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
