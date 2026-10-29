"""Level Zero Sysman ctypes integration module.

Provides hardware telemetry including real-time VRAM memory usage, frequency,
and thermal indicators for Intel Arc GPUs via the oneAPI Level Zero loader.
"""

import ctypes
import ctypes.util
import logging
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("arc_runner.devices.sysman")

ZE_STYPE_DEVICE_PROPERTIES = 0x3
ZE_STYPE_DEVICE_MEMORY_PROPERTIES = 0x7
ZE_DEVICE_TYPE_GPU = 1
ZE_INIT_FLAG_GPU_ONLY = 1
ZES_STYPE_FREQ_PROPERTIES = 0x9
ZES_STYPE_MEM_PROPERTIES = 0xB
ZES_STYPE_POWER_PROPERTIES = 0xD
ZES_STYPE_FREQ_STATE = 0x1B
ZES_STYPE_MEM_STATE = 0x1E
ZES_FREQ_DOMAIN_GPU = 0
ZES_MEM_LOC_DEVICE = 1
MIN_POWER_SAMPLE_INTERVAL_S = 0.01
MAX_POWER_SAMPLE_INTERVAL_S = 5.0
DEFAULT_POWER_SAMPLE_INTERVAL_S = 0.1
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


class _ZesPowerProps(ctypes.Structure):
    """zes_power_properties_t as declared in zes_api.h."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ctypes.c_uint8),
        ("isEnergyThresholdSupported", ctypes.c_uint8),
        ("defaultLimit", ctypes.c_int32),
        ("minLimit", ctypes.c_int32),
        ("maxLimit", ctypes.c_int32),
    ]


class _ZesEnergyCounter(ctypes.Structure):
    """zes_power_energy_counter_t: monotonic energy in microjoules and a microsecond timestamp."""

    _fields_ = [("energy", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


class _ZesFreqProps(ctypes.Structure):
    """zes_freq_properties_t as declared in zes_api.h."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ctypes.c_uint8),
        ("isThrottleEventSupported", ctypes.c_uint8),
        ("min", ctypes.c_double),
        ("max", ctypes.c_double),
    ]


class _ZesFreqState(ctypes.Structure):
    """zes_freq_state_t as declared in zes_api.h."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("currentVoltage", ctypes.c_double),
        ("request", ctypes.c_double),
        ("tdp", ctypes.c_double),
        ("efficient", ctypes.c_double),
        ("actual", ctypes.c_double),
        ("throttleReasons", ctypes.c_uint32),
    ]


class _ZesMemProps(ctypes.Structure):
    """zes_mem_properties_t as declared in zes_api.h."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("onSubdevice", ctypes.c_uint8),
        ("subdeviceId", ctypes.c_uint32),
        ("location", ctypes.c_uint32),
        ("physicalSize", ctypes.c_uint64),
        ("busWidth", ctypes.c_int32),
        ("numChannels", ctypes.c_int32),
    ]


class _ZesMemState(ctypes.Structure):
    """zes_mem_state_t as declared in zes_api.h."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("health", ctypes.c_uint32),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
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

    _sysman_ready: Optional[bool] = None
    _sysman_failure: str = ""

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
            Tuple[Optional[int], Optional[int]]: Total device memory in megabytes (core API) and free
            on-board memory in megabytes (Sysman), which is None with a reason when only shared
            system memory exists or Sysman is unavailable.
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
            return total_mb, self._query_free_memory_mb(dev_ptr)
        except Exception as exc:
            logger.debug("Error querying Level Zero memory: %s", exc)
            return None, None

    def _ensure_sysman(self, metric: str) -> bool:
        """Initialize the Sysman API once; record why it is unusable under the metric key.

        Args:
            metric: Unavailable-reason key to fill on failure.

        Returns:
            bool: True if zesInit succeeded.
        """
        if self._sysman_ready is None:
            self._sysman_ready = False
            self._sysman_failure = ""
            try:
                lib = self._lib
                lib.zesInit.argtypes = [ctypes.c_uint32]
                lib.zesInit.restype = ctypes.c_uint32
                result = int(lib.zesInit(0))
                if result == 0:
                    self._sysman_ready = True
                else:
                    self._sysman_failure = f"zesInit failed with result 0x{result:x}"
            except (AttributeError, TypeError) as exc:
                self._sysman_failure = f"Sysman API not exported by the Level Zero loader ({exc})"
        if not self._sysman_ready:
            self._unavailable_reasons[metric] = self._sysman_failure
        return bool(self._sysman_ready)

    def _enumerate_handles(self, function: str, device: Any, metric: str) -> List[int]:
        """Enumerate Sysman component handles of one kind for a device.

        Args:
            function: Name of the zesDeviceEnum* function.
            device: Device handle.
            metric: Unavailable-reason key to fill on failure.

        Returns:
            List[int]: Component handles, empty if none could be enumerated.
        """
        fn = getattr(self._lib, function, None)
        if fn is None:
            self._unavailable_reasons[metric] = f"{function} not exported by the Level Zero loader"
            return []
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = ctypes.c_uint32
        count = ctypes.c_uint32(0)
        result = int(fn(device, ctypes.byref(count), None))
        if result != 0:
            self._unavailable_reasons[metric] = f"{function} failed with result 0x{result:x}"
            return []
        if count.value == 0:
            self._unavailable_reasons[metric] = f"{function} reported no components"
            return []
        handles = (ctypes.c_void_p * count.value)()
        result = int(fn(device, ctypes.byref(count), handles))
        if result != 0:
            self._unavailable_reasons[metric] = f"{function} failed with result 0x{result:x}"
            return []
        return [int(h) for h in handles[: count.value] if h]

    def _call(self, function: str, handle: int, struct: Any, metric: str) -> bool:
        """Call a zes*Get* function taking (handle, struct*); record the result code on failure.

        Args:
            function: Function name.
            handle: Component handle.
            struct: ctypes structure instance filled by the driver.
            metric: Unavailable-reason key to fill on failure.

        Returns:
            bool: True on ZE_RESULT_SUCCESS.
        """
        fn = getattr(self._lib, function, None)
        if fn is None:
            self._unavailable_reasons[metric] = f"{function} not exported by the Level Zero loader"
            return False
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(type(struct))]
        fn.restype = ctypes.c_uint32
        result = int(fn(ctypes.c_void_p(handle), ctypes.byref(struct)))
        if result != 0:
            self._unavailable_reasons[metric] = f"{function} failed with result 0x{result:x}"
            return False
        return True

    def _query_free_memory_mb(self, device: ctypes.c_void_p) -> Optional[int]:
        """Return free on-board device memory in megabytes from Sysman, if a device module exists.

        Args:
            device: Device handle.

        Returns:
            Optional[int]: Free megabytes summed over device-local memory modules, else None with a reason.
        """
        metric = "gpu_free_memory"
        if not self._ensure_sysman(metric):
            return None
        free_bytes = 0
        found_device_module = False
        saw_system_module = False
        for module in self._enumerate_handles("zesDeviceEnumMemoryModules", device, metric):
            props = _ZesMemProps()
            props.stype = ZES_STYPE_MEM_PROPERTIES
            if not self._call("zesMemoryGetProperties", module, props, metric):
                return None
            if props.location != ZES_MEM_LOC_DEVICE:
                saw_system_module = True
                continue
            state = _ZesMemState()
            state.stype = ZES_STYPE_MEM_STATE
            if not self._call("zesMemoryGetState", module, state, metric):
                return None
            free_bytes += int(state.free)
            found_device_module = True
        if found_device_module:
            self._unavailable_reasons.pop(metric, None)
            return int(free_bytes // (1024 * 1024))
        if saw_system_module:
            self._unavailable_reasons[metric] = "only system memory modules exist: an integrated GPU has no on-board free memory"
        return None

    def query_package_power(
        self, device_index: int = 0, sample_interval_s: float = DEFAULT_POWER_SAMPLE_INTERVAL_S
    ) -> Optional[float]:
        """Measure package power in watts from two Sysman energy-counter snapshots.

        Power is (energy2 - energy1) microjoules over (timestamp2 - timestamp1) microseconds on the
        first non-subdevice power domain, which equals watts. This call blocks for sample_interval_s.

        Args:
            device_index: Hardware device index to inspect.
            sample_interval_s: Seconds between the two snapshots, clamped to [0.01, 5].

        Returns:
            Optional[float]: Average watts over the interval, or None with a reason in
            get_unavailable_reason("package_power") if no honest value exists.
        """
        interval = min(max(float(sample_interval_s), MIN_POWER_SAMPLE_INTERVAL_S), MAX_POWER_SAMPLE_INTERVAL_S)
        first = self.read_package_energy(device_index)
        if first is None:
            return None
        time.sleep(interval)
        second = self.read_package_energy(device_index)
        if second is None:
            return None
        metric = "package_power"
        if second[0] < first[0]:
            self._unavailable_reasons[metric] = "energy counter went backwards (wrap or reset); no power value"
            return None
        if second[1] <= first[1]:
            self._unavailable_reasons[metric] = "energy counter timestamps did not advance"
            return None
        self._unavailable_reasons.pop(metric, None)
        return float(second[0] - first[0]) / float(second[1] - first[1])

    def read_package_energy(self, device_index: int = 0) -> Optional[Tuple[int, int]]:
        """Read the package-domain energy counter once.

        Args:
            device_index: Hardware device index to inspect.

        Returns:
            Optional[Tuple[int, int]]: (energy in microjoules, timestamp in microseconds) of the first
            non-subdevice power domain, or None with a reason in get_unavailable_reason("package_power").
            The timestamp is only meaningful as a difference between two reads.
        """
        metric = "package_power"
        device = self._sysman_device(device_index, metric)
        if device is None or not self._ensure_sysman(metric):
            return None
        try:
            package = None
            for domain in self._enumerate_handles("zesDeviceEnumPowerDomains", device, metric):
                props = _ZesPowerProps()
                props.stype = ZES_STYPE_POWER_PROPERTIES
                if not self._call("zesPowerGetProperties", domain, props, metric):
                    return None
                if not props.onSubdevice:
                    package = domain
                    break
            if package is None:
                if metric not in self._unavailable_reasons:
                    self._unavailable_reasons[metric] = "no package-level power domain (only subdevice domains)"
                return None
            counter = _ZesEnergyCounter()
            if not self._call("zesPowerGetEnergyCounter", package, counter, metric):
                return None
        except Exception as exc:
            self._unavailable_reasons[metric] = f"power query raised {type(exc).__name__}: {exc}"
            return None
        return int(counter.energy), int(counter.timestamp)

    def query_gpu_frequency_mhz(self, device_index: int = 0) -> Optional[float]:
        """Read the resolved GPU-core frequency in MHz from Sysman.

        Args:
            device_index: Hardware device index to inspect.

        Returns:
            Optional[float]: Actual frequency in MHz of the first non-subdevice GPU frequency domain,
            or None with a reason in get_unavailable_reason("gpu_frequency").
        """
        metric = "gpu_frequency"
        device = self._sysman_device(device_index, metric)
        if device is None or not self._ensure_sysman(metric):
            return None
        try:
            for domain in self._enumerate_handles("zesDeviceEnumFrequencyDomains", device, metric):
                props = _ZesFreqProps()
                props.stype = ZES_STYPE_FREQ_PROPERTIES
                if not self._call("zesFrequencyGetProperties", domain, props, metric):
                    return None
                if props.type != ZES_FREQ_DOMAIN_GPU or props.onSubdevice:
                    continue
                state = _ZesFreqState()
                state.stype = ZES_STYPE_FREQ_STATE
                if not self._call("zesFrequencyGetState", domain, state, metric):
                    return None
                if state.actual < 0:
                    self._unavailable_reasons[metric] = "driver reports the actual frequency as unknown"
                    return None
                self._unavailable_reasons.pop(metric, None)
                return float(state.actual)
        except Exception as exc:
            self._unavailable_reasons[metric] = f"frequency query raised {type(exc).__name__}: {exc}"
            return None
        self._unavailable_reasons.setdefault(metric, "no GPU-core frequency domain at package level")
        return None

    def _sysman_device(self, device_index: int, metric: str) -> Optional[ctypes.c_void_p]:
        """Return the handle of a GPU device for Sysman calls (zes_device_handle_t is ze_device_handle_t).

        Args:
            device_index: Hardware device index.
            metric: Unavailable-reason key to fill on failure.

        Returns:
            Optional[ctypes.c_void_p]: Device handle, or None if the index or library is unusable.
        """
        if not self.is_available or not self._lib or not self._devices:
            self._unavailable_reasons[metric] = self._unavailable_reasons.get("devices", "Level Zero not available")
            return None
        if device_index < 0 or device_index >= len(self._devices):
            self._unavailable_reasons[metric] = f"device index {device_index} out of range"
            return None
        return ctypes.c_void_p(self._devices[device_index])

    def get_unavailable_reason(self, metric: str) -> Optional[str]:
        """Return why a telemetry metric last came back as None.

        Args:
            metric: Metric key such as "package_power" or "gpu_free_memory".

        Returns:
            Optional[str]: Human-readable reason, or None if no reason was recorded.
        """
        return self._unavailable_reasons.get(metric)
