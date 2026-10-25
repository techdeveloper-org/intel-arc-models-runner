"""Hardware device detector for Intel Arc GPUs, NPUs, and CPUs.

Discovers heterogeneous accelerators through the OpenVINO runtime core and
oneAPI Level Zero interfaces, constructing a complete SystemTopology model.
"""

import platform
from typing import List, Optional

from arc_runner.devices.models import (
    DeviceArchitecture,
    DeviceInfo,
    DeviceType,
    SystemTopology,
)
from arc_runner.devices.sysman import LevelZeroSysman
from arc_runner.utils.dll_loader import (
    initialize_driver_environment,
    initialize_windows_dll_environment,
)


class DeviceDetector:
    """Discovers and queries Intel compute silicon across the host platform."""

    def __init__(self, sysman_client: Optional[LevelZeroSysman] = None) -> None:
        """Initialize the DeviceDetector with pre-flight DLL resolution.

        Args:
            sysman_client: Optional Level Zero Sysman client instance.
        """
        initialize_driver_environment()
        self._sysman = sysman_client or LevelZeroSysman()

    def _classify_gpu_architecture(self, full_name: str) -> DeviceArchitecture:
        """Map GPU commercial name to its microarchitectural family.

        Args:
            full_name: Device name returned by driver.

        Returns:
            DeviceArchitecture: Detected microarchitecture enum.
        """
        name_lower = full_name.lower()
        if "b580" in name_lower or "b570" in name_lower or "battlemage" in name_lower:
            return DeviceArchitecture.XE2_HPG
        if any(model in name_lower for model in ["a770", "a750", "a580", "a380", "a310"]):
            return DeviceArchitecture.XE_HPG
        if "igpu" in name_lower or "graphics" in name_lower or "ultra" in name_lower:
            return DeviceArchitecture.XE_LPG
        return DeviceArchitecture.XE_HPG

    def _classify_npu_architecture(self, full_name: str) -> DeviceArchitecture:
        """Map NPU commercial name to its microarchitectural family.

        Args:
            full_name: Device name returned by driver.

        Returns:
            DeviceArchitecture: Detected microarchitecture enum.
        """
        name_lower = full_name.lower()
        if "4000" in name_lower or "lunar" in name_lower:
            return DeviceArchitecture.NPU_4000
        if "boost" in name_lower or "3720" in name_lower:
            return DeviceArchitecture.NPU_3720
        return DeviceArchitecture.GENERIC_NPU

    def _classify_cpu_architecture(self, full_name: str) -> DeviceArchitecture:
        """Map CPU commercial name to its microarchitectural family.

        Args:
            full_name: Device name returned by driver.

        Returns:
            DeviceArchitecture: Detected microarchitecture enum.
        """
        name_lower = full_name.lower()
        if "ultra" in name_lower:
            return DeviceArchitecture.CORE_ULTRA_CPU
        return DeviceArchitecture.GENERIC_CPU

    def detect_topology(self) -> SystemTopology:
        """Probe all available execution silicon and construct the complete SystemTopology.

        Returns:
            SystemTopology: Populated topology model with devices and fallback ordering.
        """
        devices: List[DeviceInfo] = []
        host_cpu = platform.processor() or "Unknown CPU"

        try:
            import openvino as ov
            core = ov.Core()
            available = core.available_devices
        except Exception:
            available = []

        for device_id in available:
            dev_type = DeviceType.CPU
            if "GPU" in device_id:
                dev_type = DeviceType.GPU
            elif "NPU" in device_id:
                dev_type = DeviceType.NPU

            full_name = device_id
            capabilities: List[str] = []
            try:
                full_name = str(core.get_property(device_id, "FULL_DEVICE_NAME"))
            except Exception:
                pass

            try:
                raw_caps = core.get_property(device_id, "OPTIMIZATION_CAPABILITIES")
                if isinstance(raw_caps, (list, tuple)):
                    capabilities = [str(c) for c in raw_caps]
            except Exception:
                pass

            has_usm = "GPU_USM_MEMORY" in capabilities

            architecture = DeviceArchitecture.UNKNOWN
            if dev_type == DeviceType.GPU:
                architecture = self._classify_gpu_architecture(full_name)
            elif dev_type == DeviceType.NPU:
                architecture = self._classify_npu_architecture(full_name)
            elif dev_type == DeviceType.CPU:
                architecture = self._classify_cpu_architecture(full_name)

            devices.append(
                DeviceInfo(
                    device_id=device_id,
                    device_type=dev_type,
                    full_name=full_name,
                    architecture=architecture,
                    capabilities=capabilities,
                    has_usm=has_usm,
                    is_default_target=False,
                    is_default_draft=False,
                )
            )

        has_intel_arc = any(d.device_type == DeviceType.GPU for d in devices)
        has_intel_npu = any(d.device_type == DeviceType.NPU for d in devices)

        target_device: Optional[DeviceInfo] = None
        draft_device: Optional[DeviceInfo] = None

        for dev in devices:
            if dev.device_type == DeviceType.GPU and target_device is None:
                dev.is_default_target = True
                target_device = dev

        if target_device is None:
            for dev in devices:
                if dev.device_type == DeviceType.NPU:
                    dev.is_default_target = True
                    target_device = dev
                    break

        if target_device is None:
            for dev in devices:
                if dev.device_type == DeviceType.CPU:
                    dev.is_default_target = True
                    target_device = dev
                    break

        for dev in devices:
            if dev.device_type == DeviceType.NPU and dev != target_device:
                dev.is_default_draft = True
                draft_device = dev
                break

        fallback_chain: List[str] = []
        if target_device:
            fallback_chain.append(target_device.device_id)
        if draft_device and draft_device.device_id not in fallback_chain:
            fallback_chain.append(draft_device.device_id)
        for dev in devices:
            if dev.device_type == DeviceType.CPU and dev.device_id not in fallback_chain:
                fallback_chain.append(dev.device_id)

        return SystemTopology(
            devices=devices,
            has_intel_arc=has_intel_arc,
            has_intel_npu=has_intel_npu,
            target_device=target_device,
            draft_device=draft_device,
            fallback_chain=fallback_chain,
            host_cpu_model=host_cpu,
            os_platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        )
