"""Intel AI Boost NPU silicon device detector and capability query interface.

Queries OpenVINO core devices and inspects Intel NPU architecture, hardware tile
topology, and Level Zero driver compilation capabilities.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional

from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.npu_detector")


@dataclass
class NpuInfo:
    """Hardware capability and driver descriptor for Intel AI Boost NPU accelerators."""

    device_name: str
    full_name: str
    architecture: str
    driver_version: str
    available_devices: List[str]
    is_available: bool
    properties: Dict[str, Any]


class NpuDetector:
    """Discovers and inspects Intel AI Boost NPU accelerators via OpenVINO Core."""

    def __init__(self) -> None:
        """Initialize NpuDetector instance."""
        self._cached_info: Optional[NpuInfo] = None

    def detect(self) -> Optional[str]:
        """Detect presence of Intel NPU hardware.

        Returns:
            Optional[str]: "NPU" if hardware is present and not disabled, else None.
        """
        info = self.get_npu_info()
        return "NPU" if info and info.is_available else None

    def get_npu_info(self) -> Optional[NpuInfo]:
        """Query detailed hardware telemetry and driver properties for the NPU.

        Returns:
            Optional[NpuInfo]: Populated NpuInfo descriptor or None if absent or disabled.
        """
        if os.environ.get("ARC_NPU_DISABLE") == "1":
            logger.debug("NPU detection skipped: ARC_NPU_DISABLE environment flag is set")
            return None

        try:
            import openvino as ov

            core = ov.Core()
            available = core.available_devices

            npu_present = any("NPU" in dev.upper() for dev in available)
            if not npu_present:
                return None

            full_name = "Intel(R) AI Boost"
            try:
                full_name = str(core.get_property("NPU", "FULL_DEVICE_NAME"))
            except Exception:
                pass

            dev_list: List[str] = []
            try:
                raw_devs = core.get_property("NPU", "AVAILABLE_DEVICES")
                if isinstance(raw_devs, (list, tuple)):
                    dev_list = [str(d) for d in raw_devs]
            except Exception:
                pass

            arch = "UNKNOWN"
            for d in dev_list:
                if d.startswith("37"):
                    arch = "3720"
                    break
                elif d.startswith("40"):
                    arch = "4000"
                    break
            if arch == "UNKNOWN" and "3720" in full_name:
                arch = "3720"

            driver_ver = "unknown"
            try:
                driver_ver = str(core.get_property("NPU", "DEVICE_DRIVER_VERSION"))
            except Exception:
                pass

            self._cached_info = NpuInfo(
                device_name="NPU",
                full_name=full_name,
                architecture=arch,
                driver_version=driver_ver,
                available_devices=dev_list,
                is_available=True,
                properties={},
            )
            return self._cached_info

        except Exception as exc:
            logger.debug(f"OpenVINO NPU device inspection failed: {exc}")
            return None
