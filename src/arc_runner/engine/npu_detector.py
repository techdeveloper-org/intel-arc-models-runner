"""Intel AI Boost NPU silicon device detector via OpenVINO device enumeration."""

import os
from typing import Optional


class NpuDetector:
    """Detects available Intel AI Boost NPU accelerators."""

    def __init__(self) -> None:
        """Initialize NpuDetector instance."""
        pass

    def detect(self) -> Optional[str]:
        """Detect presence of Intel NPU hardware.

        Returns:
            Optional[str]: "NPU" if hardware is present and not disabled, else None.
        """
        if os.environ.get("ARC_NPU_DISABLE") == "1":
            return None

        try:
            import openvino as ov

            core = ov.Core()
            available = core.available_devices
            for dev in available:
                if "NPU" in dev.upper():
                    return "NPU"
            return None
        except Exception:
            return None
