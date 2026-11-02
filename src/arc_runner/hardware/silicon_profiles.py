"""Multi-Tier Silicon Profile Detector and Hardware Architecture Classifier.

Classifies host hardware platforms across Laptop/Edge UMA (Meteor Lake Xe-LPG),
Workstation dGPU (Intel Arc A770, Battlemage B580), and Data Center Server tiers
(Ponte Vecchio, Gaudi 2/3), determining optimal speculative batching lookahead (gamma),
KV-cache compression, and heterogeneous CPU+GPU+NPU execution parameters.
"""

from dataclasses import dataclass
from enum import Enum
import os
from typing import Optional

from arc_runner.core.logging import get_logger
from arc_runner.engine.npu_detector import NpuDetector

logger = get_logger("arc_runner.hardware.silicon_profiles")


class SiliconTier(str, Enum):
    """Categorical classification of host hardware execution environment."""

    LAPTOP_UMA = "LAPTOP_UMA"
    WORKSTATION_DGPU = "WORKSTATION_DGPU"
    SERVER_DATACENTER = "SERVER_DATACENTER"
    GENERIC_CPU = "GENERIC_CPU"


@dataclass
class SiliconProfile:
    """Comprehensive hardware configuration and performance characteristics descriptor."""

    tier: SiliconTier
    gpu_name: str
    npu_present: bool
    npu_architecture: str
    theoretical_bandwidth_gb_s: float
    recommended_gamma: int
    recommended_kv_cache_precision: str
    supports_heterogeneous_fusion: bool


class SiliconProfiler:
    """Detects and characterizes host silicon topology for heterogeneous inference."""

    def __init__(self) -> None:
        """Initialize SiliconProfiler instance."""
        self._cached_profile: Optional[SiliconProfile] = None

    def detect_profile(self) -> SiliconProfile:
        """Query host hardware and classify into architectural performance tiers.

        Returns:
            SiliconProfile: Populated descriptor with recommended inference hyperparameters.
        """
        if self._cached_profile is not None:
            return self._cached_profile

        gpu_name = "Unknown"
        npu_present = False
        npu_arch = "NONE"

        try:
            import openvino as ov

            core = ov.Core()
            devices = core.available_devices

            for dev in devices:
                if dev.startswith("GPU"):
                    try:
                        gpu_name = str(core.get_property(dev, "FULL_DEVICE_NAME"))
                    except Exception:
                        gpu_name = dev
                    break

            npu_detector = NpuDetector()
            npu_info = npu_detector.get_npu_info()
            if npu_info and npu_info.is_available:
                npu_present = True
                npu_arch = npu_info.architecture

        except Exception as exc:
            logger.debug(f"Hardware enumeration in SiliconProfiler caught exception: {exc}")

        gpu_upper = gpu_name.upper()

        if any(token in gpu_upper for token in ("DATA CENTER", "MAX 1550", "MAX 1100", "PONTE VECCHIO", "GAUDI")):
            tier = SiliconTier.SERVER_DATACENTER
            bandwidth = 3276.8
            gamma = 6
            kv_precision = "f16"
            fused = False
        elif any(token in gpu_upper for token in ("A770", "A750", "B580", "B570", "DG2", "ALCHEMIST", "BATTLEMAGE")):
            tier = SiliconTier.WORKSTATION_DGPU
            bandwidth = 512.0
            gamma = 5
            kv_precision = "f16"
            fused = npu_present
        elif any(token in gpu_upper for token in ("ARC(TM) GRAPHICS", "XE-LPG", "METEOR LAKE", "LUNAR LAKE", "ARROW LAKE", "IGPU")):
            tier = SiliconTier.LAPTOP_UMA
            bandwidth = 89.6
            gamma = 3
            kv_precision = "u8"
            fused = npu_present
        elif gpu_name != "Unknown":
            tier = SiliconTier.LAPTOP_UMA
            bandwidth = 89.6
            gamma = 3
            kv_precision = "u8"
            fused = npu_present
        else:
            tier = SiliconTier.GENERIC_CPU
            bandwidth = 50.0
            gamma = 2
            kv_precision = "f16"
            fused = False

        self._cached_profile = SiliconProfile(
            tier=tier,
            gpu_name=gpu_name,
            npu_present=npu_present,
            npu_architecture=npu_arch,
            theoretical_bandwidth_gb_s=bandwidth,
            recommended_gamma=gamma,
            recommended_kv_cache_precision=kv_precision,
            supports_heterogeneous_fusion=fused,
        )

        logger.info(
            f"SiliconProfiler classified host as {tier.value} "
            f"(GPU: '{gpu_name}', NPU: {npu_present} [{npu_arch}], Bandwidth: {bandwidth} GB/s)"
        )
        return self._cached_profile
