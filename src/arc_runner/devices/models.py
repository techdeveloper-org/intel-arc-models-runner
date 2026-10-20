"""Data transfer objects and models for hardware devices and system topology.

Provides strongly-typed Pydantic schemas representing Intel Arc GPUs,
Intel AI Boost NPUs, and host CPUs.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Classification of heterogeneous compute silicon."""

    GPU = "GPU"
    NPU = "NPU"
    CPU = "CPU"


class DeviceArchitecture(str, Enum):
    """Specific microarchitectural family of the accelerator."""

    XE_HPG = "Xe-HPG (Alchemist dGPU)"
    XE_LPG = "Xe-LPG (Meteor Lake iGPU)"
    XE2_HPG = "Xe2-HPG (Battlemage dGPU)"
    XE2_LPG = "Xe2-LPG (Lunar Lake iGPU)"
    NPU_3720 = "Intel AI Boost (NPU 3720 / Meteor Lake)"
    NPU_4000 = "Intel AI Boost (NPU 4000 / Lunar Lake)"
    GENERIC_NPU = "Intel AI Boost NPU"
    CORE_ULTRA_CPU = "Intel Core Ultra Hybrid CPU"
    GENERIC_CPU = "Intel x86_64 CPU"
    UNKNOWN = "Unknown Architecture"


class DeviceInfo(BaseModel):
    """Detailed hardware metadata and execution capabilities for a single device."""

    device_id: str = Field(..., description="OpenVINO device identifier (e.g., GPU.0, NPU, CPU)")
    device_type: DeviceType = Field(..., description="Silicon device category")
    full_name: str = Field(..., description="Commercial product name reported by driver")
    architecture: DeviceArchitecture = Field(default=DeviceArchitecture.UNKNOWN, description="Microarchitectural family")
    capabilities: List[str] = Field(default_factory=list, description="Supported optimization capabilities (FP16, INT8, USM)")
    has_usm: bool = Field(default=False, description="Whether Unified Shared Memory zero-copy is supported")
    total_memory_mb: Optional[int] = Field(default=None, description="Total dedicated or shared memory in megabytes")
    free_memory_mb: Optional[int] = Field(default=None, description="Currently available memory in megabytes")
    driver_version: Optional[str] = Field(default=None, description="Reported driver version string")
    is_default_target: bool = Field(default=False, description="Selected as primary verification/target engine")
    is_default_draft: bool = Field(default=False, description="Selected as speculative draft engine")


class SystemTopology(BaseModel):
    """Complete heterogeneous accelerator topology and fallback routing configuration."""

    devices: List[DeviceInfo] = Field(default_factory=list, description="All discovered compute devices")
    has_intel_arc: bool = Field(default=False, description="True if one or more Intel Arc GPUs are present")
    has_intel_npu: bool = Field(default=False, description="True if an Intel AI Boost NPU is present")
    target_device: Optional[DeviceInfo] = Field(default=None, description="Primary device designated for model execution")
    draft_device: Optional[DeviceInfo] = Field(default=None, description="Draft device designated for speculative execution")
    fallback_chain: List[str] = Field(default_factory=list, description="Ordered degradation fallback sequence")
    host_cpu_model: str = Field(default="", description="Host processor name")
    os_platform: str = Field(default="", description="Operating system platform details")
