"""Testing utilities and hardware integration harness for Intel silicon.

Exports synthetic model graph builders, numerical statistics engines,
and hardware execution harnesses for Intel Arc GPUs, NPUs, and CPUs.
"""

from arc_runner.testing.hardware_harness import (
    HardwareBenchmarkResult,
    SiliconHardwareHarness,
    StudentTStatistics,
    SyntheticGraphBuilder,
    WelfordAccumulator,
)

__all__ = [
    "HardwareBenchmarkResult",
    "SiliconHardwareHarness",
    "StudentTStatistics",
    "SyntheticGraphBuilder",
    "WelfordAccumulator",
]
