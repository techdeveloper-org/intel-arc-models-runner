"""Pytest fixtures and configuration for real hardware integration testing.

Registers custom CLI flags and silicon availability fixtures for executing
tests on Intel Arc GPUs, Intel AI Boost NPUs, and Intel Core Ultra CPUs.
"""

import pytest

from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import SystemTopology


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command-line options for hardware integration testing.

    Args:
        parser: The pytest command-line parser instance.
    """
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run live hardware integration tests on real Intel silicon.",
    )


@pytest.fixture(scope="session")
def hardware_flag(request: pytest.FixtureRequest) -> bool:
    """Retrieve the status of the --hardware command-line flag.

    Args:
        request: The pytest fixture request context.

    Returns:
        True if --hardware was supplied on the CLI, False otherwise.
    """
    return bool(request.config.getoption("--hardware"))


@pytest.fixture(scope="session")
def live_topology() -> SystemTopology:
    """Detect and return physical silicon hardware topology on active host.

    Returns:
        SystemTopology instance representing available silicon accelerators.
    """
    detector = DeviceDetector()
    return detector.detect_topology()


@pytest.fixture(scope="session")
def require_intel_gpu(live_topology: SystemTopology) -> SystemTopology:
    """Enforce presence of an Intel Arc GPU accelerator on the active host.

    Args:
        live_topology: SystemTopology fixture representing discovered devices.

    Returns:
        SystemTopology instance when Intel Arc GPU is confirmed present.

    Raises:
        pytest.skip.Exception: When no Intel Arc GPU is available on host.
    """
    if not live_topology.has_intel_arc:
        pytest.skip("Intel Arc GPU accelerator not detected on current host.")
    return live_topology


@pytest.fixture(scope="session")
def require_intel_npu(live_topology: SystemTopology) -> SystemTopology:
    """Enforce presence of an Intel AI Boost NPU accelerator on the active host.

    Args:
        live_topology: SystemTopology fixture representing discovered devices.

    Returns:
        SystemTopology instance when Intel AI Boost NPU is confirmed present.

    Raises:
        pytest.skip.Exception: When no Intel AI Boost NPU is available on host.
    """
    if not live_topology.has_intel_npu:
        pytest.skip("Intel AI Boost NPU accelerator not detected on current host.")
    return live_topology
