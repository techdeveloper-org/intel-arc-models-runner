"""Unit tests for hardware device detector and topology models.

Tests device probing, classification, fallback ordering, and CLI integration
across positive, negative, and edge-case execution paths.
"""

import io
import json
import unittest.mock as mock
from typing import List

import pytest

from arc_runner.cli.main import main
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import (
    DeviceArchitecture,
    DeviceInfo,
    DeviceType,
    SystemTopology,
)


def test_detect_topology_live_silicon() -> None:
    """Verify that DeviceDetector discovers hardware devices on the active host."""
    detector = DeviceDetector()
    topology: SystemTopology = detector.detect_topology()

    assert isinstance(topology, SystemTopology)
    assert len(topology.devices) > 0
    assert topology.target_device is not None
    assert len(topology.fallback_chain) > 0
    assert topology.host_cpu_model != ""


def test_architecture_classification_gpu() -> None:
    """Verify accurate classification of Intel Arc GPU microarchitectures."""
    detector = DeviceDetector()
    assert detector._classify_gpu_architecture("Intel(R) Arc(TM) A770 Graphics") == DeviceArchitecture.XE_HPG
    assert detector._classify_gpu_architecture("Intel(R) Arc(TM) A750 Graphics") == DeviceArchitecture.XE_HPG
    assert detector._classify_gpu_architecture("Intel(R) Arc(TM) B580 Graphics") == DeviceArchitecture.XE2_HPG
    assert detector._classify_gpu_architecture("Intel(R) Arc(TM) Graphics (iGPU)") == DeviceArchitecture.XE_LPG


def test_architecture_classification_npu() -> None:
    """Verify accurate classification of Intel NPU generations."""
    detector = DeviceDetector()
    assert detector._classify_npu_architecture("Intel(R) AI Boost") == DeviceArchitecture.NPU_3720
    assert detector._classify_npu_architecture("Intel(R) AI Boost 4000") == DeviceArchitecture.NPU_4000
    assert detector._classify_npu_architecture("Unknown NPU") == DeviceArchitecture.GENERIC_NPU


def test_architecture_classification_cpu() -> None:
    """Verify accurate classification of CPU microarchitectures."""
    detector = DeviceDetector()
    assert detector._classify_cpu_architecture("Intel(R) Core(TM) Ultra 5 125H") == DeviceArchitecture.CORE_ULTRA_CPU
    assert detector._classify_cpu_architecture("Intel(R) Core(TM) i7-13700K") == DeviceArchitecture.GENERIC_CPU


def test_detect_topology_negative_no_devices() -> None:
    """Verify graceful handling when OpenVINO reports zero available devices."""
    mock_core = mock.MagicMock()
    mock_core.available_devices = []

    with mock.patch("openvino.Core", return_value=mock_core):
        detector = DeviceDetector()
        topology: SystemTopology = detector.detect_topology()

        assert topology.devices == []
        assert topology.has_intel_arc is False
        assert topology.has_intel_npu is False
        assert topology.target_device is None
        assert topology.draft_device is None
        assert topology.fallback_chain == []


def test_detect_topology_edge_case_property_exceptions() -> None:
    """Verify resilience when device property queries raise exceptions."""
    mock_core = mock.MagicMock()
    mock_core.available_devices = ["GPU", "NPU", "CPU"]
    mock_core.get_property.side_effect = RuntimeError("Driver communication fault")

    with mock.patch("openvino.Core", return_value=mock_core):
        detector = DeviceDetector()
        topology: SystemTopology = detector.detect_topology()

        assert len(topology.devices) == 3
        assert topology.has_intel_arc is True
        assert topology.has_intel_npu is True
        assert topology.target_device is not None
        assert topology.target_device.device_id == "GPU"
        assert topology.draft_device is not None
        assert topology.draft_device.device_id == "NPU"
        assert topology.fallback_chain == ["GPU", "NPU", "CPU"]


def test_cli_devices_json_output() -> None:
    """Verify that arc-runner devices --json returns valid parsable JSON."""
    captured = io.StringIO()
    with mock.patch("sys.stdout", captured):
        exit_code = main(["devices", "--json"])

    assert exit_code == 0
    output = captured.getvalue()
    parsed = json.loads(output)
    assert "devices" in parsed
    assert "fallback_chain" in parsed
    assert "has_intel_arc" in parsed


def test_cli_devices_default_run() -> None:
    """Verify that arc-runner default execution exits with code 0."""
    exit_code = main([])
    assert exit_code == 0
