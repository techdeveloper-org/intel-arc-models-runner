"""Unit, negative, and edge-case triad tests for SiliconProfiler and SiliconTier classification.

Tests hardware detection across Laptop UMA (Meteor Lake Xe-LPG), Workstation dGPU (Arc A770, B580),
and Datacenter Server tiers (Ponte Vecchio, Gaudi), verifying recommended gamma and KV precision.
"""

from unittest.mock import MagicMock, patch
import pytest

from arc_runner.engine.npu_detector import NpuInfo
from arc_runner.hardware.silicon_profiles import SiliconProfile, SiliconProfiler, SiliconTier


class TestSiliconProfiles:
    """Test suite for SiliconProfiler multi-tier hardware classification."""

    def test_should_classify_meteor_lake_laptop_uma_p_sil_1(self) -> None:
        """P-SIL-1: Verify Intel Core Ultra Meteor Lake Xe-LPG is classified as LAPTOP_UMA."""
        mock_core = MagicMock()
        mock_core.available_devices = ["GPU", "NPU", "CPU"]
        mock_core.get_property.return_value = "Intel(R) Arc(TM) Graphics (iGPU)"

        mock_npu_info = NpuInfo(
            device_name="NPU",
            full_name="Intel(R) AI Boost",
            architecture="3720",
            driver_version="1.0",
            available_devices=["3720"],
            is_available=True,
            properties={},
        )

        with patch("openvino.Core", return_value=mock_core):
            with patch("arc_runner.engine.npu_detector.NpuDetector.get_npu_info", return_value=mock_npu_info):
                profiler = SiliconProfiler()
                profile = profiler.detect_profile()

                assert profile.tier == SiliconTier.LAPTOP_UMA
                assert profile.theoretical_bandwidth_gb_s == 89.6
                assert profile.recommended_gamma == 3
                assert profile.recommended_kv_cache_precision == "u8"
                assert profile.npu_present is True
                assert profile.npu_architecture == "3720"
                assert profile.supports_heterogeneous_fusion is True

    def test_should_classify_discrete_arc_workstation_dgpu_p_sil_2(self) -> None:
        """P-SIL-2: Verify Intel Arc A770 / Battlemage B580 is classified as WORKSTATION_DGPU."""
        mock_core = MagicMock()
        mock_core.available_devices = ["GPU", "CPU"]
        mock_core.get_property.return_value = "Intel(R) Arc(TM) A770 Graphics"

        with patch("openvino.Core", return_value=mock_core):
            with patch("arc_runner.engine.npu_detector.NpuDetector.get_npu_info", return_value=None):
                profiler = SiliconProfiler()
                profile = profiler.detect_profile()

                assert profile.tier == SiliconTier.WORKSTATION_DGPU
                assert profile.theoretical_bandwidth_gb_s == 512.0
                assert profile.recommended_gamma == 5
                assert profile.recommended_kv_cache_precision == "f16"

    def test_should_classify_server_datacenter_ponte_vecchio_p_sil_3(self) -> None:
        """P-SIL-3: Verify Intel Data Center GPU Max 1550 (Ponte Vecchio) is classified as SERVER_DATACENTER."""
        mock_core = MagicMock()
        mock_core.available_devices = ["GPU", "CPU"]
        mock_core.get_property.return_value = "Intel(R) Data Center GPU Max 1550"

        with patch("openvino.Core", return_value=mock_core):
            with patch("arc_runner.engine.npu_detector.NpuDetector.get_npu_info", return_value=None):
                profiler = SiliconProfiler()
                profile = profiler.detect_profile()

                assert profile.tier == SiliconTier.SERVER_DATACENTER
                assert profile.theoretical_bandwidth_gb_s == 3276.8
                assert profile.recommended_gamma == 6

    def test_should_fallback_to_generic_cpu_when_no_gpu_n_sil_1(self) -> None:
        """N-SIL-1: Verify absence of GPU accelerator silicon falls back to GENERIC_CPU tier."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]

        with patch("openvino.Core", return_value=mock_core):
            with patch("arc_runner.engine.npu_detector.NpuDetector.get_npu_info", return_value=None):
                profiler = SiliconProfiler()
                profile = profiler.detect_profile()

                assert profile.tier == SiliconTier.GENERIC_CPU
                assert profile.theoretical_bandwidth_gb_s == 50.0

    def test_should_handle_runtime_exceptions_safely_e_sil_2(self) -> None:
        """E-SIL-2: Verify unexpected exceptions during OpenVINO querying fall back to GENERIC_CPU safely."""
        with patch("openvino.Core", side_effect=RuntimeError("Level Zero runtime missing")):
            profiler = SiliconProfiler()
            profile = profiler.detect_profile()

            assert profile.tier == SiliconTier.GENERIC_CPU
            assert profile.gpu_name == "Unknown"
