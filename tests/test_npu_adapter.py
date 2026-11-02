"""Unit, negative, and edge-case triad tests for Intel NPU Model Adapter and Pre-Flight Verifier.

Tests NPU device enumeration, driver capability extraction, model XML dequantization inspection,
and unsupported unsigned 4-bit (u4) rejection on Meteor Lake NPU37XX silicon.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from arc_runner.engine.npu_adapter import IncompatibleModelHardwareError, NpuCompatibilityReport, NpuModelAdapter
from arc_runner.engine.npu_detector import NpuDetector, NpuInfo


class TestNpuDetector:
    """Test suite for NpuDetector discovery and device property extraction."""

    def test_should_detect_npu_when_hardware_present_p_npu_1(self) -> None:
        """P-NPU-1: Verify detect() returns 'NPU' and instantiates NpuInfo when OpenVINO reports NPU."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU", "NPU"]
        mock_core.get_property.side_effect = lambda dev, prop: {
            "FULL_DEVICE_NAME": "Intel(R) AI Boost",
            "AVAILABLE_DEVICES": ["3720"],
            "DEVICE_DRIVER_VERSION": "32.0.100.3104",
        }.get(prop, "")

        with patch("openvino.Core", return_value=mock_core):
            detector = NpuDetector()
            device = detector.detect()
            info = detector.get_npu_info()

            assert device == "NPU"
            assert info is not None
            assert info.is_available is True
            assert info.architecture == "3720"
            assert info.device_name == "NPU"

    def test_should_return_none_when_npu_disabled_by_env_n_npu_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """N-NPU-2: Verify detect() returns None when ARC_NPU_DISABLE environment flag is active."""
        monkeypatch.setenv("ARC_NPU_DISABLE", "1")
        detector = NpuDetector()
        assert detector.detect() is None
        assert detector.get_npu_info() is None

    def test_should_handle_openvino_runtime_exception_safely_n_npu_3(self) -> None:
        """N-NPU-3: Verify openvino exceptions during device enumeration return None without crashing."""
        with patch("openvino.Core", side_effect=RuntimeError("Level Zero driver initialization failed")):
            detector = NpuDetector()
            assert detector.detect() is None
            assert detector.get_npu_info() is None

    def test_should_handle_dual_tile_npu_enumeration_e_npu_3(self) -> None:
        """E-NPU-3: Verify multi-tile NPU reported as list resolves base microarchitecture correctly."""
        mock_core = MagicMock()
        mock_core.available_devices = ["NPU"]
        mock_core.get_property.side_effect = lambda dev, prop: {
            "FULL_DEVICE_NAME": "Intel(R) AI Boost",
            "AVAILABLE_DEVICES": ["3720.0", "3720.1"],
            "DEVICE_DRIVER_VERSION": "1.0",
        }.get(prop, "")

        with patch("openvino.Core", return_value=mock_core):
            detector = NpuDetector()
            info = detector.get_npu_info()
            assert info is not None
            assert info.architecture == "3720"


class TestNpuModelAdapter:
    """Test suite for NpuModelAdapter XML graph inspection and hardware pre-flight rules."""

    def test_should_approve_symmetric_int4_model_p_npu_2(self, tmp_path: Path) -> None:
        """P-NPU-2: Verify model with symmetric INT4 (i4) dequantization passes NPU pre-flight check."""
        xml_content = """<?xml version="1.0"?>
        <net name="test_model" version="11">
            <layers>
                <layer id="0" name="input_ids" type="Parameter">
                    <output><port id="0" precision="I64"><dim>1</dim><dim>128</dim></port></output>
                </layer>
                <layer id="1" name="weights_compressed" type="Constant">
                    <data element_type="i4" shape="896,128"/>
                    <output><port id="0" precision="I4"><dim>896</dim><dim>128</dim></port></output>
                </layer>
                <layer id="2" name="dequant_convert" type="Convert">
                    <data destination_type="f16"/>
                    <input><port id="0" precision="I4"><dim>896</dim><dim>128</dim></port></input>
                    <output><port id="1" precision="FP16"><dim>896</dim><dim>128</dim></port></output>
                </layer>
            </layers>
        </net>
        """
        xml_file = tmp_path / "openvino_model.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        adapter = NpuModelAdapter(target_architecture="3720")
        report = adapter.inspect_model_compatibility(tmp_path)

        assert report.is_compatible is True
        assert len(report.incompatible_layer_names) == 0
        assert report.rejection_reason is None

    def test_should_reject_asymmetric_u4_dequantize_on_npu37xx_n_npu_1(self, tmp_path: Path) -> None:
        """N-NPU-1: Verify model with asymmetric unsigned 4-bit (u4) ops is rejected for NPU 3720."""
        xml_content = """<?xml version="1.0"?>
        <net name="incompatible_qwen" version="11">
            <layers>
                <layer id="1" name="layer_0_matmul_weights" type="Constant">
                    <data element_type="u4" shape="896,128"/>
                    <output><port id="0" precision="U4"><dim>896</dim><dim>128</dim></port></output>
                </layer>
                <layer id="2" name="dequant_u4_to_f16" type="Convert">
                    <data destination_type="f16"/>
                    <input><port id="0" precision="U4"><dim>896</dim><dim>128</dim></port></input>
                    <output><port id="1" precision="FP16"><dim>896</dim><dim>128</dim></port></output>
                </layer>
            </layers>
        </net>
        """
        xml_file = tmp_path / "openvino_model.xml"
        xml_file.write_text(xml_content, encoding="utf-8")

        adapter = NpuModelAdapter(target_architecture="3720")
        report = adapter.inspect_model_compatibility(tmp_path)

        assert report.is_compatible is False
        assert "dequant_u4_to_f16" in report.incompatible_layer_names or "layer_0_matmul_weights" in report.incompatible_layer_names
        assert "Unsupported DequantizeOp configuration on arch NPU37XX" in str(report.rejection_reason)
        assert report.recommended_precision == "INT4_SYM"

    def test_should_return_incompatible_when_xml_file_missing_n_npu_missing(self, tmp_path: Path) -> None:
        """Verify non-existent model path returns incompatible report with clear rejection reason."""
        adapter = NpuModelAdapter()
        report = adapter.inspect_model_compatibility(tmp_path / "non_existent_folder")

        assert report.is_compatible is False
        assert "not found" in str(report.rejection_reason).lower()

    def test_should_handle_malformed_xml_syntax_e_npu_1(self, tmp_path: Path) -> None:
        """E-NPU-1: Verify malformed XML syntax returns incompatible report without unhandled parse crash."""
        bad_xml = tmp_path / "openvino_model.xml"
        bad_xml.write_text("<unclosed_tag>broken syntax", encoding="utf-8")

        adapter = NpuModelAdapter()
        report = adapter.inspect_model_compatibility(tmp_path)

        assert report.is_compatible is False
        assert "malformed xml" in str(report.rejection_reason).lower()

    def test_should_produce_static_shape_bounds_e_npu_2(self) -> None:
        """E-NPU-2: Verify static shape bounds dictionary conforms to NPU VPUX compiler requirements."""
        adapter = NpuModelAdapter(target_architecture="3720")
        bounds = adapter.get_static_reshape_bounds(max_prompt_len=512, batch_size=1)

        assert bounds["input_ids"] == [1, 512]
        assert bounds["attention_mask"] == [1, 512]
        assert bounds["position_ids"] == [1, 512]
