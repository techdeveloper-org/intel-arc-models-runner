"""Intel NPU Model Adapter and Pre-Flight Hardware Compatibility Verifier.

Inspects OpenVINO computational graph IR XML definitions before compilation to identify
unsupported dequantization patterns (such as unsigned u4 on NPU37XX) and provides static
shape configuration hints required by the Intel VPUX compiler.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.npu_adapter")


class IncompatibleModelHardwareError(ArcRunnerError):
    """Raised when an OpenVINO model contains operations unsupported on target accelerator silicon."""

    def __init__(self, message: str, error_code: str = "INCOMPATIBLE_MODEL_HARDWARE", status_code: int = 422) -> None:
        """Initialize incompatible model hardware error.

        Args:
            message: Descriptive error message explaining the incompatible layer or precision.
            error_code: Unique error code token.
            status_code: HTTP status code.
        """
        super().__init__(message, error_code, status_code)


@dataclass
class NpuCompatibilityReport:
    """Pre-flight hardware verification assessment for model deployment on Intel NPU silicon."""

    is_compatible: bool
    model_path: Path
    detected_architecture: str
    incompatible_layer_names: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    recommended_precision: str = "INT4_SYM"
    suggested_compilation_properties: Dict[str, Any] = field(default_factory=dict)


class NpuModelAdapter:
    """Adapts and verifies OpenVINO models for compilation on Intel AI Boost NPUs."""

    def __init__(self, target_architecture: str = "3720") -> None:
        """Initialize NpuModelAdapter with target NPU microarchitecture family.

        Args:
            target_architecture: NPU silicon generation (e.g. '3720' for Meteor Lake, '4000' for Lunar Lake).
        """
        self.target_architecture = target_architecture

    def inspect_model_compatibility(self, model_dir_or_xml: Union[Path, str]) -> NpuCompatibilityReport:
        """Parse OpenVINO model XML and verify tensor operations against NPU constraints.

        Args:
            model_dir_or_xml: Path to model folder or direct path to openvino_model.xml.

        Returns:
            NpuCompatibilityReport detailing compatibility, defective operations, and compiler flags.
        """
        path = Path(model_dir_or_xml)
        xml_path = path if path.is_file() and path.suffix.lower() == ".xml" else path / "openvino_model.xml"

        if not path.exists():
            return NpuCompatibilityReport(
                is_compatible=False,
                model_path=path,
                detected_architecture=self.target_architecture,
                rejection_reason=f"Model directory not found at {path}",
            )

        if not xml_path.exists():
            return NpuCompatibilityReport(
                is_compatible=True,
                model_path=xml_path,
                detected_architecture=self.target_architecture,
                rejection_reason=None,
            )

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as exc:
            return NpuCompatibilityReport(
                is_compatible=False,
                model_path=xml_path,
                detected_architecture=self.target_architecture,
                rejection_reason=f"Malformed XML syntax: {exc}",
            )

        incompatible_layers: List[str] = []
        has_asymmetric_u4 = False

        for layer in root.iter("layer"):
            layer_type = layer.get("type", "")
            layer_name = layer.get("name", "")

            if layer_type in ("Convert", "Subtract", "Constant"):
                for port in layer.iter("port"):
                    elem_type = port.get("precision", "").lower()
                    if elem_type == "u4":
                        has_asymmetric_u4 = True
                        incompatible_layers.append(layer_name)

            data_elem = layer.find("data")
            if data_elem is not None:
                dst_type = data_elem.get("destination_type", "").lower()
                elem_type = data_elem.get("element_type", "").lower()
                if "u4" in dst_type or "u4" in elem_type:
                    has_asymmetric_u4 = True
                    incompatible_layers.append(layer_name)

        if has_asymmetric_u4 and self.target_architecture.startswith("37"):
            return NpuCompatibilityReport(
                is_compatible=False,
                model_path=xml_path,
                detected_architecture=self.target_architecture,
                incompatible_layer_names=incompatible_layers[:10],
                rejection_reason=(
                    f"Unsupported DequantizeOp configuration on arch NPU37XX: "
                    f"Found asymmetric unsigned 4-bit (u4) tensor operations across {len(incompatible_layers)} layers. "
                    f"Intel NPU3720 requires symmetric signed INT4 (i4 / INT4_SYM) or INT8."
                ),
                recommended_precision="INT4_SYM",
                suggested_compilation_properties={
                    "NPU_USE_NPUW": "YES",
                },
            )

        compilation_properties: Dict[str, Any] = {
            "PERFORMANCE_HINT": "LATENCY",
            "NPU_COMPILATION_MODE_PARAMS": "compute-layers-in-parallel=YES",
        }

        return NpuCompatibilityReport(
            is_compatible=True,
            model_path=xml_path,
            detected_architecture=self.target_architecture,
            incompatible_layer_names=[],
            rejection_reason=None,
            recommended_precision="INT4_SYM",
            suggested_compilation_properties=compilation_properties,
        )

    def get_static_reshape_bounds(self, max_prompt_len: int = 512, batch_size: int = 1) -> Dict[str, List[int]]:
        """Return static shape tensors required for Intel NPU pipeline compilation.

        Args:
            max_prompt_len: Maximum prompt token sequence boundary.
            batch_size: Fixed inference batch dimension.

        Returns:
            Dictionary of tensor input names to static shape dimensions.
        """
        return {
            "input_ids": [batch_size, max_prompt_len],
            "attention_mask": [batch_size, max_prompt_len],
            "position_ids": [batch_size, max_prompt_len],
        }
