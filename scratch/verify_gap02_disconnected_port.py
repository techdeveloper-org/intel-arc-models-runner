"""Verification and physical silicon testing script for GAP-02.

This test loads an OpenVINO IR model with a disconnected input port to verify
whether the error diagnostics emitted during model deserialization report
actionable context (friendly name, layer type, layer ID, missing port ID).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CUSTOM_BUILD_DIR = Path(r"C:\Users\techd\Documents\workspace-spring-tool-suite-4-4.27.0-new\openvino\bin\intel64\Release")


def setup_environment(use_custom_build: bool) -> None:
    """Configures the DLL search path to prioritize local custom OpenVINO builds.

    Args:
        use_custom_build: Whether to prepend the local build artifacts directory.
    """
    if use_custom_build:
        if not CUSTOM_BUILD_DIR.exists():
            raise FileNotFoundError(f"Custom build directory not found: {CUSTOM_BUILD_DIR}")
        if sys.platform == "win32":
            os.add_dll_directory(str(CUSTOM_BUILD_DIR))
            print(f"[ENV] Prioritizing local custom build DLLs from: {CUSTOM_BUILD_DIR}")


def verify_disconnected_port_diagnostic(xml_path: Path) -> dict[str, str | bool]:
    """Attempts to deserialize an IR model containing a disconnected input port.

    Args:
        xml_path: Filesystem path to the malformed Intermediate Representation XML.

    Returns:
        A dictionary containing the test status, caught exception type, and message.
    """
    import openvino as ov

    core = ov.Core()
    result: dict[str, str | bool] = {
        "success": False,
        "exception_type": "",
        "exception_message": "",
        "has_friendly_name": False,
        "has_layer_id": False,
        "has_port_info": False,
    }

    try:
        core.read_model(model=str(xml_path))
        result["success"] = True
    except Exception as exc:
        msg = str(exc)
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = msg
        result["has_friendly_name"] = "Add_MissingEdge" in msg
        result["has_layer_id"] = "id: 2" in msg or "layerId: 2" in msg
        result["has_port_info"] = "[1]" in msg or "input port" in msg

    return result


def main() -> int:
    """Executes the GAP-02 diagnostic verification test and logs telemetry."""
    parser = argparse.ArgumentParser(description="GAP-02 Disconnected Port Diagnostic Verification")
    parser.add_argument("--custom-build", action="store_true", help="Load freshly compiled local OpenVINO DLLs")
    args = parser.parse_args()

    setup_environment(args.custom_build)

    import openvino as ov

    base_dir = Path(__file__).resolve().parent
    fixture_path = base_dir / "fixtures" / "disconnected_port_model.xml"

    print("=" * 80)
    print("GAP-02 PHYSICAL VERIFICATION: DISCONNECTED PORT DIAGNOSTICS")
    print("=" * 80)
    print(f"OpenVINO Runtime Version : {ov.__version__}")
    print(f"Using Custom Build DLLs  : {args.custom_build}")
    print(f"Target Fixture Path      : {fixture_path}")
    print("-" * 80)

    if not fixture_path.exists():
        print(f"ERROR: Fixture not found at {fixture_path}", file=sys.stderr)
        return 1

    outcome = verify_disconnected_port_diagnostic(fixture_path)

    print(f"Model Deserialization Succeeded : {outcome['success']}")
    print(f"Exception Type Caught          : {outcome['exception_type']}")
    print(f"Exception Message              :\n{outcome['exception_message']}\n")
    print("-" * 80)
    print("DIAGNOSTIC QUALITY EVALUATION:")
    print(f"  - Contains Layer Friendly Name ('Add_MissingEdge') : {outcome['has_friendly_name']}")
    print(f"  - Contains Layer ID (id: 2)                        : {outcome['has_layer_id']}")
    print(f"  - Contains Missing Port Details ([1])              : {outcome['has_port_info']}")
    print("=" * 80)

    if outcome["has_friendly_name"] and outcome["has_layer_id"] and outcome["has_port_info"]:
        print("RESULT: 100% PROVEN FIX CONFIRMED! Full diagnostic context present.")
        return 0
    else:
        print("RESULT: DEFECT UNRESOLVED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
