"""Live empirical silicon verification of Heterogeneous NPU-GPU-CPU Fusion and Profiler.

Tests SiliconProfiler, NpuDetector, and NpuModelAdapter on physical Intel Core Ultra 5 125H
Meteor Lake hardware.
"""

import json
from pathlib import Path

from arc_runner.engine.npu_adapter import NpuModelAdapter
from arc_runner.engine.npu_detector import NpuDetector
from arc_runner.hardware.silicon_profiles import SiliconProfiler
from arc_runner.quant.model_transpiler import HardwareCompatibilityTranspiler


def main() -> None:
    """Run live hardware verification on physical Meteor Lake silicon."""
    print("=== LIVE EMPIRICAL HETEROGENEOUS FUSION VERIFICATION ===")

    profiler = SiliconProfiler()
    profile = profiler.detect_profile()
    print(f"[1] Silicon Tier: {profile.tier.value}")
    print(f"    GPU Accelerator: {profile.gpu_name}")
    print(f"    NPU Hardware Active: {profile.npu_present} (Arch: {profile.npu_architecture})")
    print(f"    Theoretical Bus Bandwidth: {profile.theoretical_bandwidth_gb_s} GB/s")
    print(f"    Recommended Lookahead Gamma: {profile.recommended_gamma}")
    print(f"    Recommended KV Precision: {profile.recommended_kv_cache_precision}")
    print(f"    Heterogeneous Fusion Supported: {profile.supports_heterogeneous_fusion}")

    detector = NpuDetector()
    npu_info = detector.get_npu_info()
    if npu_info:
        print(f"[2] NPU Telemetry: {npu_info.full_name}")
        print(f"    Driver Version: {npu_info.driver_version}")
        print(f"    Available Tiles/Devices: {npu_info.available_devices}")

    hf_model_path = Path(
        r"C:\Users\techd\.cache\huggingface\hub\models--hsuwill000--Qwen2.5-0.5B-Instruct-openvino-4bit\snapshots\50b455248a3c33f66052b2169031bf216edf4d18"
    )
    if hf_model_path.exists():
        adapter = NpuModelAdapter(target_architecture=profile.npu_architecture or "3720")
        report = adapter.inspect_model_compatibility(hf_model_path)
        print("[3] Live Model Compatibility Pre-Flight:")
        print(f"    Model: {hf_model_path.name}")
        print(f"    Compatible on NPU 3720: {report.is_compatible}")
        if not report.is_compatible:
            print(f"    Rejection Reason: {report.rejection_reason}")
            print(f"    Incompatible Op Count: {len(report.incompatible_layer_names)}")
            print(f"    Recommended Transpilation Target: {report.recommended_precision}")

    transpiler = HardwareCompatibilityTranspiler()
    npu_strat = transpiler.select_strategy("NPU")
    gpu_strat = transpiler.select_strategy("GPU")
    print("[4] Model Transpiler Strategies:")
    print(f"    NPU Strategy: {npu_strat.name} (Precision: {npu_strat.target_precision})")
    print(f"    GPU Strategy: {gpu_strat.name} (Precision: {gpu_strat.target_precision})")

    results = {
        "tier": profile.tier.value,
        "gpu_name": profile.gpu_name,
        "npu_present": profile.npu_present,
        "npu_architecture": profile.npu_architecture,
        "bandwidth_gb_s": profile.theoretical_bandwidth_gb_s,
        "recommended_gamma": profile.recommended_gamma,
        "fused_supported": profile.supports_heterogeneous_fusion,
        "npu_report_compatible": report.is_compatible if hf_model_path.exists() else None,
    }
    out_path = Path("scratch/live_heterogeneous_verification.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved live verification report to {out_path}")


if __name__ == "__main__":
    main()
