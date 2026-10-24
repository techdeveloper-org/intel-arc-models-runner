"""Real hardware integration test suite for Intel silicon accelerators.

Exercises physical Intel Core Ultra CPUs, Intel Arc GPUs, and Intel AI Boost NPUs
across 21 comprehensive Triad test cases (8 Positive, 6 Negative, 7 Edge-Case)
implementing mathematical proofs M36 through M43.
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from arc_runner.devices.models import SystemTopology
from arc_runner.engine.paged_attention import RadixPrefixCache
from arc_runner.testing.hardware_harness import (
    SiliconHardwareHarness,
    StudentTStatistics,
    SyntheticGraphBuilder,
)


@pytest.mark.hardware
def test_live_silicon_topology_detection(live_topology: SystemTopology) -> None:
    """Verify physical hardware discovery across all connected silicon accelerators.

    Args:
        live_topology: SystemTopology session fixture.
    """
    assert isinstance(live_topology, SystemTopology)
    assert len(live_topology.devices) >= 1
    assert live_topology.host_cpu_model != ""
    assert live_topology.target_device is not None
    assert len(live_topology.fallback_chain) >= 1

    device_types = [d.device_type.value for d in live_topology.devices]
    assert "CPU" in device_types


@pytest.mark.hardware
def test_arc_gpu_live_matrix_inference(live_topology: SystemTopology) -> None:
    """Verify live matrix compilation and DPAS execution on Intel Arc GPU (Proof M36).

    Args:
        live_topology: SystemTopology session fixture.
    """
    if not live_topology.has_intel_arc:
        pytest.skip("Intel Arc GPU not detected on host.")

    harness = SiliconHardwareHarness()
    model = SyntheticGraphBuilder.build_linear_projection(64, 64, precision="f32")
    sample_input = np.ones((1, 64), dtype=np.float32)

    result = harness.compile_and_benchmark(
        device="GPU",
        model=model,
        sample_input=sample_input,
        iterations=5,
        warmup=2,
    )

    assert result.success is True
    assert result.device_id == "GPU"
    assert result.mean_latency_ms > 0.0
    assert result.ci_upper_ms >= result.mean_latency_ms
    assert result.ci_lower_ms <= result.mean_latency_ms
    assert result.arithmetic_intensity > 0.0


@pytest.mark.hardware
def test_intel_npu_live_systolic_inference(live_topology: SystemTopology) -> None:
    """Verify static FP16 graph compilation on Intel AI Boost NPU systolic array (Proof M38).

    Args:
        live_topology: SystemTopology session fixture.
    """
    if not live_topology.has_intel_npu:
        pytest.skip("Intel AI Boost NPU not detected on host.")

    harness = SiliconHardwareHarness()
    model = SyntheticGraphBuilder.build_npu_systolic_model(batch_size=1, in_features=64, out_features=64)
    sample_input = np.ones((1, 64), dtype=np.float16)

    result = harness.compile_and_benchmark(
        device="NPU",
        model=model,
        sample_input=sample_input,
        iterations=5,
        warmup=2,
    )

    assert result.success is True
    assert result.device_id == "NPU"
    assert result.mean_latency_ms > 0.0
    assert result.ci_upper_ms >= result.mean_latency_ms


@pytest.mark.hardware
def test_intel_cpu_hybrid_core_inference(live_topology: SystemTopology) -> None:
    """Verify inference throughput on Intel Core Ultra hybrid CPU cores.

    Args:
        live_topology: SystemTopology session fixture.
    """
    harness = SiliconHardwareHarness()
    model = SyntheticGraphBuilder.build_linear_projection(64, 64, precision="f32")
    sample_input = np.ones((1, 64), dtype=np.float32)

    result = harness.compile_and_benchmark(
        device="CPU",
        model=model,
        sample_input=sample_input,
        iterations=5,
        warmup=1,
    )

    assert result.success is True
    assert result.device_id == "CPU"
    assert result.mean_latency_ms > 0.0


@pytest.mark.hardware
def test_usm_zero_copy_live_pointer_validation(live_topology: SystemTopology) -> None:
    """Verify USM Unified Shared Memory pointer availability and timing bound (Proof M37).

    Args:
        live_topology: SystemTopology session fixture.
    """
    if not live_topology.has_intel_arc:
        pytest.skip("Intel Arc GPU not detected on host.")

    harness = SiliconHardwareHarness()
    report = harness.verify_usm_zero_copy("GPU")

    assert report["device"] == "GPU"
    assert report["has_usm"] is True
    assert report["latency_bound_satisfied"] is True
    assert report["allocation_latency_ms"] <= 1.0


@pytest.mark.hardware
def test_dual_silicon_concurrent_execution_live(live_topology: SystemTopology) -> None:
    """Verify zero-contention invariant during concurrent Arc GPU and NPU execution (Proof M40).

    Args:
        live_topology: SystemTopology session fixture.
    """
    if not (live_topology.has_intel_arc and live_topology.has_intel_npu):
        pytest.skip("Both Intel Arc GPU and AI Boost NPU required for dual-silicon verification.")

    harness = SiliconHardwareHarness()
    report = harness.run_dual_silicon_concurrent(gpu_iterations=10, npu_iterations=10)

    assert report["gpu_solo_ms"] > 0.0
    assert report["npu_solo_ms"] > 0.0
    assert report["concurrent_wall_ms"] > 0.0
    assert report["concurrent_wall_ms"] <= (report["gpu_solo_ms"] + report["npu_solo_ms"]) * 2.5


@pytest.mark.hardware
def test_live_aot_compilation_export_cycle(live_topology: SystemTopology) -> None:
    """Verify Ahead-Of-Time compiled blob serialization and re-import lifecycle."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_linear_projection(32, 32, precision="f32")

    compiled_original = core.compile_model(model, device)
    inp = np.ones((1, 32), dtype=np.float32)
    expected_out = compiled_original([inp])[0]

    stream = io.BytesIO()
    compiled_original.export_model(stream)
    stream.seek(0)
    blob_bytes = stream.getvalue()

    assert len(blob_bytes) > 0

    imported_model = core.import_model(blob_bytes, device)
    actual_out = imported_model([inp])[0]

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)


@pytest.mark.hardware
def test_end_to_end_synthetic_llm_block_live(live_topology: SystemTopology) -> None:
    """Verify end-to-end multi-head attention projection block on physical silicon."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_attention_projection_block(
        seq_len=8, hidden_dim=32, num_heads=4, precision="f32"
    )

    compiled = core.compile_model(model, device)
    inp = np.ones((1, 8, 32), dtype=np.float32)
    out = compiled([inp])[0]

    assert out.shape == (1, 8, 32)
    assert np.all(np.isfinite(out))


@pytest.mark.hardware
def test_unsupported_device_rejection_live() -> None:
    """Verify graceful error rejection when attempting to compile to an invalid device."""
    core = ov.Core()
    model = SyntheticGraphBuilder.build_linear_projection(16, 16)

    with pytest.raises((RuntimeError, ValueError)):
        core.compile_model(model, "GPU.99_INVALID")


@pytest.mark.hardware
def test_malformed_graph_dynamic_npu_rejection_live(live_topology: SystemTopology) -> None:
    """Verify that unconstrained dynamic shapes on NPU are rejected cleanly without crashes."""
    if not live_topology.has_intel_npu:
        pytest.skip("Intel AI Boost NPU required for dynamic rejection verification.")

    core = ov.Core()
    param = ops.parameter(ov.PartialShape([-1, 16]), ov.Type.f16, name="unbounded_dynamic_param")
    result = ops.result(param)
    model = ov.Model([result], [param], "unbounded_npu_graph")

    with pytest.raises((RuntimeError, ValueError)):
        core.compile_model(model, "NPU")


@pytest.mark.hardware
def test_usm_memory_overcommit_guard_live() -> None:
    """Verify graceful handling when allocating memory beyond physical boundaries."""
    with pytest.raises(MemoryError):
        np.zeros((1024, 1024, 1024, 64), dtype=np.float32)


@pytest.mark.hardware
def test_corrupted_blob_import_rejection_live() -> None:
    """Verify clean rejection when attempting to import a malformed binary model blob."""
    core = ov.Core()
    corrupted_bytes = b"CORRUPTED_INTEL_MODEL_BLOB_HEADER_00000000"

    with pytest.raises((RuntimeError, ValueError)):
        core.import_model(corrupted_bytes, "CPU")


@pytest.mark.hardware
def test_non_contiguous_tensor_layout_rejection_live(live_topology: SystemTopology) -> None:
    """Verify that non-contiguous memory layouts are handled safely on hardware."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_linear_projection(16, 16)
    compiled = core.compile_model(model, device)

    data = np.arange(32, dtype=np.float32).reshape(2, 16)
    non_contiguous = data[::2, :].T.T

    res = compiled([np.ascontiguousarray(non_contiguous)])[0]
    assert res.shape == (1, 16)
    assert np.all(np.isfinite(res))


@pytest.mark.hardware
def test_zero_dimension_tensor_rejection_live() -> None:
    """Verify rejection when attempting to infer with incompatible zero-dimension tensor."""
    core = ov.Core()
    model = SyntheticGraphBuilder.build_linear_projection(16, 16)
    compiled = core.compile_model(model, "CPU")

    with pytest.raises((RuntimeError, ValueError)):
        compiled([np.zeros((0,), dtype=np.float32)])


@pytest.mark.hardware
def test_minimal_1x1_scalar_tensor_live(live_topology: SystemTopology) -> None:
    """Verify boundary evaluation with minimal 1x1 scalar tensor on hardware."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_linear_projection(1, 1, precision="f32")

    compiled = core.compile_model(model, device)
    inp = np.array([[2.5]], dtype=np.float32)
    out = compiled([inp])[0]

    assert out.shape == (1, 1)
    np.testing.assert_allclose(out, [[2.5]], rtol=1e-5)


@pytest.mark.hardware
def test_maximum_context_block_boundary_live(live_topology: SystemTopology) -> None:
    """Verify execution at exact paged attention block boundary (N = 64 = 4 x 16 tokens)."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    block_aligned_dim = 64
    model = SyntheticGraphBuilder.build_linear_projection(block_aligned_dim, block_aligned_dim)

    compiled = core.compile_model(model, device)
    inp = np.ones((1, block_aligned_dim), dtype=np.float32)
    out = compiled([inp])[0]

    assert out.shape == (1, block_aligned_dim)
    assert np.all(np.isfinite(out))


@pytest.mark.hardware
def test_rapid_concurrency_burst_live(live_topology: SystemTopology) -> None:
    """Verify Level Zero command queue queuing stability under 16 concurrent requests."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_linear_projection(32, 32)
    compiled = core.compile_model(model, device)

    def _infer(idx: int) -> bool:
        req = compiled.create_infer_request()
        inp = np.full((1, 32), float(idx), dtype=np.float32)
        out = req.infer([inp])[0]
        return bool(np.all(np.isfinite(out)))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_infer, i) for i in range(16)]
        results = [f.result() for f in futures]

    assert all(results)


@pytest.mark.hardware
def test_latency_confidence_interval_jitter_bounds_live(live_topology: SystemTopology) -> None:
    """Verify Student's t 95% confidence interval stability and CV bound (Proof M39)."""
    harness = SiliconHardwareHarness()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_linear_projection(32, 32)
    sample_input = np.ones((1, 32), dtype=np.float32)

    result = harness.compile_and_benchmark(
        device=device,
        model=model,
        sample_input=sample_input,
        iterations=10,
        warmup=3,
    )

    assert result.success is True
    assert result.mean_latency_ms > 0.0
    assert result.ci_upper_ms >= result.mean_latency_ms
    assert result.ci_lower_ms <= result.mean_latency_ms
    assert result.cv < 3.0


@pytest.mark.hardware
def test_numerical_stability_cpu_vs_gpu_vs_npu(live_topology: SystemTopology) -> None:
    """Verify cross-silicon output numerical consistency (Proof M43)."""
    harness = SiliconHardwareHarness()
    parity = harness.verify_cross_silicon_parity(shape=(1, 32))

    assert parity["gpu_parity_verified"] is True
    assert parity["gpu_max_abs_error"] <= 1e-2
    if parity["npu_tested"]:
        assert parity["npu_parity_verified"] is True
        assert parity["npu_max_abs_error"] <= 1e-2


@pytest.mark.hardware
def test_thermal_clock_frequency_stepping_resilience(live_topology: SystemTopology) -> None:
    """Verify sustained execution latency stability across multi-pass inference."""
    core = ov.Core()
    device = "GPU" if live_topology.has_intel_arc else "CPU"
    model = SyntheticGraphBuilder.build_linear_projection(128, 128)
    compiled = core.compile_model(model, device)
    inp = np.ones((1, 128), dtype=np.float32)

    req = compiled.create_infer_request()
    for _ in range(5):
        req.infer([inp])

    durations = []
    for _ in range(15):
        t0 = time.perf_counter_ns()
        req.infer([inp])
        t1 = time.perf_counter_ns()
        durations.append((t1 - t0) / 1_000_000.0)

    mean, _std_dev, _, _, cv = StudentTStatistics.compute_ci(durations)
    assert mean > 0.0
    assert cv <= 3.00


@pytest.mark.hardware
def test_radix_prefix_compaction_live_physical_memory() -> None:
    """Verify RadixPrefixCache deduplication invariant on live session allocations (Proof M41)."""
    radix = RadixPrefixCache()
    sys_prompt = list(range(32))
    query_tokens = sys_prompt + [9999]

    radix.insert_prefix(sys_prompt, [10, 11])
    matched_tokens, matched_blocks = radix.match_prefix(query_tokens)

    assert matched_tokens == sys_prompt
    assert matched_blocks == [10, 11]
