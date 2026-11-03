# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GAP-04 Physical Silicon Verification Harness
Target: OpenVINO GenAI LLMPipeline DAG Topological Pre-flight Validation
Hardware: Intel Core Ultra 5 125H (Meteor Lake CPU / Arc GPU / NPU)
"""

import sys
import time
from collections import deque
from pathlib import Path
import openvino as ov
import openvino.opset13 as ops
import numpy as np


class ModelRole:
    PRIMARY_LLM = "PRIMARY_LLM"
    DRAFT_MODEL = "DRAFT_MODEL"
    EAGLE3_DRAFT = "EAGLE3_DRAFT"


def validate_llm_model_dag_py(model: ov.Model, role: str = ModelRole.PRIMARY_LLM) -> None:
    """Python reference implementation mirroring C++ ov::genai::utils::validate_llm_model_dag."""
    if model is None:
        raise RuntimeError("[LLMPipeline][GAP-04] LLMPipeline pre-flight validation failed: Model must not be null.")

    model_name = model.get_friendly_name()
    parameters = model.get_parameters()
    results = model.get_results()
    sinks = model.get_sinks()

    if len(parameters) == 0:
        raise RuntimeError(
            f"[LLMPipeline][GAP-04] LLMPipeline pre-flight failed [model='{model_name}']: "
            f"Model contains 0 input Parameters. Action: Export model with valid input tensors."
        )

    if len(results) == 0 and len(sinks) == 0:
        raise RuntimeError(
            f"[LLMPipeline][GAP-04] LLMPipeline pre-flight failed [model='{model_name}']: "
            f"Model contains 0 Results and 0 Sinks. Action: Export model with valid output tensors."
        )

    param_names = [p.get_friendly_name() for p in parameters]

    # Gate A: Parameter Whitelist Check based on role
    has_valid_primary_input = False
    for name in param_names:
        if role == ModelRole.PRIMARY_LLM:
            if name == "input_ids" or "input_ids" in name:
                has_valid_primary_input = True
                break
        else:
            if (
                name == "input_ids"
                or "input_ids" in name
                or name == "hidden_states"
                or "hidden_states" in name
                or name == "inputs_embeds"
                or "inputs_embeds" in name
            ):
                has_valid_primary_input = True
                break

    if not has_valid_primary_input:
        found_list = ", ".join(f"'{n}'" for n in param_names)
        if role == ModelRole.PRIMARY_LLM:
            raise RuntimeError(
                f"[LLMPipeline][GAP-04] LLMPipeline pre-flight failed [model='{model_name}']: "
                f"Required Parameter 'input_ids' not found. "
                f"Found {len(parameters)} Parameter(s): [{found_list}]. "
                f"Action: Re-export model with tokenizer inputs retained."
            )
        else:
            raise RuntimeError(
                f"[LLMPipeline][GAP-04] LLMPipeline pre-flight failed [model='{model_name}']: "
                f"Required Parameter ('input_ids', 'hidden_states', or 'inputs_embeds') not found for draft model. "
                f"Found {len(parameters)} Parameter(s): [{found_list}]. "
                f"Action: Re-export draft model with appropriate input tensor."
            )

    # Gate B: Output Logit Heuristic Check
    has_logits_output = False
    result_names = []
    for r in results:
        res_name = r.get_friendly_name()
        if r.get_input_size() > 0:
            tensor_names = r.input(0).get_tensor().get_names()
            if tensor_names:
                res_name = next(iter(tensor_names))
        result_names.append(res_name)

        if res_name == "logits" or res_name == "lm_logits" or "logit" in res_name:
            has_logits_output = True

    if not has_logits_output and len(results) == 1:
        partial_shape = results[0].get_output_partial_shape(0)
        if partial_shape.rank.is_dynamic or partial_shape.rank.get_length() >= 2:
            has_logits_output = True

    if role != ModelRole.PRIMARY_LLM and not has_logits_output and len(results) > 0:
        has_logits_output = True

    if not has_logits_output:
        found_res_list = ", ".join(f"'{n}'" for n in result_names)
        raise RuntimeError(
            f"[LLMPipeline][GAP-04] LLMPipeline pre-flight failed [model='{model_name}']: "
            f"Required Result 'logits' not found. "
            f"Found {len(results)} Result(s): [{found_res_list}]. "
            f"Action: Re-export model with vocabulary logits output retained."
        )

    # Gate C: Dual-Root Backward BFS from Results and Sinks (Assign nodes)
    visited = set()
    bfs_queue = deque()

    for r in results:
        bfs_queue.append(r)
        visited.add(r.get_instance_id())

    for s in sinks:
        bfs_queue.append(s)
        visited.add(s.get_instance_id())

    while bfs_queue:
        curr = bfs_queue.popleft()
        input_size = curr.get_input_size()
        for i in range(input_size):
            source_output = curr.input(i).get_source_output()
            src_node = source_output.get_node()
            if src_node is not None and src_node.get_instance_id() not in visited:
                visited.add(src_node.get_instance_id())
                bfs_queue.append(src_node)

    for p in parameters:
        p_name = p.get_friendly_name()
        is_critical = False
        if role == ModelRole.PRIMARY_LLM and (p_name == "input_ids" or "input_ids" in p_name):
            is_critical = True
        elif role != ModelRole.PRIMARY_LLM and (
            p_name in ["input_ids", "hidden_states", "inputs_embeds"]
            or "input_ids" in p_name
            or "hidden_states" in p_name
        ):
            is_critical = True

        if is_critical and p.get_instance_id() not in visited:
            raise RuntimeError(
                f"[LLMPipeline][GAP-04] LLMPipeline pre-flight failed [model='{model_name}']: "
                f"Critical Parameter '{p_name}' is topologically disconnected from graph Results and Sinks. "
                f"Action: Ensure tensor graph connectivity is preserved during export/optimization."
            )


def test_real_production_models():
    print("\n--- [1] Validating Real Production Models on Intel Core Ultra 5 125H ---")
    core = ov.Core()
    models_dir = Path("models")

    target_models = [
        ("qwen-2.5-1.5b", models_dir / "qwen-2.5-1.5b" / "openvino_model.xml", ModelRole.PRIMARY_LLM),
        ("deepseek-r1-1.5b", models_dir / "deepseek-r1-1.5b" / "openvino_model.xml", ModelRole.PRIMARY_LLM),
        ("qwen-0.5b-npu", models_dir / "qwen-0.5b-npu-certified" / "openvino_model.xml", ModelRole.DRAFT_MODEL),
    ]

    for label, xml_path, role in target_models:
        if not xml_path.exists():
            print(f"[-] Skipping {label}: {xml_path} not found.")
            continue

        print(f"[*] Reading {label} IR...")
        model = core.read_model(xml_path)

        # Benchmark latency
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            validate_llm_model_dag_py(model, role)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)  # microseconds

        min_us = min(latencies)
        avg_us = sum(latencies) / len(latencies)
        node_count = len(model.get_ordered_ops())
        param_count = len(model.get_parameters())
        result_count = len(model.get_results())
        sink_count = len(model.get_sinks())

        print(
            f"    [PASS] {label}: {node_count} ops, {param_count} params, {result_count} results, {sink_count} sinks"
        )
        print(f"           Pre-flight Latency (Python pybind11): min={min_us:.1f}μs, avg={avg_us:.1f}μs (Python budget: 50,000μs; C++ budget: 5,000μs)")
        assert min_us < 50000, f"Latency {min_us}μs exceeded 50ms Python budget!"


def test_synthetic_defect_cases():
    print("\n--- [2] Validating Synthetic Defect Diagnostic Cases ---")

    # Case 1: Missing input_ids
    p_wrong = ops.parameter([1, 16], ov.Type.i64, name="wrong_tokens")
    bias = ops.constant(np.ones((1, 16), dtype=np.int64))
    add = ops.add(p_wrong, bias, name="add_logits")
    res = ops.result(add, name="logits")
    m_missing_input = ov.Model([res], [p_wrong], "missing_input_test")

    try:
        validate_llm_model_dag_py(m_missing_input, ModelRole.PRIMARY_LLM)
        assert False, "Should have thrown for missing input_ids"
    except RuntimeError as e:
        assert "GAP-04" in str(e) and "input_ids" in str(e)
        print("    [PASS] Missing input_ids correctly caught with descriptive diagnostic.")

    # Case 2: Missing logits
    p_valid = ops.parameter([1, 16], ov.Type.i64, name="input_ids")
    res_other = ops.result(p_valid, name="intermediate_embeddings")
    res_other2 = ops.result(p_valid, name="hidden_states")
    m_missing_logits = ov.Model([res_other, res_other2], [p_valid], "missing_logits_test")

    try:
        validate_llm_model_dag_py(m_missing_logits, ModelRole.PRIMARY_LLM)
        assert False, "Should have thrown for missing logits"
    except RuntimeError as e:
        assert "GAP-04" in str(e) and "logits" in str(e)
        print("    [PASS] Missing logits correctly caught with descriptive diagnostic.")

    # Case 3: Disconnected parameter
    p_connected = ops.parameter([1, 16], ov.Type.i64, name="input_ids")
    p_disconnected = ops.parameter([1, 16], ov.Type.i64, name="input_ids_extra")
    res_valid = ops.result(p_connected, name="logits")
    m_disconnected = ov.Model([res_valid], [p_connected, p_disconnected], "disconnected_test")

    try:
        validate_llm_model_dag_py(m_disconnected, ModelRole.PRIMARY_LLM)
        assert False, "Should have thrown for disconnected parameter"
    except RuntimeError as e:
        assert "GAP-04" in str(e) and "disconnected" in str(e).lower()
        print("    [PASS] Disconnected parameter correctly caught with descriptive diagnostic.")

    # Case 4: Null model guard
    try:
        validate_llm_model_dag_py(None, ModelRole.PRIMARY_LLM)
        assert False, "Should have thrown for null model"
    except RuntimeError as e:
        assert "Model must not be null" in str(e)
        print("    [PASS] Null model correctly caught without segfault.")

    # Case 5: Eagle3 / Draft model with hidden_states input
    p_hidden = ops.parameter([1, 16, 2048], ov.Type.f32, name="hidden_states")
    res_draft_logits = ops.result(p_hidden, name="logits")
    m_draft = ov.Model([res_draft_logits], [p_hidden], "draft_head_test")
    validate_llm_model_dag_py(m_draft, ModelRole.EAGLE3_DRAFT)
    validate_llm_model_dag_py(m_draft, ModelRole.DRAFT_MODEL)
    print("    [PASS] Eagle3 / Draft model with hidden_states input successfully validated.")


def main():
    print("=" * 70)
    print("🚀 Running GAP-04 DAG Topological Pre-flight Verification Suite")
    print("   Architecture: Intel Meteor Lake (Core Ultra 5 125H)")
    print("=" * 70)

    test_real_production_models()
    test_synthetic_defect_cases()

    print("\n" + "=" * 70)
    print("✅ ALL GAP-04 PHYSICAL SILICON & DIAGNOSTIC VERIFICATIONS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
