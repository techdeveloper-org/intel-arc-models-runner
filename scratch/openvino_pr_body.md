## Description
Malformed or corrupt OpenVINO IR models (e.g. models missing `input_ids`, missing vocabulary `logits`, or with topologically disconnected parameters from aggressive pruning/quantization tools) currently bypass `ov::Core::read_model` and fail deep inside pipeline instantiation with cryptic internal runtime errors such as `std::out_of_range: invalid vector subscript` or null pointer dereferences.

This PR introduces topological pre-flight validation `ov::genai::utils::validate_llm_model_dag` in `src/cpp/src/utils.hpp` and `src/cpp/src/utils.cpp`, called during `StatefulLLMPipeline` construction (in both stateful and static pipeline paths) before KV-cache initialization and plugin compilation:

1. **Gate 0 (Null & Zero-Element Guard)**:
   Asserts model pointer is non-null and contains at least one parameter and at least one result or sink.
2. **Gate A (Role-Aware Input Validation)**:
   Verifies primary LLMs possess `input_ids`. For speculative draft models (`ModelRole::DRAFT_MODEL` / `EAGLE3_DRAFT`), whitelists `input_ids`, `hidden_states`, or `inputs_embeds`.
3. **Gate B (Heuristic Logits Output Verification)**:
   Inspects output friendly names and tensor names for `logits`, `lm_logits`, rank $\ge 2$ single-result fallback, and draft heads.
4. **Gate C (Dual-Root Backward BFS Traversal)**:
   Traverses backward from both `ov::Result` nodes and `ov::Sink` nodes (`ov::op::v6::Assign` in stateful KV-cache loops) to confirm that critical parameters are topologically connected to outputs, preventing disconnected-subgraph crashes.

When validation fails, an informative `ov::Exception` is raised detailing the model name, exact missing tensor, list of existing tensors, and actionable advice to re-export or fix the graph.

Fixes https://github.com/techdeveloper-org/intel-arc-models-runner/issues/63

## Checklist:
- [x] This PR follows [GenAI Contributing guidelines](https://github.com/openvinotoolkit/openvino.genai?tab=contributing-ov-file#contributing).
- [x] Tests have been updated or added to cover the new code: Added 16 GTest test cases in `tests/cpp/test_llm_pipeline_preflight.cpp` covering null models, empty models, missing inputs, missing logits, disconnected parameters, draft models, stateful sink loops, thread safety, and latency budget (<5ms).
- [x] This PR fully addresses the ticket.
- [x] I have made corresponding changes to the documentation (N/A - internal pre-flight validation without public API contract changes).

## AI Usage Disclosure:
AI assistance used: yes
If yes: Assisted in drafting initial boilerplate and exhaustive test permutations for GTest.
Human validation performed: Complete manual architectural review, MSVC 2026 build verification, 16/16 GTest suite passing, regression suite passing, and physical silicon validation on Intel Meteor Lake Core Ultra 5 125H (Arc GPU & NPU) with Optimum-Intel models (Qwen2.5-1.5B, DeepSeek-R1-1.5B, Qwen-0.5B NPU).
