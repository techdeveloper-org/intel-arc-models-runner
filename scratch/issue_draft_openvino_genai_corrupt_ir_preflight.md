# [Feature Proposal] Graceful pre-flight diagnostics for malformed or corrupted IR models in LLMPipeline

## 1. Problem Description
When loading an invalid, corrupted, or non-standard OpenVINO IR model (for example, models exported with missing `input_ids`, missing output `logits`, or topological disconnections from experimental quantization/pruning pipelines) into `ov::genai::LLMPipeline`:
1. `ov::Core::read_model()` successfully parses the XML/BIN container without errors.
2. However, upon constructing `StatefulLLMPipeline` or `StaticLLMPipeline`, the pipeline attempts to inspect inputs and bind KV-cache states (`m_cache_state(model)` or `utils::compile_decoder_for_npu`).
3. If primary inputs or logits are missing or disconnected, the process aborts deep in execution with cryptic low-level C++ runtime exceptions:
   ```text
   std::out_of_range: invalid vector subscript
   ```
   or segmentation faults / null pointer dereferences without indicating which tensor or layer is malformed.

## 2. Motivation & Use Case
In production deployment environments and automated LLM benchmarking suites, developers and end-users often test custom quantized models (e.g. NPU-adapted, INT4-quantized via Optimum-Intel, or pruned variants). When an export fails or produces a topologically incomplete graph, users receive no diagnostic hints about what went wrong (e.g. missing `input_ids`, missing `logits`, or disconnected parameters).

Providing actionable, user-friendly error diagnostics before plugin compilation and KV-cache allocation significantly reduces debugging time and improves developer experience across CPU, GPU, and NPU targets.

## 3. Proposed Solution / Discussion Points
We would like to gather maintainer feedback on the preferred architectural location for validating LLM graph integrity:

### Option A: Lightweight Pre-flight Check in `ov::genai` (`LLMPipelineImplBase` / `StatefulLLMPipeline`)
Perform a quick, lightweight check prior to plugin compilation:
- **Input Guard**: Verify the model contains required primary inputs (`input_ids`, or `inputs_embeds` / `hidden_states` for speculative/draft heads).
- **Logits Guard**: Verify that at least one output corresponds to vocabulary logits.
- **Diagnostics**: If validation fails, throw an informative `ov::Exception` detailing model name, expected tensors, and actual tensors found.

### Option B: OpenVINO Core Runtime Validation (`ov::pass::Validate`)
Alternatively, if `openvino.genai` prefers to keep pipeline constructors free of graph inspection, should this diagnostic check be contributed upstream to `openvino` core (e.g., via a transformation pass or frontend validation)?

## 4. Willingness to Contribute
We have a working C++ implementation with comprehensive GTest test coverage (handling standard models, speculative draft models, and malformed graph permutations) ready to submit as a clean PR once the maintainers align on the preferred design and scope.
