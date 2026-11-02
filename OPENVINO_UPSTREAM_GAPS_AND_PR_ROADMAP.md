# OpenVINO & OpenVINO GenAI Upstream Gaps and PR Roadmap

**Target Architectures**: Intel Arc Xe-LPG Graphics (iGPU) & Intel AI Boost NPU 3720 (`NPU37XX`) on Meteor Lake (Intel Core Ultra 5 125H)  
**OpenVINO Version**: 2026.4.1 / OpenVINO GenAI 2026.4.1.0  
**Status**: Empirically verified on physical silicon

---

## Executive Summary

During our physical hardware profiling and implementation of dual-silicon heterogeneous speculative co-execution (Intel Arc GPU Target + Intel AI Boost NPU Draft), we uncovered 5 critical upstream limitations in the OpenVINO runtime, VPUX compiler, Intel GPU plugin, and OpenVINO GenAI C++ speculative decoding engines.

This document details each gap, root cause, exact stack traces, our localized workarounds in `intel-arc-models-runner`, and actionable upstream PR proposals for Intel's `openvino` and `openvino.genai` repositories.

---

## Upstream Gap Matrix

| ID | Component | File / Pass | Symptom / Error | Upstream Status |
|:---|:---|:---|:---|:---|
| **GAP-1** | `vpux-compiler` (NPU) | `OneShotBufferizeVPU2VPUIP` | `Unsupported DequantizeOp configuration: !quant.uniform<u4:f16> on arch NPU37XX` | Missing compiler lowering for asymmetric u4 |
| **GAP-2** | `openvino-core` (NPU) | `src/core/src/partial_shape.cpp:261` | `to_shape was called on a dynamic shape` on `[?, 2, ?, 64]` | CMX scratchpad static layout requirement |
| **GAP-3** | `openvino.genai` | `fast_draft_strategy.cpp:437` | `Check 'config.is_greedy_decoding()' failed: Currently only greedy decoding is supported` | Speculative decoding lacks stochastic sampling |
| **GAP-4** | `intel_gpu` plugin | `multi_tensor_variable_state.cpp:216` | `[GPU] get_state API is supported only when KV-cache compression is disabled` | GPU state API blocks compressed KV-cache |
| **GAP-5** | `intel_npu` plugin | `ze_graph_ext_wrappers.cpp:433` | `Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER (0x78000007)` | Concurrency & driver initialization contention |
| **GAP-6** | `openvino.genai` / NPU | `fast_draft_strategy.cpp` | Synchronous ping-pong latency bottleneck on NPU 3720 ($T_{\text{draft}} \gg T_{\text{target}}$) | Serial execution blocks GPU while NPU drafts |

---

## Detailed Gap Analysis & Upstream PR Specifications

### Gap 1: NPU 3720 Rejection of Asymmetric Unsigned INT4 (`u4`) Dequantization
- **Repository**: `openvinotoolkit/openvino` (`src/plugins/intel_npu/src/compiler_adapter/` & `vpux-compiler`)
- **Observed Traceback**:
  ```text
  [ERROR] [vpux-compiler] OneShotBufferizeVPU2VPUIP failed : Unsupported DequantizeOp configuration:
  !VPU.DistributedTensor<896x128x1x1x!quant.uniform<u4:f16, 1.000000e+00>, affine_map<...>, @CMX_NN, ...> ->
  !VPU.DistributedTensor<896x128x1x1xf16, affine_map<...>, @CMX_NN, ...> on arch NPU37XX
  [ERROR] [vpux-compiler] Failed Pass OneShotBufferizeVPU2VPUIP on Operation loc(fused<...>)
  ```
- **Physical Root Cause**:
  The Intel NPU 3720 matrix multiplication hardware units (DPUs) lack runtime hardware integer offset subtraction datapaths ($Z \neq 0$). They natively accelerate symmetric signed INT4 (`i4` / `!quant.uniform<i4:f16>`), where $Z \equiv 0$. The VPUX compiler pass `OneShotBufferizeVPU2VPUIP` rejects any `u4` DequantizeOp on `NPU37XX` rather than lowering it.
- **Local Workaround in Arc Runner**:
  `NpuGraphRewriter` converts `u4` Const tensors to `i4`, decouples the 169 `Subtract` zero-point layers from the DAG, reroutes upstream Convert outputs to downstream Multiply layers, and toggles the MSB nibble (`^ 0x88`) in the weights binary.
- **Proposed Upstream PR**:
  Add an upstream compiler pass in `vpux-compiler` (before `OneShotBufferizeVPU2VPUIP`) that detects `!quant.uniform<u4:f16>` on `NPU37XX` and automatically rewrites it to `!quant.uniform<i4:f16>` by absorbing the integer offset into the bias or performing pre-CMX tensor decomposition.

---

### Gap 2: Dynamic Shape Collapse on Stateful KV-Cache in NPU Compiler
- **Repository**: `openvinotoolkit/openvino` (`src/core/src/partial_shape.cpp` & `intel_npu/src/plugin`)
- **Observed Traceback**:
  ```text
  [ERROR] [IE::FrontEnd::importNetwork] Upper bounds are not specified for node 'ShapeOf_108309':
  input '0' bounds are '[9223372036854775807, 9223372036854775807]'
  Compiler returned msg: Exception from src/core/src/partial_shape.cpp:261: to_shape was called on a dynamic shape.
  ```
- **Physical Root Cause**:
  Models exported with dynamic KV cache states have `ReadValue` and `Assign` operators with shapes `[?, 2, ?, 64]` and `variable_shape="?,2,?,64"`. The NPU 3720 compiler requires bounded scratchpad memory layouts in CMX SRAM. Calling `.to_shape()` on an unbounded dynamic dimension triggers a hard crash.
- **Local Workaround in Arc Runner**:
  `NpuGraphRewriter` inspects all `ReadValue` layers and sets `variable_shape="?,2,0..128,64"` and statically bounds `ReadValue` and `Assign` port dimensions to `[1, 2, 128, 64]`.
- **Proposed Upstream PR**:
  Enhance OpenVINO's `openvino.tools.ovc` and NPU plugin compiler adapter to automatically infer or accept default context window upper bounds (`--max_seq_len`) for stateful transformer models during compilation, preventing the unhandled `to_shape` exception.

---

### Gap 3: OpenVINO GenAI Speculative Decoding Limited to Greedy Decoding
- **Repository**: `openvinotoolkit/openvino.genai` (`src/cpp/src/speculative_decoding/stateful/fast_draft_strategy.cpp`)
- **Observed Traceback**:
  ```text
  RuntimeError: Check 'config.is_greedy_decoding()' failed at
  openvino.genai\src\cpp\src\speculative_decoding\stateful\fast_draft_strategy.cpp:437:
  Currently only greedy decoding is supported
  ```
- **Root Cause**:
  `FastDraftStrategy::verify_draft_tokens()` currently only implements deterministic token equality matching:
  $$\text{token}_{\text{draft}} == \text{token}_{\text{target}}$$
  When `temperature > 0.0` or `do_sample=True`, the engine throws a `RuntimeError` rather than executing speculative rejection sampling.
- **Local Workaround in Arc Runner**:
  Enforce greedy decoding configuration (`temperature=0.0`, `do_sample=False`) in `HybridSpeculativeEngine` when speculative dual-silicon mode is active.
- **Proposed Upstream PR**:
  Implement Leviathan et al. (2023) speculative sampling in `fast_draft_strategy.cpp`:
  $$P(\text{accept}) = \min\left(1, \frac{q(x)}{p(x)}\right)$$
  allowing full temperature and top_p sampling support during speculative generation.

---

### Gap 4: Intel GPU Plugin Disallows `get_state()` with Compressed KV-Cache
- **Repository**: `openvinotoolkit/openvino` (`src/plugins/intel_gpu/src/plugin/multi_tensor_variable_state.cpp`)
- **Observed Traceback**:
  ```text
  RuntimeError: Exception from src\inference\src\cpp\variable_state.cpp:45:
  Exception from src\plugins\intel_gpu\src\plugin\multi_tensor_variable_state.cpp:216:
  [GPU] get_state API is supported only when KV-cache compression is disabled
  ```
- **Physical Root Cause**:
  When speculative decoding runs with GPU target and NPU draft, the engine synchronizes transformer state using `variable_state.get_state()`. The Intel GPU plugin's multi-tensor variable state implementation only supports raw uncompressed tensors (`f16`/`f32`), throwing when KV-cache compression (`u8`/`u4`) is enabled.
- **Local Workaround in Arc Runner**:
  Explicitly configure the target GPU pipeline with `{"KV_CACHE_PRECISION": "f16"}` when speculative draft mode is active.
- **Proposed Upstream PR**:
  Add decompression support to `MultiTensorVariableState::get_state()` in `intel_gpu` plugin so that compressed KV caches can be read back and synchronized across accelerators without throwing.

---

### Gap 5: Level Zero Driver Initialization Race & Null Pointer in NPU Compiler
- **Repository**: `openvinotoolkit/openvino` (`src/plugins/intel_npu/src/compiler_adapter/src/ze_graph_ext_wrappers.cpp`)
- **Observed Traceback**:
  ```text
  Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER, code 0x78000007 - pointer argument may not be nullptr
  ```
- **Physical Root Cause**:
  Under concurrent thread initialization or when host background services (e.g. Intel Graphics Command Center or telemetry agents) interact with the Level Zero driver, `pfnCreate2` can fail with an unhandled null pointer error rather than waiting or providing a clear driver busy error code.
- **Local Workaround in Arc Runner**:
  Safeguard 3: Wrapped Level Zero compiler instantiation in driver lock and timeout guards, with automatic graceful fallback to CPU draft execution.
- **Proposed Upstream PR**:
  Add synchronization mutex and retry loops around Level Zero driver initialization handles in the Intel NPU plugin.

### Gap 6: Synchronous Ping-Pong Latency Bottleneck on NPU 3720 in Speculative Co-Execution
- **Repository**: `openvinotoolkit/openvino.genai` (`src/cpp/src/speculative_decoding/stateful/fast_draft_strategy.cpp`)
- **Observed Physical Performance**:
  On Intel Core Ultra 5 125H (Meteor Lake):
  - **Standalone Governor GPU**: **36.60 tok/s** ($T_{\text{target}} \approx 27.3\text{ ms/token}$).
  - **Standalone NPU 3720 Draft**: **~4.5 tok/s** ($T_{\text{draft}} \approx 220\text{ ms/token}$).
  - **Fused Speculative (GPU Target + NPU Draft)**: **3.51 - 3.96 tok/s** ($T_{\text{cycle}} \approx 680\text{ ms}$).
- **Physical Root Cause**:
  `FastDraftStrategy` executes synchronous serial drafting: the main thread blocks and waits while the NPU performs $\gamma$ autoregressive decode iterations. Because the NPU 3720 DPU dispatch and CMX buffer synchronization overhead is $\approx 200\text{ ms}$ per single-token iteration ($M=1$), drafting $\gamma=3$ tokens on NPU takes $\approx 600\text{ ms}$. Even if the GPU verifies all 3 tokens in $28\text{ ms}$, the total cycle time is dominated by $T_{\text{draft}}$, pulling throughput down to $\approx 3.96\text{ tok/s}$.
- **Proposed Upstream PR**:
  1. Implement **Asynchronous Pipelined Speculative Decoding**: The GPU should proceed with speculative verification of the *current* window while the NPU asynchronously generates proposals for the *next* window over disjoint Level Zero command queues without stalling.
  2. Implement **NPU Token-Tree Draft Parallelism** (Medusa/Eagle style): Compile the NPU draft model with parallel multi-head prediction to emit all $\gamma$ candidate tokens in a single forward pass ($O(1)$ DPU dispatches rather than $O(\gamma)$ serial iterations).

---

## Empirical Benchmark Table (Physical Intel Core Ultra 5 125H)

```text
==========================================================================================
PHYSICAL HARDWARE BENCHMARK COMPARISON: INTEL CORE ULTRA 5 125H (METEOR LAKE)
Target Model: Qwen2.5-1.5B (Intel Arc Xe-LPG iGPU) | Draft Model: Qwen2.5-0.5B (Intel AI Boost NPU 3720)
==========================================================================================
Workload Tier    | Baseline GPU     | Governor GPU     | Fused GPU+NPU    | Speedup Delta 
------------------------------------------------------------------------------------------
Tier 1 (Low)     | 28.25 tok/s      | 36.12 tok/s      | 3.51 tok/s       | -87.6% (Power Save)
Tier 2 (Medium)  | 31.33 tok/s      | 36.60 tok/s      | 3.96 tok/s       | -87.4% (Power Save)
Tier 3 (Hard)    | 34.02 tok/s      | 36.48 tok/s      | 2.51 tok/s       | -92.6% (Power Save)
==========================================================================================
```

---

## Upstream Contribution Roadmap

- [ ] **PR 1**: `openvino`: Add `u4` to `i4` zero-point absorption pass in `vpux-compiler` for NPU37XX.
- [ ] **PR 2**: `openvino`: Bounded static range inference for `ReadValue`/`Assign` in NPU compiler frontend.
- [ ] **PR 3**: `openvino.genai`: Stochastic rejection sampling in `FastDraftStrategy` for non-greedy speculative decoding.
- [ ] **PR 4**: `openvino`: Support `get_state()` for compressed KV-cache in Intel GPU plugin.
- [ ] **PR 5**: `openvino`: Level Zero driver initialization retry guard in Intel NPU compiler adapter.
