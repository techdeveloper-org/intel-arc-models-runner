---
doc_id: DOC-NPU-027-STAGE
title: "Release 0.2.7 - NPU draft graph rewriter correctness, performance and fallback"
stage: NPU-0.2.7
version: 1.0.0
status: draft
created: 2026-11-07
updated: 2026-11-07
authors: [technical-writer-agent]
reviewers: [user]
inputs: [src/arc_runner/quant/zero_point_math.py, src/arc_runner/quant/npu_graph_rewriter.py, src/arc_runner/engine/speculative.py]
---

# Release 0.2.7 - NPU draft graph rewriter correctness, performance and fallback

Gate status: not assessed in this document. Every number below is copied from the commit messages of the listed commits; none was re-measured here. Issues: #67 (rewriter correctness), #68 (rewriter performance), #69 (draft fallback).

## 1. Problem statement and findings fixed

| Commit | Issue | Finding | Change |
|:--|:--|:--|:--|
| 411cde5 | #67 | The rewriter deleted every Subtract and XORed every u4 Const, silently changing the model: 139 u4 and 58 u8 zero-points on Qwen3-0.6B are not symmetric, and u8 was never converted. | New `zero_point_math` module: nibble codec (low nibble first), symmetric detection, exact XOR conversion (0x88 for u4, 0x80 for u8), per-group minimax requantisation with an f16 scale floor and a per-element error bound. |
| e4152c2 | #67 | The rewriter removed every Subtract without checking the zero-point, flipped u4 Consts that kept their Subtract, ignored u8, trusted unvalidated offsets and wrote a hardcoded INT4_SYM manifest. | One-pass graph index; Subtract inputs traced through Convert to Consts by declared port id; offsets, sizes and overlaps validated before writing; exact conversion when the zero-point is symmetric, otherwise requantisation (`conversion_mode` lossy, exact or off; env `ARC_NPU_CONVERSION_MODE`); orphan zero-point chains removed; the manifest carries measured `weight_conversion` statistics, `certified_exact` and `quality=unmeasured` when requantised. |
| 4c492c2 | #68 | Rewriting Qwen3-0.6B took 39 s, 15 s of it in full-size float64 temporaries; a bytearray copy of the weights; a non-atomic write left partial output on failure; re-runs refreshed nothing. | `requantize_symmetric` works in 2M-element row chunks and reduces errors per group; `rewrite()` copies the weights file and edits it through mmap, stages output in a sibling temp directory and swaps it in atomically, checks free disk >= 2x weights, skips when the manifest source signature (size, mtime, mode, bounds) matches, and overwrites stale tokenizer assets. |
| 1fedcca | #69 | `_create_draft_model` caught every Exception from the rewriter, logged free text and fell back to CPU, hiding programming errors. | Only `NpuGraphRewriterError` is caught and logged at ERROR with error code and details as structured fields; the CPU fallback is kept; the reason is exposed as `HybridSpeculativeEngine.draft_fallback_reason`. Other exceptions propagate. |
| 96653c1 | #69 | The fallback knew only preferred -> CPU; after a successful NPU rewrite later tiers would have received the requantised NPU copy. | `_draft_device_chain` returns NPU, GPU, CPU (GPU, CPU; CPU alone; other devices then CPU); each tier starts from the original draft path; a compile failure records `draft_fallback_reason` and logs structured fields before the next tier. |
| b2cc880 | #67 | State freezing assumed rank 4 and axis 2, so latent (rank 3) or sequence-first caches were skipped or corrupted; the Assign bound could differ from its ReadValue bound; non-numeric dims raised a raw ValueError. | `find_growth_axis` picks the axis with static initial size 0 that matches the Concat axis downstream of the ReadValue (through Gather and Convert, negative axes normalised) and raises when candidates are ambiguous or disagree; `freeze_state_variable_shape` bounds only that axis and requires all other non-batch dims fixed; the Assign reuses its ReadValue bound; `verify_bounded` reads the rewritten model back with OpenVINO and reports variables with an unbounded non-batch dim (`NpuGraphRewriter(verify=True)` fails on them). |

## 2. Evidence stated in the commit messages

- 411cde5: 28 tests, including a hypothesis property against an independent float64 oracle.
- e4152c2: 91 tests pass. Qwen3-0.6B IR: 197 zero-points requantised (139 u4 and 58 u8), 0 unconverted, max relative error 0.06709, 56 KV states frozen. The rewrite took 39 s before #68.
- 4c492c2: 4 new tests plus 91 existing pass. The Qwen3 IR gives identical error statistics (max_abs 0.0737, max_rel 0.06709) and rewrites in 18.0 s and 19.7 s in two runs (39 s before).
- 1fedcca: 2 new tests; `test_speculative_triad` 17 passed. The full non-hardware suite had 4 failures, none caused by this change: docs-sync (stale diagram stamps), two known pre-existing speculative tests, and a `test_transpiler` CLI json-keys test that passes alone and with the change stashed (order-dependent, cause not investigated).
- 96653c1: 8 new tests; `test_speculative_triad` 25 passed. Not run on hardware.
- b2cc880: 50 tests in `test_npu_graph_rewriter` (including `verify_bounded` on real OpenVINO models) and the conversion and math suites pass. Real Qwen3-0.6B: source 56 of 56 variables unbounded, rewritten 0 of 56, all shapes `?,8,0..128,128`, 112 layers frozen. Two old rank-3 tests encoded the previous rank rule and were replaced on purpose.

## 3. What is NOT done

- Quality of requantised models is unmeasured: the manifest says `quality=unmeasured`; only weight-level error statistics exist, no task-level metric.
- NPU compile of the rewritten model has not been run. `verify_bounded` is a read-back check, not an NPU compile.
- Parameter and Result bounds and the reshape-corner check are not done; the batch axis (axis 0) stays dynamic in the variable shape.
- The mutation campaign has not been run for this release.
- 96653c1 is not run on hardware.
- The fallback reason is not surfaced in the API telemetry.
- Peak memory of the rewrite was not measured; the skip path is not timed.
- Not done in #68: per-output lock file, E741 renames in the KV freeze code.
- The 4 non-hardware suite failures listed in section 2 are not resolved by this release; `check_docs_sync` reports stale diagrams.

## 4. Traceability

| Requirement source | Code | Tests |
|:--|:--|:--|
| #67 | `src/arc_runner/quant/zero_point_math.py`, `src/arc_runner/quant/npu_graph_rewriter.py` | `tests/test_zero_point_math.py`, `tests/test_npu_graph_rewriter.py` |
| #68 | `src/arc_runner/quant/npu_graph_rewriter.py` | `tests/test_npu_graph_rewriter.py` |
| #69 | `src/arc_runner/engine/speculative.py` | `tests/test_speculative_triad.py` |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-11-07 | technical-writer-agent | first version | release 0.2.7 documentation for 411cde5, e4152c2, 4c492c2, 1fedcca, 96653c1, b2cc880 | pending |
