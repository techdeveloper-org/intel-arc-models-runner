---
doc_id: DOC-PERF-WP05-STAGE
title: "WP-05 speculative decode engine"
stage: PERF-WP05
version: 1.0.0
status: draft
created: 2026-10-31
updated: 2026-10-31
authors: [CLI session]
reviewers: [user]
inputs: [docs/architecture.md, C:\Users\techd\.claude\plans\snoopy-soaring-book.md]
---

# WP-05 Speculative Decode Engine

Gate status: `PASS`. Reliability score: `0.98`. Red tests: `tests/test_speculative.py` (25 passed, including CC-1..CC-5).

## 1. Problem Statement & Findings Fixed

Autoregressive token generation on Intel Arc GPUs is constrained by memory bandwidth during single-token forward passes, leaving compute units underutilized.
- **Root cause:** Fetching full target model weights for every decode step bounds throughput to memory bandwidth.
- **Remedy:** Heterogeneous dual-silicon speculative decoding: offload draft token generation to the Intel AI Boost NPU (48 TOPS) running an INT4 draft model at ultra-low power ($< 5\text{ W}$), and verify $K=4$ proposed tokens in parallel on the Intel Arc GPU in a single matrix evaluation. If the empirical token acceptance rate drops below `SPECULATION_MIN_ACCEPT_RATE=0.6` after warmup, or if NPU hardware is absent or disabled (`ARC_NPU_DISABLE=1`), automatically fall back to single-model execution without interruption.

## 2. Core Contracts Implemented

1. **CC-1 — Module Structure & Import:**
   - `SpeculativeDecodeEngine`, `DraftEngine`, `VerifierEngine`, and `AdjustedRejectionSampler` exposed from `arc_runner.engine.speculative`.

2. **CC-2 — Intel AI Boost NPU Silicon Detection:**
   - `NpuDetector` in `arc_runner.engine.npu_detector` enumerates devices via OpenVINO Core.
   - Returns `"NPU"` (case-sensitive) iff hardware is present and `ARC_NPU_DISABLE!=1`, else returns `None`.

3. **CC-3 — Adjusted Rejection Sampling:**
   - `AdjustedRejectionSampler.sample(p_target, p_draft)` implements Leviathan et al. (2023) rejection sampling ($\alpha_i = \min(1.0, p_t / p_d)$ with zero-division clamping to $1.0$).
   - Rejects mismatched probability sequence lengths with `ValueError`.

4. **CC-4 — Environment Disable & Fallback:**
   - When `ARC_NPU_DISABLE=1`, `SpeculativeDecodeEngine` marks speculation unavailable (`is_available() == False`) and routes generation through single-model verification without error.

5. **CC-5 — Efficiency Gate & Warning Fallback:**
   - Maintains an exponential moving average (EMA) of token acceptance rate.
   - If acceptance falls below `SPECULATION_MIN_ACCEPT_RATE=0.6` after `SPECULATION_WARMUP_STEPS=50`, logs a `WARNING` and switches execution to the fallback path.

## 3. Test Evidence

- **Test Suite:** `tests/test_speculative.py` (25 tests passing: 20 existing lifecycle tests + 5 contract tests CC-1..CC-5).
- **Checks Verified:**
  - `test_speculative_decode_engine_import_cc1`: Validates class exports and module interfaces.
  - `test_npu_detector_detection_cc2`: Validates device detection, non-NPU systems, and environment variable suppression.
  - `test_adjusted_rejection_sampler_dimension_mismatch_cc3`: Validates vector dimension matching and rejection sampling mathematics.
  - `test_npu_disabled_by_env_falls_back_cc4`: Validates graceful fallback without exceptions when disabled.
  - `test_accept_rate_below_threshold_triggers_warning_fallback_cc5`: Validates accept-rate threshold enforcement and WARNING logging.

## 4. Modified Components

- `src/arc_runner/engine/npu_detector.py`: New Intel AI Boost NPU detector.
- `src/arc_runner/engine/speculative.py`: New dual-silicon speculative decode engine, draft engine, verifier engine, and rejection sampler.
- `src/arc_runner/core/constants.py`: Added `SPECULATION_DRAFT_K`, `SPECULATION_MIN_ACCEPT_RATE`, `NPU_DRAFT_DEVICE`, `SPECULATION_WARMUP_STEPS`.
- `src/arc_runner/engine/__init__.py`: Exposed speculative decoding classes.
- `tests/test_speculative.py`: Appended contract test suite CC-1..CC-5.

## Revision history

| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-31 | CLI session | WP-05 speculative decode engine stage record | Deliver WP-05 CC-1..CC-5 | pending |
