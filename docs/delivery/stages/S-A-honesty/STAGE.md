---
doc_id: DOC-SA-STAGE
title: Slice S-A - remove fabricated values
stage: S-A
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session]
reviewers: [user]
inputs: [docs/plans/MASTER_PLAN_v4.md sections 2.2, 2.3, 17.4, 21.1]
---

# Slice S-A: remove fabricated values

Gate status: `cloud-green`. Commits: `23c2c60` (13 failing tests), `3102488` (fix).

## 1. Findings fixed
L1 (constant 22.0 W), L2 (free GPU memory from system RAM / `total // 2`), E1/E6 (word-count tokens, `elapsed * 200/400/500`), H1 (`REFERENCE_SWEEP` rows), H3 (speculative acceptance and speed-up from constants), H4 (Ollama-compatible duration split 0.2/0.8, word-count token counts), H5 (synthetic heavy-model constants and unrun Ollama OOM), H6 (README Ollama figures not matching the JSON).

## 2. Red evidence
`tests/test_honesty_no_fabrication.py`: 13 tests, all failing on c7312e0 (missing `get_unavailable_reason` and telemetry fields, `ttft_ms = 0.237` for a result with no counters, a grep gate that found `return 22.0`).

## 3. Changes
`engine/telemetry.py` (measured-or-unavailable builders), new `EngineTelemetry` flags, `sysman.py` (power `None`, free memory `None`, reasons), speculative engine and Ollama routes without invented numbers, benchmark script and JSON without projected rows, README table rebuilt from the JSON with a provenance notice.

## 4. Results
589 non-hardware tests at the end of the slice, 3 baseline failures (Dockerfile health URL, Level Zero absent, ffmpeg), 6 skipped.

## 5. Open items
Docs prose audit (done in S-E as provenance notes); real arc-runner rows need a hardware run.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version, written after the slice | complete the delivery record | pending |
