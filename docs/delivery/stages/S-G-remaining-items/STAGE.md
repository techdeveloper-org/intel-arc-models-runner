---
doc_id: DOC-SG-STAGE
title: Slice S-G - remaining technical items
stage: S-G
version: 1.4.0
status: draft
created: 2026-10-29
updated: 2026-10-30
authors: [main session]
reviewers: [user]
inputs: [docs/delivery/RESUME.md open technical items, docs/plans/MASTER_PLAN_v4.md sections 21, 22]
---

# Slice S-G: remaining technical items

Gate status: `cloud-green`. Everything that needs Intel silicon, model weights or a Windows machine is still pending and is listed in RESUME.md.

## 1. Changes
| Item | Change |
|:--|:--|
| Speculative context window | `engine/context_budget.ContextBudgetMixin` shared by both engines; speculative generate/stream reject before any pipeline call and report `prompt_tokens` |
| `try_pin` | `FlashStreamWeightManager(pin_resident_layers)` / `ARC_PIN_RING=1`; pin on READY, unpin on recycle, release, close; first OS refusal disables pinning; `_pin_lock` prevents a leaked lock under concurrent eviction |
| USM/DMA | `engine/flash_stream/device_stager.DeviceLayerStager` (opt-in `device_stager=`); counted bytes, seconds and CPU-fallback copies; first failure disables staging; no consumer reads the pointer yet |
| Sysman | package power from the energy counter, GPU frequency, free memory from device-local modules; structs and stypes from upstream `zes_api.h` v1.9; `PowerMeter`; `probe_hardware.py` section |
| Schema and runners | `workload`/`metrics` fields; `nonllm` runners for LLM, embeddings, VLM, audio, image, video; scripts rewritten; VLM script replaces the fixture audit |
| Honesty audit | image/video engines, studio pipeline, co-benchmark, scratch scripts: invented stage splits, power, memory, RTF, cloud ROI removed (None when unmeasured); fabrication gate extended |
| u8 KV gate | `benchmark/kv_quality`, `scripts/bench_kv_quality.py`, hypothesis H-D; default stays opt-in |
| Mermaid | all 55 blocks in `uml/`, architecture, diagrams and research docs render with mermaid-cli 10.9.1; four old diagrams and the state-diagram generator fixed; `scripts/check_mermaid.py` and a CI job added |
| E5 | OpenVINO GenAI 2026.4.1 source read (`perf_metrics.cpp`): `get_throughput` = 1000 / mean TPOT, TPOT excludes the first token, so it is decode-only; `telemetry.py` and METHOD.md state this |
| E6 | streamed `total_tokens` now comes from tokenizing the emitted text (`token_count`); timing remains a wall-clock estimate; re-tokenizing decoded text can differ slightly from the generated ids |
| Plan test list | all 24 named plan tests mapped to repository tests in `TRACEABILITY.md`; added boundary (4 KiB / 64 KiB), 80-layer, lock-refusal, byte-identical-fallback tests and two `@hardware` tests (`ARC_TEST_MODEL_DIR`) |
| Provenance worklist | `scripts/list_unverified_numbers.py` lists documents with UNVERIFIED banners and their numeric table rows (6 files, ~170 rows) |
| Library use | `llm-benchmark-analyst` read: its rules target accuracy benchmarks (Wilson CI, McNemar, contamination); nothing in them changed the speed protocol except confirming that evaluation configuration, CI reporting and "no unverified vendor numbers" are already rules here. Other plan agents (`model-capability-profiler`, `gpu-driver-engineer`, `kernel-driver-engineer`, `supply-chain-engineer`, `release-engineering-specialist`, `technical-writer-agent`) were not read |

## 2. Results
Clean-venv run of the CI Linux steps (`pip install -e ".[dev]"`, native build, 800+ non-hardware tests, ring model, docs sync, delivery docs, benchmark validators) passes. The Linux 3.10 job, the Windows job and the new diagrams job have not run on GitHub: the workflow triggers only on `main` pushes and pull requests, and no pull request exists.

## 3. Not verified here
- No script that loads a model or reads a real counter has run (no Intel silicon or weights here); behaviour of Sysman on real drivers (permissions, Windows enablement) is untested.
- The KV gate and every benchmark script need hardware and weights.
- The staged USM pointer has no consumer; no bandwidth is claimed.

## 4. Decisions
| DL | Decision | Why |
|:--|:--|:--|
| 056 | Non-LLM results extend schema v1 additively (workload + metrics) instead of a schema v2 | existing files and tooling stay valid |
| 057 | Power is measured with the Sysman energy counter or is None | wattage constants are fabrication |
| 058 | A script that printed fixed numbers (`run_full_catalog_benchmark_audit.py`) is deleted, not patched | there was no measurement to keep |
| 059 | u8 KV stays opt-in whatever the gate says | a default change is a human decision |
| 060 | Mermaid validity is checked by rendering, in CI | `validate.py`-style checks cannot see syntax errors |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | S-G implementation | pending |
| 1.1.0 | 2026-10-29 | main session | E5, E6, test-list audit, provenance worklist | close plan open items | pending |
| 1.2.0 | 2026-10-30 | main session | WP-03: SerialWorker, EngineUnavailableError, TIMEOUT_GRACEFUL_SHUTDOWN_S, /health worker state | fix event-loop stall and stream token loss | pending |
| 1.3.0 | 2026-10-30 | main session | WP-02 02a/02e/02f: power_mode + external_artifacts in schema + capture_environment; validate_model_dir; probe_hardware --out flag | measurement support items | pending |
| 1.4.0 | 2026-10-30 | CLI session | WP-02 02b: bench_arms.py paired gate runner, bootstrap_paired_ratio_ci, evaluate_paired_gate, decide_adoption (AD-1..AD-6), 32 new tests | measurement support item 02b | pending |
