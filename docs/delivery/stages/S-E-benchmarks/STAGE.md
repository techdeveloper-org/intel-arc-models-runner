---
doc_id: DOC-SE-STAGE
title: Slice S-E - benchmark protocol, statistics, and provenance
stage: S-E
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session applying the benchmark roles from the plan]
reviewers: [user]
inputs: [docs/plans/MASTER_PLAN_v4.md section 18]
---

# Slice S-E: benchmark protocol, statistics, and provenance

Gate status: `cloud-green, local-pending`. No arc-runner or Ollama result file exists yet; every benchmark that needs the real hardware is the author's to run.

## 1. Library use (honest record)
`hardware-performance-engineer`, `performance-testing-engineer` and `llm-benchmark-analyst` agent files and skills were not read; the work follows plan section 18.

## 2. New finding B11 (fixed)
`BenchmarkRunner` looked up the Student-t multiplier in a table that skipped df 11-19 and rounded up to the next key (df 11 returned 2.086 instead of 2.201), and `MultiModelCoBenchmarker` used `table.get(df, 2.0)`. Both under-reported confidence intervals for 12-20 samples (a 5-8% too narrow half-width). Both now call one implementation (`arc_runner.benchmark.stats.student_t_ci95`) with the exact table.

## 3. Changes
| File | Purpose |
|:--|:--|
| `src/arc_runner/benchmark/stats.py` | exact t table, median/IQR/CV, bootstrap CI and ratio CI (seeded), warmup convergence, roofline check, `BenchSample`, `evaluate_speedup_gate`, `evaluate_noninferiority_gate` |
| `src/arc_runner/benchmark/results.py` | schema-v1 documents: environment capture, validation, write, load |
| `docs/benchmarks/schema/bench_result.v1.json`, `scripts/validate_bench_json.py` | schema and stdlib validator |
| `scripts/render_bench_docs.py` + marker block in `docs/benchmarks/README.md` | doc table generated only from result files; `--check` for CI |
| `docs/benchmarks/METHOD.md` | metric definitions, protocol, claim gates, hypotheses registered before running |
| `scripts/bench_native_micro.py` | B8: tensor lookup latency, mapped read bandwidth vs file read, ring acquire latency at 1/4/8 threads |
| `scripts/bench_regression.py` | B10: interleaved baseline-vs-current runs (git worktree), schema-v1 output, non-inferiority verdicts; workloads `flash_acquire` (anywhere) and `llm` (real model) |
| `scripts/probe_hardware.py` | prints memory bandwidth, Level Zero devices (type, id, name), OpenVINO devices, PerfMetrics accessors, memory module info on Windows |
| live benchmark script | exports a schema-v1 smoke file; legacy arc-runner rows without a source are skipped; the 6 stored legacy arc-runner rows in `scratch/live_real_models_benchmark_results.json` are marked `source: legacy_unverified` (their producing script could fall back to word-count tokens and elapsed*200 TTFT) |

## 4. Results
- 40 new tests (`tests/test_benchmark_stats.py` 24, `tests/test_bench_results.py` 11 plus script and docs checks); non-hardware suite: 733 tests, same 3 baseline failures, 6 skipped.
- `scripts/probe_hardware.py` runs on this Linux VM (CPU model, 20.9 GB/s copy and 14.7 GB/s triad on a 4-vCPU VM; no Level Zero), which is not the target machine.
- A smoke run of `bench_regression.py --workload flash_acquire` against c7312e0 (4 runs, noisy cloud VM): the current ring buffer completes layer loads about 280 times faster than the old one on this synthetic workload. The old code called `gc.collect()` on every eviction while holding the lock (finding X2); this number is an artefact of that on a toy container and must not be quoted as a model speed-up. TTFT of the first acquire was too noisy (CV above 10%) for the gate, which refused it.

## 5. Open items (for the author, on the real machine)
1. `python scripts/probe_hardware.py > probe.json` and commit the JSON (bandwidth feeds the roofline check).
2. Run the LLM matrix with at least 3 warmups and 10 timed runs per system and write schema-v1 files to `docs/benchmarks/results/`; then `python scripts/render_bench_docs.py`.
3. `python scripts/bench_regression.py --baseline c7312e0 --workload llm --model-dir <model> --device GPU` for the no-regression gate (B10).
4. The multi-domain scripts (embeddings, VLM, audio, image, video) still write their legacy JSON; converting them to schema v1 needs those models and is part of the local run.
5. Quality gate for the u8 KV cache (KL on a fixed prompt set) is not implemented.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | S-E1 implementation | pending |
