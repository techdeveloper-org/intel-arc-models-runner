---
doc_id: DOC-RESUME
title: Resume checkpoint
stage: all
version: 1.12.0
status: draft
created: 2026-10-29
updated: 2026-10-30
authors: [main session]
reviewers: [user]
---

# Resume checkpoint

Read this first in a new session. Tracking issue: [#42](https://github.com/techdeveloper-org/intel-arc-models-runner/issues/42). Branch: `claude/nice-volta-bs7mnf`. Plan: `docs/plans/MASTER_PLAN_v4.md` (sections 17-23 are current; 23 is the performance programme).

## State (2026-10-30)

| Slice | Record | State |
|:--|:--|:--|
| S-A honesty | `stages/S-A-honesty/STAGE.md` | done in the cloud |
| S-B memory safety | `stages/S-B-memory-safety/STAGE.md` | done in the cloud; Windows DLL rebuild pending (local) |
| S-C ring buffer and lifetime | `stages/S-C-ring-and-lifetime/STAGE.md` | done in the cloud; TLC run pending (local) |
| S-D engine and Level Zero | `stages/S-D-engine-and-level-zero/STAGE.md` | done against a fake loader; real-driver run pending (local) |
| S-E benchmarks | `stages/S-E-benchmarks/STAGE.md` | statistics, gates, schema, harnesses done; no result files yet (local) |
| S-F docs, version, CI | `stages/S-F-docs-and-release-prep/STAGE.md` | done; release not done |
| S-G remaining items | `stages/S-G-remaining-items/STAGE.md` | done in the cloud; needs hardware to run |
| PERF performance programme | plan section 23 (v9.7), DL-061..DL-095 | planned only; no source changed; v9.0 and v9.3 **simplified** the design and v9.1, v9.2, v9.4, v9.5 and v9.6 and v9.7 were fix-only revisions after four independent review rounds (round 10 scored 9, 9, 9, 8 with two design-level defects, both resolved in v9.7; resolution logs 23.9.1 to 23.9.11; gate re-review of v9.7 pending); v9.3 keeps only mechanisms with PROVEN-impact evidence (DL-088): design contract CC-1..CC-4 (23.5.0): one serial inference worker (a daemon thread, FIFO), **a request is one job and no route holds an engine object**, engine-free blocking routes run through `to_thread`, a minimal `GET /health` (`ok` or `wedged`, `busy`, `busy_for_s`, open without the API key, detects only a join expiry), route error mapping by `LoadPhaseError` tagging (DL-089; whole-call routes and `DELETE /api/delete` get explicit clauses, DL-092), a bounded stop (`timeout_graceful_shutdown`, DL-092; its real bound and client-visible effects, and jobs submitted as `(run, fail)` pairs so a refused queued stream answers 503, DL-093; the worker's `_admit` lock and its stop-versus-start ordering, DL-094) and a timing contract; stall limits, `/ready`, exit codes and earlier stop (DF-10), on-disk atomicity (DF-11) and process-global device health (DF-12) are deferred with their gates and corrected designs; ten active packages in one build order (plan 23.5.1): 02g (the first commit), WP-01 (honesty, not a performance lever), WP-03 (event-loop stall, token loss, leaked generation, telemetry crosstalk, `GET /health`, bounded stop), WP-12a, WP-02 (measurement support, items 02a-02f), WP-17 (measurement programme for decode and first-request levers, with `scripts/bench_arms.py`), then WP-05, WP-07 (a classifier at four fallback handlers) and WP-06, WP-14, WP-15; everything else is deferred with exact gates (plan 23.5.3, DF-01..DF-12); the pre-merge block of plan 23.4.0 (N-02, N-01 part 1, N-04 on GPU, N-12, N-13) must run before WP-03 and WP-12a merge; **the ordered minimal run list for the one Windows laptop is plan 23.4.0** (after that block, order M-09, M-01, M-17, M-18, M-19, M-10, M-05, M-04; about 3.4 h of derived machine time before confirmations (3.9 h if the M-01 `--seed 1` repeat is needed; step 4 counted, 3.3 h and 3.8 h without it) and about 5.9 h with one winner per device (6.4 h with the repeat), every sweep decided by the `bench_arms.py --evaluate SPEC` calls written out in 23.4.0 (DL-094) and every adoption by the cross-tier and cross-length rule of 23.4.0 (DL-095), which needs an A-B-B-A soak bracket run with 20 s pauses, plus unmeasured loads and compiles; hardware tests need `ARC_MODEL_DIR_QWEN15B`; M-28 is not a step; risky runs M-07 hard and M-25 are optional and last); every `python scratch/run_live_*` command needs `--keep`; open questions in 23.8 carry provisional defaults the maintainer may override |

Version is 0.2.0 in the tree but nothing is tagged or published; tagging, merging and publishing need the maintainer's explicit approval.

## What only the maintainer can do (real machine)

1. Rebuild the Windows DLL and commit it with its sidecar: `python src/arc_runner/native/build_native.py`, then `pytest -m "not hardware"` and `pytest -m hardware`; commit `libarc_native_sys.dll` and `libarc_native_sys.dll.build.json`.
2. `python scripts/probe_hardware.py --out probe_new.json` (the option is delivered by plan WP-02 item 02f; until then `cmd /c "python scripts\probe_hardware.py > probe_new.json"`; never use `>` in Windows PowerShell 5.1, which writes UTF-16 and would overwrite the committed `probe.json`), review the diff against `probe.json`, then replace it and commit.
3. Run the benchmark matrix with at least 3 warmups and 10 timed runs per system, write schema-v1 files to `docs/benchmarks/results/`, run `python scripts/render_bench_docs.py`.
4. `python scripts/bench_regression.py --baseline c7312e0 --workload llm --model-dir <dir> --device GPU`.
5. Run TLC on the ring model: `java -cp tla2tools.jar tlc2.TLC -workers 4 -config docs/formal/RingSlots.cfg docs/formal/RingSlots.tla`.
6. Decide on merge, tag and release.

## Open technical items

All need a real machine or model weights: run `scratch/run_live_*_benchmarks.py` (LLM, embeddings, VLM, audio, image, video) and `scripts/bench_kv_quality.py`; check Sysman power/frequency on real drivers; give the USM-staged buffers a consumer; push to a pull request so the three CI jobs run. Independent test authorship is still open (tests were written by the implementing sessions, checked by mutation).

## Commands that must pass before a push

```bash
python -m pytest -m "not hardware" -q
python scripts/check_docs_sync.py
python scripts/check_delivery_docs.py
python scripts/validate_bench_json.py
```

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | session entry point | 93868c6 |
| 1.1.0 | 2026-10-29 | main session | S-G added, open items updated | remaining technical items closed in the cloud | 8605588 |
| 1.2.0 | 2026-10-29 | main session | PERF row, plan sections 17-23 | performance bottleneck programme added | d6043f3 |
| 1.3.0 | 2026-10-29 | main session | PERF row: plan v8.4, DL-061..DL-070, new first packages and gates | reviews G1-G4 resolved | 58ff19e |
| 1.4.0 | 2026-10-29 | main session | PERF row: plan v8.5, WP-02 split, round-2 resolution; probe command no longer redirects with `>` | round-2 reviews G1-G4 resolved | d0744e3 |
| 1.5.0 | 2026-10-29 | main session | PERF row: plan v9.0 (simplified), DL-071..DL-079, minimal run list in plan 23.4.0; probe option now WP-02 item 02f | round-3 reviews G1-G4 resolved by simplification | ae428e4 |
| 1.6.0 | 2026-10-29 | main session | PERF row: plan v9.1 (fix only), DL-080..DL-084, one request = one job, stall detection, run list reordered (3.6-4.6 h), WP-12a after WP-03 | round-4 reviews G1-G4 resolved | 3aa9f72 |
| 1.7.0 | 2026-10-29 | main session | PERF row: plan v9.2 (fix only), DL-085..DL-087, route error mapping, timing contract, `/health` and `/ready`, process-global device health, run list about 5.4 h with one winner (A-B-A bracket) | round-5 reviews G1-G4 resolved | f4db269 |
| 1.8.0 | 2026-10-29 | main session | PERF row: plan v9.3 (simplify and fix), DL-088..DL-091, deferred stall, exit, on-disk and device-health mechanisms (DF-10..DF-12), error mapping by phase tagging, run list about 3.3 h and 5.7 h with one winner (A-B-B-A bracket) | round-6 reviews G1-G4 resolved | 3d95ee0 |
| 1.9.0 | 2026-10-29 | main session | PERF row: plan v9.4 (fix only), DL-092, whole-call route mapping, bounded stop, one build order, run order M-09 first with A/A for M-18 and TTFT-800, N-04 verdict, hardware-test model location, totals 3.3 h (3.8 h) and 5.7 h (6.25 h) | round-7 reviews G1-G4 resolved | 8cdf045 |
| 1.10.0 | 2026-10-29 | main session | PERF row: plan v9.5 (fix only), DL-093, refused queued streams, the stop limit as run, a pre-merge block before the renumbered run list M-09, M-01, M-17, M-18, M-19, M-10, M-05, M-04, N-04 first-request TTFT, `--evaluate` file lists | round-8 reviews G1-G4 resolved | a312ee1 |
| 1.11.0 | 2026-10-30 | main session | PERF row: plan v9.6 (fix only), DL-094, worker stop-versus-start ordering, `engine_key` and the tests it breaks, join constants in `core/constants.py`, `--evaluate SPEC`, N-02 numbers, the M-09 tier, N-04 per device, totals 3.4 h and 5.9 h with `m18_a` | round-9 reviews G1-G4 resolved | 1a4ccfe |
| 1.12.0 | 2026-10-30 | main session | PERF row: plan v9.7 (fix only), DL-095, the adoption rule across tiers and lengths (medium tier decides a GPU decode arm, both lengths decide an NPU arm, NPU decode no-harm cell), the M-09 bytes interval with the tie-embedding premise UNVERIFIED, `unload_path` and the hub delete call, `--evaluate` cell rule with confirmation calls and K, M-07 with `--out`, the bounded stop as run | round-10 reviews G1-G4 resolved | next revision |
