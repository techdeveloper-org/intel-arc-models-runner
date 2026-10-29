---
doc_id: DOC-RESUME
title: Resume checkpoint
stage: all
version: 1.6.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session]
reviewers: [user]
---

# Resume checkpoint

Read this first in a new session. Tracking issue: [#42](https://github.com/techdeveloper-org/intel-arc-models-runner/issues/42). Branch: `claude/nice-volta-bs7mnf`. Plan: `docs/plans/MASTER_PLAN_v4.md` (sections 17-23 are current; 23 is the performance programme).

## State (2026-10-29)

| Slice | Record | State |
|:--|:--|:--|
| S-A honesty | `stages/S-A-honesty/STAGE.md` | done in the cloud |
| S-B memory safety | `stages/S-B-memory-safety/STAGE.md` | done in the cloud; Windows DLL rebuild pending (local) |
| S-C ring buffer and lifetime | `stages/S-C-ring-and-lifetime/STAGE.md` | done in the cloud; TLC run pending (local) |
| S-D engine and Level Zero | `stages/S-D-engine-and-level-zero/STAGE.md` | done against a fake loader; real-driver run pending (local) |
| S-E benchmarks | `stages/S-E-benchmarks/STAGE.md` | statistics, gates, schema, harnesses done; no result files yet (local) |
| S-F docs, version, CI | `stages/S-F-docs-and-release-prep/STAGE.md` | done; release not done |
| S-G remaining items | `stages/S-G-remaining-items/STAGE.md` | done in the cloud; needs hardware to run |
| PERF performance programme | plan section 23 (v9.1), DL-061..DL-084 | planned only; no source changed; v9.0 **simplified** the design and v9.1 is a **fix-only** revision after a fourth round of four independent reviews (resolution logs 23.9.1, 23.9.3, 23.9.4 and 23.9.5; gate round 5 pending); design contract CC-1..CC-4 (23.5.0): one serial inference worker (a daemon thread, FIFO) replaces the lease, grant, ticket, reaper and budget machinery, **a request is one job and no route holds an engine object**, and a hang is seen through progress timestamps in `GET /health` (open without the API key; opt-in `ARC_JOB_STALL_EXIT_S`); ten active packages: WP-01 (honesty, not a performance lever, includes the speculative config-validity fix), WP-02 (measurement support, items 02a-02g, parallel with WP-01), WP-17 (measurement programme for decode and first-request levers, with `scripts/bench_arms.py`), WP-03 (the PROVEN items: event-loop stall, token loss, leaked generation, telemetry crosstalk, `GET /health`), then WP-12a (device-keyed compile cache and opt-in warmup, small, decided by N-04, ordered after WP-03), WP-05, WP-07, WP-06 (follows WP-03), WP-14, WP-15; everything else is deferred with exact gates (plan 23.5.3, DF-01..DF-09); N-02 must run before WP-03 merges; **the ordered minimal run list for the one Windows laptop is plan 23.4.0** (about 3.6 h of derived machine time before confirmations and 4.6 h with one winner per device, plus unmeasured loads and compiles; M-02 was removed; risky runs M-07 hard and M-25 are optional and last); every `python scratch/run_live_*` command needs `--keep`; open questions in 23.8 carry provisional defaults the maintainer may override |

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
| 1.6.0 | 2026-10-29 | main session | PERF row: plan v9.1 (fix only), DL-080..DL-084, one request = one job, stall detection, run list reordered (3.6-4.6 h), WP-12a after WP-03 | round-4 reviews G1-G4 resolved | pending |
