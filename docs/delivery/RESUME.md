---
doc_id: DOC-RESUME
title: Resume checkpoint
stage: all
version: 1.4.0
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
| PERF performance programme | plan section 23 (v8.5), DL-061..DL-070 | planned only; no source changed; revised after two rounds of four independent reviews (resolution logs in 23.9.1 and 23.9.3; design contract CC-1..CC-14 in 23.5.0); first packages: WP-01 (honesty, not a performance lever), WP-02L (lever prerequisites: schema fields, `--backends`, backend-scoped `--ov-prop`/`--arm`, `--strict-markers`) and WP-02h (shared fakes and fake clock), then WP-12a (device-keyed compile cache and opt-in warmup) and WP-17 (decode and first-request levers), which wait only for WP-02L (WP-02R is the rest); WP-03..WP-07 are correctness work; WP-08..WP-10 are measure-first and unscheduled (no served model uses the flash-stream engine); N-02 must run before WP-03 merges; Windows measurements M-01..M-28 and N-01..N-18 are listed in 23.4 (every `scratch/run_live_*` command needs `--keep`, or the runner deletes the model); open questions in 23.8 carry provisional defaults the maintainer may override |

Version is 0.2.0 in the tree but nothing is tagged or published; tagging, merging and publishing need the maintainer's explicit approval.

## What only the maintainer can do (real machine)

1. Rebuild the Windows DLL and commit it with its sidecar: `python src/arc_runner/native/build_native.py`, then `pytest -m "not hardware"` and `pytest -m hardware`; commit `libarc_native_sys.dll` and `libarc_native_sys.dll.build.json`.
2. `python scripts/probe_hardware.py --out probe_new.json` (the option is delivered by plan WP-02i; until then `cmd /c "python scripts\probe_hardware.py > probe_new.json"`; never use `>` in Windows PowerShell 5.1, which writes UTF-16 and would overwrite the committed `probe.json`), review the diff against `probe.json`, then replace it and commit.
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
| 1.0.0 | 2026-10-29 | main session | first version | session entry point | pending |
| 1.1.0 | 2026-10-29 | main session | S-G added, open items updated | remaining technical items closed in the cloud | pending |
| 1.2.0 | 2026-10-29 | main session | PERF row, plan sections 17-23 | performance bottleneck programme added | pending |
| 1.3.0 | 2026-10-29 | main session | PERF row: plan v8.4, DL-061..DL-070, new first packages and gates | reviews G1-G4 resolved | pending |
| 1.4.0 | 2026-10-29 | main session | PERF row: plan v8.5, WP-02 split, round-2 resolution; probe command no longer redirects with `>` | round-2 reviews G1-G4 resolved | pending |
