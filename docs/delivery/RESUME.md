---
doc_id: DOC-RESUME
title: Resume checkpoint
stage: all
version: 1.14.0
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
| PERF performance programme | plan section 23 (v9.8), DL-061..DL-096 | plan v9.8; **partly implemented, see 'Code since the plan' below**; v9.0 and v9.3 **simplified** the design and v9.1, v9.2, v9.4, v9.5, v9.6, v9.7 and v9.8 were fix-only revisions after four independent review rounds (round 10 scored 9, 9, 9, 8 with two design-level defects, both resolved in v9.7; the round-11 final checks of v9.7 scored G1 9 and G4 8 with one more design-level defect, the adoption rule, resolved in v9.8 as the procedure AD-1..AD-6 with `decide_adoption` and verified by enumeration (DL-096); resolution logs 23.9.1 to 23.9.12; gate re-review of v9.8 pending); v9.3 keeps only mechanisms with PROVEN-impact evidence (DL-088): design contract CC-1..CC-4 (23.5.0): one serial inference worker (a daemon thread, FIFO), **a request is one job and no route holds an engine object**, engine-free blocking routes run through `to_thread`, a minimal `GET /health` (`ok` or `wedged`, `busy`, `busy_for_s`, open without the API key, detects only a join expiry), route error mapping by `LoadPhaseError` tagging (DL-089; whole-call routes and `DELETE /api/delete` get explicit clauses, DL-092), a bounded stop (`timeout_graceful_shutdown`, DL-092; its real bound and client-visible effects, and jobs submitted as `(run, fail)` pairs so a refused queued stream answers 503, DL-093; the worker's `_admit` lock and its stop-versus-start ordering, DL-094) and a timing contract; stall limits, `/ready`, exit codes and earlier stop (DF-10), on-disk atomicity (DF-11) and process-global device health (DF-12) are deferred with their gates and corrected designs; ten active packages in one build order (plan 23.5.1): 02g (the first commit), WP-01 (honesty, not a performance lever), WP-03 (event-loop stall, token loss, leaked generation, telemetry crosstalk, `GET /health`, bounded stop), WP-12a, WP-02 (measurement support, items 02a-02f), WP-17 (measurement programme for decode and first-request levers, with `scripts/bench_arms.py`), then WP-05, WP-07 (a classifier at four fallback handlers) and WP-06, WP-14, WP-15; everything else is deferred with exact gates (plan 23.5.3, DF-01..DF-12); the pre-merge block of plan 23.4.0 (N-02, N-01 part 1, N-04 on GPU, N-12, N-13) must run before WP-03 and WP-12a merge; **the ordered minimal run list for the one Windows laptop is plan 23.4.0** (after that block, order M-09, M-01, M-17, M-18, M-19, M-10, M-05, M-04; about 3.4 h of derived machine time before confirmations (3.9 h if the M-01 `--seed 1` repeat is needed; step 4 counted, 3.3 h and 3.8 h without it) and about 5.9 h with one winner per device (6.4 h with the repeat), every sweep decided by the `bench_arms.py --evaluate SPEC` calls written out in 23.4.0 (DL-094) and every adoption by the cross-tier and cross-length rule of 23.4.0 (DL-095), which needs an A-B-B-A soak bracket run with 20 s pauses, plus unmeasured loads and compiles; hardware tests need `ARC_MODEL_DIR_QWEN15B`; M-28 is not a step; risky runs M-07 hard and M-25 are optional and last); every `python scratch/run_live_*` command needs `--keep`; open questions in 23.8 carry provisional defaults the maintainer may override |

Version is 0.2.0 in the tree but nothing is tagged or published; tagging, merging and publishing need the maintainer's explicit approval.

## Code since the plan (2026-10-30), all pushed and each with a red-first test

| Commit | What it did | Plan item |
|:--|:--|:--|
| `c3b69cb` | the benchmark runners keep the model by default (`--delete-after` opts in, `--keep` stays as a no-op), the generation-config builder fails loudly with `ConfigurationError`, token usage is measured or omitted (OpenAI, embeddings, Ollama `prompt_eval_count`) | WP-01 subset |
| `2185129` | absolute, device-keyed compile cache for GPU, NPU and CPU; `ARC_CACHE_DIR`; opt-in `ARC_WARMUP_ON_LOAD` (logs `warmup_on_load=on` at WARNING) | WP-12a core |
| `4cf5913` | Anthropic Messages API: `POST /v1/messages` (blocking and SSE), `count_tokens`, `HEAD/GET /api/hello`, `x-api-key` or Bearer auth; guide in `docs/claude-code.md` | new, not in the plan |
| `b260757` | context window from `ARC_CONTEXT_WINDOW`, else the model's `config.json` `max_position_embeddings`, else 4096 | new, not in the plan |
| `d397869`, `f6d8bf0` | native tests correct on Windows (fake loader counted naps instead of time); Sysman reports why the Level Zero loader failed to load | test and Sysman fixes |
| `ebd5b6c`, `1d0eb64` | the KV gate records a runtime rejection as a result; measured: the Arc GPU rejects `KV_CACHE_PRECISION=u8` with default settings (`docs/benchmarks/quality/kv_u8_qwen-2.5-1.5b.json`) | M-15 and H-D |
| `283064c`, `9d685a2` | real `GET /health`; clean `ARC_CONTEXT_WINDOW` error in `serve`; `VisionEngine` and `EmbeddingEngine` logging fixed (they hid the real load error), guarded by `tests/test_logging_call_contract.py` | docs and logging |
| `0cc0d6d` | security: Host validation (DNS rebinding), browser CSRF guards (Origin, Content-Type, Sec-Fetch-Site), `/api/pull` path traversal, no filesystem paths in error bodies, tool-parser recursion, exact open paths, bounded `X-Request-ID`; new env `ARC_ALLOWED_HOSTS` | security review S1 |

The plan's WP-03 (serial worker, `to_thread`, stream bridge, bounded stop), WP-02, WP-17, WP-05, WP-06, WP-07, WP-14 and WP-15 are **not implemented**. After the security commit the full non-hardware suite exits 0 (2058 passed, 8 skipped, 24 deselected in the agent's run).

## Measured on the maintainer's machine (Core Ultra 5 125H, Arc iGPU, NPU)

- Hardware tests (`pytest -m hardware`, `ARC_TEST_MODEL_DIR=models\qwen-2.5-1.5b`): 21 passed, 1 skipped (NOT RUN), 1 failed before `ebd5b6c` (the u8 gate, now reported as NOT RUN with the reason).
- `scratch/run_live_sequential_model_benchmarks.py --model qwen-2.5-1.5b --runs 30 --warmups 3 --max-warmups 5 --interleave --cooldown-s 20 --seed 0` wrote `llm_qwen-2.5-1.5b_20261030T070037Z.v1.json` (315 samples). Derived in the plan: GPU decode medians 46.24 / 41.39 / 36.33 tok/s with CV 17.6 / 16.0 / 12.3 %, CPU 18.20 / 13.32 / 13.10, NPU 15.02 / 14.68 / 13.52. No claim gate was evaluated.
- `scratch/run_live_image_benchmarks.py --model stable-diffusion-v1-5 --tiers low --runs 3 --warmups 1 --no-cpu-baseline`: 7 samples, none failed, about 10.6 to 12.0 s per image (the earlier `CL_OUT_OF_RESOURCES` did not occur on the low tier). Medium and hard tiers were not run; hard is risky (GPU reset or out-of-memory).
- `bench_regression.py --baseline 76b0f26` refused a verdict on both gates (CV above 10 %). Baseline `c7312e0` cannot run on this GPU because it defaults to u8.
- The two Windows Sysman tests pass after linking the test fake statically; the two ASan/UBSan tests skip because mingw has no `libasan`.

## Handoff for the next (CLI) session on the maintainer's hardware

Start here: `git pull origin claude/nice-volta-bs7mnf`, then the commands in "Commands that must pass before a push". Next, in this order:

1. **Try Claude Code against the runner** (`docs/claude-code.md`; the biggest unknown): `ARC_CONTEXT_WINDOW=32768`, `arc-runner serve`, then `ANTHROPIC_BASE_URL=http://127.0.0.1:11434`, `ANTHROPIC_AUTH_TOKEN=local`, `ANTHROPIC_MODEL=<name from /v1/models>`. UNVERIFIED so far: Claude Code's behaviour (empty `usage`, a 404 from `count_tokens`, first-run login), tool-call quality of small models, TTFT with its large system prompt, memory at a large window on 15 GB shared memory. Send back the server log and the first error.
2. **Plan pre-merge block and run list** (plan 23.4.0) once WP-03, WP-12a and WP-02 are built; WP-12a's cache effect on first-inference cost (CPU 8.86 to 13.87 s, NPU 0.25 to 2.65 s, GPU 0.06 to 0.20 s in the committed LLM file) is a hypothesis until N-04 runs.
3. **Re-run regression** with `--baseline 76b0f26`, AC power, nothing else running, after cooling down; a CV above 10 % again would be a finding.
4. Implement the plan's WP-03 first (it fixes the proven event-loop stall: a 52 ms probe took 2077 ms behind one blocking `/api/generate`, and 67 of 300 stream tokens were lost when the loop stalled for 2 s; both measured on a Linux VM), then WP-02 and the rest in the order of plan 23.5.1.

Decisions left to the maintainer (not made by any session): `--host 0.0.0.0` without `ARC_API_KEY` (only a WARNING exists); a queue cap and cancel on the blocking path (60 abandoned `max_tokens=32768` requests all ran and starved a real one); arbitrary Hugging Face pulls and disk fill (M2); the engine cache keyed by the raw model string (M3); body-size and tool-count caps (M5); CORS allowing any local port (M6); chat-template delimiter injection (M7); `/api/ollama/local` and `/api/ollama/inspect` return filesystem paths in successful responses; one adoption per CPU/NPU device in plan 23.4.0 (AD-5; the reviewer wanted a bracketed second winner); license file; merge, tag and release.

Not verified anywhere: the whole plan section 23 v9.8 was last independently scored at v9.7 (G1 9, G4 8); the v9.8 adoption procedure was checked only by enumeration.

## What only the maintainer can do (real machine)

1. (Done on 2026-10-30, commit by the maintainer) Rebuild the Windows DLL and commit it with its sidecar: `python src/arc_runner/native/build_native.py`, then `pytest -m "not hardware"` and `pytest -m hardware`; commit `libarc_native_sys.dll` and `libarc_native_sys.dll.build.json`.
2. `python scripts/probe_hardware.py --out probe_new.json` (the option is delivered by plan WP-02 item 02f; until then `cmd /c "python scripts\probe_hardware.py > probe_new.json"`; never use `>` in Windows PowerShell 5.1, which writes UTF-16 and would overwrite the committed `probe.json`), review the diff against `probe.json`, then replace it and commit.
3. Run the benchmark matrix with at least 3 warmups and 10 timed runs per system, write schema-v1 files to `docs/benchmarks/results/`, run `python scripts/render_bench_docs.py`.
4. `python scripts/bench_regression.py --baseline 76b0f26 --workload llm --model-dir models\\qwen-2.5-1.5b --device GPU --runs 30 --warmup 3` (not `c7312e0`: it defaults to u8, which this GPU rejects).
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
| 1.12.0 | 2026-10-30 | main session | PERF row: plan v9.7 (fix only), DL-095, the adoption rule across tiers and lengths (medium tier decides a GPU decode arm, both lengths decide an NPU arm, NPU decode no-harm cell), the M-09 bytes interval with the tie-embedding premise UNVERIFIED, `unload_path` and the hub delete call, `--evaluate` cell rule with confirmation calls and K, M-07 with `--out`, the bounded stop as run | round-10 reviews G1-G4 resolved | 2abe58f |
| 1.13.0 | 2026-10-30 | main session | PERF row: plan v9.8 (fix only), DL-096, the adoption procedure AD-1..AD-6 (one adoption per device and key, a second GPU winner bracketed against the first, CPU and NPU one per device, one 30-round repeat), the code baseline 1a4ccfe versus c3b69cb..1d0eb64 with done items marked, the measured u8 KV rejection (`m17_e` blocked), the owner's new LLM and SD-1.5 result files | round-11 final checks G1 and G4 resolved | 35b2dc3 |
| 1.14.0 | 2026-10-30 | main session | added 'Code since the plan', 'Measured on the maintainer's machine' and 'Handoff for the next (CLI) session'; PERF row says partly implemented; maintainer list updated | the maintainer continues in the CLI on real hardware | next revision |
