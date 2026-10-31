---
doc_id: DOC-R07-REVIEW
title: R07 Faithfulness audit of repository claims
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R07 Faithfulness audit of repository claims

Date 2026-10-29. Roles applied: `hallucination-detector` (claim extraction, per-claim verdicts, FactScore) and `context-faithfulness-engineer` (source-grounding, citation to code/test/result evidence). Scope: README, CHANGELOG, SRS, benchmarks README + METHOD, architecture, testing, deployment, RESUME, TRACEABILITY, DECISION_LOG, 7 STAGE.md, MASTER_PLAN section 21.8. Evidence base is the repository only; `source` = the files, Python recomputation, tests and harnesses I ran.

## 1. Method and honest limits
- **Skills.** Agent files read in full. Mandatory skills read only in part (first 100-150 lines each), not completely: `hallucination-detection-core` (1-150), `uncertainty-quantification-core` (1-100), `rag-faithfulness-core` (1-100), `context-faithfulness-core` (1-100), `self-consistency-core` (1-100). Proof lines: "faithfulness(g, S) = (1/|C(g)|) × Σ_{c∈C(g)} max_{s∈S} P(E|s,c)"; "ECE = Σ_{m=1}^{M} (|B_m|/n) × |acc(B_m) - conf(B_m)|"; "F = |{s_i : v(s_i, C) = Entailment}| / |S|"; "**Definition:** Faithful(g, D) = 1 iff ∀ claim c ∈ g: D ⊨ c — the source document D must entail every claim in the generated text g."; "CS(q) = max_c p_c". Not used: NLI models, SelfCheckGPT, semantic entropy, calibration (no model access; verdicts are by code/file/test evidence, which is stricter than NLI for numbers).
- **Deviation from agent rule** "must NOT suggest corrections": the task asked for suggested wording, so each defect has one; it is a proposal, not an applied edit. No existing file was edited.
- **Sampling.** Systematic, not random: every checkable atomic claim (number, count, named file/function/flag/route, "measured/verified/closed/fixed", FR/NFR reference, performance statement) in reading order of each in-scope file; prose adjectives skipped. Result: **239 claims** (>=120). Selection bias: claims cheap to check are over-represented, and the 21 benchmark-table rows are 21 separate claims. Documents carrying `Provenance status: UNVERIFIED` were not sampled except for banner accuracy (section 4); README tables under its own provenance notice were sampled only for the notice and internal consistency.
- **Verdicts.** SUPPORTED = evidence found and re-checked; UNSUPPORTED = no code/test/result evidence found after grep; CONTRADICTED = evidence says otherwise (includes internal contradictions); UNVERIFIABLE-HERE = needs Intel hardware, network/registry, mermaid-cli, or VCS history (this checkout has no `.git`). UNVERIFIABLE is excluded from FactScore.
- **Commands run** (venv python): `pytest --collect-only` (all / `-m "not hardware"` / `-m hardware`, `--ignore=tests/independent` because another reviewer is mid-edit there); `pytest -m "not hardware"` -> **878 passed, 8 skipped, 23 deselected, 41 s**; `-m hardware` -> 15 passed, 7 skipped, 1 failed (tests are not Intel-only, see 5); mmap harness built with ASan/UBSan -> `checks=1000078 failures=0`; L0 harness vs fake loader -> `checks=72 failures=0`; `check_ring_model.py` -> 750420 states / 3606900 transitions OK; `validate_bench_json.py` 3 files 0 invalid; `render_bench_docs.py --check` exit 0; `list_unverified_numbers.py` 6 files 168 rows; numpy recomputation of every row of the generated benchmark table; `evaluate_speedup_gate` run on the real LLM file; sha256 of the DLL vs sidecar; grep/AST existence checks of every named test, symbol, script, flag, route.

## 2. Scores
| Measure | Value |
|:--|--:|
| Claims sampled | 239 |
| SUPPORTED / UNSUPPORTED / CONTRADICTED / UNVERIFIABLE-HERE | 159 / 16 / 51 / 13 |
| **FactScore** = S / (S+U+C) | **0.704** (159 / 226) |
| Contradictions | **51** (14 of them are stale snapshot counts in STAGE/plan records, labelled "drift") |
| FactScore excluding drift-type contradictions | 0.750 (159 / 212) |
| FactScore excluding the 21 recomputed benchmark-table rows | 0.673 |
| Hallucination severity (agent scale) | **MEDIUM**: no invented numbers in generated tables (all 21 recomputed exactly), but several hand-written numbers and counts are false or stale |

| File | S | U | C | V |
|:--|--:|--:|--:|--:|
| README.md | 29 | 3 | 13 | 3 |
| CHANGELOG.md | 17 | 0 | 2 | 2 |
| SRS.md | 11 | 0 | 0 | 0 |
| docs/benchmarks/README.md | 24 | 2 | 5 | 0 |
| docs/benchmarks/METHOD.md | 16 | 5 | 2 | 1 |
| docs/architecture.md | 8 | 5 | 1 | 0 |
| docs/testing.md | 3 | 0 | 4 | 1 |
| docs/deployment.md | 9 | 1 | 3 | 1 |
| docs/delivery/RESUME.md | 3 | 0 | 3 | 1 |
| docs/delivery/TRACEABILITY.md | 5 | 0 | 3 | 0 |
| docs/delivery/DECISION_LOG.md | 15 | 0 | 0 | 0 |
| docs/delivery/stages/*/STAGE.md | 16 | 0 | 12 | 3 |
| docs/plans/MASTER_PLAN_v4.md | 3 | 0 | 3 | 1 |

## 3. UNSUPPORTED and CONTRADICTED claims (file:line, evidence, suggested wording)
Severity: HIGH = misleads a user or deployer; MED = wrong number/status in a headline place; LOW = stale or cosmetic. `drift` = true when written, stale now.

| ID | Where | Claim | Verdict | Sev | Evidence | Suggested wording |
|:--|:--|:--|:--|:--|:--|:--|
| D05 | docs/deployment.md:150,173 | liveness probe and HEALTHCHECK on /v1/health | CONTRADICTED | HIGH | no /v1/health route in src/arc_runner/server; Dockerfile:103 and docker-compose.yml:23 probe `/` (S-F STAGE says the same) | path: /  (or add the route) |
| R29 | README.md:42 | run 32B/72B on 1.24 GB free DRAM with zero-RAM OOM crashes | CONTRADICTED | HIGH | README:345, METHOD.md:59 and the JSON: only a synthetic 8-layer container ran; 'real 32B/72B weights not run' | 'Designed to stream 32B/72B weights through a ring buffer; no real 32B/72B run exists yet' |
| A03 | docs/architecture.md:47 | 43.4% power reduction (Proof M6) | UNSUPPORTED | MED | power is never measured for speculation (README:43, CHANGELOG:52); no result file | delete the percentage |
| A13 | docs/architecture.md:189 | radix prefix dedup cuts TTFT by up to 63.2% | UNSUPPORTED | MED | no result file, no test | delete or measure |
| B28 | docs/benchmarks/README.md:88 | evaluated: curated 25-model catalog + Qwen 32B/72B Flash-Stream | CONTRADICTED | MED | measured files cover Qwen2.5-1.5B, Whisper-tiny, Qwen2-VL-7B; no 32B/72B run exists (METHOD.md:59) | 'Measured so far: Qwen2.5-1.5B, Whisper-tiny, Qwen2-VL-7B' |
| B30 | docs/benchmarks/README.md:76 | '30.8x TTFT' head-to-head | CONTRADICTED | MED | see R43 | see R43 |
| D03 | docs/deployment.md:49 | arc-runner server start --host 0.0.0.0 --port 8000 | CONTRADICTED | MED | no `server` subcommand; the command is `serve` (main.py:1559) | arc-runner serve --host 0.0.0.0 --port 8000 |
| E01 | docs/delivery/RESUME.md:22 | S-B: Windows DLL rebuild pending (local) | CONTRADICTED | MED | see R17 | 'DLL rebuilt (ABI 3, sidecar sha matches)' |
| E02 | docs/delivery/RESUME.md:24-27 | S-E: 'no result files yet' | CONTRADICTED | MED | docs/benchmarks/results has 3 files; probe.json committed | 'LLM, audio STT, VLM results committed; embeddings, image, video, Ollama baseline pending' |
| E03 | docs/delivery/RESUME.md:33-34 | maintainer to-do 1 (rebuild DLL) and 2 (probe.json) | CONTRADICTED | MED | both done (see R17, probe.json) | mark done |
| G18 | stages/S-E:16,42 | no result file exists yet; probe/matrix to run | CONTRADICTED (drift) | MED | probe.json and 3 result files exist | update after the local run |
| H08 | CHANGELOG.md:21 | removed the legacy result JSONs holding formula-derived numbers | CONTRADICTED | MED | scratch/live_real_models_benchmark_results.json remains with 6 legacy_unverified arc-runner rows, all power_w = 22.0 (the constant CHANGELOG:52 says was removed) | 'legacy rows are labelled legacy_unverified; remaining file scratch/live_real_models_benchmark_results.json'; or strip them |
| H18 | CHANGELOG.md:55 | 'Needs a real machine (not done)': rebuild DLL, probe, benchmark matrix | CONTRADICTED | MED | DLL rebuilt (see R17); probe.json committed; 3 schema-v1 result files exist (LLM, audio STT, VLM) | move DLL and probe to Done; list embeddings/image/video, Ollama baseline, regression, TLC, @hardware as open |
| M06 | docs/benchmarks/METHOD.md:28 | environment recorded: driver versions, AC power, Ollama version | UNSUPPORTED | MED | all 3 result files: driver_versions {}, on_ac_power null, ollama_version null; AC state is a precondition of the protocol | populate the fields or say 'not captured' |
| M07 | docs/benchmarks/METHOD.md:29 | idle CPU < 10% for 30 s precondition; violating run flagged noisy | UNSUPPORTED | MED | grep src/scratch/scripts: nothing sets BenchSample.noisy or checks CPU idle; all samples noisy=false | 'noisy is set by the operator; no idle-CPU check is implemented' |
| M09 | docs/benchmarks/METHOD.md:31 | warmups until last 3 TTFT CV <= 5% (max 5) | CONTRADICTED | MED | exactly 3 warmups per group; stats.warmup_converged is never called; warm-up TTFT CV is 133% (GPU low) ... 6.4% (NPU hard), 8 of 9 groups above 5% | '3 fixed warmups discarded' |
| M11 | docs/benchmarks/METHOD.md:31 | systems interleaved per iteration in seeded random order; 20 s cooldown | CONTRADICTED | MED | result files: all GPU samples, then NPU, then CPU (LLM) / GPU then CPU (audio, VLM); no cooldown code | 'systems are run in sequential blocks'; thermal/boost drift can bias GPU vs CPU |
| M16 | docs/benchmarks/METHOD.md:38 | no-regression gate also requires resident memory growth <= 5% | UNSUPPORTED | MED | evaluate_noninferiority_gate (stats.py:346-382) has no memory term; bench_regression.py has none either | 'decode TPS and TTFT only; memory check not implemented' |
| P04 | docs/plans/MASTER_PLAN_v4.md:1074 | open: probe, benchmark matrix, DLL rebuild | CONTRADICTED (drift) | MED | probe.json, 3 result files, rebuilt DLL present | update |
| R01 | README.md:7 | badge: 569 tests passing (100%) | CONTRADICTED | MED | collect-only: 909 tests/58 modules; -m 'not hardware': 886 collected, 878 passed, 8 skipped, 0 failed | Tests-886%20collected%20(878%20passed%2C%208%20skipped)  or generate the badge from a run |
| R02 | README.md:29 | 741 non-hardware tests (733 passed, 8 skipped, 0 failures) | CONTRADICTED | MED | same run: 886 / 878 passed; only '8 skipped' and '0 failures' hold | 886 non-hardware tests (878 passed, 8 skipped, 0 failures on Linux, <date>) |
| R03 | README.md:373-375 | 533 automated tests across 33 modules, 100% passing | CONTRADICTED | MED | 58 modules, 909 tests collected | state the count measured by `pytest --collect-only` or drop the number |
| R04 | README.md:383 | '533 passed in ~24.5s' | CONTRADICTED | MED | 878 passed, 8 skipped in 41.4 s (this VM) | '# -> 878 passed, 8 skipped (Linux, 41 s)' |
| R17 | README.md:306 | prebuilt DLL predates ABI 3 and is refused | CONTRADICTED | MED | src/arc_runner/native/libarc_native_sys.dll sha256 68d64d75... equals the sidecar; sidecar abi_version 3, built on Windows-11-10.0.26220 with MinGW gcc 15.2 | 'The committed libarc_native_sys.dll is ABI 3 and carries a matching sidecar (rebuilt on Windows); rebuild after any native change' |
| R28 | README.md:362-363 | '(84.1 Spec)' / '(68.5 Spec)' speculative tok/s | CONTRADICTED | MED | README:43 says speculative speed-up/acceptance were never measured and the earlier figures were removed; speculative engine reports 0 / 1.0 when unmeasured (DL-036) | delete the Spec figures |
| R31 | README.md:59 | 8.3 GB peak VRAM (studio) | UNSUPPORTED | MED | no constant, result file or test; CHANGELOG:15 says fixed memory figures were removed from the studio pipeline | remove or label projected |
| R32 | README.md:59 | Level Zero USM zero-copy bridge <= 50 us | UNSUPPORTED | MED | no measurement; USM path has no consumer (architecture.md:104) | remove or label projected |
| R43 | README.md:26 | empirical_silicon.md: 'Arc GPU vs CPU (30.8x TTFT)' | CONTRADICTED | MED | measured medians (qwen-2.5-1.5b, same prompt): CPU/GPU TTFT 17.7x low, 2.9x medium, 2.3x hard; speed-up gate fails (CPU CV > 10%) | 'banner says the 30.8x figure is not reproduced; measured ratios 2.3x-17.7x, not gate-passing' |
| T01 | docs/testing.md:9 | 741 tests (733 passed, 8 skipped, 0 failures) | CONTRADICTED | MED | see R02 | 886 / 878 / 8 / 0 |
| T02 | docs/testing.md:13,53 | 569 tests / 36 modules | CONTRADICTED | MED | 909 collected in 58 modules; matrix rows sum to 569 but omit 22 modules | regenerate the matrix from collect-only |
| A04 | docs/architecture.md:63 | Level Zero immediate commands tau <= 2.1 us | UNSUPPORTED | LOW | no measurement or constant | delete |
| A05 | docs/architecture.md:45 | package arc_runner.engines | CONTRADICTED | LOW | package is arc_runner.engine | arc_runner.engine |
| A10 | docs/architecture.md:151 | LoRA swap overhead < 0.20 ms | UNSUPPORTED | LOW | no measurement; lora tests are functional | delete or measure |
| A11 | docs/architecture.md:152 | K_max = 8 implies <= 512 MB | UNSUPPORTED | LOW | K_max=8 exists (lora_manager DEFAULT_MAX_CACHED) but the 512 MB bound has no source (adapter sizes vary) | 'at most 8 cached adapters' |
| B24 | docs/benchmarks/README.md:83-84 | 14 cores; 7 Xe-cores, XMX/DPAS engines, GPU_USM_MEMORY | UNSUPPORTED | LOW | probe.json cpu section unavailable and has no Xe-core field; comparative_matrix.md:88 says '8 Xe Cores', legacy JSON says 7; empirical_silicon banner says XMX is unverified | drop or fill from probe output |
| B25 | docs/benchmarks/README.md:86 | Windows 11 AMD64 build 26100 | CONTRADICTED | LOW | probe.json platform and result environments: Windows-11-10.0.26220 | build 26220 |
| B27 | docs/benchmarks/README.md:87 | Intel Graphics Driver 32.0.101.6127 / Level Zero 1.3 | UNSUPPORTED | LOW | every result file has driver_versions = {}; probe.json has no driver version | record driver version in the result environment, then quote it |
| B29 | docs/benchmarks/README.md:75 | 76 formal proofs (M59-M134) | CONTRADICTED | LOW | 70 distinct proofs (README.md:25 says 70) | 70 formal proofs |
| B31 | docs/benchmarks/README.md:16-18 | Source column 'measured' for decode tok/s | CONTRADICTED | LOW | METHOD.md:18 defines decode TPS as derived from counters; benchmark/nonllm.py:299 labels it measured | label decode_tps 'derived' or amend METHOD.md |
| D02 | docs/deployment.md:45 | arc-runner devices discover | CONTRADICTED | LOW | cli/main.py:1556-1557: `devices` has only --json, no subcommand | arc-runner devices |
| D14 | docs/deployment.md:172 | cgroup ceiling with P(OOM) = 0 margin | UNSUPPORTED | LOW | no mechanism or test shows zero OOM probability | 'memory limit set in the pod spec' |
| G02 | stages/S-A:22 | test_honesty_no_fabrication.py has 13 tests | CONTRADICTED (drift) | LOW | 15 collected now | '13 at slice end, 15 now' |
| G05 | stages/S-B:45 | test_gguf_untrusted_input.py 9 tests | CONTRADICTED (drift) | LOW | 10 collected | 10 |
| G07 | stages/S-B:79 | open item: rebuild/commit Windows DLL | CONTRADICTED (drift) | LOW | done (see R17) | closed |
| G09 | stages/S-C:45 | 25 new tests in test_flash_stream_ring.py | CONTRADICTED (drift) | LOW | 32 collected (pin tests added in S-G) | 25 at slice end |
| G13 | stages/S-D:53 | L0 harness: 66 checks, 0 failures | CONTRADICTED (drift) | LOW | re-run: checks=72 failures=0 | 72 |
| G14 | stages/S-D:56 | test counts 13 / 10 / 5 (L0 bridge, context+KV, telemetry details) | CONTRADICTED (drift) | LOW | 13 / 13 / 8 now | update |
| G15 | stages/S-D:72,74 | open: speculative engine lacks context check; zesInit/power/frequency not implemented | CONTRADICTED (drift) | LOW | ContextBudgetMixin (S-G) and sysman.py power/frequency/zes preinit implemented | closed in S-G |
| G17 | stages/S-E:38 | 40 new tests (stats 24, results 11) | CONTRADICTED (drift) | LOW | test_benchmark_stats 28, test_bench_results 26 | update |
| G20 | stages/S-E:46 | multi-domain scripts still write legacy JSON | CONTRADICTED (drift) | LOW | run_live_*_benchmarks.py use nonllm schema-v1 runners; audio and VLM result files exist | closed in S-G |
| G21 | stages/S-F:33 | 741 tests (733 passed, 8 skipped); docs-sync checks exit 0 | CONTRADICTED (drift) | LOW | 886 / 878 now | snapshot; mark as of S-F |
| G24 | stages/S-F:26 | DECISION_LOG DL-001..053 | CONTRADICTED (drift) | LOW | DL-001..060 | 060 |
| M10 | docs/benchmarks/METHOD.md:31 | timed runs extend up to 30 until CI half-width <= 3% | UNSUPPORTED | LOW | no extension logic; every group has exactly 10 timed samples | '10 timed runs' |
| M13 | docs/benchmarks/METHOD.md:32 | row flagged non-normal when t and bootstrap intervals differ > 5% | UNSUPPORTED | LOW | no such flag in stats.py, results.py or render_bench_docs.py | remove or implement |
| P02 | docs/plans/MASTER_PLAN_v4.md:1072 | L1-L12 'closed in code' | CONTRADICTED | LOW | L12 USM path has no consumer (TRACEABILITY:43, architecture.md:104); L11 'documentation only' | 'L1-L11 closed; L12 opt-in path, no consumer' |
| P03 | docs/plans/MASTER_PLAN_v4.md:1073 | benchmarks B1-B6 'harness done, not run' | CONTRADICTED (drift) | LOW | LLM, audio STT, VLM runs exist (3 result files) | B1 LLM / B4 audio / B3 VLM run; others pending |
| R11 | README.md:287 | lock_range 'not yet called by the streaming path' | CONTRADICTED | LOW | weight_streamer.py:370-386 `_pin_layer` calls native.lock_range (opt-in pin_resident_layers / ARC_PIN_RING=1, CHANGELOG [UNRELEASED]) | 'Called by the streaming path only when pin_resident_layers / ARC_PIN_RING=1 is set; effect not measured' |
| R27 | README.md:365 | Whisper-Base 5.5x: Ollama CPU 1.944 s vs iGPU 0.354 s | CONTRADICTED | LOW | RTF 0.486 implies a 4.0 s clip, 28.2x RT at 0.354 s implies a 10.0 s clip; comparative_matrix.md:96-100 shows 1.944 s is the 4.0 s Low tier and 0.354 s the 10.0 s Medium tier (same-tier ratios 6.61x / 4.31x) | compare same-tier rows or drop the cell |
| R34 | README.md:22,222,369 | 25 curated models | CONTRADICTED | LOW | ModelRegistry._MODELS has 34 entries (11 llm, 10 image, 5 stt, 5 video, 2 tts, 1 vision) | '34 registry entries' (generate from the registry) |
| R38 | README.md:198 | arc-runner benchmark ... --runs 5 | CONTRADICTED | LOW | cli/main.py:1582 defines --iterations (default 5); no --runs | arc-runner benchmark llama-3.2-3b --device GPU --iterations 5 |
| R39 | README.md:219 | arc-runner embed "text" --model nomic-embed-text --device NPU | CONTRADICTED | LOW | cli/main.py:1603-1607: model and text are positional (model first), no --model flag | arc-runner embed nomic-embed-text "What is Intel Arc GPU?" --device NPU |
| R44 | README.md:26 | Flash-Stream memory trace (dM <= 1 MB) listed under 'Real Silicon Hardware Telemetry' | UNSUPPORTED | LOW | empirical_silicon banner: mapped bytes, synthetic container | list it as synthetic-container trace |
| T03 | docs/testing.md:15-52 | per-module counts (36 rows) | CONTRADICTED | LOW | 33 rows match; test_image_engine 24 vs 25, test_video_engine 24 vs 25, test_multimodal_orchestrator 5 vs 7 | regenerate |
| T04 | docs/testing.md:62 | 733 passed, 8 skipped in ~40 s | CONTRADICTED | LOW | 878 passed; '8 skipped' and ~41 s hold | 878 passed |
| X04 | docs/delivery/TRACEABILITY.md:28 | C10 test test_compile_library_produces_loadable_abi3_library_with_matching_sidecar | CONTRADICTED | LOW | tests/test_native_bounds.py:283 is named ..._abi2_library_... | cite the real name (or rename the test; it asserts ABI 3) |
| X05 | docs/delivery/TRACEABILITY.md:25 | C7 'partial (not wired into the streamer)' | CONTRADICTED | LOW | row 64 of the same file says wired; weight_streamer._pin_layer calls lock_range; symbol `try_pin` (row 64) does not exist in src | closed (opt-in); cite _pin_layer |
| X06 | docs/delivery/TRACEABILITY.md:42 | L11 zesInit order 'n/a, documentation only' | CONTRADICTED | LOW | devices/sysman.py:158-183 implements zesInit-before-zeInit (zes preinit); probe.json shows the pre-fix 0x78000001 | 'fixed: zesInit called first; verified on real driver by the maintainer' |

## 4. UNVERIFIED-banner accuracy (documents known unverified)
**docs/benchmarks/comparative_matrix.md** banner (items 1-5) is accurate: item 5 holds (Llama-3.2-3B Ollama 1,420 ms / 11.4 tok/s vs stored JSON 2,238 ms / 14.7 tok/s); items 1-4 match README/JSON/CHANGELOG. It is **incomplete**: real measured rows (docs/benchmarks/results, same Core Ultra 5 125H) now contradict numbers it does not list. Matrix TTFT uses a 512-token prompt, measured prompts are 20-42 tokens, so TTFT is only indicative; decode and audio rows are comparable:
- :32 NPU Qwen2.5-1.5B 10.3 tok/s vs measured NPU 14.4 / 14.7 / 15.4 tok/s (CI up to +-1.1). :33 iGPU 54.8 tok/s vs measured 40.1 / 41.4 / 52.2 (all three below). :31 Ollama CPU 17.0 tok/s / 920 ms: no Ollama Qwen run exists (JSON status UNAVAILABLE, HTTP 404); measured OpenVINO-CPU 13.1-13.7 tok/s.
- :49 DeepSeek-R1-1.5B on NPU 41.3 tok/s (1.5B model, same NPU) vs measured 14.4-15.4 for the Qwen 1.5B; it also disagrees with the matrix's own :32 (10.3).
- :96-97 whisper-tiny iGPU Medium 0.155 s (54.69x RT), CPU 0.886 s, 5.70x; Hard 0.225 s (80.0x RT) vs measured Medium 0.080 s (106x RT), CPU 0.308 s, 3.85x; Hard 0.0882 s (204x RT), CPU 0.331 s. :91 (and :116,:144,:169) still say "Empirically measured ... Zero synthetic data" and "8 Xe Cores" (hub README says 7).
**docs/benchmarks/empirical_silicon.md** banner items 1-4 are accurate (the 30.8x TTFT is not reproduced: measured CPU/GPU TTFT is 17.7x low, 2.9x medium, 2.3x hard, and the speed-up gate fails on CPU CV > 10%). Not covered by the banner: :31 GPU TTFT 139 ms vs measured 20.1 / 25.0 / 73.7 ms; :138 GPU 154.6-162.9 ms; :140 GPU decode 27.5-32.0 tok/s vs measured 40.1-52.2; :154 NPU 10.26 tok/s vs measured 14.4-15.4; :138 NPU/GPU TTFT 14.7x vs measured 112x / 91x / 30x; NPU TTFT 2,394-2,475 ms is consistent only with the medium tier (2,271 +- 232 ms); :176-181 flash-stream rows (843 MB resident, 14.37 tok/s, "Ollama CRASH: OOM") vs JSON (4.12 MB working set, 8.51 tok/s, Ollama "not run"); the banner mentions only the memory trace. README:362-367 (provenance-noticed) has the same Llama-1B NPU/iGPU figures (890 ms, 19.8 tok/s, 72.4 tok/s); no Llama-3.2-1B result exists, measured Qwen-1.5B NPU is 2,233-2,271 ms TTFT.
`ollama_showdown.md`, `multimodal_studio.md`, `research/07` banners were only listed by `list_unverified_numbers.py` (6 files, 168 rows); their content was not re-audited.

## 5. Other observations
- **Hardware-marked tests are not silicon-only.** 15 of the 21 `test_real_hardware.py` tests pass on an x86 Linux VM with no Intel GPU/NPU (they use CPU); 5 skip, 1 fails (`test_numerical_stability_cpu_vs_gpu_vs_npu`: OpenVINO RuntimeError at core.cpp:117, device absent). README:376 "physical silicon tests" and the CLAUDE.md "never counted as passed" rule are only safe if the marker is deselected, as CI does. Suggest: skip, not pass, when no GPU/NPU; fix the failing test to skip.
- **Delivery gate currently red** (`check_docs_sync.py`, `check_delivery_docs.py` exit 1): `docs/delivery/reviews/R04_secrets_and_dependencies.md` lacks front matter. Every review report under `docs/delivery/reviews/` (this one too: no front matter by instruction) trips it. Suggest: exempt `reviews/` or add front matter at merge.
- **Reproduced real results are honest**: all 21 table rows recompute to printed precision; sources `measured`, `fallback_used` false, 0 noisy. But the claim gates do not pass for GPU vs CPU on any tier (CPU CV > 10%); GPU vs NPU passes 4 of 6; no Ollama result exists, so H-A/H-B (defined against Ollama) are still open.
- Result files cover 3 of 6 workloads (no embeddings, image, video); LLM tier "low" generates only 18-20 tokens, so decode tok/s there rests on ~17 decode tokens.
- Stale-by-design snapshots: stage records keep slice-time counts (13/9/25/66/741 ...). Suggest a "as of" line per record or generated counts.

## 6. SUPPORTED ledger (evidence in the claims table of the generator; one line per file)
- **README.md** (29): R05 collect: 33 / 38 / 42; R06 tests/integration/test_real_hardware.py: 21 collected; R08 built tests/native_c/mmap_harness.c with -fsanitize=address,undefined ; R09 test_windows_branch_compiles_clean_with_werror ran and passed (not in ; R10 test_native_views_are_read_only, test_close_with_live_view_defers_unma; R12 scripts/check_ring_model.py -> 750420 states, 3606900 transitions, OK; R14 openvino_engine.py:42,145,163; R15 openvino_engine.py:156,160; R16 stats.py is_within_roofline (249) and evaluate_speedup_gate SUSPECT br; R18 sidecar JSON of dll and so; R19 bridge.py:33; R20 scratch/live_real_models_benchmark_results.json 14.69 / 2237.79; R21 JSON 17.64 / 70.01; R22 JSON 17.43 / 959.14; R23 JSON: arc rows source legacy_unverified, Ollama status UNAVAILABLE (40; R24 JSON model_kind synthetic_container_8_layers_hidden_dim_32; R25 no result file contains these values; R26 arithmetic re-derived; R30 speculative_engine.py:39-40; R33 orchestrator/muxer.py:39 convergence_tolerance_sec=0.05; R35 mathematical_proofs.md has 70 distinct 'Proof Mxx' (M59-M114, M121-M13; R36 pyproject classifiers and requires-python >=3.10; R37 cli/main.py:1559-1565; R40 subcommands list has no `run` (only run-heavy); R41 argparse definitions 1648-1712; R42 routes_ollama.py:102,259,358; R45 test_nextgen_orchestration_triad.py: 42 collected; R46 dist/ has the 0.1.0 and 0.2.0 wheels; R48 scripted link check of README, benchmarks README, architecture, testin
- **CHANGELOG.md** (17): H01 engine/context_budget.py; H02 weight_streamer.py:34,135,370; H03 benchmark/power.py:89-97; H04 files exist; H05 kv_quality.py, scripts exist; H07 file absent; H09 header ABI 3u; H10 symbols and files found; H11 engine/models.py:78-91; H12 all five exist; H13 arc_level_zero_bridge.h:33-34; H14 build_native.py:23-24; H15 student_t_critical_95: 11->2.201, 15->2.131, 19->2.093 (published valu; H16 gguf_parser.py _read_declared; H17 grep src/scripts/scratch: no `return 22.0`, no REFERENCE_SWEEP; H19 app.py:7,94; H21 .github/workflows/ci.yml jobs linux, windows, diagrams, redfirst
- **SRS.md** (11): Q01 all three found; Q02 test_honesty_no_fabrication, test_telemetry_details, test_level_zero_b; Q03 test_engine_context_and_kv.py (13 tests pass); Q04 stats.py:15-16, 302-343; Q05 test_flash_stream_ring.py 32 tests pass; Q06 header constants; Q07 gguf_parser.py:20,275; Q08 test_native_and_python_paths_return_identical_tensor_bytes passes; Q09 render_bench_docs.py --check exit 0; Q10 sidecar flags; Q11 VERSION, _version.py, pyproject.toml
- **docs/benchmarks/README.md** (24): B01 recomputed from the result file: all metrics equal to printed precisio; B02 recomputed from the result file: all metrics equal to printed precisio; B03 recomputed from the result file: all metrics equal to printed precisio; B04 recomputed from the result file: all metrics equal to printed precisio; B05 recomputed from the result file: all metrics equal to printed precisio; B06 recomputed from the result file: all metrics equal to printed precisio; B07 recomputed from the result file: all metrics equal to printed precisio; B08 recomputed from the result file: all metrics equal to printed precisio; B09 recomputed from the result file: all metrics equal to printed precisio; B10 recomputed from the result file: all metrics equal to printed precisio; B11 recomputed from the result file: all metrics equal to printed precisio; B12 recomputed from the result file: all metrics equal to printed precisio; B13 recomputed from the result file: all metrics equal to printed precisio; B14 recomputed from the result file: all metrics equal to printed precisio; B15 recomputed from the result file: all metrics equal to printed precisio; B16 recomputed from the result file: all metrics equal to printed precisio; B17 recomputed from the result file: all metrics equal to printed precisio; B18 recomputed from the result file: all metrics equal to printed precisio; B19 recomputed from the result file: all metrics equal to printed precisio; B20 recomputed from the result file: all metrics equal to printed precisio; B21 recomputed from the result file: all metrics equal to printed precisio; B22 --check exit 0; B23 probe.json openvino_devices + logical_cpus 18; B26 scratch JSON testbed.ram_total_gb 15.43 (not in probe.json, which has 
- **docs/benchmarks/METHOD.md** (16): M01 stats.SampleSource Literal; M02 probe.json accessor list; M03 telemetry.py _decode_rate; M04 prompt_tokens from get_num_input_tokens; M08 prompt_tokens identical across backends per tier (20/35/42); M12 README table medians recomputed on all 10 non-warmup samples; M14 both scripts exit 0; M15 evaluate_speedup_gate; M17 kv_quality.py:18-19; M18 power.py:94-97; M19 probe.json free_mb null, total_mb 6976; M20 run_live_image/video_benchmarks.py:383,422 / 418,457; M21 vision_engine.py:208-229; M22 run_live_audio_benchmarks.py:6,220; M23 0.08822/18 = 0.00490 (file 0.0049); M24 test_flash_stream_engine_telemetry_is_labelled_synthetic passes
- **docs/architecture.md** (8): A01 all 20 class names found by grep; A02 speculative_engine.py:57; A06 weight_streamer.py SlotState; A07 tests + check_ring_model; A08 probe.json; A09 device_stager has no consumer; A12 continuous_batcher.py:69, cli/main.py:836; A14 13 files each
- **docs/testing.md** (3): T06 both harnesses built and run clean; T07 ci.yml:108-123 (workflow_dispatch); T08 validate/render exit 0
- **docs/deployment.md** (9): D01 pyproject.toml:74-75; D04 main.py:1548; D06 Dockerfile:10,41,71-75,83-88; D07 Dockerfile:86-88; D08 Dockerfile:100; D09 index.html: 7 tab-pane sections; D10 header 1900u; D11 docker-entrypoint.sh:18; D12 sysman.py:199-200
- **docs/delivery/RESUME.md** (3): E04 VERSION etc.; E06 headings 17..22 exist; E07 stage records and tests agree
- **docs/delivery/TRACEABILITY.md** (5): X01 all symbols and test names found; X02 symbols and tests found; X03 symbols/tests found; X07 device_stager has no consumer; X08 all right-hand names exist
- **docs/delivery/DECISION_LOG.md** (15): L01 build_native.py has no mavx2; L02 VERSION; L03 GGUF_MAX_DIMENSIONS=4; L04 <lib>.build.json files; L05 weight_streamer.py:526; L06 bridge.py:93,105; L07 header; L08 openvino_engine.py; L09 file exists; L10 file absent; L11 models.py:76-77; L12 scratch JSON model_kind; L13 ContextWindowExceededError tests; L14 ci.yml diagrams job; L15 13/13 stamps
- **docs/delivery/stages/*/STAGE.md** (16): G03 re-run: checks=1000078 failures=0; G04 41 collected; G06 header _Static_assert and limits; G08 re-run identical; G10 SLOT_WAIT_TIMEOUT_SECONDS = 30.0; G11 arc_ze_types.h:24-33 equal the table (values themselves are external t; G12 header/source constants (1<<36 = 64 GiB); G16 2.201 is t(11); G22 ci.yml; G23 grep + Dockerfile:103; G25 weight_streamer.py:179,379,429; G26 55 in uml/*.md, architecture, diagrams README, research/*.md; G28 list_unverified_numbers.py: 6 files, 168 rows; G29 878 passed, 8 skipped (existing venv, tests/independent excluded); G30 22 in audit table + 2 same-name tests; G31 ci.yml:3-7 (plus workflow_dispatch)
- **docs/plans/MASTER_PLAN_v4.md** (3): P01 run above; P05 test_honesty gate passes (15 tests); P07 openvino_engine opt-in

## 7. UNVERIFIABLE-HERE (excluded from FactScore)
- R07 README.md:376,26 those 21 run on live Intel silicon and pass -- needs Intel hardware; on this x86 Linux VM the same 21 give 15 passed, 5 skipped (no GPU/NPU), 1 failed (test_numerical_stability_cpu_vs_gpu_vs_npu, OpenVINO RuntimeError)
- R13 README.md:289 Level Zero USM/DMA not used by streaming path, not run against a real driver -- first half true (no consumer for DeviceLayerStager); real-driver run needs the Windows machine (probe.json shows real L0 enumeration only)
- R47 README.md:394,8 wheel published on PyPI / container on GHCR -- needs network and registry; CHANGELOG:25 says 0.2.0 not published
- H06 CHANGELOG.md:16 four Mermaid diagrams + state generator fixed so they parse -- needs mermaid-cli (mmdc absent here)
- H20 CHANGELOG.md:25 0.2.0 not published -- needs registry access
- M05 docs/benchmarks/METHOD.md:24 get_throughput = 1000/mean TPOT in GenAI 2026.4.1 (perf_metrics.cpp) -- version 2026.4.1.0 confirmed by probe.json; the C++ source is not in this checkout/offline
- T05 docs/testing.md:68 '21 passed in 4.82 s on physical Intel Core Ultra silicon' -- needs the hardware; here 15 pass / 5 skip / 1 fail on CPU-only Linux, so 15 of the 21 do not need an Intel GPU/NPU
- D13 docs/deployment.md:171 Proof M117: >= 59.9% footprint reduction -- needs a Docker build of both images
- E05 docs/delivery/RESUME.md:29 nothing tagged or published -- no VCS metadata in this checkout
- G01 stages/S-A:16 13 failing tests at c7312e0; commits 23c2c60, 3102488 -- no VCS metadata here, cannot verify
- G19 stages/S-E:40 flash_acquire ~280x faster than old ring -- needs the baseline commit (no VCS); stage itself calls it an artefact
- G27 stages/S-G:28 all blocks render with mermaid-cli 10.9.1 -- mmdc not installed here
- P06 docs/plans/MASTER_PLAN_v4.md:1075 five legacy result JSONs removed -- no VCS; one legacy JSON still present (H08)

Generator and claim table: `/tmp/claude-0/-home-user/9f5d9574-39ea-55de-8475-bf2e506385a2/scratchpad/gen_r07.py` (not part of the repo).

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
