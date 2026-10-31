---
doc_id: DOC-R08-REVIEW
title: R08 Benchmark methodology review
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R08 Benchmark methodology review

Reviewer roles: llm-benchmark-analyst + harness-evaluation-engineer. Scope: METHOD.md, `benchmark/{stats,results,workloads,nonllm,power,harness_common,kv_quality,runner,multimodal_runner}.py`, `scripts/bench_{regression,native_micro,kv_quality}.py`, `scripts/render_bench_docs.py`, `scratch/run_live_*`, and the three real result files. Everything below was read in code or reproduced; items I could not verify are marked SUSPECTED.

## 0. Skills and agents (proof of reading)

Read (first chunk only, not the M1-M6 derivation sections, which I did not need):
- llm-benchmarking-core, lines 1-120: "- **Temperature**: 0.0 for deterministic evaluation (HumanEval pass@1, MMLU); only use temperature > 0 for self-consistency sampling."
- model-capability-profiling-core, lines 1-60: "- **HumanEval**: 164 Python problems, greedy decoding (T=0), pass@1 strictly."
- context-decay-analysis-core, lines 1-60: "Effective_Context_Length(M) = max L such that score(M, L) >= threshold" (the file has the unicode >= sign).
- eval-harness-design-core, lines 1-120: "A bare pass-rate (`83%`) is uninterpretable without a sample size and an interval."
- deterministic-replay-trace-core, lines 1-80: "1. **Determinism** — the same recorded run replays the same way every time, so a test asserts against a fixed baseline instead of a fresh stochastic sample."
- agent-regression-harness-core, lines 1-80: "### 1.2 The Reframing: Pass-Rate, Not Pass/Fail"

Not read: all optional skills; remaining lines of all six. The agents delegate statistics to `genai-routing-mathematician` / `harness-mathematics-expert`; I did not invoke them. Every statistic below was computed with the repo functions and re-computed independently (numpy bootstrap with 20000 resamples and a different seed; a delta-method log-ratio interval with a Welch t quantile that I integrated numerically). Treat that as my own check, not a delegated proof.

## 1. What the three result files are

All three: Core Ultra 5 125H, Windows 11, OpenVINO GenAI 2026.4.1.0, git 9cfb3ab, 3 warmups + 10 timed per (tier, backend), all rows `source=measured`, `fallback_used=false`, `noisy=false`, no `failures`. `scripts/validate_bench_json.py`: 3 valid. `tests/test_benchmark_stats.py`: 28 passed. `render_bench_docs.py --check`: clean.

Comparison identity: **arc-runner = OpenVINO GenAI on the iGPU; cpu-baseline = the same OpenVINO GenAI pipeline, same model directory, on the CPU**. There is no Ollama row in any schema-v1 file. The only Ollama numbers in the repo are in `scratch/live_real_models_benchmark_results.json`: llama-3.2-3b, n=1 per tier, 14.69/17.64/17.43 tok/s, with the arc side UNAVAILABLE in that same file, and the CPU/"fallback" label is hard-coded (see F9). They support nothing. **No statement in this report is a statement about Ollama.** H-A and H-B are undecidable today; H-C and H-D have never been run (no regression result file, no `docs/benchmarks/quality/`).

## 2. Speed-up arc-runner(GPU) vs OpenVINO-CPU, qwen-2.5-1.5b, 95% CIs

Ratio = mean(arc)/mean(cpu) for decode tok/s; mean(cpu)/mean(arc) for TTFT (as `evaluate_speedup_gate` defines it). CI = repo `bootstrap_ratio_ci` (percentile, 2000 resamples, seed 0). "Indep" = my numpy bootstrap (20000, seed 12345). "Delta" = log-ratio normal approximation with Welch df. All n=10 vs 10.

| Tier | Metric | arc mean (CV) | cpu mean (CV) | Ratio | Repo CI | Indep CI | Delta CI | Gate verdict (repo function) |
|:--|:--|:--|:--|--:|:--|:--|:--|:--|
| low | decode tok/s | 50.83 (5.3%) | 12.04 (27.2%) | 4.22 | 3.64-5.07 | 3.65-5.08 | 3.47-5.14 | FAIL: cpu CV 27.2% > 10% |
| low | TTFT ms | 20.46 (4.3%) | 411.6 (38.0%) | 20.11 | 16.43-25.40 | 16.36-25.22 | 15.31-26.42 | FAIL: cpu CV 38.0% > 10% |
| medium | decode tok/s | 41.11 (2.9%) | 12.62 (12.4%) | 3.26 | 3.04-3.52 | 3.05-3.54 | 2.98-3.56 | FAIL: cpu CV 12.4% > 10% |
| medium | TTFT ms | 25.37 (4.0%) | 72.05 (12.0%) | 2.84 | 2.63-3.05 | 2.64-3.05 | 2.60-3.10 | FAIL: cpu CV 12.0% > 10% |
| hard | decode tok/s | 40.18 (1.0%) | 13.81 (11.1%) | 2.91 | 2.73-3.11 | 2.73-3.11 | 2.69-3.15 | FAIL: cpu CV 11.1% > 10% |
| hard | TTFT ms | 74.39 (3.3%) | 173.4 (27.9%) | 2.33 | 1.96-2.71 | 1.96-2.73 | 1.91-2.85 | FAIL: cpu CV 27.9% > 10% |

Which conditions fail and why:
- Failing in all 6: CV <= 10% on the CPU side only. The arc side passes every condition.
- Passing in all 6: measured / fallback-free / not noisy; at least 10 timed values (n is exactly 10, the minimum); the ratio lower bound is above 1.0.
- Diagnostic only (not a claim): re-running the repo gate with `max_cv=1.0` returns passed=True for all six, with the same lower bounds as the Repo CI column. So the only thing standing between these data and a "speed-up claim" under METHOD.md is CPU-side noise, and the direction of the effect is not in doubt. The magnitude is in doubt, see F1-F3.
- Roofline: `roofline_tps` is never supplied by any caller, so the SUSPECT check did not run.
- Bootstrap vs delta: the percentile bootstrap lower bounds are 2-5% higher than the delta/t bounds (low decode 3.64 vs 3.47; hard TTFT 1.96 vs 1.91), consistent with F4.
- Robustness: median-ratio gives 3.89 / 17.7 (low), 3.16 / 2.88 (medium), 2.94 / 2.31 (hard) for decode / TTFT. Dropping the first 3 timed runs of both series moves the low-tier TTFT ratio 20.1 -> 16.5 and low decode 4.22 -> 3.86; the medium and hard ratios move by less than 3% (decode medium 3.26 -> 3.27, hard 2.91 -> 2.82; TTFT medium 2.84 -> 2.78, hard 2.33 -> 2.34).
- NPU (not requested, for completeness): decode 14.2-15.0 tok/s (~ CPU), TTFT 2.2-2.4 s, nearly independent of prompt length. NPU decode is not faster than the CPU baseline here.

VLM and audio (same files, no gate possible, see F5). Latency ratio cpu/arc, repo bootstrap CI, 10 vs 10:

| Workload/tier | arc mean s (CV) | cpu mean s (CV) | Ratio | Repo CI | Note |
|:--|:--|:--|--:|:--|:--|
| vlm low | 1.218 (3.0%) | 8.608 (11.9%) | 7.07 | 6.53-7.59 | 12 chunks both |
| vlm medium | 2.137 (2.4%) | 13.22 (17.7%) | 6.19 | 5.50-6.82 | 15 chunks both |
| vlm hard | 15.75 (19.3%) | 83.61 (19.6%) | 5.31 | 4.46-6.18 | arc chunks 95-172 vs cpu 96 (F6); per-chunk ratio 6.14 (5.41-6.95) |
| stt low | 0.061 (5.5%) | 0.289 (5.0%) | 4.72 | 4.52-4.94 | 3.0 s tone |
| stt medium | 0.080 (10.2%) | 0.316 (5.7%) | 3.94 | 3.68-4.21 | 8.5 s tone |
| stt hard | 0.088 (8.1%) | 0.335 (4.5%) | 3.80 | 3.60-4.01 | 18.0 s tone |

## 3. Statistical validity (verified)

- Student-t table (`stats.py:19-91`): I inverted the t CDF numerically; the table and the 1/df interpolation agree with the exact quantile to < 0.0004 for df 1..120. No defect for the n <= 30 the protocol allows.
- `student_t_critical_95` for df > 120 returns 1.960 (`stats.py:91`): true t(121)=1.980, t(200)=1.972, t(1000)=1.962. Anti-conservative by up to 1.0%. Unreachable under the protocol (max 30 runs). LOW.
- Percentile bootstrap at n=10: my simulation (lognormal data, 1500 reps, MC error about +-0.6 points) gives two-sided coverage of the nominal 95% interval of 0.90-0.93, and a lower bound that is valid with probability 0.955-0.977 versus the nominal 0.975. So the gate's false-pass rate at the boundary can be about 4.5% instead of 2.5%. Harmless for ratios of 2-20, relevant for H-C (3% margin).
- Independence: `bootstrap_ratio_ci` resamples each system independently, i.i.d. Time-ordered series are not i.i.d.: lag-1 autocorrelation +0.60 (GPU low decode; Spearman vs iteration -0.85, -10% first-vs-last-three), +0.49 (CPU low decode), +0.54 (GPU low TTFT). n=10 makes these estimates noisy (SUSPECTED, not proven) but the monotone GPU decay and CPU hump are visible by eye in section 2's raw series. Effective n is below 10, so the true CIs are wider than reported.
- Gate reports only the first failing reason (`stats.py:331-334`: arc checked before ref, then stops). Fine as a gate, poor as diagnostics.
- CV gate on raw values (no detrending, no outlier policy by design). It correctly caught the CPU non-stationarity. Gate semantics are sound; the problem is that nothing in the pipeline makes a CPU series stationary (F1).
- No multiplicity control: H-A is "for each model in the matrix". k independent 95% tests give family-wise error roughly 1-(0.975)^k one-sided. Irrelevant at ratios 2-20, relevant for H-C.

## 4. Findings (severity, file:line, failure scenario)

**F1 HIGH. Warmup rule of METHOD.md is not implemented; the CPU baseline is timed before steady state.** METHOD.md:31 requires warmups until the last 3 TTFT values have CV <= 5% (max 5). `warmup_converged` (`stats.py:236`) is called by nothing outside tests (grep). Scripts use a fixed 3 (`run_live_sequential_model_benchmarks.py:436`, `run_live_vlm_benchmarks.py:138`). Applying the function to the recorded warmups: converged=False for all 9 LLM series (CV of last 3 between 0.06 and 1.69). Concrete: CPU low-tier first call TTFT 14224 ms, then 634, 630; the first timed runs are 810, 504, 412 ms against a steady ~300 ms. First-call costs on GPU 221 vs 20 ms, NPU 3419 vs 2240 ms. Effect on the headline: low-tier TTFT speed-up 20.1x falls to 16.5x with the first 3 timed runs dropped (-18%). The bias inflates the speed-up (CPU slower in the timed window).

**F2 HIGH. Interleaving, cooldown and preconditions of METHOD.md:29,31 are not implemented anywhere in the live scripts.** `collect_samples` runs all tiers of backend 1, then backend 2, then CPU (`run_live_sequential_model_benchmarks.py:410-418`, `run_live_vlm_benchmarks.py:108-120`). No random order, no 20 s cooldown, no AC/idle-CPU check, `noisy` is never set to True by any script (grep: only the dataclass field and the gate read it), `on_ac_power` is null in all three files because `capture_environment` is never given it (`results.py:61`), `driver_versions` is `{}` (`results.py:79`) although METHOD.md:28 requires them. Threats this leaves open (all SUSPECTED as causes, the symptoms are verified): thermal throttling and power-state drift (GPU low-tier decode decays 53 -> 45 tok/s over the last 4 timed runs, TTFT 20.0 -> 22.4 ms; CPU low-tier decode 7 -> 15 -> 7 tok/s), CPU block measured last after tens of minutes of load on a shared-power SoC (direction of the bias unknown, probably unfavourable to CPU), unattested AC power. The `noisy` exclusion in the gate is therefore vacuous: it can never fire.

**F3 MED. CPU baseline is untuned and configured asymmetrically.** `openvino_engine.py:155-160` sets CACHE_DIR, PERFORMANCE_HINT=LATENCY and SDPA optimisation only for GPU; the CPU gets defaults on 18 logical cores (P/E hybrid, no thread count or pinning). The CPU CV of 11-38% is partly a scheduling artefact. The baseline is "OpenVINO default on CPU", which is a legitimate label only if printed as such. Also: LLM label is "OpenVINO on GPU" with no quantisation, VLM labels GPU "(INT4)" and CPU without it (both load the same INT4 directory).

**F4 MED. H-C is under-powered by design.** bench_regression defaults `--runs 12 --warmup 2` (`bench_regression.py:103-104`), and the 3% margin is the same size as the n=10-12 percentile-bootstrap error (section 3). It also resamples unpaired although iterations are interleaved pairs. Derived sample size: to prove a lower bound > 0.97 when the truth is no change needs half-width 1.96*sqrt(2)*cv/sqrt(n) <= 0.03, i.e. n >= (92.4*cv)^2: cv 3% -> 8, cv 5% -> 22, cv 10% -> 85.

**F5 MED. No claim gate exists for any non-LLM workload.** `_timed_values` reads `getattr(sample, metric)` (`stats.py:274`). On VLM samples `metric="decode_tps"` returns "arc has 0 timed values, need 10"; `metric="latency_s"` raises AttributeError (reproduced). So VLM/audio/image/video ratios cannot be gated, and `evaluate_speedup_gate` is not called by any script, the docs renderer or the CLI (grep): the METHOD.md claim gate is enforced by convention only. The README table prints no ratio, which is the safe state. Also `evaluate_speedup_gate` never checks that both systems ran the same prompt/max tokens/completion tokens.

**F6 MED. VLM output length varies and changes latency.** Hard tier, GPU, greedy: chunk counts 96,118,96,96,118,95,95,112,118,172 (CPU: 96 every run, and all 3 GPU warmups: 96). Correlation(latency, chunks) = 0.95. Raw latency ratio 5.31 understates the per-chunk ratio 6.14, so the error here is conservative, but the metric is length-confounded and run-to-run GPU output is not deterministic at "temperature 0". Cause note (verified in the installed wheel, 2026.4.1.0): `GenerationConfig().do_sample` defaults to False and no repository code sets `do_sample` (grep), so `temperature=0.0` is ignored and greedy comes from the default; the protocol field `temperature: 0.0` is nominal. Same-prompt, same-budget outputs also differ across devices in the LLM file (low tier: 18 completion tokens on GPU, 20 on CPU and NPU), so "same workload" means same prompt, not same output. Tiers also change image size and token budget together (224/448/896 px with 32/96/192 tokens), so vision-encoder time and decode time are not separable. `ignore_eos` and `min_new_tokens` exist in the wheel and are the way to force equal length.

**F7 MED. Cold start and compile time are not recorded in result files.** METHOD.md:31 says cold start is reported separately. `load_model` time (GPU/NPU compile) is not captured by the schema-v1 scripts (only the legacy script records `compile_time_s`). First-warmup rows do preserve first-inference cost. Verified: compile and CACHE_DIR effects do not leak into timed rows for GPU/NPU (load happens outside the timed loop, warmups 2-3 are flat); NPU TTFT of 2.2 s is steady-state, not compile. GPU CACHE_DIR is `.cache/ov_cache` relative to the current directory (`openvino_engine.py:156`), so the first-ever run compiles, later runs hit the cache, and cwd changes silently cold-start; it is not recorded. NPU has no cache dir.

**F8 HIGH. Legacy `BenchmarkRunner` (the `arc-runner benchmark` CLI, `cli/main.py:278-312`) can emit an invented number and ignores the method.** Reproduced with a fake engine yielding 0 tokens: `completion_tokens=1`, `tokens_per_sec=1000000.0`, TPOT 0.01 ms (`runner.py:151-157`: `max(1, tokens)`, `max(1e-3, decode_ms)`, `max(0.01, ...)`). One token gives 840336 tok/s. Also: temperature 0.7/top_p 0.9 (`runner.py:118-123`; greedy only because `do_sample` is unset), default 5 iterations and 2 warmups (`cli/main.py:280-282`) against METHOD's 10 and 3, chunk callbacks counted as tokens (`runner.py:137-140`), numerator includes the first token while the decode window excludes it (bias +1/(n-1)), prompt tokens estimated by `split()` (`runner.py:135`). The scorecard says "95% Confidence Intervals" at n=5. Failure scenario: a user passes `--tokens 1` or a model emits EOS immediately and reads "1,000,000 tokens/s".

**F9 HIGH (console output) / MED. Unmeasured values still printed by `scratch/run_live_real_models_benchmark_and_compare.py`.** `:572` prints the literal "~1.3 GB USM" as RAM footprint for every model row; `:518` hard-codes "RAM: 15.43 GB"; `:234` labels every Ollama run "Intel Core Ultra CPU Fallback" without checking where Ollama ran; `:224-229` defaults a missing `eval_duration` to 1 ns (tok/s = eval_count/1e-9 if the field is absent) and a missing `prompt_eval_duration` to 0 (TTFT 0.0 reported as PASS); `:381` divides by `max(0.1, ...)`. `:505-506` writes a schema-v1 smoke file with `warmup_runs: 0, max_new_tokens: 0`, n=1 and an `ollama` backend; `render_bench_docs.py` would render it into the README as measured. The gate refuses it (n<10), the renderer does not. Not run in the three real files.

**F10 MED. Method label inconsistencies.**
- `render_bench_docs.py:40-49` prints "median +/- 95% CI" where the half-width is the Student-t interval of the mean. It reports no IQR, no bootstrap interval, no non-normal flag, although METHOD.md:33 promises all of them. Example: CPU hard TTFT shown as "170 +/- 34.5": 170 is the median, 34.5 is the mean's half-width.
- Derived values stored with `source: "measured"`: `rtf`, `throughput_audio_s_per_s`, `audio_duration_s` (a generator constant) in `nonllm.py:83-88` / `workloads.py:98-108`. METHOD.md:9 defines these as derived. `llm_samples` marks wall-counter decode rates "measured" (`nonllm.py:297`).
- Decode series can mix two definitions inside one sample set: `decode_tokens_per_second` if > 0, else OpenVINO `get_throughput` (`nonllm.py:298`).
- `docs/benchmarks/README.md` documentation-index row for `empirical_silicon.md` still advertises "30.8x TTFT"; the measured TTFT ratios are 20.1 / 2.8 / 2.3. The target doc carries an UNVERIFIED banner; the hub row does not.
- LLM result protocol records `max_new_tokens: 300` for all tiers (`run_live_sequential_model_benchmarks.py:472`, max over tiers); actual per-tier budgets are 30/200/300.

**F11 MED. STT construct validity.** `rtf = latency/audio_duration` (`nonllm.py:86`) is not a throughput for Whisper, which pads to a fixed 30 s window: GPU latency 0.061 / 0.080 / 0.088 s for 3 / 8.5 / 18 s clips (rtf 0.0207 / 0.0094 / 0.0049). Input is a synthetic tone (no transcript quality, unknown decode length, `completion_tokens` 0). The engine returns a fabricated `latency_s=0.001`, `throughput = duration/0.001` on a VAD-gated input (`audio_engine.py:247-259`); the benchmark times its own clock, and the tone RMS is 0.187 (>> thresholds 1e-4 / 0.005) so these runs were not gated. But nothing in `stt_samples` guards against a gated call, and the API path emits an invented number (HIGH if exposed to server clients; I did not trace the route).

**F12 LOW. Legacy `MultiModelCoBenchmarker`** (`multimodal_runner.py:99-100`) averages `usm_zero_copy_latency_us`, which `pipeline.py:206` defaults to 0.0 when no transfer stats exist, so absent becomes a measured 0.00 us with no validity flag (SUSPECTED; USM bridge not traced). ROI inputs are caller-supplied (good).

**F13 LOW/MED. `bench_regression.py`.** Each iteration starts a fresh subprocess whose single generation is its first in-process inference (`:49-67`), so every timed value is a cold-process first run and the `--warmup` iterations warm only the disk cache and CACHE_DIR, not the process. TTFT stops at the first non-empty text chunk, decode token count comes from re-tokenising the joined text (`:63`), both approximations equal across versions so the ratio is fair but not METHOD's definition. The `flash_acquire` workload stores layer loads/s in the `decode_tps` column with `source=measured` (`:46,132`), only the protocol label says otherwise; a renderer would print it under "Decode tok/s". Failures raise and write nothing (good). Fail-closed on `decode_tps=None` (good).

**F14 LOW. `bench_native_micro.py`.** No warmup: the first of 15 repeats pays page-fault cost yet the metric is named `mapped_read_warm_gbs` (`:150`); mapped read (sum) vs `file.read` (alloc + copy) are not like-for-like (`:150,154-158`); t-interval on non-normal timings; thread timings averaged. It claims nothing and commits nothing, so low impact.

**F15 LOW. `kv_quality.py` / H-D.** 8 prompts (`bench_kv_quality.py:20-29`): `ceil(0.9*8)=8`, so "90%" silently means 8/8, and 8/8 has a Wilson lower bound of only 0.676. No noise floor: the GPU is not run-to-run deterministic (F6), so default-vs-default disagreement must be measured before attributing divergence to u8. Short prompts and 64 new tokens under-expose KV quantisation (error grows with context). Comparison uses re-tokenised text (`:99-100`), acceptable for prefix agreement.

No defect found: `power.py` (counter backwards / non-advancing timestamps correctly yield None with a reason; all package power is null in the real files because the testbed exposes no counter, so no wattage appears anywhere); `results.py` validation (None handling, finite non-negative checks); `workloads.measure_workload`; `collect_tiers` failure recording; `target_device` fallback flagging; the CV and quantile helpers.

## 5. Can any script still emit a number that is not measured?

Yes, five paths: F8 (1,000,000 tok/s and a fabricated token count), F9 (the "~1.3 GB USM" literal, the 1-ns / 0-ns defaults), F11 (VAD-gated `latency_s=0.001` in the API path), F12 (0.0 us default), F13 (`flash_acquire` rates in `decode_tps`). None of these appears in the three committed result files; the schema-v1 live scripts (`nonllm.py`, `workloads.py`, `harness_common.py`) take latency from the harness clock and counters from the engine or None.

## 6. What makes H-A..H-D decidable (minimum protocol)

Common to all: implement METHOD.md as written before any new run: warmup-until-converged (use `warmup_converged`, last 3 CV <= 5%, 3-5 warmups, otherwise mark the series non-converged and exclude it from claims); per-iteration interleaving in seeded random order with 20 s cooldown; AC power and idle-CPU precondition check that sets `noisy`; record `on_ac_power`, driver versions, load/compile time and package temperature or frequency per run; >= 30 timed runs per cell (CI half-width target 3%); report gate verdict, CV and ratio CI next to each ratio; also compute a detrended or block-bootstrap interval and, for H-C, a paired bootstrap. Set CPU thread count explicitly and print it.

- **H-A / H-B (need new Ollama data; nothing usable exists).** Same machine and session, Ollama version pinned and recorded. Verify with `ollama ps` where it ran and record it (do not hard-code "CPU"). Match models by quantisation (Ollama Q4_K_M vs OpenVINO INT4 printed on every row; they are not the same weights). Prompt parity: send `raw: true` (or compare `prompt_eval_count`) so Ollama's template does not add tokens; arc-runner's prompts are un-templated (prompt_tokens 20/35/42 recorded). Pin `temperature 0, seed, top_k 40, repeat_penalty 1.1, num_ctx`, `keep_alive` constant. Same TTFT definition on both sides: streamed request, client wall-clock to first chunk (record Ollama `prompt_eval_duration` as a secondary column only; it excludes tokenisation and overhead and can collapse to near 0 on prefix-cache hits, so prefix a per-run nonce of fixed token length). Same decode definition: (n-1)/(t_last - t_first) from streamed timestamps. Force equal completion length (`ignore_eos` + `min_new_tokens` on OpenVINO; confirm the Ollama equivalent or discard runs with different counts). Interleave arc/Ollama per iteration. Matrix: at least qwen-2.5-1.5b (qwen2.5:1.5b) and llama-3.2-3b (llama3.2:3b) x low/medium/hard. Decide with `evaluate_speedup_gate` per cell (TTFT with old/new = ollama/arc) and, because the claim is "each model", report the number of cells tested.
- **H-C.** Two checkouts (v0.1.0, v0.2.0), real model on GPU, `bench_regression.py --workload llm`, but change the worker to do 3 in-process warmups then K timed generations per process so it measures steady state, and run cold start as a separate series. `--runs >= 30` (n >= 22 is needed at cv 5%, section 4 F4), paired analysis, margin 3%. The `flash_acquire` workload alone cannot decide H-C (it is not a model throughput).
- **H-D.** Run `bench_kv_quality.py` on the real model on GPU with: >= 50 prompts including long-context ones, a default-vs-default control to estimate the intrinsic disagreement floor, 128+ new tokens, and the verdict stated as a Wilson lower bound (50/50 gives 0.929, 48/50 gives 0.865, so "90%" is only claimable with a lower bound >= 0.90 at about 50/50). Prefix agreement stays the criterion (no logits exposed); add the u8 throughput and memory deltas as a separate non-inferiority run if u8 is ever to become default.

## 7. Verdict

The harness code (stats, results, workloads, nonllm, power) is sound and honest at the level of individual numbers; the real result files are valid and internally consistent. The published comparison supports only this statement: on this machine, OpenVINO GenAI on the iGPU was faster than OpenVINO GenAI on the CPU for qwen-2.5-1.5b by roughly 3-4x decode and 2.3-20x TTFT (point estimates above), but the METHOD claim gate fails for every LLM cell because the CPU series are noisy, and the protocol steps that would make the CPU series trustworthy (converged warmup, interleaving, preconditions) are not implemented in the scripts. Nothing here is a claim against Ollama.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
