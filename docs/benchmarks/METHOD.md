# Benchmark Method

[← Benchmarks hub](README.md)

How every number in the benchmark documentation is produced, which claims are allowed, and how they are checked. Thresholds below are fixed before running; a claim that is not met is published as not met.

## Labels

Every number is one of: **measured** (produced by a benchmark run, present in a schema-v1 file under `results/`), **derived** (computed by a stated formula from measured values), **projected** (a model or expectation, never a result). Hardware that is not part of the testbed (for example a discrete Arc A770) can only appear as projected, or as a community-submitted measured file (see [community benchmarks](../community_benchmarks.md)).

## Metric definitions

| Metric | Definition | Source field |
|:--|:--|:--|
| TTFT | time from request start to the first generated token, milliseconds | `ov_perf_metrics` (`get_ttft`) or Ollama `prompt_eval_duration` |
| TPOT | mean time per output token, milliseconds | `ov_perf_metrics` (`get_tpot`) |
| Engine throughput | the engine's own `get_throughput` mean; the API does not state whether it includes prefill | `ov_perf_metrics` |
| Decode TPS | `(generated tokens - 1) / (generate duration - TTFT)`; for Ollama `eval_count / eval_duration` | derived from documented counters |
| End-to-end TPS | `generated tokens / total wall time` | wall clock |
| Stream estimates | chunk-callback counts and wall-clock times; labelled `wallclock_stream`, `is_estimate = true` | wall clock |

Tokens are counted with the model's tokenizer, never by splitting words.

`tokens_per_second` is OpenVINO's `get_throughput()`: in GenAI 2026.4.1 it is `1000 / mean TPOT`, where TPOT excludes the first (prefill) token, so it is decode-only (`src/cpp/src/perf_metrics.cpp`, tag 2026.4.1.0). `decode_tokens_per_second` is the wall-counter view `(tokens - 1) / (generate duration - TTFT)`, which also includes sampling and detokenization and is therefore slightly lower. Both exclude prefill; Ollama's `eval_count / eval_duration` is the comparable definition on the other side.

## Protocol (all systems)

Each step says whether code enforces it. "Implemented" means the shared harness (`arc_runner.benchmark.harness_common`, `workloads`, `nonllm`, `stats`) does it and the live scripts under `scratch/run_live_*` call it; anything else is a stated manual requirement or not done.

1. Environment is recorded in the result file: OS, Python, git SHA, hardware description, OpenVINO GenAI version, AC power, and the pre-run check results (implemented; `environment.on_ac_power`, `environment.preconditions`). The Ollama version is recorded only when a script passes it. Driver versions are not collected: `driver_versions` is an empty object today.
2. Preconditions (implemented in `check_preconditions`, run once per script run before the first measurement; the 30 s idle window is `--idle-window-s`):
   - AC power: read from the OS (Windows `GetSystemPowerStatus`, Linux `/sys/class/power_supply` Mains entries). On battery is a violation.
   - Idle CPU: whole-machine CPU busy percent over a 30 s window, computed from the OS cumulative CPU counters. Above 10% is a violation.
   - Memory gate: available physical memory must be at least 1.2 times the bytes the run keeps resident (model files on disk, times the number of engines held at once when `--interleave` is used). The factor 1.2 is a fixed convention for runtime overhead, not a measured requirement.
   - A violation sets `noisy` on every row of the run, which the claim gates refuse. A value the OS could not report is recorded as `null` with the reason and is not a violation (so an unread check does not block a claim); `--skip-preconditions` records all three as skipped. On a Linux machine without a Mains entry (for example a desktop VM) AC power is therefore `null`.
   - Not checked by any code: "no other benchmark process". The operator must ensure it.
3. Same prompt text, same prompt length (+-2 tokens), same `max_new_tokens`, `temperature = 0`, same context length for every system; quantization and device labels (for example Q4_K_M vs OpenVINO INT4, CPU vs iGPU) are printed next to every row. Prompt equality is by construction (one prompt per tier shared by all backends); token-count parity is not checked by code.
4. Warmup and timed runs:
   - Converged warmup (implemented, default on): at least `--warmups` (3) discarded warmup runs, then more, up to `--max-warmups` (5), until the last three values of the warmup metric have a coefficient of variation of at most 5%. The metric is TTFT for LLM runs and run latency for every other workload; a TTFT that is not available cannot count as converged. `--fixed-warmups` runs exactly `--warmups`. A series that has not converged after the maximum still runs; the outcome is recorded per series as `warmup_converged` in `protocol.series_quality` (recomputed from the warmup rows; `null` with fewer than three warmups). Non-convergence is recorded, not a claim-gate condition.
   - Timed runs: exactly `--runs` (default 10). The "up to 30, stopping at a 95% CI half-width of 3% of the mean" rule is not implemented; a longer series needs a larger `--runs` chosen by hand.
   - Interleaving (implemented as an option, default off): `--interleave` runs the systems of one tier once per iteration in a seeded random order (`--seed`, order = `random.Random("<seed>:<iteration>")` shuffle), warmups included, with every engine held in memory at once. It exists for the LLM and VLM scripts only. The embeddings, audio, image and video scripts always run backend after backend. A result file records `protocol.harness.interleaved`.
   - Cooldown (implemented): `--cooldown-s` (default 20) seconds of sleep when switching between backends (sequential mode) or between tiers (interleaved mode), not before the first. `protocol.harness.cooldown_s` records it; 0 disables it.
   - Cold start is not reported: model load and compile time are not recorded in the schema-v1 files.
5. No outlier removal. Result files keep every sample, so median, IQR, mean, the Student-t 95% interval and the bootstrap 95% interval can be computed from them; `scripts/render_bench_docs.py` currently prints only the median and the Student-t half-width of the mean. Non-normal flag (implemented, `stats.non_normal_flag`): the 95% Student-t interval of the mean and the 95% percentile bootstrap interval of the mean (2000 resamples, seed 0) are computed; the series is flagged non-normal when either endpoint differs by more than 5% of the sample mean. It is `null` for fewer than three values or a non-positive mean. The flag is recorded per series and metric in `protocol.series_quality`; it does not change any gate.
6. Result files follow `schema/bench_result.v1.json` and are validated by `scripts/validate_bench_json.py`; tables in the docs are generated by `scripts/render_bench_docs.py` and checked with `--check`.

## Claim gates (implemented in `arc_runner.benchmark.stats`; document-level entry point `arc_runner.benchmark.results.evaluate_document_claim`)

- **Speed-up claim** (`evaluate_speedup_gate`): every timed row of both systems is `measured`, fallback-free and not noisy; at least 10 timed values each; coefficient of variation at most 10%; the lower bound of the 95% bootstrap interval of the ratio is above 1.0. A median above the memory-bandwidth ceiling `B / (W + KV)` is SUSPECT and needs a human decision.
- **VLM and audio claims** (`evaluate_workload_gate`): same conditions as the speed-up claim (measured, fallback-free, not-noisy rows; at least 10 timed values each; coefficient of variation at most 10%; ratio lower bound above 1.0) on a lower-is-better metric with the ratio reference/arc: `latency_s` for `vlm` and `audio_stt` (same clip for both systems), `rtf` for `audio_tts` (its output length varies). Additionally all timed rows must be of the stated workload and of one model/tier id, and for `vlm` every timed row of both systems must carry the same `output_chunk_count`, because latency scales with output length; a VLM series whose output length varies cannot support a claim. Embeddings, image and video have no gate, so no claim about them can pass (`evaluate_document_claim` refuses them). No script calls the speed-up or workload gates automatically (only `scripts/bench_regression.py` calls a gate, the no-regression one): a published speed claim is allowed only if someone runs the gate on the result file and it passes.
- **No-regression claim** (`evaluate_noninferiority_gate`): the new version's lower ratio bound versus the old version is above `1 - 0.03` for decode TPS and TTFT. The 5% resident-memory growth limit is a requirement that no code checks yet.
- **Hypotheses registered before the first run** (the claim fails if its bound does not clear):
  - H-A: arc-runner on the iGPU has higher decode TPS than Ollama on CPU for each model in the matrix (ratio lower bound > 1.0).
  - H-B: arc-runner TTFT is lower than Ollama's for each model (old/new ratio lower bound > 1.0).
  - H-C: v0.2.0 is not slower than v0.1.0 by more than 3% on decode TPS and TTFT.
  - H-D: with `KV_CACHE_PRECISION=u8`, greedy generations agree with the runtime-default precision for at least 32 tokens (or entirely, if shorter) on at least 90% of the fixed prompt set (`scripts/bench_kv_quality.py`, `arc_runner.benchmark.kv_quality`). The criterion is prefix agreement only; KL divergence and perplexity need logits the pipeline does not expose and are not computed. u8 stays opt-in (`ARC_KV_CACHE_PRECISION`) whatever the outcome; changing the default is a separate human decision.

## Power and memory

Package power and energy are measured, not assumed: `benchmark.power.PowerMeter` reads the Level Zero Sysman package-domain energy counter before and after each run (energy J = delta microjoules / 1e6, average W = delta microjoules / delta microseconds). The counter covers the whole SoC (CPU and iGPU), so the figures describe the machine under load, not the GPU alone. If a machine exposes no such counter, or the counter goes backwards or its timestamp does not advance, `package_power_w` and `package_energy_j` are `null` with the reason recorded by Sysman (`get_unavailable_reason`); no wattage is ever filled in. Free device memory comes from Sysman only for on-board device memory modules (an integrated GPU has none, so it is `null`). Memory is otherwise reported as resident working set; mapped (virtual) size is never presented as RAM use.

## Tiers that fail on the device

A tier that raises (for example `CL_OUT_OF_RESOURCES` when an integrated GPU runs out of shared memory) is written to the result file's `failures` list with the error text; its samples are absent, never estimated. After `CL_OUT_OF_RESOURCES` the Intel driver may hang on any further OpenCL call, so the image and video scripts skip the remaining tiers of that backend, do not unload the engine, still run the CPU baseline, write the file, and leave with exit code 3 without normal shutdown. `--tiers low,medium` restricts a run to tiers that fit.

## Other workloads (embeddings, VLM, audio, image, video)

These runs use the same protocol (environment capture, warmups discarded, timed runs one sample each, CPU baseline as a real second run, `fallback_used` when the wanted device is missing) and the schema-v1 `workload` and `metrics` fields. Metric keys carry their unit (`latency_s`, `rtf`, `frames_per_s`, `texts_per_s`, `megapixels_per_s`, `package_power_w`, `package_energy_j`). Latency and power are measured per run; rates and real-time factors are derived per sample from the measured latency and a workload constant stated in the metric name's formula (for example `rtf = latency_s / audio_duration_s`, `frames_per_s = num_frames / latency_s`) and stored with the measured sample. Stage timings (`step_latency_ms`, `iteration_rate_it_s`) are only the engine's own OpenVINO perf metrics and are `null` when OpenVINO reported none; they are never apportioned from total time. VLM requests report a streamer-callback chunk count (`output_chunk_count`), not tokens, and no TTFT, because `VisionEngine` yields only after generation completes. Speech-recognition input is a generated test tone, so timing is valid but transcript quality is not evaluated; image and video quality are not scored here.

## Heavy models

Results for 32B/72B-class models require the real weights. The streaming demonstrator in this repository emits canned tokens (`metrics_source = synthetic_pipeline`) and is never a throughput result.
