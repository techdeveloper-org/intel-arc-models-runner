---
doc_id: DOC-DECISION-LOG
title: Decision log
stage: all
version: 1.2.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session]
reviewers: [user]
---

# Decision log

Append-only. A changed decision gets a new entry that names the one it supersedes; earlier rows are never edited.

| DL | Decision | Why | Alternative rejected | Stage |
|:--|:--|:--|:--|:--|
| DL-001 | Gemini plan stays the base spec | user instruction; most work done and tests pass | restart from scratch | S00 |
| DL-002 | Route to silicon-architecture, math expert for derivations | discovery tool and rules | generic systems-programming route | S00 |
| DL-003 | Library agents read all their skills with quote proof | user rule | sampling skills | S00 (see 023, 039 for what applied in practice) |
| DL-004 | Do not build non-temporal-load L3 bypass | ineffective on write-back pages; no CPU copy exists | implement as planned | S00 |
| DL-005 | Host USM is the default residency | iGPU shares system memory; device USM adds a copy | device or shared USM | S00 |
| DL-006 | Sysman returns None plus a reason, never fabricated values | machine reports no power domains / memory modules | keep 22.0 W and RAM-as-VRAM | S-A |
| DL-007 | Sealed-after-registration flag (release/acquire) | table immutable after load; zero reader cost | rwlock | S-B |
| DL-008 | Direct-mapped ring slots (layer % slots) | cyclic scan gives LRU a 0% hit rate | LRU | S-C |
| DL-009 | Remove global -mavx2 | no AVX2 code exists; SIGILL risk | runtime-dispatched AVX2 | S-B |
| DL-010 | Commit a prebuilt DLL with ABI/hash safeguards | user decision | gitignore and CI build | S00 (see 040) |
| DL-011 | Release and tag only on the maintainer's explicit OK | PyPI and GHCR publishing is irreversible | auto-release | S00 |
| DL-012 | Keep tracked uml/ and drawio/ | repo precedent | untrack | S00 |
| DL-013 | Leave root CONTRIBUTING.md | pre-existing, out of scope | move to docs/ | S00 |
| DL-014 | Keep float telemetry fields and add validity flags | avoid breaking consumers | Optional fields | S-A |
| DL-015 | Tests written by a different author than the implementer | avoid Goodhart on weak tests | implementers write tests | S00 (not achieved in cloud sessions; mutation checks used instead) |
| DL-016 | Agents write their own report files | token cost and audit trail | main session transcribes | S00 |
| DL-017 | Version 0.1.0 to 0.2.0 | new feature and behaviour fixes | patch bump | S-F |
| DL-018 | Restate or remove the unverified theorem and gain claims | math re-checked | keep | S-E |
| DL-019 | LEAN track skips optional BA/PM and quality stages | proportionality | FULL track | S00 |
| DL-020 | Regenerate all 13 UML/drawio types once | new package | selective regeneration | S-F |
| DL-021 | Rules for unrelated languages and frameworks marked not applicable | not touched | re-audit each session | S00 |
| DL-022 | LEAN is the track | token cost vs scope | FULL by default | S00 |
| DL-023 | Library agents and MCPs optional; absence logged | cloud session lacked the library at that time | pretend they ran | S00 (superseded in part by section 22 of the plan) |
| DL-024 | Work in slices S-A..S-F, honesty first | smallest harm-first units | one big wave | S00 |
| DL-025 | POSIX branch fixed with the same helper; sanitizer harness is the memory-safety proof | runnable without Windows | rely on review only | S-B |
| DL-026 | One probe script for hardware facts; claims stay unverified until its JSON is committed | hardware not reachable from the cloud | guess values | S-E |
| DL-027 | Canonical DLL build is a CI cross-compile | stale-binary risk | commit DLL separately | S00 (see 040) |
| DL-028 | Baseline for red-first is c7312e0 | Gemini work was already committed | extra snapshot commit | S00 |
| DL-029 | Benchmark scope: all domains plus native micro, roofline inputs, regression | requested coverage | LLM-only | S-E |
| DL-030 | One statistics implementation, both runners delegate | avoid drifting CI code | duplicate implementations | S-E |
| DL-031 | Results are schema-validated JSON; doc tables generated from it | root cause of hand-edited numbers | manual edits | S-E |
| DL-032 | A770 and other unowned hardware are projected or community-submitted | not in the testbed | leave as measured | S-E |
| DL-033 | Claim thresholds registered in METHOD.md before runs | prevents post-hoc tuning | decide after seeing data | S-E |
| DL-034 | CPU-only/Linux micro-benchmark subset as CI smoke, no claims | catch harness breakage | local only | S-E |
| DL-035 | POSIX native branch is best-effort; Windows x64 is the supported DLL platform | users and hardware are Windows | drop POSIX; claim full support | S-F |
| DL-036 | Speculative acceptance and speed-up are reported as 0 and 1.0 until measured | no verified source | keep constants | S-A |
| DL-037 | Shared telemetry builders in engine/telemetry.py | one place enforces measured-or-unavailable | per-engine code | S-A |
| DL-038 | Heavy-model benchmark rows carry model_kind synthetic_container until real weights run | they describe a toy model | leave unlabelled | S-A |
| DL-039 | India regulatory skills, integration-testing agent not applied; threat model scoped to GGUF parser and DLL loader; C SAST via clang-tidy, analyzer, sanitizers | no Indian data, local runner, no Semgrep in the cloud image | apply all agent defaults | S-B |
| DL-040 | Windows DLL is not rebuilt in the cloud; the maintainer rebuilds locally and commits it with its sidecar | a cross-compiled binary cannot be executed here; the ABI check refuses the old DLL | commit an untested binary | S-B |
| DL-041 | get_tensor_buffer seals the table on first use | keeps call sites working and makes the table immutable before concurrent reads | require explicit seal | S-B |
| DL-042 | GGUF parser caps dimensions at 4 and validates declared lengths | hostile input; matches GGML_MAX_DIMS | trust the file | S-B |
| DL-043 | Sidecar file name is the full library file name plus .build.json | .dll and .so share a stem | shared stem name | S-B |
| DL-044 | No pin counts in the ring buffer; arrays keep the mapping alive | colliding layers could make a thread wait for itself | blocking pins with timeout | S-C |
| DL-045 | Slot is always layer % slots; prefetch ignores its slot argument | deterministic, no timestamps | LRU | S-C |
| DL-046 | BF16 raw uint16, quantized raw uint8 in both paths | old fallback misread BF16 and inflated quantized data to float32 | convert | S-C |
| DL-047 | Row-major shape is the reversed GGUF dimension list | ne[0] is the fastest dimension | use dimensions as stored | S-C |
| DL-048 | close() defers the native unmap while views are alive | prevent use after unmap | unmap immediately | S-C |
| DL-049 | ARC_NATIVE_ABI_VERSION 3 | Level Zero entry points changed | keep 2 | S-D |
| DL-050 | u8 KV cache is opt-in | no quality gate exists | default on | S-D |
| DL-051 | ARC_ZE_LOADER_PATH and ARC_ZE_SYNC_TIMEOUT_MS are supported overrides | testability without hardware; clamp below TDR | no overrides | S-D |
| DL-052 | Prompt plus completion over the window raises | caller learns the request cannot be honoured | clamp silently | S-D |
| DL-053 | Version lives in src/arc_runner/_version.py | importing it from the package root created a circular import | duplicate strings | S-F |
| DL-054 | Diagrams are extended additively with a generated native-layer section instead of regenerating all earlier content | a repository-wide hand-built model was not rebuilt; the generated parts are tied to the code by ast and checked for drift | full regeneration of all earlier content | S-F |
| DL-055 | Stale tests that failed before this work are corrected or skipped with a reason rather than left red | CI must be able to go green; hardware-dependent behaviour is reported as skipped, never as passed | leave them failing | S-F |
| DL-056 | Non-LLM results extend schema v1 additively (workload + metrics) | existing files and tooling stay valid | schema v2 | S-G |
| DL-057 | Power is measured with the Sysman energy counter or is None | wattage constants are fabrication | assumed TDP | S-G |
| DL-058 | The fixture audit script is deleted, not patched | there was no measurement to keep | keep with a banner | S-G |
| DL-059 | u8 KV stays opt-in whatever the gate says | a default change is a human decision | auto-enable on pass | S-G |
| DL-060 | Mermaid validity is checked by rendering, in CI | structural checks cannot see syntax errors | none | S-G |
| DL-061 | One shared async bridge: dedicated executor for blocking work, one worker thread per stream, loop-side buffer bounded by max_new_tokens, finish as a flag, cancel event plus disconnect watcher | token loss (P4-03), event-loop stall (P4-01), leaked generation (P4-04), three copies of the bridge | blocking producer backpressure (P1-03 option 2); bare to_thread keeping three bridges | PERF |
| DL-062 | Manager concurrency: short state lock, single-flight load per key, one compile per device, lease counting, load-before-evict, canonical key on the resolved path, request-scoped telemetry | one lock spans compile (P4-02); eviction can unload an in-flight engine (P4-05); telemetry race (P1-11) | one lock with I/O moved out; a single asyncio manager task | PERF |
| DL-063 | Memory and request limits fail closed and are data; no default numeric limit and no calibrated factor until a result file exists | no budget (P5-01), no pixel or frame bound (P5-04), numbers must not be guessed | static guessed limits; auto-degrade of resolution | PERF |
| DL-064 | Prefetch and advice in two stages: Python pre-fault, wrap-around prefetch, persistent worker, pool and lock narrowing first; native advice only if measured necessary; sync timeout not raised | cold faults inside the timed DMA (P2-03), no I/O in native prefetch (P2-02), thread per prefetch (P2-07) | native advice immediately; raise the sync timeout; LRU ring | PERF |
| DL-065 | One absolute cache root for every engine, device and the AOT manager; NPU static-shape knobs only after a measured win | CWD-relative GPU-only cache (P1-08), AOT cache disconnected (P5-07) | keep GPU-only but absolute; record load time only | PERF |
| DL-066 | At most one ABI bump (to 4), bundling only exports justified by hardware numbers; internal C changes ship without a bump | each bump forces a local DLL rebuild (DL-040) | one bump per export; no native change at all | PERF |
| DL-067 | Honesty-fix policy: fabricated constant becomes None plus validity flag plus reason; swallowed errors become typed errors; additive Optional fields; published figures relabelled, never silently changed | P4-16, P5-08, P5-10, P1-09, P1-13, P5-12 | label as estimate and keep; document only | PERF |
| DL-068 | Speculative decoding is gated by a committed calibration; a draft that is not cheaper per step than the target (c >= 1) cannot speed up | derived c = 2.82 on the committed same-model data (P1-07) | leave enabled with a warning; default NPU draft | PERF |

## ADR detail: performance programme (DL-061..DL-068)

Status of all eight: proposed on 2026-10-29 as part of plan section 23; no source changed; effective when the named work package is accepted. Evidence ids refer to the discovery reports P1-P5 described in plan section 23.1; evidence class is stated per claim (PROVEN, READ, READ+replica, DERIVED, HYPOTHESIS). Work packages are WP-nn in plan section 23.5.

### DL-061 Shared async bridge (WP-03)
- Context: blocking load, compile, generate, tokenize, embed, GGUF parse and pull run inside `async def` handlers (P4-01, READ+replica: a probe needing about 52 ms took 2077 ms during one 1 s load plus 1 s generate). The thread-to-loop bridge uses `call_soon_threadsafe(queue.put_nowait)`, whose `QueueFull` handler is dead code, so chunks and the sentinel are dropped (P4-03, READ+replica: 67 of 300 tokens lost under a 2 s stall). A disconnect leaves the worker generating (P4-04, READ+replica: 102 tokens one second after the client left at token 5). Three copies of the bridge exist (P4-03, P1-03). Per-token JSON plus timestamp cost is 3.7 us (P4-12, replica) against a 24.9 ms GPU token (DERIVED from P3 data), so per-token micro-cost is not the problem.
- Options: (A) `asyncio.to_thread` at each call site and keep the per-route bridges (smallest diff, leaves the loss bug in three places); (B) one helper module with a dedicated named executor, a single worker thread per stream, an unbounded loop-side buffer bounded by `max_new_tokens`, finish as a flag, cancel by event plus a disconnect watcher; (C) a dedicated engine thread or process with a request queue (also removes concurrency issues, largest change and changes the sync API used by the CLI). A fourth variant, a blocking `run_coroutine_threadsafe(...).result()` for true backpressure (P1-03 option 2), was rejected inside B: it keeps the single engine busy for as long as a slow client stalls and turns a loop stall into a device stall.
- Decision: B. Executor size starts at `max_loaded_models + 2` (P4-01 option 1) and is revisited after N-01. Context variables are copied into the worker so request ids survive. An interim per-engine lock ships with it so offloading cannot expose concurrent pipeline calls before WP-04.
- Consequences: one place to test loss, cancel and ordering; routes shrink; unbounded buffer is a memory policy (bounded by generation length, asserted); a slow client no longer slows generation; the interim lock makes concurrent requests to one model wait, as the single compute engine does anyway.
- Reversibility: high. The helper is additive; routes can be pointed back at the old code in one commit. No ABI or data format involved.

### DL-062 Manager concurrency model (WP-04, WP-05)
- Context: `ModelManager._lock` is held across path resolution, eviction, unload and compile, and shared with `record_generation_metrics`, `get_telemetry` and listing (P4-02, P1-02, P5-03, READ). First-request cost after load is PROVEN (P3-03: CPU TTFT 9.0-14.2 s vs 0.07-0.36 s steady; GPU 134-221 ms vs 20-74 ms); compile duration itself is a HYPOTHESIS (P4-02 "O(10-100 s)", unmeasured). The engine returns one shared object to all callers with one mutable `_telemetry` (P4-05, P1-11, READ), eviction sets `_pipeline=None` under an in-flight request (P4-05, P5-02, READ), the cache key uses the raw string and includes gamma (P4-06, P1-06, READ), eviction precedes a load that may fail (P4-06, P1-02, READ). Two simultaneous compiles double peak memory on a 6976 MiB shared ceiling (P5-03, probe total_mb PROVEN, effect HYPOTHESIS).
- Options: (A) keep one lock, move disk scans and unload outside it (small, does not fix compile blocking); (B) short state lock, single-flight future per key, a semaphore of one compile per device, lease counting that blocks eviction of in-use engines, canonical key on the resolved path, load-before-evict, telemetry returned from the call; (C) one asyncio manager task owning all lifecycle through a queue (cleanest, rewrites the synchronous API the CLI uses).
- Decision: B, with the sync API kept and an async variant added. Serialization per engine is the default because concurrent `LLMPipeline.generate` safety is a HYPOTHESIS (N-03 decides whether it is mandatory or merely conservative). `warmup_on_load` ships off and is flipped only by a committed N-04 result.
- Consequences: cache hits and telemetry no longer wait behind a compile; same-key requests share one compile; compiles stay sequential per device; request latency under contention rises and correctness rises with it; engine API gains a result object carrying telemetry; unload of a leased engine waits or raises `EngineBusyError`; eviction before load is still used when the count limit forces it.
- Reversibility: medium. State is in memory only, but the telemetry-in-result API change touches callers; keep `get_telemetry()` as last-value for one release.

### DL-063 Memory budget and request limits (WP-06, WP-07)
- Context: three LRU caches are each capped at two by count with no byte accounting (P5-01, READ); the probe shows a 6976 MiB shared ceiling (PROVEN) and `free_mb` null for the iGPU. Image and video requests have no pixel or frame budget; the derived unfused-attention bound is 0.54 GB at 512x512, 2.72 GB at 768x768 (matches the recorded SD-1.5 failure, R09) and 8.59 GB at 1024x1024, upper bounds only because SDPA fusion is unverified (P5-04, DERIVED/HYPOTHESIS). The SD-1.5 failure has no committed result row (P3-12) and no passing size is recorded either. Auto-downscale changes output silently (P5-04 option C), which the project forbids.
- Options: (A) fail-closed static budget and limits from configuration, typed errors, nothing evicted when a model alone exceeds the budget; (B) calibrated per-model-class factors produced by a measurement run and shipped with the result files; (C) one engine per device with sequential loading, documented as the iGPU profile.
- Decision: A now as mechanism, B as data once N-10 and M-07 results are committed, C as a documented profile. No default numeric limit and no calibration factor is shipped before its result file exists; an unset limit warns once and does not reject. The on-disk-bytes factor is labelled a convention (`budget_basis`) until replaced. Enforcement flag is off at first.
- Consequences: loads that used to succeed may be refused once enabled (typed error); no guessed numbers in the repository; the first release of the mechanism protects only users who configure limits. Open question for the maintainer: whether to ship the R09 derived 768x768 bound as a hard cap before the measurement.
- Reversibility: high. Limits and the budget are configuration and a flag.

### DL-064 Prefetch and advice strategy (WP-08, WP-09, WP-11)
- Context: the native path performs no I/O in "prefetch" without stager or pin (P2-02, READ); cold faults of mmap sources happen inside the DMA copy under a 1.5 s synchronous wait whose expiry marks the context dead and disables staging for the manager's life (P2-03, READ; impact HYPOTHESIS); layer 0 of the next token is never prefetched, a thread is created per prefetch, and depth is 1 of R-1 (P2-07, READ); the Windows open uses `FILE_FLAG_RANDOM_ACCESS` (P2-02, READ, effect HYPOTHESIS); Windows is the supported platform (DL-035). A cyclic scan over N>R layers misses every layer under any replacement policy (P2-07), consistent with DL-008 and DL-045. No native or flash-stream result exists (P2, P3), so every size of effect is a HYPOTHESIS.
- Options: (A) Python-level pre-fault of the registered tensor ranges before the timed copy, wrap-around prefetch and a persistent worker, no ABI change; (B) a native `arc_mmap_advise` (madvise or `PrefetchVirtualMemory`) now, ABI 4; (C) raise the sync timeout (rejected: it is clamped below the driver TDR by design, DL-051, so relief is partial).
- Decision: A first, behind `ARC_PREFAULT` default off until N-07 shows a benefit; B only if N-07 shows A does not recover the cold-cache cost on Windows. Keep the direct-mapped ring. Close the slot-reuse hazard (a slot is reusable only after its layer is released) and reflect any eviction-rule change in RingSlots.tla.
- Consequences: no DLL rebuild for stage 1; the optional flag keeps behaviour identical until measured; TLC must be re-run locally if the eviction rule changes; a Python pre-touch burns a CPU thread and gives no asynchronous kernel readahead (P2-02 option 3 trade-off).
- Reversibility: high for A (flag and commit revert); B is covered by DL-066.

### DL-065 Cache unification (WP-12)
- Context: `CACHE_DIR` is set only for GPU and relative to the working directory (P1-08, READ); NPU, CPU, the speculative target and draft, and every non-LLM pipeline get no properties (P1-08, P5-07, READ); `AOTCompilationManager` writes `~/.arc-runner/cache/blobs/<DEV>/<model>`, so an AOT compile does not warm what engines read (P5-07, READ); `warm_load_time_ms` is a constant on failure (P5-08, READ, honesty). NPU TTFT is a flat 2.23-2.27 s across prompt lengths (P3-02, PROVEN); its cause and whether a cache removes it are HYPOTHESES. Compile and load time are in no result file (P3-03).
- Options: (A) one absolute cache root with a device-keyed layout used by every engine and by the AOT manager; (B) keep GPU-only but make it absolute; (C) do nothing but record cold load time.
- Decision: A, preceded by recording cold and warm load seconds in the schema (WP-02) so the benefit is measured, not assumed. NPU static-shape properties are added per workload only where a committed result shows a win, with property names verified against the installed GenAI. The user-supplied `CACHE_DIR` override is kept.
- Consequences: one directory to warm, size and purge; every load path changes (medium risk); stale-blob behaviour after driver or OpenVINO updates must be tested (HYPOTHESIS that OpenVINO hashes model plus config); an unwritable root degrades with a logged reason.
- Reversibility: high. Cache directories are disposable and the change is a helper plus call sites.

### DL-066 ABI 4 batching policy (WP-10, WP-11)
- Context: any change to an exported function requires bumping `ARC_NATIVE_ABI_VERSION` in four places and rebuilding both libraries with sidecars in the same commit (CLAUDE.md); the Windows DLL cannot be rebuilt in the cloud (DL-040), so each bump costs a local rebuild. Candidate exports from P2: `arc_mmap_advise` (P2-02), `arc_mmap_get_layer_ptrs` and `arc_ze_dma_copy_batch` (P2-08 option 3), `arc_ze_host_copy` (P2-06 option 2), immediate-list submit and wait (P2-05 option 3). Several improvements need no new export: memmove out of the spinlock (P2-06 option 3), a layer index table (P2-08 option 2), a copy-only queue ordinal found internally (P2-05 option 2). The Python side of per-tensor overhead was replica-measured at 2.95-11.09 us per tensor (P2-08) and is small against I/O and DMA.
- Options: (A) bump per export as each is justified; (B) one bump to 4 bundling only the exports justified by then, with internal C changes shipped earlier without a bump; (C) never change the C ABI, only Python.
- Decision: B. An export with no supporting measurement (N-07, N-08, N-09) is not added; ABI 3 stays through WP-10. The four ABI strings (header, `bridge.py`, `build_native.py`, `arc_native_build_info`) change in one commit with both libraries and sidecars.
- Consequences: at most one local DLL rebuild for exports; hardware numbers gate native work; if no export is justified the ABI never changes; a stale library still falls back to the Python path through the existing ABI check.
- Reversibility: medium. Reverting the commit restores ABI 3; the ABI check prevents a mixed state from loading.

### DL-067 Honesty-fix policy (WP-01)
- Context: invented or constant values reach APIs and result fields: SQNR 34.6 and 99.0 (P4-16), word-count tokens in `usage` and `prompt_eval_count=0` without a flag (P4-16, residual of H4), `0.15` and `12.0` ms in the AOT report (P5-08), constants 0.001, 0.038, 9.5, 0.22, 0.25, 4.5 in media telemetry (P5-10), swallowed configuration errors (P1-13, P5-12), a TPS numerator off by one (P1-09, DERIVED +5.9 % at 18 tokens, +0.5 % at 200). All READ. Earlier decisions already set the pattern: DL-006, DL-014, DL-036, DL-037, DL-056, DL-057.
- Options: (A) mark the value "estimate" and keep it; (B) keep and document; (C) replace by `None` plus a validity flag plus a reason, with additive Optional fields, and replace swallowed errors by typed `ArcRunnerError` subclasses.
- Decision: C. Published figures produced by an old formula are relabelled with the formula, never silently changed. Each fix is red-first with the literal added to the grep guard in `tests/test_honesty_no_fabrication.py`. Client-visible nulls are listed in the CHANGELOG.
- Consequences: some API fields become null or omitted (clients must tolerate it); a request that used to succeed with dropped sampling parameters now fails loudly; no number is shown that was not measured.
- Reversibility: high per item, but reverting reintroduces a fabricated value and is not recommended; the fields stay additive either way.

### DL-068 Speculative decoding gate (WP-13)
- Context: committed same-model data gives a step-cost ratio c = t_draft/t_target = GPU tps / NPU tps = 41.42/14.67 = 2.82 (P1-07, DERIVED from P3 medians); with the standard model S <= (g+1)/(g*c+1), g=4 gives 5/12.28 = 0.41. g+1 > g*c+1 reduces to c < 1, so the bound exceeds 1 if and only if the draft is cheaper per step than the target, whatever g is (DERIVED here); c for a real small draft is UNMEASURED and no speculative result exists (P1-07, P3). The engine ignores the requested device (P1-05, P5-13, READ), recompiles when gamma changes (P1-06, READ), has dead dynamic-gamma code (P1-06, READ), and the draft can silently fall back to CPU (P1-07, P5-13, READ). Acceptance and speed-up are reported as 0 and 1.0 until measured (DL-036).
- Options: (A) leave speculative decoding enabled and add a warning; (B) gate it on a committed calibration (measured c < 1 and a measured speed-up with a lower confidence bound above 1 under the stats module), otherwise fall back to the plain target with a clear log; (C) default the draft to GPU or CPU and leave NPU experimental.
- Decision: B. No acceptance rate is ever inferred; the dead gamma code is deleted unless the pipeline exposes counters. The override for unmeasured use is a maintainer decision (plan 23.8 question 3).
- Consequences: speculative decoding is off by default for any model without calibration; docs lose any NPU-draft speed-up wording that lacks a result file (H-15); one new workload `llm_speculative` and a hardware calibration run (N-16) are needed before it can be enabled.
- Reversibility: high. The gate is a pre-check on enabling.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | DL-001..DL-055 consolidated from the plan and the stage records | create the log | pending |
| 1.1.0 | 2026-10-29 | main session | DL-056..DL-060 | S-G decisions | pending |
| 1.2.0 | 2026-10-29 | main session | DL-061..DL-068 and ADR detail | performance bottleneck programme, plan section 23 | pending |
