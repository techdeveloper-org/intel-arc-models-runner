# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/). Tracking issue: [#42](https://github.com/techdeveloper-org/intel-arc-models-runner/issues/42).

## [UNRELEASED]

### Added
- Context-window enforcement in `HybridSpeculativeEngine` (shared `ContextBudgetMixin`).
- Opt-in best-effort layer pinning (`pin_resident_layers`, `ARC_PIN_RING=1`) and opt-in Level Zero USM staging (`DeviceLayerStager`) in the flash-stream weight manager.
- Sysman package power (energy-counter delta), GPU frequency and free device memory, each `None` with a reason when unavailable; `benchmark.power.PowerMeter`.
- Schema-v1 `workload` and `metrics` fields (additive), `benchmark/workloads`, `benchmark/nonllm`, `benchmark/harness_common`; schema-v1 runners for LLM, embeddings, VLM (`scratch/run_live_vlm_benchmarks.py`), audio, image and video.
- u8 KV-cache quality gate (`benchmark/kv_quality`, `scripts/bench_kv_quality.py`, hypothesis H-D); `scripts/check_mermaid.py` and a CI job that renders every Mermaid block.

### Changed
- Image and video stage timings, step rate, power, energy yield and peak memory are `None` when the engine did not measure them; the studio pipeline and co-benchmark no longer attach fixed memory, RTF, tok/s or cloud-price figures (ROI is computed only from caller-supplied prices).
- Four hand-written Mermaid diagrams (communication, component, composite, object) and the state-diagram generator were fixed so they parse.

- Streamed `total_tokens` use the pipeline tokenizer; `scripts/list_unverified_numbers.py`; plan test-list audit with new boundary, 80-layer, lock-refusal and `@hardware` tests.

### Removed
- `scratch/run_full_catalog_benchmark_audit.py` (fixed timings and a constant PASS) and the legacy result JSONs holding formula-derived numbers.

## [0.2.0] - 2026-10-29

Not published yet: tagging and publishing (PyPI, GHCR) need the maintainer's explicit approval, and the items under "Needs a real machine" are not done.

### Added
- Native memory-mapping layer ABI version 3: `arc_range_ok`, status codes, sealed layer table, UTF-8/long-path file open on Windows, build info and sidecar (`<library>.build.json` with sha256, compiler, flags, ABI) verified by the Python bridge.
- `NativeLayerError`, `NativeBoundsError`, `ARC_NATIVE_DISABLE`, `ARC_ZE_LOADER_PATH`, `ARC_ZE_SYNC_TIMEOUT_MS`, `ARC_KV_CACHE_PRECISION`.
- Direct-mapped ring buffer with a single loader per slot, safe shutdown, deferred unmap while NumPy views are alive, `get_resident_memory_bytes()`; model in `docs/formal/RingSlots.tla` and `scripts/check_ring_model.py`.
- `arc_ze_types.h` and a rewritten Level Zero bridge: GPU-only selection, bounded waits (< Windows TDR), tracked allocations, dead-context handling.
- `EngineTelemetry` fields `metrics_source`, `is_estimate`, `is_ttft_valid`, `is_tps_valid`, `prompt_tokens`, `tpot_ms`, `decode_tokens_per_second`, `kv_cache_precision`; `engine/telemetry.py`.
- Context-window enforcement in the OpenVINO engine (prompt and prompt + `max_new_tokens`), `tokenize()`.
- Benchmark statistics and claim gates (`arc_runner.benchmark.stats`), schema-v1 result files, `docs/benchmarks/METHOD.md`, `scripts/{validate_bench_json,render_bench_docs,probe_hardware,bench_native_micro,bench_regression}.py`.
- C sanitizer harnesses (`tests/native_c/`), CI workflows for Linux and Windows, `scripts/check_docs_sync.py`, `scripts/check_delivery_docs.py`, `SRS.md`, `CLAUDE.md`, `docs/delivery/`.

### Changed
- The u8 KV cache is no longer the default on GPU; opt in with `KV_CACHE_PRECISION=u8` or `ARC_KV_CACHE_PRECISION=u8`.
- `generate()` uses one pipeline call form; failures propagate instead of being retried with another form.
- Tensors are viewed with their own dtype (bfloat16 as raw uint16, quantized types as raw uint8) and row-major reversed GGUF dimensions in both native and Python paths.
- Level Zero structure types now match `ze_api.h`; queue priority is NORMAL; the synchronize timeout is 1.5 s (max 1.9 s) instead of 10 s.
- `-mavx2` removed from the native build; native sources build with `-std=c17 -Wall -Wextra -Wpedantic -Werror`.
- Student-t multipliers for 11-19 degrees of freedom were too small in both runners; both now use one exact table.
- Version strings of the Ollama-compatible endpoint and the API app follow the package version.

### Fixed
- Out-of-bounds pointer from wrapped `offset + size` in the mapping layer (also in lock/unlock), unchecked tensor counts, silent truncation of tensor counts, names and dimensions, negative integers wrapping in ctypes.
- GGUF parser crashed with `MemoryError` / `OverflowError` on hostile length fields.
- Duplicate loads and shutdown races in the ring buffer.

### Removed
- Fabricated telemetry: constant `22.0` W package power, system RAM reported as GPU free memory, word-count token estimates, `elapsed * 200/400/500` TTFT guesses, speculative acceptance and speed-up generated from constants, Ollama-compatible duration splits of 0.2/0.8, `hash(word)` prefix-cache token ids, hard-coded `REFERENCE_SWEEP` benchmark rows and hard-coded heavy-model constants. Unverified published numbers are labelled in the docs (see `docs/benchmarks/`).

### Needs a real machine (not done)
- Rebuild and commit the Windows DLL, run `pytest -m hardware`, run `scripts/probe_hardware.py`, run the benchmark matrix and the regression benchmark, run TLC on `docs/formal/RingSlots.tla`.

## [0.1.0]

Initial release (PyPI wheel and container).
