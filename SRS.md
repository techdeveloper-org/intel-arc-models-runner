# Software Requirements Specification: arc-runner native layer, telemetry and benchmarking

Version: 0.2.0 | Status: draft | Source issue: [#42](https://github.com/techdeveloper-org/intel-arc-models-runner/issues/42) | Append-only: requirements are never edited in place; a changed requirement gets a new ID and the old one is marked superseded.

## 1. Introduction

### 1.1 Purpose
Specifies the behaviour required of the native acceleration layer (memory-mapped weight streaming, Level Zero bridge), the engine telemetry, and the benchmark process of `intel-arc-models-runner` (package `arc-runner`).

### 1.2 Scope
Covers `src/arc_runner/native/`, `engine/flash_stream/weight_streamer.py`, `engine/openvino_engine.py`, `engine/telemetry.py`, `devices/sysman.py`, `quant/gguf_parser.py` (hostile input), `src/arc_runner/benchmark/`, and `docs/benchmarks/`. Other components of the runner are out of scope.

### 1.3 Definitions
- **Extent**: a byte range `[offset, offset + size)` inside the model file.
- **Measured / derived / projected**: provenance labels of every published number (see `docs/benchmarks/METHOD.md`).
- **TDR**: Windows Timeout Detection and Recovery, 2 s for a GPU stall.

### 1.4 References
`docs/plans/MASTER_PLAN_v4.md`, `docs/benchmarks/METHOD.md`, `docs/formal/RingSlots.tla`, `docs/delivery/`, `ze_api.h` (oneapi-src/level-zero).

## 2. Overall description

### 2.1 Product perspective
The native layer is optional: if the library is missing, disabled (`ARC_NATIVE_DISABLE`), built with another ABI version, or fails its sidecar hash check, the Python path serves identical bytes.

### 2.2 Users
Operators running local models on Intel Arc GPUs/NPUs; contributors benchmarking other Intel silicon.

### 2.3 Constraints
Windows x64 is the supported platform for the native DLL (DL-035); POSIX branches get the same bounds checks, are covered by sanitizer tests, and make no performance claims. GCC is required to build.

### 2.4 Assumptions
Model files are untrusted input. The Intel GPU driver provides a Level Zero loader at run time.

## 3. Specific requirements

### 3.1 Functional requirements

| ID | Requirement | Source | Acceptance criteria (test) |
|:--|:--|:--|:--|
| FR-001 | Native tensor access shall never form a pointer outside the mapped file: every extent is validated against the file size at registration and at read time, counts, names, dimensions and integers are validated without truncation or wrapping, the layer table is sealed before concurrent reads, and views never outlive the mapping. | #42 | `tests/native_c/mmap_harness.c` (1,000,000-case fuzz, ASan/UBSan), `tests/test_native_bounds.py`, `tests/test_flash_stream_ring.py::test_close_with_live_view_defers_unmap_until_view_is_released` |
| FR-002 | Engine telemetry shall contain only values measured by the engine or derived from documented counters; unmeasured values are zero with their validity flag False and `metrics_source` states the origin. | #42 | `tests/test_honesty_no_fabrication.py`, `tests/test_telemetry_details.py` |
| FR-003 | Sysman and Level Zero queries shall report real values or `None` with a reason: no constant power, no system RAM as GPU memory, GPU devices only (never the NPU), correct structure types. | #42 | `tests/test_honesty_no_fabrication.py`, `tests/test_level_zero_bridge.py`, `tests/native_c/ze_harness.c` |
| FR-004 | The OpenVINO engine shall reject a request whose prompt, or prompt plus `max_new_tokens`, exceeds the context window before any pipeline call, using the pipeline tokenizer. | #42 | `tests/test_engine_context_and_kv.py` |
| FR-005 | Published benchmark numbers shall come from schema-v1 result files with a source label; a speed claim shall pass the claim gates (measured, no fallback, at least 10 timed values, CV <= 10%, bootstrap ratio lower bound > 1). | #42 | `tests/test_benchmark_stats.py`, `tests/test_bench_results.py`, `python scripts/render_bench_docs.py --check` |
| FR-006 | The ring buffer shall load a layer at most once concurrently per slot, shall not deadlock when layers collide on a slot, and shall shut down without leaving a loader running. | #42 | `tests/test_flash_stream_ring.py`, `scripts/check_ring_model.py`, `docs/formal/RingSlots.tla` |
| FR-007 | A synchronize call on the Level Zero bridge shall return within the configured timeout (default 1.5 s, maximum 1.9 s) and shall mark the context dead on timeout or device loss. | #42 | `tests/test_level_zero_bridge.py`, `tests/native_c/ze_harness.c` |
| FR-008 | The system SHALL measure token counts at the engine boundary using the OpenVINO tokenizer and set `is_estimate: False` in telemetry. When the tokenizer is unavailable, the system MAY emit an estimate with `is_estimate: True`. | GitHub Issue #45 (WP-14) | `tests/test_token_accounting.py` |
| FR-009 | The `arc_runner` package SHALL defer all heavy imports (OpenVINO, numpy, torch) until first attribute access via PEP 562 `__getattr__`, reducing cold-start latency without breaking existing import paths. | GitHub Issue #45 (WP-15) | `tests/test_lazy_imports.py` |
| FR-010 | The system SHALL record and publish GPU decode throughput (tok/s) for supported models using both the 30-run harness protocol (schema-v1, `*.v1.json`) and the quick-API-benchmark protocol (schema `quick_v1`, `*.quick.json`). Published values SHALL be derived from measured result files only; no value is hardcoded or estimated. | GitHub Issue #49 | `docs/benchmarks/results/llm_*.json`, `scripts/validate_bench_json.py`, `tests/test_bench_results.py::TestQuickBenchSchema` |

### 3.2 Non-functional requirements

| ID | Requirement | Source | Acceptance criteria |
|:--|:--|:--|:--|
| NFR-001 | Input validation: model files, paths, sizes and counts are treated as hostile; the GGUF parser rejects lengths larger than the remaining file and more than four dimensions. | #42 | `tests/test_gguf_untrusted_input.py` |
| NFR-002 | Portability: missing, disabled, stale-ABI or hash-mismatched native libraries fall back to the Python path with identical tensor bytes. | #42 | `tests/test_native_bounds.py::test_native_and_python_paths_return_identical_tensor_bytes` and the ABI/sidecar tests |
| NFR-003 | Reproducible measurements: environment, protocol and provenance are recorded with every result; documentation tables are generated, not edited. | #42 | `python scripts/validate_bench_json.py`, `python scripts/render_bench_docs.py --check` |
| NFR-004 | Documentation traceability: stage records, decisions and version history live under `docs/delivery/`, and code, docs and diagrams are checked for drift. | #42 | `python scripts/check_delivery_docs.py`, `python scripts/check_docs_sync.py` |
| NFR-005 | Build hygiene: native sources compile with `-std=c17 -Wall -Wextra -Wpedantic -Werror` (gcc and mingw-w64) and ship no instruction-set extensions beyond the baseline. | #42 | `tests/test_native_bounds.py::test_compile_library_*`, CI |

## 4. System models
Architecture and diagrams: `docs/architecture.md`, `uml/`, `drawio/`.

## 5. Appendices
Findings and their mapping to requirements: `docs/delivery/TRACEABILITY.md`.

## 6. Change log

| Date | Version | Change | Source |
|:--|:--|:--|:--|
| 2026-10-29 | 0.2.0 | initial requirements FR-001..FR-007, NFR-001..NFR-005 | #42 |
| 2026-10-31 | 0.2.0 | WP-14 engine token accounting | Add FR-008: measured token counts with is_estimate flag | #45 |
| 2026-10-31 | 0.2.0 | WP-15 lazy package import | Add FR-009: PEP 562 __getattr__ lazy imports | #45 |
| 2026-10-31 | 0.2.0 | fix(tests): Windows parametrize ids | Change log row only (fix type per rule 44 §3) | #45 |
| 2026-10-31 | 0.2.0 | fix(scripts,hub): Windows test failures | Change log row only (fix type per rule 44 §3) | #45 |
| 2026-10-31 | 0.2.0 | feat(benchmarks): publish GPU tok/s for qwen-2.5-1.5b and deepseek-r1-1.5b | Add FR-010: benchmark throughput publication via both full-harness and quick-API protocols | #49 |
