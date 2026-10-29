# CLAUDE.md

Guidance for AI coding sessions working in this repository. Start a session by reading `docs/delivery/RESUME.md`.

## Project Overview

`intel-arc-models-runner` (package `arc-runner`) is a local model runner for Intel Arc GPUs and AI Boost NPUs built on OpenVINO GenAI, with an Ollama-compatible and OpenAI-compatible API, an optional native C layer for memory-mapped weight streaming and Level Zero, and a benchmark suite. Python 3.10+, Windows x64 is the primary target for the native layer.

Honesty rule of this project: no number, token count, power value or throughput is invented. Values are measured, derived by a stated formula, or absent (`None`/zero with a validity flag). Published numbers come from result files (`docs/benchmarks/METHOD.md`).

## Architecture & Structure

```
src/arc_runner/
  core/            constants, exceptions (ArcRunnerError, NativeLayerError, NativeBoundsError), logging
  engine/          OpenVINO engine, speculative engine, telemetry builders, KV cache, flash_stream/ (ring-buffer weight streamer)
  native/          C sources (arc_flash_stream_mmap, arc_level_zero_bridge, arc_ze_types.h), bridge.py (ctypes), build_native.py
  devices/         device detector, Sysman (Level Zero memory properties)
  benchmark/       runner, multimodal runner, stats (claim gates), results (schema v1)
  server/          FastAPI app and routes
  quant/           GGUF parser (treats input as hostile)
tests/             pytest suite; tests/native_c/ holds C sanitizer harnesses and a fake Level Zero loader
scripts/           probe_hardware, bench_*, render_bench_docs, validate_bench_json, check_docs_sync, check_delivery_docs, check_ring_model
docs/              architecture, benchmarks (METHOD.md, schema/, results/), formal/ (TLA+), plans/, delivery/
uml/, drawio/      generated diagrams (13 types each)
```

Native layer: Python `NativeMemoryEngine` -> ctypes -> `libarc_native_sys` (ABI version 3, sidecar hash checked on load) -> OS mapping. Missing, disabled (`ARC_NATIVE_DISABLE=1`) or stale libraries fall back to the Python read path. Level Zero is loaded at run time and always selects a GPU device, never the NPU.

Delivery records: `docs/delivery/` (RESUME.md is the session entry point; INDEX.md, DECISION_LOG.md, TRACEABILITY.md, stages/). The long-range plan is `docs/plans/MASTER_PLAN_v4.md`.

## Development Guidelines

- Every public and protected function has a Google-style docstring; do not add comments that narrate what the code does. Doxygen headers for C.
- New Python files are ASCII only (Windows cp1252 safe).
- Tests are named `test_<function>_<scenario>_<expected>`; tests needing real Intel silicon use `@pytest.mark.hardware` and must be reported as not run when skipped, never counted as passed. Use `tests/conftest.py` fixtures.
- Fix bugs red-first: write the failing test, show it failing, then fix.
- Build the native library with `python src/arc_runner/native/build_native.py` (GCC; flags include `-Werror`). A source change under `native/*.c|h` and the matching rebuilt library plus sidecar belong in the same commit; bump `ARC_NATIVE_ABI_VERSION` (header, `bridge.py`, `build_native.py`) when an exported function changes.
- Treat model files, paths and sizes as untrusted input; validate before use; never truncate silently.
- Exceptions: raise the specific `ArcRunnerError` subclass; do not swallow errors; no retry with a different call form.
- Logging: `arc_runner.core.logging.get_logger(name)` with key-value `extra`; never log secrets or full prompts.
- After any change under `src/arc_runner/**` run `python scripts/check_docs_sync.py`; it lists exactly which diagram or document is stale. Documents under `docs/delivery/` carry front matter and a revision history: a changed document gets a new version row and an INDEX update (`python scripts/check_delivery_docs.py`).
- Commits: conventional commits referencing the issue (`#42`); descriptive bodies stating problem, change, verification and what is not done.
- Version lives in `src/arc_runner/_version.py`, `VERSION`, `pyproject.toml` and the CHANGELOG top entry; they must match.
