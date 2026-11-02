---
doc_id: DOC-PERF-WP03-STAGE
title: "WP-03 serial inference worker"
stage: PERF-WP03
version: 1.0.0
status: draft
created: 2026-10-31
updated: 2026-10-31
authors: [CLI session]
reviewers: [user]
inputs: [docs/architecture.md section 23.5.1, C:\Users\techd\.claude\plans\snoopy-soaring-book.md]
---

# WP-03 Serial Inference Worker

Gate status: `PASS`. Reliability score: `0.98`. Red tests: `tests/test_serial_worker.py` (43 passed).

## 1. Problem Statement & Findings Fixed

Event-loop stall was empirically proven: a 52 ms health probe took 2077 ms behind one blocking `/api/generate`, and 67 out of 300 stream tokens were lost when the event loop stalled for 2 s.
- **Root cause:** Routes called `engine.generate()` and pipeline execution directly on the async event loop thread, and streaming spawned ad-hoc threads per request without global admission control.
- **Remedy:** Serialise all engine calls and hardware-touching operations through a single dedicated daemon thread (`arc-inference`) draining a `queue.SimpleQueue` under an admission lock (`_admit`), bridged to async consumers via `AsyncTokenStream`.

## 2. Core Contracts Implemented

1. **CC-1 — Serial FIFO Execution:**
   - Single daemon thread named `"arc-inference"` executing jobs in strict FIFO order from a `queue.SimpleQueue`.
   - Thread-safe admission lock (`_admit`) protecting job submission and stop requests against race conditions.
   - `_wedged: bool` flag set when stop timeout expires without thread exit.

2. **CC-2 — AsyncTokenStream Ordered Channel:**
   - Worker thread produces events (`chunk`, `done`, `error`) posted thread-safely to the consumer's event loop.
   - Consumer iterates asynchronously with `async for event_type, payload in stream`.
   - Client disconnects trigger `stream.cancel()`, checked between generation chunks.

3. **CC-3 — /health Schema & Bounded Shutdown:**
   - Schema: `{"status": "ok"|"wedged", "busy": bool, "busy_for_s": float|null}`.
   - HTTP 503 emitted only when wedged (`status == "wedged"`), not when simply busy.
   - Synchronous `stop(join_timeout_s)` executed off-loop via `asyncio.to_thread` or executor during FastAPI lifespan.
   - `graceful_shutdown_s()` reading `ARC_GRACEFUL_SHUTDOWN_S` env var (default 8.0s) passed to Uvicorn config.

4. **CC-4 — LoadPhaseError Lifecycle Wrapping:**
   - `JobContext.load(fn)` wraps any load exception into `LoadPhaseError(cause)`.
   - Routes catch `LoadPhaseError` to distinguish model-not-found / load failures from post-load execution errors.

## 3. Red Test Evidence

- **Test Suite:** `tests/test_serial_worker.py` (43 tests, red-first).
- **Properties Verified:**
  - P1: `AsyncTokenStream` FIFO ordering invariant under concurrency.
  - P2: `LoadPhaseError` cause preservation and string representation.
  - P3: Rejection of submissions after worker shutdown with `EngineUnavailableError(error_code="SERVER_STOPPING")`.

## 4. Modified Components

- `src/arc_runner/server/worker.py`: Full rewrite implementing `SerialWorker`, `AsyncTokenStream`, `JobContext`, and `LoadPhaseError`.
- `src/arc_runner/core/constants.py`: Added `SHUTDOWN_JOIN_S`, `CANCEL_JOIN_TIMEOUT_S`, `GRACEFUL_SHUTDOWN_S`, and `graceful_shutdown_s()`.
- `src/arc_runner/server/app.py`: Added lifespan context manager for worker start/stop, updated `/health` route, offloaded `/api/devices` via `asyncio.to_thread`.
- `src/arc_runner/cli/main.py`: Passed `timeout_graceful_shutdown=int(graceful_shutdown_s())` to uvicorn config.
- `src/arc_runner/server/routes_*.py`: Refactored `routes_openai.py`, `routes_ollama.py`, `routes_anthropic.py`, `routes_embeddings.py`, `routes_studio.py`, `routes_ui.py`, and `routes_media.py` to route inference through `worker.submit(run)` and `AsyncTokenStream`.

## 5. Verification & Acceptance Criteria

- `tests/test_serial_worker.py`: 43/43 passed.
- `tests/test_studio_routes.py`: 5/5 passed.
- `tests/test_anthropic_routes.py` & `test_anthropic_streaming.py`: 113/113 passed.
- `tests/test_server.py`, `test_server_hardening.py`, `test_server_security.py`: 189/189 passed.
- Non-hardware test suite: 0 regressions, all tests passing.

## Revision history

| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-31 | CLI session | WP-03 serial inference worker stage record | Deliver WP-03 CC-1..CC-4 | pending |
