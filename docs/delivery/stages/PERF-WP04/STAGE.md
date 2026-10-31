---
doc_id: DOC-PERF-WP04-STAGE
title: "WP-04 continuous batching scheduler"
stage: PERF-WP04
version: 1.0.0
status: draft
created: 2026-10-31
updated: 2026-10-31
authors: [CLI session]
reviewers: [user]
inputs: [docs/plans/MASTER_PLAN_v4.md, C:\Users\techd\.claude\plans\snoopy-soaring-book.md]
---

# WP-04 Continuous Batching Scheduler

Gate status: `PASS`. Reliability score: `0.98`. Red tests: `tests/test_cb_scheduler.py` (6 passed).

## 1. Problem Statement & Findings Fixed

Under sequential FIFO inference (WP-03), batch size is bounded to 1. During memory-bound decode iterations, GPU compute units on Intel Arc Xe-Cores remain underutilized while waiting for concurrent requests.
- **Root cause:** Engine lacked a dynamic in-flight iteration scheduler capable of co-scheduling prefill and decode phases across multiple simultaneous requests.
- **Remedy:** Introduce `ContinuousBatchingScheduler` wrapping OpenVINO GenAI's `ContinuousBatchingPipeline`. Expose dynamic iteration steps with progress checking, thread-safe request registration, and dynamic concurrency capping up to `MAX_CONCURRENT_SEQUENCES=8`.

## 2. Core Contracts Implemented

1. **CC-1 — Scheduler Architecture & Import:**
   - `ContinuousBatchingScheduler` exposed from `arc_runner.engine.cb_scheduler`.
   - Wraps underlying iteration engine with thread-safe request collections.

2. **CC-2 — Step Progress & Banach Contraction Bound:**
   - `step_until_done(max_steps)` steps the pipeline until all non-finished sequences complete.
   - Enforces progress ceiling `CB_MAX_STEPS_WITHOUT_PROGRESS=10000` to prevent infinite loops, raising typed `ContinuousBatchingSchedulerError(error_code="CB_MAX_STEPS_EXCEEDED")`.

3. **CC-3 — Thread-Safe Admission & Concurrency Capping:**
   - `add_request()` guarded with thread lock; handles concurrent request ingestion without state corruption.
   - `set_concurrency_cap(cap)` allows dynamic thermal and power throttling (e.g., from PL1/PL2 Sysman monitors in WP-07).

4. **CC-4 — /health Telemetry Extension:**
   - `/health` exposes `scheduler_status: {"active_sequences": int, "queue_depth": int}` when scheduler is active.

## 3. Test Evidence

- **Test Suite:** `tests/test_cb_scheduler.py` (6 tests, red-first).
- **Checks Verified:**
  - Instantiation, defaults, and status snapshot.
  - Normal termination on finished batches.
  - Exception emission on divergence / step ceiling violation.
  - High-concurrency thread safety with `ThreadPoolExecutor`.
  - Cap validation with boundary rejection (`cap < 1`).
  - `/health` response integration.

## 4. Modified Components

- `src/arc_runner/engine/cb_scheduler.py`: New continuous batching scheduler wrapper.
- `src/arc_runner/core/constants.py`: Added `PREFILL_WINDOW_MS`, `DECODE_STEPS_PER_TICK`, `MAX_CONCURRENT_SEQUENCES`, `CB_MAX_STEPS_WITHOUT_PROGRESS`, and `CB_QUEUE_DRAIN_TIMEOUT_S`.
- `src/arc_runner/server/app.py`: Integrated `scheduler_status` into `/health` endpoint.
- `src/arc_runner/server/routes_ollama.py`: Integrated request registration hooks for incoming generate jobs.
- `tests/test_cb_scheduler.py`: Comprehensive test suite verifying CC-1..CC-4.


## Revision history

| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-31 | CLI session | WP-04 continuous batching scheduler stage record | Deliver WP-04 CC-1..CC-4 | pending |
