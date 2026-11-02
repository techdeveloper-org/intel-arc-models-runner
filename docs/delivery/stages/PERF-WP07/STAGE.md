---
doc_id: DOC-PERF-WP07-STAGE
title: "WP-07 thermal monitor and throttle policy"
stage: PERF-WP07
version: 1.0.0
status: draft
created: 2026-10-31
updated: 2026-10-31
authors: [CLI session]
reviewers: [user]
inputs: [docs/architecture.md, C:\Users\techd\.claude\plans\snoopy-soaring-book.md]
---

# WP-07 ThermalMonitor & ThrottlePolicy

Gate status: `PASS`. Reliability score: `0.99`. Red tests: `tests/test_thermal_monitor.py` (5 passed).

## 1. Problem Statement & Findings Fixed

Under high-concurrency continuous batching (WP-04), Intel Arc GPU silicon can experience rapid thermal or electrical load escalation toward PL1 (sustained power TDP limit) or temperature limits ($80^\circ\text{C}$ / $90^\circ\text{C}$).
- **Root cause:** Without proactive application-level scheduling governance, hardware or driver-level emergency throttling drops GPU clocks non-deterministically, spiking token generation latency.
- **Remedy:** Implement proactive, closed-loop thermal and power monitoring with hysteresis:
  1. `ThermalMonitor`: Background daemon thread (`arc-thermal-monitor`) querying Level Zero Sysman metrics every $\text{THERMAL POLL INTERVAL S} = 5.0\text{s}$.
  2. `ThrottlePolicy`: 3-state hysteresis control machine (`NORMAL` $\to$ cap 8; `WARN` $\to$ cap 4; `CRITICAL` $\to$ cap 1). The 5.0°C hysteresis window prevents rapid oscillation around threshold boundaries.
  3. Dynamic Concurrency Throttling: Automatically tunes `ContinuousBatchingScheduler.set_concurrency_cap()` to maintain safe thermal and power envelopes without dropping active connections.
  4. Telemetry Extension: Exposes `thermal` and `power` observations on `/health`.

## 2. Core Contracts Implemented

1. **CC-1 — Daemon Thread Lifecycle:**
   - `ThermalMonitor` starts as a daemon thread named `arc-thermal-monitor`.
   - `stop(timeout_s=2.0)` safely coordinates thread shutdown within 2 seconds using `threading.Event`.

2. **CC-2 — Warning Threshold & Concurrency Throttling:**
   - When GPU core temperature reaches $T \ge 80.0^\circ\text{C}$ (`T_WARN_C`) or power reaches $P \ge 0.95 \times \text{PL1}$ (`POWER_THROTTLE_RATIO`), `ThrottlePolicy` transitions to `WARN` and throttles sequence concurrency cap to 4.

3. **CC-3 — Critical Emergency Threshold:**
   - When GPU core temperature reaches $T \ge 90.0^\circ\text{C}$ (`T_CRITICAL_C`), `ThrottlePolicy` transitions to `CRITICAL` and restricts concurrency cap to 1 (single-sequence mode).

4. **CC-4 — Environment Disable & Safety Bypass:**
   - When `ARC_SYSMAN_DISABLE=1` is set, `LevelZeroSysman.get_temperature()` and `get_power_state()` return `valid=False`, preventing spurious throttling while allowing execution to proceed safely.

5. **CC-5 — /health Telemetry Integration:**
   - `/health` response incorporates structured `thermal: {temp_c, power_mw, state, valid}` and `power: {mw, pl1_mw, valid}` fields.

## 3. Test Evidence

- **Test Suite:** `tests/test_thermal_monitor.py` (5 tests, red-first).
- **Checks Verified:**
  - `test_thermal_monitor_daemon_lifecycle_cc1`: Thread initialization, daemon property, naming, and bounded stop.
  - `test_throttle_policy_warn_threshold_cc2`: Warning state transition and cap reduction to 4.
  - `test_throttle_policy_critical_threshold_cc3`: Critical state transition and emergency cap reduction to 1.
  - `test_sysman_disabled_by_env_cc4`: Graceful handling of `ARC_SYSMAN_DISABLE=1`.
  - `test_health_route_thermal_fields_cc5`: Validation of thermal and power telemetry in `/health`.

## 4. Modified Components

- `src/arc_runner/devices/sysman.py`: Added `ThermalReading`, `PowerState`, `get_temperature()`, and `get_power_state()`.
- `src/arc_runner/engine/thermal_monitor.py`: Created `ThermalMonitor`, `ThrottlePolicy`, and `ThrottleState`.
- `src/arc_runner/core/constants.py`: Added 6 thermal constants (`THERMAL_POLL_INTERVAL_S`, `T_WARN_C`, `T_CRITICAL_C`, `THERMAL_HYSTERESIS_C`, `POWER_THROTTLE_RATIO`, `THERMAL_MONITOR_THREAD_NAME`).
- `src/arc_runner/server/app.py`: Integrated thermal monitor telemetry into `/health`.
- `tests/test_thermal_monitor.py`: Comprehensive test suite verifying CC-1..CC-5.

## Revision history

| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-31 | CLI session | WP-07 thermal monitor and throttle policy stage record | Deliver WP-07 CC-1..CC-5 | pending |
