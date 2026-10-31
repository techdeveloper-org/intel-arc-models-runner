---
doc_id: DOC-UI-01-STAGE
title: Modernization of Intel Arc Model Studio UI & Telemetry
stage: UI-01
version: 1.0.0
status: draft
created: 2026-10-31
updated: 2026-10-31
authors: [Gemini 3.8 Flash, maintainer]
reviewers: [user]
inputs: [src/arc_runner/server/templates/index.html, src/arc_runner/server/static/css/tokens.css, src/arc_runner/server/static/js/app.js]
---

# STAGE.md — UI-01: Modernization of Intel Arc Model Studio UI & Telemetry

## 1. Executive Summary
- **Stage ID**: `UI-01`
- **Domain**: `frontend-ui-ux` & `silicon-architecture`
- **Lead Agents**: `ui-ux-mathematics-engineer`, `ui-ux-designer`, `frontend-codegen-engineer`, `foundation-model-mathematics-expert`, `gpu-architecture-specialist`, `as-built-doc-generator`
- **Model Engine**: Gemini 3.8 Flash (Antigravity Environment)
- **Status**: Completed & Verified

## 2. Deliverables & Architectural Modifications
1. **Mathematical Design Tokens (`src/arc_runner/server/static/css/tokens.css`)**:
   - 8pt Baseline Spatial Grid (`--space-4` through `--space-96`).
   - Golden Ratio Layout Partition ($\Phi \approx 1.618034$).
   - Major Third Modular Typographic Scale ($r = 1.250$, base 16px).
   - OKLCH Perceptually Uniform Palette with APCA $L_c \ge 75$ dark console contrast.
   - Damped harmonic motion spring physics ($\zeta = 0.85$, `cubic-bezier(0.2, 0.0, 0.0, 1.0)`).
2. **Template Overhaul & Proof Purge (`src/arc_runner/server/templates/index.html`)**:
   - Eradicated hardcoded `1.86x Active`, `Intel NPU 5W`, and `Proof M135 Cloud ROI` claims.
   - Cleaned internal proof assertions (`Proof M1`, `Proof M2`, `Proof M131-M135`).
   - Integrated dynamic header chips (`#header-device-chip`, `#header-model-chip`, `#header-server-status`).
3. **Dynamic Client Telemetry Engine (`src/arc_runner/server/static/js/app.js`)**:
   - `HardwareTelemetryPoller`: Polls `/api/devices` every 3000ms, mapping Level Zero GPU name, USM shared memory state, and Intel NPU presence.
   - `StreamTelemetryTracker`: Uses microsecond `performance.now()` timestamps to compute real TTFT, TPOT, and live TPS during streaming.
   - Purged dead `updateRoiMetrics` calculations.
4. **Architectural UML Diagram**:
   - Created `uml/webui_telemetry_flow.puml` mapping asynchronous hardware heartbeat and streaming inference sequence.

## 3. Quality & Standards Conformance
- `rules/12-docstrings-only.md`: Zero inline narration comments; JSDoc contract docstrings.
- Strict non-BOM UTF-8 validation across all files.
- Automated API test suite `tests/test_ui_endpoints.py`: 24/24 passed (100%).

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-31 | Gemini 3.8 Flash | Modernization of WebUI and telemetry | UI-01 implementation | pending |
