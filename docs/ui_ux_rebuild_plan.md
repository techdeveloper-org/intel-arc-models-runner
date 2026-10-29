# Arc Runner UI/UX rebuild plan

**Status:** Browser routing and urgent Chat safeguards implemented; broader screen rebuild remains planned.  
**Primary evidence:** current HTML/CSS/JS and backend routes, plus the checked-in desktop and 390 px chat captures.  
**Runtime note:** `127.0.0.1:11435` was not listening during this audit and no in-app browser was available. The captures are not a reliable representation of the current source: for example, the capture places New Chat in a different location from the current HTML. Runtime behavior remains to be confirmed after the service is available.

## What the audit found

The app currently serves one HTML document at `/` and uses click handlers to swap seven tab sections. Navigation does not update the URL, so a screen cannot be bookmarked, refreshed, or reached with browser Back/Forward. The API is already organized under `/api` and `/v1`; those contracts should stay stable while browser screens get named routes.

The Chat Playground puts model/device setup, LoRA blending, conversation history, context compaction, archiving, telemetry, image attachment, and message entry on the same surface. On mobile, the fixed-height pane and horizontal navigation make the conversation and secondary controls hard to reach. Several messages imply capabilities or measurements the current request path does not provide.

### Confirmed functional defects

- **Image input is not implemented end to end.** The browser reads an image and shows a preview, but `/api/chat` receives neither image data nor a vision-capable engine request. The file input is being made visibly unavailable until that path is implemented.
- **Model discovery failure was not a send gate.** An empty or failed `/api/tags` request disabled the model picker but left the composer enabled. The composer is being gated on an available model and the recovery state gets Retry and Model Hub actions.
- **Benchmark results are fabricated.** The browser inserts fixed TTFT, TPOT, and throughput figures after a timeout instead of calling a benchmark endpoint. Treat this as P0; remove the figures and connect the real benchmark flow before presenting results as measured.
- **Model Hub pull has no user-visible progress or result.** It calls `/api/pull`, but discards the response stream after showing an alert that says the request was sent. Replace this with real progress, completion, failure, and retry states.
- **The current screenshots are stale versus current HTML.** They show a header New Chat control while current source places the original control in history. A new persistent header action is being added.
- **Metric and status copy needs trust cleanup.** Chat rates are browser text estimates, not hardware token telemetry. All estimated values must say so, and missing data must stay blank instead of using a plausible-looking constant.

## Browser route map

`/` remains the compatibility entry point and opens Cockpit. Named routes render the same shell and activate the matching screen on first load, refresh, or direct link. Back/Forward changes the active screen. `/ui` remains supported. Existing API endpoints do not move.

| Screen | Route |
|---|---|
| Silicon Cockpit | `/cockpit` (also `/`) |
| Chat Playground | `/chat` |
| Ollama & GGUF | `/models/ollama` |
| Model Hub | `/models/hub` |
| Benchmarks | `/benchmarks` |
| NPU Embeddings | `/embeddings` |
| Multi-Modal Studio | `/studio` |

This explicit allow-list avoids a catch-all route that could intercept `/api/*`, `/v1/*`, or static assets.

## Delivery phases

### P0 — Correctness and trust

1. **Implemented:** named, directly reachable browser routes; server-rendered active navigation/title plus client Back/Forward synchronization.
2. Make Chat model discovery a real gate. Provide retry and a clear path to Model Hub when the model list is empty or unavailable.
3. Keep image input disabled with an explanation until the model-capability contract, request payload, backend vision engine, and response flow all work together.
4. Replace simulated benchmark output with actual backend measurements. Show configuration, warm-ups, measured run count, confidence information only when calculated, and explicit failures.
5. Replace Model Hub's alert with a streamed pull state: queued, downloading, verifying, ready, failed, cancel/retry as supported by the server.
6. Replace blocking alerts and silent no-ops with inline status and recovery actions.

### P1 — Chat Playground first

1. **Primary path:** choose a ready model → type prompt → send → see streamed answer → stop or retry when needed.
2. Give the conversation the main area. Keep the composer visible, use the remaining height for messages, and remove fixed-height clipping on mobile.
3. Keep history in a discoverable drawer. Put LoRA, archive, context compaction, and diagnostics behind Settings/Advanced.
4. Only show generation status during work. Put performance readings below the conversation or behind Diagnostics, with “estimated from displayed text” where applicable.
5. Define loading, no-model, offline, unsupported-device, streaming, stopped, failed, storage-unavailable, and empty-history states.
6. Ensure every action has a visible result, every disabled action explains why, and every failure offers a next step.

### P2 — Model management and evaluation

- **Model Hub:** real catalog metadata, compatibility, disk requirement, installed state, download progress, verification, cancellation, retry, and launch-to-chat action.
- **Ollama & GGUF:** separate discovery, inspect, and convert steps; use a side panel for tensor details and a persistent conversion progress/result area.
- **Benchmarks:** real API-backed runs only; default to an honest empty state; preserve reproducible model/device/run configuration and label measured versus derived values.

### P3 — Remaining product screens

- **Embeddings:** two clearly named text fields, model/device choices under Options, one primary action, then similarity score with interpretation and copy action. Remove “NPU READY” unless device detection confirms it.
- **Multi-Modal Studio:** guided brief and output settings, estimated resource use clearly separated from measurements, explicit run/cancel/retry states, output-first completion.
- **Cockpit:** concise, live hardware overview with links into Chat, Model Hub, and Benchmarks. Remove hard-coded “active” speedups and readiness claims.
- **App shell:** group navigation into Run, Manage, Evaluate, and Create; keep it keyboard reachable and make the active route obvious. Move repeated inline styling into shared semantic tokens and components.

## Shared design and accessibility rules

- Use one shared page header, status badge, empty state, error notice, async action, metric card, progress panel, drawer, and data table pattern.
- Maintain a content-led layout at 320, 390, 768, 1024, and 1440 px; no clipped required controls or horizontal overflow.
- Keep touch controls at least 44 × 44 px. Preserve visible keyboard focus and a logical tab order.
- Use landmarks and route-aware links. Announce concise status changes, not every streamed token. Dialogs/drawers close with Escape and return focus to their trigger.
- Hide advanced configuration until requested. Avoid presenting unsupported hardware choices as ready.
- Replace invalid/misleading labels, hard-coded “live” values, and visible encoding artifacts before visual polish.
- Use semantic CSS tokens for color, spacing, type, radius, and elevation; remove the current inline-style drift as each screen is rebuilt.

## Acceptance checks

1. Every route opens directly, survives refresh, sets the correct screen title/active link, and works with Back/Forward. `/`, `/ui`, `/api/tags`, `/api/chat`, `/v1/*`, and `/static/*` keep their current purpose.
2. Empty, failed, and ready model discovery states all behave correctly; Send cannot issue a request without a ready model.
3. New Chat visibly clears the active transcript while preserving prior chats; archive/restore, storage failure, stop, and retry states are understandable.
4. Image input remains unavailable until an image is demonstrably sent to and processed by a supported vision model.
5. Benchmarks never show canned measurements. Model downloads show the server's real stream and terminal status.
6. At 320/390 px, navigation and composer are reachable, model setup is not squeezed beside device setup, and the conversation does not hide behind a fixed viewport.
7. Keyboard-only operation, screen-reader labels/status, focus restoration, and target size are reviewed for each rebuilt screen.

## Expert review

- `ui-ux-designer` reviewed the screen hierarchy, responsive behavior, and phase order.
- `usability-testing-engineer` reviewed screenshots and source behavior. No participant study or SUS score was claimed.
- `ui-ux-mathematics-engineer` dispatch was attempted through the library, but its required Opus-tier runtime and supported fallback were unavailable in this session. Numeric color/contrast/type calculations remain a design gate before final CSS tokens.
