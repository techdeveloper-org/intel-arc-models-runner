---
doc_id: DOC-SD-STAGE
title: Slice S-D - engine context window and Level Zero bridge
stage: S-D
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [main session applying gpu-driver-engineer / llm-serving-engineer roles from the plan]
reviewers: [user]
inputs: [DOC-SB-STAGE@1.0.0, docs/plans/MASTER_PLAN_v4.md sections 10.2, 10.5, 10.6]
---

# Slice S-D: engine context window and Level Zero bridge

Gate status: `cloud-green, local-pending` (Level Zero behaviour was tested against a fake loader; no real driver ran).

## 1. Scope
E2, E3, E4, E5 (documented counters), E6/E1 follow-ups, L3-L12, H7, plus new H8.

## 2. Library use (honest record)
Agent files and skills of `gpu-driver-engineer`, `kernel-driver-engineer`, `llm-serving-engineer` were NOT read in this stage; the work follows the plan's section 10.2/10.5/10.6 and the upstream `ze_api.h` (downloaded read-only from oneapi-src/level-zero and used as the source of truth for constants and struct layouts).

## 3. Verified against ze_api.h (L3)
| Constant | Old code | Real value |
|:--|:--|:--|
| context desc stype | 0x1 | 0xd |
| command queue desc stype | 0x2 | 0xe |
| command list desc stype | 0x3 | 0xf |
| host mem alloc desc stype | 0x4 | 0x16 |
| device mem alloc desc stype | 0x5 | 0x15 |
| device memory properties stype (Sysman) | 0x11 | 0x7 |
| device properties stype | not used | 0x3 (deprecated alias of 0x00020006 since 1.17, still valid) |
| ZE_DEVICE_TYPE_GPU / VPU (NPU) | not used | 1 / 5 |
Drivers that do not validate `stype` accepted the wrong values on the author's machine, which is why the hardware tests passed; stricter drivers would not.

## 4. Changes
| Area | Change |
|:--|:--|
| `arc_ze_types.h` (new) | named constants and struct layouts copied from ze_api.h (`ze_device_properties_t` complete) |
| `arc_level_zero_bridge.c/.h` | opaque context with a per-context function table (no globals); `arc_ze_init_ex` selects the first `ZE_DEVICE_TYPE_GPU` across all drivers and devices (NPU never chosen), uses `zeInit(GPU_ONLY)`, checks every driver call and cleans up on any failure; queue priority NORMAL; allocation table (1024) with size/type guards (`0 < size <= 64 GiB`), failed tracking frees the new allocation; copy falls back to memcpy only between tracked host/shared buffers and returns `ARC_ZE_FALLBACK_MEMCPY`; `arc_ze_synchronize` polls in 250 ms slices up to 1500 ms (hard max 1900 ms < TDR), marks the context dead on timeout or device loss and never resets a possibly in-flight list; `arc_ze_destroy` frees outstanding USM; accessors; `ARC_ZE_LOADER_PATH` override; symbols bound by copying pointer bytes (no object/function pointer cast, so `-Wpedantic -Werror` is clean on gcc and mingw) |
| ABI | `ARC_NATIVE_ABI_VERSION` 3 (L0 API changed); bridge, build script and tests updated |
| `bridge.py` | `NativeLevelZeroBridge`: `arc_ze_init_ex`, `init_status`, `is_dead`, `device_type`, `last_status`, default USM type host (DL-005), size validation, `free_usm`/`destroy` idempotent |
| `devices/sysman.py` | GPU-only device selection through `zeDeviceGetProperties`, memory properties stype 0x7, `zeInit(GPU_ONLY)`, loader override env, reasons for empty device list |
| `openvino_engine.py` | `_build_ov_properties` (u8 KV cache no longer default; opt-in via `KV_CACHE_PRECISION` or `ARC_KV_CACHE_PRECISION`), `_count_prompt_tokens`, `_enforce_context_window` (prompt, and prompt + `max_new_tokens`, raise `ContextWindowExceededError` before any pipeline call), single list call form without swallowed retries, empty prompt resets telemetry, `tokenize()` |
| `telemetry.py`, `models.py` | prompt tokens from `get_num_input_tokens`, `tpot_ms`, `decode_tokens_per_second` derived as (tokens-1)/(generate duration - TTFT) |
| routes | prefix-cache probing uses real token ids or is skipped (hash-derived pseudo ids removed from 4 places) |
| flash engine | documented as a demonstrator emitting canned tokens; telemetry labelled `synthetic_pipeline` (H8) |

## 5. Results
| Check | Result |
|:--|:--|
| C harness for the L0 bridge vs fake loader (ASan+UBSan, `-Werror`) | 66 checks, 0 failures: GPU chosen when the NPU driver is first, NO_GPU when only an NPU exists, each failing driver call reported without leaks, wrong stype detected by an independent oracle (mutant with the old context stype is caught), oversize/zero/invalid-type allocations refused, table-full frees the new allocation, double free ignored, copy fallback, timeout 300 ms measured, device lost, clamp at 1900 ms |
| mingw-w64 compile of the Win32 loader path, `-Wpedantic -Werror` | clean (compile only) |
| `gcc -fanalyzer` on both sources | no findings |
| Python tests | `tests/test_level_zero_bridge.py` 13, `tests/test_engine_context_and_kv.py` 10, `tests/test_telemetry_details.py` 5 |
| Full non-hardware suite | 693 tests, 3 failures (same as baseline), 6 skipped |

## 6. E5 (OpenVINO throughput definition)
The installed `openvino_genai` 2026.4.1 stubs say only "mean and standard deviation of throughput in tokens per second"; they do not say whether prefill is included. The code therefore keeps `tokens_per_second` as the engine's own number (source `ov_perf_metrics`) and reports a separately derived `decode_tokens_per_second`. Benchmarks must quote which of the two they use.

## 7. Decisions
| DL | Decision | Why |
|:--|:--|:--|
| 049 | `ARC_NATIVE_ABI_VERSION` 3 | L0 entry points changed; older libraries must be refused |
| 050 | u8 KV cache is opt-in | no quality gate exists (E4); default behaviour of the runtime is used |
| 051 | `ARC_ZE_LOADER_PATH` and `ARC_ZE_SYNC_TIMEOUT_MS` are supported environment overrides | testability without hardware; clamp keeps waits below TDR |
| 052 | prompt+completion over the window raises instead of clamping | the caller learns the request cannot be honoured; the Ollama-style clamping behaviour can be added at the route layer if wanted |

## 8. Open items
1. LOCAL gate: after rebuilding the DLL (S-B open item 1) run `pytest -m hardware`, the L0 tests with the real loader, and confirm the GPU is selected on the Core Ultra (driver0 = GPU, driver1 = NPU).
2. The speculative engine does not call the context-window check yet.
3. E4 quality gate (KL divergence of u8 vs f16 KV on a fixed prompt set) is not implemented; it needs a real model.
4. `zesInit`, power domains and frequency telemetry via the Sysman API are not implemented (this machine reports no power domains); L11 stays documentation-only.
5. USM/DMA paths are still not called by the streaming code (L12); they are tested through the fake loader only.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | S-D implementation | pending |
