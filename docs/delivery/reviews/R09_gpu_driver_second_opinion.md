---
doc_id: DOC-R09-REVIEW
title: R09 - Level Zero / Sysman / OpenVINO device handling: second opinion
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R09 - Level Zero / Sysman / OpenVINO device handling: second opinion

Reviewers (roles played): gpu-driver-engineer, kernel-driver-engineer (library agents).
Scope: native/arc_level_zero_bridge.c, arc_ze_types.h, bridge.py (NativeLevelZeroBridge), devices/sysman.py,
benchmark/power.py, flash_stream/device_stager.py, scratch/run_live_image|video_benchmarks.py,
engine/openvino_engine.py, docs/benchmarks/results/*.json, probe.json.
Method: read all files; compared every struct and stype field by field against upstream
`level-zero v1.17.2` `ze_api.h` / `zes_api.h` (curl into the scratchpad, `.../scratchpad/zehdr/`);
reproduced the engine-lifetime defect with a fake engine (`.../scratchpad/r09_repro.py`); analysed the three
result JSONs numerically. No hardware, no OpenVINO source, no Windows available here: everything about
live driver behaviour is marked "suspected".

## Skills read (proof) and not read
- gpu-driver-stack-core (lines 1-120 read; lines 160-170 and 660-700 only via grep): "**Windows TDR (WDDM):** the OS imposes a 2-second timeout on any GPU command." (line 164, text after the heading is on that line.)
- kernel-driver-development-core (lines 1-80 read; DMA section only via grep): "### 6.3 DMA Coherent vs DMA Streaming" (line 893).
- Not read: embedded-linux-core, bare-metal-architecture-core, embedded-c-core and all optional skills.

## Verdict table (answers to the asked questions)

| Question | Answer | Status |
|---|---|---|
| Struct layouts / stype / enum values vs upstream | All match, no defect (details below) | VERIFIED against v1.17.2 |
| `ze_bool_t = uint8_t` and padding | `ze_api.h:95 typedef uint8_t ze_bool_t`; ctypes/C natural alignment reproduces the same padding (u8 then u32 etc.) | VERIFIED |
| Is casting ze handles to zes handles valid? | Yes by the header: `zes_api.h:30,34` `typedef ze_driver_handle_t zes_driver_handle_t; typedef ze_device_handle_t zes_device_handle_t;`. zesDriverGet is not needed for device-level queries. | VERIFIED (header); live behaviour not shown by any file in the repo |
| zesInit before zeInit | Spec text (`zes_api.h` ~572-583): "There is no requirement to call this function before or after zeInit" and alternatively `zeInit` with env `ZES_ENABLE_SYSMAN` set. So the 0x78000001-after-zeInit seen on Windows is implementation behaviour, not spec; ordering first is the right workaround, but see F3/F4 | VERIFIED (spec), suspected (cause) |
| Package power from first non-subdevice domain on Meteor Lake | Unproven. The code never identifies the domain type (see F5) | SUSPECTED |
| Energy counter wrap | Handling is correct and honest (see F9); no defect | VERIFIED by reading |
| TDR / 1.5 s sync timeout | Logic partly wrong (F2, F6) | VERIFIED by reading |
| Polled synchronize safe? | The polling itself is safe: `zeCommandQueueSynchronize` doc says "may call this function from simultaneous threads". What follows a timeout is not (F2) | VERIFIED |
| After CL_OUT_OF_RESOURCES / `os._exit(3)` | Right idea for a one-shot script, but the "do not unload" protection is defeated by Python refcounting (F1) | VERIFIED (fake-engine repro) |
| NPU TTFT ~2.2 s = compile leaking into timed runs? | No. It is a steady-state, prompt-length-independent floor (F7) | VERIFIED from JSON |

## Prioritised findings

### F1 - HIGH - "engine not unloaded after CL_OUT_OF_RESOURCES" is not true; destructor runs before results are written
Files: `scratch/run_live_image_benchmarks.py:350-360` (video twin: `run_live_video_benchmarks.py:385-395`),
`src/arc_runner/benchmark/harness_common.py:101-147`, tests `tests/test_tier_failures.py` (fake engine, only counts `unload()` calls).
Mechanism: on `lost`, the loop does `continue` and skips `engine.unload()`. But the next iteration executes
`engine = engine_factory(dev)` (line 350) and rebinds `make`, which drops the last references to the GPU engine; CPython
destroys it immediately. If the lost iteration was the last plan item, the engine dies when `collect_samples` returns.
Repro (`scratchpad/r09_repro.py`, fake engine with `__del__`): event order `DESTROYED GPU` -> `unload CPU` -> `DESTROYED CPU` -> `collect_samples returned`.
So the real `Text2ImagePipeline`/`VideoEngine` destructor (OpenCL/L0 release on a context the code itself says may hang
the driver) runs inside `collect_samples`, before `export_workload_result` (line 413) and before `os._exit` (line 422).
If that release hangs, the whole result file, including the good low/medium tiers, is lost. Whether OpenVINO's release really
hangs after CL_OUT_OF_RESOURCES is suspected (the project's own premise), not measured.
Fix: quarantine the object so it is never collected before `os._exit`: module-level `_QUARANTINE: list = []`, and in the `if lost:` branch `_QUARANTINE.append(engine); _QUARANTINE.append(make)` before `continue`. Add a test with a `__del__` that records destruction (the existing test cannot see this). Better fix: see F8.

### F2 - HIGH - after a sync timeout / error the bridge frees memory and destroys queue objects the GPU may still be using
File: `arc_level_zero_bridge.c:327-348` (`arc_teardown`), `:548-570` (`arc_ze_free_usm`, no `is_dead` check), `:538-541` (timeout), `device_stager.py:133-135` (frees `ptr` on error).
Spec: `zeMemFree` "The application must ensure the device is not currently referencing the memory before it is freed ... may immediately free"; same wording for `zeCommandQueueDestroy`, `zeCommandListReset`, `zeContextDestroy` (ze_api.h v1.17.2).
Scenario: a copy of host-USM layer N is queued behind a long OpenVINO compute packet on the same engine (F6), 1.5 s elapses, `arc_ze_synchronize` returns `ARC_ZE_ERR_TIMEOUT` and sets `is_dead`; `stage_layer` then calls `free_usm(ptr)` and later `arc_ze_destroy` -> `zeMemFree`/`zeCommandQueueDestroy` while the copy engine is still reading or writing that buffer. Outcome: GPU page fault, device reset, or driver hang; there is no TDR to save it because the app timeout (1.5 s) fires before the OS's 2 s.
Fix: when `is_dead` is set by a timeout (not DEVICE_LOST), `arc_ze_free_usm` and `arc_teardown` must leak (skip zeMemFree/Destroy) and only unload the loader if the device reported DEVICE_LOST; document the leak. Add a C harness test with the fake loader (`tests/native_c/`) where `zeCommandQueueSynchronize` always returns NOT_READY and assert no `zeMemFree` call occurs.

### F3 - HIGH - concurrent `stage_layer` calls share one command list (spec violation, real race)
Files: `device_stager.py:119-135` (append x N then `synchronize`, no submit lock; `self._lock` guards only dict/counters), caller `flash_stream/weight_streamer.py:392-405,516` (`_stage_layer` runs outside `self._lock`, from the caller thread and from `prefetch_layer_async` threads, :521-545), `arc_level_zero_bridge.c:484-546` (single `cmdlist_handle`).
Spec: `zeCommandListClose/Reset/Append*`: "must not call this function from simultaneous threads with the same command list handle".
Scenario: prefetch thread A appends copy 1 of layer 7 while thread B (layer 8) runs `zeCommandListClose`+Execute: A's append hits a closed list (error -> "USM staging disabled" permanently, silently degrading the run), or B's Reset races A's append (undefined). No data race protection exists in C either (`last_status` atomics only).
Fix: in `DeviceLayerStager.stage_layer` take a dedicated `self._submit_lock` around the whole append-all + synchronize sequence (cheap, copies are already serialised on one queue), or make `arc_ze_*` hold a mutex per context. Add a two-thread test with a fake bridge that asserts non-overlap.

### F4 - MED - Sysman init order still depends on nothing in-process having called zeInit first; `ZES_ENABLE_SYSMAN` is never set
Files: `sysman.py:166-186,207-218`, `bridge.py:666-668`, `arc_level_zero_bridge.c:371`. `grep ZES_ENABLE_SYSMAN` over the repo: no hits.
`zes_preinit()` is called only by the bridge and `scripts/probe_hardware.py`; `LevelZeroSysman._init_devices` (`sysman.py:281`) does zesInit before its own zeInit, so scripts that construct `DeviceDetector()` first (it builds a `LevelZeroSysman()`, `detector.py:33`) are ordered correctly. But any code path that creates `ov.Core()` / loads an OpenVINO NPU or GPU plugin before the first `LevelZeroSysman()` can run zeInit in the same loader module first (Windows deduplicates `ze_loader.dll` by module name), reproducing the 0x78000001 failure. The spec offers a second, order-independent route.
Fix: in `zes_preinit()` and at the top of `arc_ze_init_ex` (and `LevelZeroSysman.__init__`), `os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")` / `setenv`/`_putenv_s` before any load. On Linux this is the documented switch; note power/frequency domains on an iGPU may still be empty there (sysfs hwmon, not verified). Keep zesInit-first as well.
Evidence gap: all 234 samples in the three result files have `package_power_w = null` (VERIFIED), and `probe.json` still shows `zesInit failed with result 0x78000001` (it predates the fix). Nothing in the repo proves the fix works on hardware; re-run `scripts/probe_hardware.py` and one benchmark and commit the output.

### F5 - MED - package-power domain is never identified; docstring asserts "whole SoC (CPU and iGPU)" without evidence
Files: `sysman.py:539-547` (first `onSubdevice == false` domain), `power.py:27-29`.
The spec defines `zes_power_ext_properties_t` (domain = package / card / ...) reachable via `pNext` of `zes_power_properties_t`, and `zesDeviceGetCardPowerDomain`; the code uses neither, so it cannot tell package from card, nor SoC from GPU-only. On Meteor Lake (no subdevices) "first non-subdevice" is merely "first". The statement in `power.py:27` is an unverified claim and violates the project's own honesty rule once any power number is published.
Fix: chain `zes_power_ext_properties_t` (read its stype value from the header, do not hard-code from memory) and record domain type, `canControl`, and domain count in the result `environment` block; until then, label the metric "first Sysman power domain" and drop the SoC claim. Also enumerate once and cache the domain handle: `read_package_energy` re-enumerates domains and properties on every snapshot (`power.py:52` calls it twice per sample), which widens the counter window and adds driver calls.

### F6 - MED - timeout design: TDR constant, slice overshoot, copy runs on the compute engine
Files: `arc_level_zero_bridge.h:33-34` (1500 / 1900 ms), `arc_level_zero_bridge.c:29,203-214,528-542`, `:283-288,293-299` (ordinal 0 / group 0).
1. The clamp "below the TDR limit" is only true by luck: the loop calls Synchronize in fixed 250 ms slices and checks the deadline after each, so the real wait is `ceil(timeout/250)*250`. Default 1500 -> 1500 ms (OK); the maximum 1900 -> 2000 ms, i.e. at or over the 2 s WDDM TDR the skill cites (`gpu-driver-stack-core:164`). Fix: `slice = min(ARC_SYNC_SLICE_MS, deadline - now)`.
2. 2 s is the Windows constant. On Linux hang detection is driver-configurable (skill lines 667/695), so an app timeout tied to "TDR" has no meaning there; a timeout is a policy, name it as such.
3. Queue group ordinal 0 is whatever the driver exposes first (normally the compute/universal group), not a copy engine; `zeDeviceGetCommandQueueGroupProperties` is never queried. Staging copies therefore queue behind OpenVINO compute work on the same engine and can hit the app timeout without any hang (feeds F2). Fix: query groups, prefer a copy-only group; fall back to ordinal 0.
4. After a failed `zeCommandQueueExecuteCommandLists` that is not DEVICE_LOST (`:522-526`) the list stays closed and `is_dead` stays false; every later append fails with an API error. Reset the list on that path.

### F7 - NPU TTFT ~2.2 s is NOT compile cost leaking into timed runs
Data: `docs/benchmarks/results/llm_qwen-2.5-1.5b_20261029T111424Z.v1.json`, backend `npu`.
| tier (prompt tokens) | warmup 0 | warmup 1 | warmup 2 | timed 3..12 median (min..max) ms |
|---|---|---|---|---|
| low (20) | 3418.9 | 2246.8 | 2240.8 | 2252 (2230..2499) |
| medium (35) | 4922.6 | 2280.9 | 2290.5 | 2263 (2231..3207) |
| hard (42) | 2480.7 | 2231.5 | 2215.0 | 2238 (2201..2490) |
First-run extra cost (about 1.2 s, 2.6 s, 0.26 s) sits in warmup 0 and is discarded; warmups 1-2 and all timed runs sit at the same ~2.23-2.29 s regardless of prompt length 20 vs 42 (2.5 % apart). A compile/first-inference cost would be prompt-independent too, but it would vanish after warmup 0, and it does not. So 2.2 s is a steady-state floor. Hypothesis (not verified, no OpenVINO source or NPU available here): static-shape NPU prefill processes a fixed padded prompt length every call (OpenVINO GenAI NPU defaults, e.g. `MAX_PROMPT_LEN`, are not set anywhere: `openvino_engine.py:154-165` passes nothing NPU-specific). Test to settle it: TTFT at prompt lengths 1, 200, 800 (flat => fixed padded prefill), then repeat with a smaller `MAX_PROMPT_LEN`; also confirm what `ttft_ms` source is (perf_metrics). Stray timed outliers (medium it3 3207 ms, low it12 2499, medium it12 2772, hard it6 2490) are +10..+40 % and not flagged `noisy` (all `noisy=false`).
Also: NPU (and CPU) never get `CACHE_DIR` (`openvino_engine.py:155-160` is GPU-only) so every process start recompiles the NPU model; that costs load time, not TTFT.

### F8 - MED - what `os._exit(3)` should and should not be
Files: `run_live_image_benchmarks.py:417-422`, video `:452-457`.
- Keeping `os._exit` after the file is written and streams flushed is defensible for a one-shot benchmark: it skips Python finalisation and atexit; DLL_PROCESS_DETACH still runs under `ExitProcess`, which well-behaved user-mode drivers skip. If a hang is ever seen at exit, `TerminateProcess(GetCurrentProcess(), 3)` is the stronger form. It is NOT acceptable for the API server (`src/arc_runner/server/`): a lost context there needs process-level isolation.
- Better design (both): run each GPU tier (or each backend) in a child process (`multiprocessing`/`subprocess`, spawn), return samples as JSON to the parent; the parent never touches the GPU, writes the file, then runs the CPU baseline and the remaining tiers in fresh children. This also removes F1 and lets later tiers be retried with a fresh context if the allocation failure turns out to be recoverable.
- CL_OUT_OF_RESOURCES is ambiguous (allocation failure vs kernel failure after a TDR reset); the code treats it as fatal, which is the safe choice. To tell them apart cheaply, log the Windows Event Log "Display" 4101 events during the run window and the L0 `ZE_RESULT_ERROR_DEVICE_LOST` state; by arithmetic the 768x768 SD-1.5 failure looks like allocation (below), not TDR.
- Only the exact string `CL_OUT_OF_RESOURCES` marks a lost context (`harness_common.py:76,113`). Equivalent errors (`CL_MEM_OBJECT_ALLOCATION_FAILURE`, `CL_OUT_OF_HOST_MEMORY`, `DEVICE_LOST`) would be treated as ordinary tier failures and the next tier would run on a possibly dead context. LOW; add them.
- `finally: shutil.rmtree(model_dir, ignore_errors=True)` (image :408-410) runs while the engine may still hold the weight files mapped; on Windows deletion then fails silently and gigabytes stay on disk with no message. LOW; check that the directory is gone and print a warning if not.

### F9 - no defect - energy counter and power arithmetic
`sysman.py:514-521`, `power.py:91-97`: `zes_power_energy_counter_t` is `{u64 energy [uJ], u64 timestamp [us]}` (`zes_api.h` ~4955-4966), power = delta-energy / delta-timestamp = W exactly as the header states; the timestamp is only used as a delta, as the header requires. A true 64-bit wrap of microjoules needs about 1.8e13 J (never in practice). The realistic backwards case is a driver reset / suspend, and returning `None` with a reason is correct and consistent with the honesty rule; do not "unwrap" modulo 2^64 unless a narrower counter is actually observed. Meter snapshots sit outside the workload timer (`workloads.py:62-67`), so latency is not polluted. Caveat only: counter update granularity is unknown, so for runs of ~0.3 s (LLM low tier) the energy may be quantised; record the counter resolution (min observed timestamp step) or gate on a minimum window.

### F10 - LOW - smaller items
- `arc_level_zero_bridge.c:499-500`: an untracked source pointer (numpy memory from `device_stager.py:125`) is treated as "not CPU-visible", so the memcpy fallback can never trigger for the stager's own copies; only `found && type == DEVICE` should be non-visible. Whether the driver accepts pageable host pointers in `zeCommandListAppendMemoryCopy` is not verified here.
- `arc_level_zero_bridge.c:253-257`: `maxMemAllocSize` and `ZE_DEVICE_PROPERTY_FLAG_INTEGRATED` (`ze_api.h:1429`, bit 0) are read and discarded. `zeMemAllocHost` requires `size <= maxMemAllocSize` (header); `ARC_MAX_USM_BYTES` is a fixed 64 GiB (`arc_level_zero_bridge.h:32`). Keep both in the context, reject oversize requests with `ARC_ZE_ERR_TOO_LARGE`, and let `usm_type` default to host only when `INTEGRATED` (device_stager default "host" is wrong on a dGPU). Also record both in `probe.json`.
- `sysman.py` device index 0 vs OpenVINO "GPU" index: L0 enumeration order is not guaranteed to equal OpenVINO's; harmless on this iGPU-only machine, wrong on iGPU+dGPU.
- `openvino_engine.py:156`: `Path(".cache/ov_cache").resolve()` is relative to the process CWD, so a different CWD silently gives a cold GPU kernel cache. Anchor it under the repo/user cache dir. `:161-163`: `ARC_KV_CACHE_PRECISION` is applied to every device including NPU, where an unsupported property will raise at load.
- `power.py` only reads a power counter; `zes_freq_state_t.throttleReasons` is declared (`sysman.py:111`) but never recorded per sample. GPU timed runs 10-12 of the LLM file drift (tps 52.6 -> 45.4, latency +10..17 %) with `noisy=false`; throttle reasons would explain or refute thermal limits.
- Methodology: CPU baseline warmup=3 is not enough: LLM-low CPU latency in timed runs 3..12 is 3.49, 2.59, 1.84, 1.57, 1.58, 1.55, 1.64, 1.74, 2.22, 3.10 s (still converging at timed it 3-5), so the baseline is partly cold, which flatters GPU speedups. Raise warmups or discard until settle; the `noisy` flag did not fire.
- Spec dicts hard-code model `size_bytes` (image :47,88..., video :83,132,181), not measured; derive from the on-disk `.bin` sizes instead (honesty rule). The CogVideoX/LTX values look low for models with a large text encoder, unverified.
- Mixed loader instances: `ctypes.CDLL("ze_loader.dll")` (name-only, default DLL search dirs) vs the C `LoadLibraryA` (full default order incl. PATH) can resolve different copies if `ze_loader.dll` exists in both System32 and the OpenVINO folder. Windows dedups by module name once one is loaded, so the first loader wins. Suspected only; log the resolved module path of both.
- NPU risk: bridge and `LevelZeroSysman` call `zeInit(ZE_INIT_FLAG_GPU_ONLY)`. If the same loader instance later serves the OpenVINO NPU plugin, a GPU-only init could hide the VPU driver. The recorded run initialises NPU fine (llm file has `npu` samples), so this is a regression-test item, not a defect.

## Struct / constant comparison (VERIFIED vs v1.17.2, field by field)
Matched exactly (names, types, order, sizes, enum values):
- ze: `ze_device_properties_t` (stype..name[256], uuid 16 x u8, 2x u64 alignment), `ze_device_memory_properties_t`, `ze_context_desc_t`, `ze_command_queue_desc_t` (ordinal,index,flags,mode,priority), `ze_command_list_desc_t`, `ze_host_mem_alloc_desc_t`, `ze_device_mem_alloc_desc_t` (flags, ordinal).
- stype: DEVICE_PROPERTIES 0x3, DEVICE_MEMORY_PROPERTIES 0x7, CONTEXT_DESC 0xd, COMMAND_QUEUE_DESC 0xe, COMMAND_LIST_DESC 0xf, DEVICE_MEM_ALLOC_DESC 0x15, HOST_MEM_ALLOC_DESC 0x16; `ZE_RESULT_NOT_READY` 1, `ERROR_DEVICE_LOST` 0x70000001, `ERROR_UNINITIALIZED` 0x78000001; `INIT_FLAG_GPU_ONLY` bit 0; `DEVICE_TYPE_GPU` 1, `VPU` 5.
- zes: `zes_power_properties_t` (u8,u32,u8,u8,i32 x3), `zes_power_energy_counter_t`, `zes_freq_properties_t` (type u32, three ze_bool_t, double min/max), `zes_freq_state_t` (5 doubles then u32 flags), `zes_mem_properties_t`, `zes_mem_state_t`; stype FREQ_PROPERTIES 0x9, MEM_PROPERTIES 0xb, POWER_PROPERTIES 0xd, FREQ_STATE 0x1b, MEM_STATE 0x1e; `ZES_FREQ_DOMAIN_GPU` 0; `ZES_MEM_LOC_DEVICE` 1 (SYSTEM 0).
- Only cosmetic difference: `zes_freq_state_t.pNext` and `zes_mem_state_t.pNext` are `const void*` upstream, `void*` in ctypes; same size and offset.
- Caveat: checked against v1.17.2 only; installed Windows driver/loader versions are newer, but these structs are ABI-frozen core structures (extensions are chained by pNext), so no risk found.

## iGPU shared-memory budget: what fits (derived, not measured)
Inputs: `probe.json` `total_mb = 6976` (VERIFIED: it is `zeDeviceGetMemoryProperties.totalSize` on a 15 GB host, i.e. a ceiling shared with the OS, the CPU baseline and every other process, not free memory; `free_mb` is null by design on an iGPU). Weights are also in host RAM while being copied, so peak during load can approach 2x the weight size.
Formula for unfused self-attention scores: `bytes = tokens^2 * 2 (FP16) * heads * batch`, tokens = (H/8)(W/8); SD-1.5: 8 heads, batch 2 (CFG).
- SD-1.5 512x512: tokens 4096 -> 33.6 MB/head -> 0.54 GB total. Fits with 1.6 GiB weights.
- SD-1.5 768x768: tokens 9216 -> 170 MB/head -> 2.72 GB for the score tensor alone; a second same-size softmax buffer would add 2.72 GB, i.e. ~5.4 GB + 1.6 GiB weights + VAE exceeds 6976 MiB. Consistent with the recorded failure of the 768x768 tier. This depends on the GPU plugin NOT fusing SDPA; unverified. Not a TDR (the arithmetic is ~0.1 TFLOP, far below a 2 s packet).
- SDXL FP16 (spec constant 6.5 GB = 6.05 GiB): leaves < 1 GiB of 6976 MiB for activations; expect failure or heavy paging at any resolution. SDXL INT8 / turbo (3.2 GB = 2.98 GiB): attention max tokens 4096 at 1024x1024 (no attention at the highest SDXL level), 10 heads, batch 2 -> 0.67 GB, so it fits.
- Video (assumption: one full-resolution VAE activation with 128 channels, FP16): 33 frames 768x512 -> 33*768*512*128*2 = 3.3 GB per tensor, several live at once, does not fit; 16 frames 512x512 -> 1.07 GB per tensor, borderline; 8 frames 256x256 -> 0.13 GB, fits. Treat as order-of-magnitude, the real channel counts are model specific. `--tiers low,medium` is the right advice.
- Single-allocation limit (`maxMemAllocSize`) is not recorded anywhere (F10); capture it, it often caps one buffer well below the total.

## Suggested order of work
1. F1 quarantine (or F8 child processes) and a `__del__`-based test.
2. F3 submit lock, F2 leak-on-timeout.
3. F4 env var + re-run probe and commit real Sysman output; F5 domain identification before any power number is published.
4. F6 slice clamp, copy-engine group, list reset; F10 items.
5. F7 NPU prompt-length sweep.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
