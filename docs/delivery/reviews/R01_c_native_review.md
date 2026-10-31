---
doc_id: DOC-R01-REVIEW
title: R01 - Independent review of the native C layer and ctypes bridge
stage: S-G
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [library agent via main session]
reviewers: [user]
---

# R01 - Independent review of the native C layer and ctypes bridge

Reviewer: c-cpp-systems-engineer (library agent). Scope: `src/arc_runner/native/` (`arc_flash_stream_mmap.c/.h`, `arc_level_zero_bridge.c/.h`, `arc_ze_types.h`, `bridge.py`, `build_native.py`), plus `tests/native_c/*` and the callers in `engine/flash_stream/` that decide whether the thread-safety contracts hold.
Standard: C17 (ISO/IEC 9899:2018). Toolchain: gcc 13.3 (Linux), x86_64-w64-mingw32-gcc 13-win32, clang. All probes live in `/tmp/claude-0/-home-user/9f5d9574-39ea-55de-8475-bf2e506385a2/scratchpad/review_c/` (`probe_ze.c`, `probe_mm.c`, `probe_py.py`, `probe_py2.py`, `probe_py3.py`); nothing was written into the repo except this report.

Status words: VERIFIED = reproduced by running code. TRACED = read from file:line, not run. SUSPECTED = depends on driver or OS behaviour I could not execute here (no Intel hardware, no Windows, no wine).

## Skills read (partial, stated honestly)

Agent definition read in full. Skill files read only in part, so nothing below claims full coverage of any skill:

| Skill | Lines read | Quoted line |
|---|---|---|
| c-language-foundations-core | 1-300 of 597 | "Compilers are legally permitted to assume UB never occurs. This enables:" |
| cpp-language-foundations-core | 1-150 of 617 | "Failure in the **function body** (not immediate context) -> hard error (not SFINAE)." |
| embedded-c-core | 1-150 of 689 | "Embedded recommendation: avoid malloc() — use static allocation or memory pools" |
| clean-architecture | 1-100 of 444 | "D = \|A + I - 1\|" |
| algorithm-design-complexity-core | 1-100 of 2508 | "Time: O(k·W), Space: O(W)" |

Not read: c-language lines 301-597 (so the formal C11 atomics section after 7.4 and the ABI tail), the rest of the other four. Only the C11 memory model section (7.2-7.4) and UB/provenance sections (2, 6) were applied here.

## Summary

No integer-overflow, out-of-bounds or UB defect was found in the mmap layer's bounds logic, and the sealed-table atomics are correct. The defects that exist are in the Level Zero fallback path, in lifetime protection around `close()`, in timeout accounting versus the documented limit, and in silent integer narrowing at the ctypes boundary. Counts: 1 HIGH (D1), 5 MED (D2-D6), 8 LOW entries (D7-D14, each grouping several small items). Each entry is marked VERIFIED, TRACED or SUSPECTED.

## Defects

### D1 [HIGH, VERIFIED] DMA fallback `memcpy` is not bounded by the allocation
`arc_level_zero_bridge.c:497-506` and `:426-440`. `arc_find_alloc` returns true when only the *start* pointer lies inside a tracked allocation (`p >= base && p - base < size`, line 432). The fallback then runs `memcpy(dst, src, size_bytes)` (line 502) with the caller's `size_bytes`, never checking `dst + size <= base + alloc_size` (nor `src`). The fallback is entered for any non-success driver result (line 493), including a result caused by a bad size.
Scenario: two 64-byte host allocations, driver rejects the copy, caller passes 4096. Reproduced with `probe_ze_asan 1`: ASan reports `memcpy-param-overlap ... located 0 bytes after 64-byte region` at `arc_level_zero_bridge.c:502` (a 4096-byte copy out of 64-byte buffers). Reachable from the public `NativeLevelZeroBridge.dma_copy_async(dst, src, n)` (`bridge.py:732`). The in-repo caller `DeviceStager` computes in-range sizes, so exposure today is the public API, not the stager.
Secondary faults on the same lines: (a) the check releases the table lock before `memcpy`, so a concurrent `arc_ze_free_usm` is a use-after-free window; (b) pointer comparison across distinct objects is unspecified (C17 6.5.8p5) - use `uintptr_t`; (c) `dst`/`src` overlap is UB for `memcpy` (C17 7.24.2.1p2), should be `memmove` or rejected.
Fix: in the lookup return base+size and require `arc_range_ok(alloc_size, p - base, size_bytes)` for both pointers inside one critical section that also performs the copy, or drop the fallback.

### D2 [MED, TRACED] Fallback copy reorders against queued GPU work
`arc_level_zero_bridge.c:501-505`. The fallback copies synchronously at call time, while earlier `zeCommandListAppendMemoryCopy` commands are still only recorded and run at `arc_ze_synchronize`. If an earlier queued copy writes into `src`, the fallback copies stale bytes; if it reads `dst`, it sees the new data. In the stager's pattern (distinct regions) this is benign, but the API header (`arc_level_zero_bridge.h:97-108`) documents the fallback as equivalent to the async copy.

### D3 [MED, VERIFIED] Synchronize can exceed the configured timeout and the 2 s TDR limit
`arc_level_zero_bridge.c:528-542`. The deadline is only tested after each 250 ms slice, and the last slice is not shortened to the remaining time. Measured against the fake loader (`never` mode, fake driver adds no extra latency): `ARC_ZE_SYNC_TIMEOUT_MS=1900` returned `-9` after **2055 ms**; the default 1500 ms returned after **1558 ms**. `arc_level_zero_bridge.h:6-7` says all waits are bounded below the 2 s TDR limit, `ARC_ZE_MAX_SYNC_TIMEOUT_MS` is 1900, and `SRS.md` FR-007 says the call "shall return within the configured timeout (default 1.5 s, maximum 1.9 s)". Worst case is timeout + 249 ms. The harness only tests 300 ms (`ze_harness.c:224-230`, bound `< 1200`), so the stated maximum is untested. `GetTickCount64` (line 111) also has 10-16 ms granularity.
Fix: `slice = min(250, deadline - now)` for each wait; test with the 1900 ms maximum and assert `<= 1900 + one tick`.

### D4 [MED, SUSPECTED] Stager frees in-flight memory after a timeout, and shares one command list across threads
`device_stager.py:125-135` and `arc_level_zero_bridge.c:572-576, 327-348`. On `sync_status < 0` (including `-9` timeout, where the C layer deliberately leaves the list un-reset because work may still run) the stager calls `free_usm(ptr)` and then lets `contiguous` (the source host arrays) go out of scope. The Level Zero spec makes the application responsible for ensuring the device is done with memory before `zeMemFree`; I could not execute a real driver to see the result. `arc_ze_destroy` likewise frees all USM and destroys queue and list without a final idle wait.
Separately, `arc_ze_dma_copy_async`/`arc_ze_synchronize` append to, close, execute and reset one shared command list with no lock (only the allocation table has the spin lock), and `DeviceStager.stage_layer` runs those calls outside any lock while `acquire_layer_weights` is called from several prefetch threads (`weight_streamer.py:521-545`). Command lists are not thread-safe in Level Zero; thread A's `zeCommandListClose` can land between thread B's appends. The header has no single-thread contract for the Ze context (the mmap header does). Staging is opt-in, so exposure is limited.

### D5 [MED, VERIFIED] `NativeMemoryEngine` has no protection between pointer lookup and `close()`
`bridge.py:560-582`, `:405-651`. `get_tensor_buffer` obtains the raw pointer from C (line 560) and only afterwards registers the view with the guard (line 582). If `close()` runs on another thread in that gap, `request_close` sees `_views == 0` and unmaps immediately; the view then points at unmapped memory. Forced interleave in `probe_py.py` (hook `add_view` to call `close()` first): the subprocess died with rc -11 (SIGSEGV) on the first read.
The same class affects every method that passes `self._ctx` to C (`lock_range` line 609, `unlock_range` 630, `last_error` 448, `register_layer` 516): `close()` can free the native context while they run. In production the manager joins prefetch threads with a timeout before `native.close()` (`weight_streamer.py:604-635`), so the race needs a worker outliving `PREFETCH_JOIN_TIMEOUT_SECONDS`, or a caller thread outside the prefetch set; the engine itself has no lock.
Fix: take the view count before the C call (`guard.acquire()` rejecting after `request_close`), release it on failure; make the other calls hold a short lock or a refcount on the context.

### D6 [MED, VERIFIED] External truncation of the model file crashes the process
`arc_flash_stream_mmap.c:217` (`MAP_SHARED`, `PROT_READ`). The extents are validated against the size at open, never re-checked against the live file. `probe_mm 2`: map 8192 bytes, register the tensor at 4096..8192, `truncate` the file to 100 bytes, read the returned pointer: `Bus error` (exit 135). Same class on Windows is `EXCEPTION_IN_PAGE_ERROR` from a network drive or removed USB disk (TRACED: no `__try` anywhere). Windows `CreateFileW` omits `FILE_SHARE_WRITE` (line 165), so local writers are blocked there; POSIX has no equivalent without advisory locking. Candidate mitigations: document that model files must not be modified in place, take a shared `flock`, or copy-on-open for removable media. This is a design risk to record, not a coding slip.

### D7 [LOW, VERIFIED] File descriptor 0 is leaked by `arc_mmap_close`
`arc_flash_stream_mmap.c:224` stores `(void*)(intptr_t)fd`; `:444` closes only `if (ctx->file_handle)`. If the process has stdin closed (a service or daemon), `open` returns fd 0, the handle is NULL, and the fd is never closed. `probe_mm 1`: after `close(0)`, open, `arc_mmap_close`, `/proc/self/fd/0` is still present. Fix: store `fd + 1`, or keep a separate `int fd = -1` field and test `>= 0`. POSIX branch only.

### D8 [LOW, VERIFIED] ctypes narrows integers silently, against the project's no-silent-truncation rule
- `bridge.py:306`: `_validate_u64` is applied to `tensor_type`, but the C field is `uint32_t`. `tensor_type=2**32+5` is stored as `5` (`probe_py.py`: "tensor_type stored: 5"). Needs a u32 validator.
- `bridge.py:560-561`: `layer_idx` is not range-checked and goes through an `argtypes` `c_int32`. `get_tensor_buffer(2**32, ...)` returned layer 0's tensor (`probe_py.py`).
- `bridge.py:745-747`: `dma_copy_async` accepts any `int` and `bool`; `ctypes.c_size_t(2**64+4096).value == 4096` (verified), so a huge size wraps to a small valid one. `allocate_usm` (line 728) correctly caps at `2**63-1`; `dma_copy_async` and the pointer arguments do not.
- `bridge.py:281, 310-311`: missing `offset` or `size_bytes` default to 0, so `register_layer(0, [{"name": "x"}])` returns True (verified). A malformed record becomes a zero-length tensor instead of an error.
- Names with an embedded NUL: registering `"a\0b"` and looking up `"a"` matches (verified), because both sides are C strings.
- `bridge.py:584`: `flags.writeable = False` can be undone with `setflags(write=True)` (the base is a writable ctypes array); a write to the `PROT_READ` mapping then raises SIGSEGV (verified rc -11, `probe_py3.py`).

### D9 [LOW, TRACED] Silent 127-byte name truncation in C
`arc_flash_stream_mmap.c:256` overwrites byte 127 with NUL, so a 128-byte caller name collides with its 127-byte prefix. Python limits names to 127 bytes, so only direct C callers are affected. The harness asserts the truncation as intended (`mmap_harness.c:221-243`); an error would match the project rule better.

### D10 [LOW, VERIFIED] Error-state handling in `arc_ze_synchronize`
`arc_level_zero_bridge.c:524`: `atomic_store(&is_dead, res == ZE_RESULT_ERROR_DEVICE_LOST)` stores `false` for other errors, which can clear a `true` set by a concurrent thread; it should only ever set true. `probe_ze 5`: execute failure returns `-5` and `is_dead == 0`. After an execute (or `Close`) failure that is not device loss, the list stays closed and is never reset (lines 518-525), so later appends or `Close` calls will fail on a spec-invalid state; there is no recovery path other than destroy (SUSPECTED, needs a real driver). Either mark dead or `Reset` on these paths.

### D11 [LOW, VERIFIED] Env parsing and bookkeeping nits
- `arc_level_zero_bridge.c:209-213`: `strtoull("-1")` returns `ULLONG_MAX`, which clamps to 1900 rather than falling back to the default (`probe_ze 4`: `cfg(-1)=1900`). Reject a leading `-`/whitespace and check `errno`.
- `arc_flash_stream_mmap.c:400, 421`: `locked_bytes` adds the requested size per call; neither `mlock` nor `VirtualLock` nests and both round to pages, so overlapping or repeated locks double-count. Diagnostic only.
- `arc_ze_alloc_count` (arc_level_zero_bridge.c:594-596) and `arc_mmap_layer_count` read plain `uint32_t` fields that other threads may write: a data race by C17 5.1.2.4p25 when used concurrently, benign in practice.
- `last_status`/`last_result` are per-context, so the status read after a failing call can belong to another thread's call. Diagnostic by documentation; `bridge.py:461` builds exception text from it.

### D12 [LOW, TRACED] Windows library search path
`arc_level_zero_bridge.c:20, 185-190`: `LoadLibraryA("ze_loader.dll")` uses the default search order, which includes the current directory (after system directories). `ARC_ZE_LOADER_PATH` (line 180) loads any path from the environment. The POSIX side uses the default `ld.so` search. For a server process, `LoadLibraryExA(..., LOAD_LIBRARY_SEARCH_SYSTEM32)` or an absolute path removes the planting vector. Low because it needs write access to the working directory or environment.

### D13 [LOW, VERIFIED] ABI/declaration drift risks
- `arc_flash_stream_mmap.c:486` hard-codes `"abi=3"` in `arc_native_build_info`; the macro `ARC_NATIVE_ABI_VERSION` (header:37), `_EXPECTED_ABI_VERSION` (bridge.py:32) and `ABI_VERSION` (build_native.py:19) are three more copies. A bump of the macro leaves the string stale; nothing cross-checks the four. Stringize the macro, and have a test grep the header.
- `arc_flash_stream_mmap.h:63-64`: `_Static_assert` at file scope breaks C++ inclusion although the header wraps itself in `extern "C"`: `g++ -std=c++17 -Wpedantic` fails with "expected constructor, destructor, or type conversion before '(' token" (verified). Use a macro that maps to `static_assert` under `__cplusplus`.
- `arc_ze_types.h` has no `_Static_assert` for its structs. I derived the upstream layout by hand and the compiled sizes match (`ze_device_properties_t` 368 bytes, `uuid` at offset 96, `name` at 112; descriptors 24/40/24/24/24), and all `ZE_STYPE_*`, `ZE_DEVICE_TYPE_*`, `ZE_RESULT_*` values match ze_api.h as I know it, but the driver writes `ze_device_properties_t` through the pointer, so a layout assert is cheap protection.
- Python side: `arc_mmap_is_sealed`, `arc_mmap_file_size`, `arc_mmap_layer_count`, `arc_native_build_info` have no `argtypes`/`restype` in `_configure_signatures` (none is called from Python today; a future call would default to a 32-bit int return).

### D14 [LOW, TRACED] Build and sidecar provenance
- `build_native.py:171-174`: `--windows` without `--cc` still picks native `gcc` (line 40), producing a Linux ELF named `libarc_native_sys.dll` plus a valid-looking sidecar. Require `--cc` or a target-triple check when `--windows` is given.
- The sidecar records the binary sha256 and `git rev-parse HEAD` (line 153) but not a hash of the C sources and not whether the tree was dirty, so "binary matches the source in this commit" cannot be proven from the repo. Concretely: the committed `.so` is byte-identical to a fresh rebuild from the current source (VERIFIED, sha256 `63e783f4...`), but the committed `.dll` sidecar says git_sha `0fe6981...` and compiler MinGW-Builds 15.2.0, while the `.so` says `d5a2507...`; I cannot tell whether C changed between them (no git history here, different toolchain). Add `sha256(sources)` to the sidecar and assert it in a test.
- No link hardening flags (`-Wl,-z,relro,-z,now`, `-fstack-protector-strong`) on POSIX; Windows DLL imports only KERNEL32 and the UCRT api-sets (VERIFIED with `objdump -p`), so no runtime DLL dependency issue.
- `CC="gcc -m64"` fails (`shutil.which` on the whole string).
- Sidecar and library are replaced non-atomically as a pair; a crash between them leaves a mismatch that makes the loader disable native silently (safe direction).
- The sidecar check protects against corruption, not tampering: whoever can replace the library can replace the sidecar, and a missing sidecar loads unverified (`bridge.py:145-146`). State it as integrity, not security.

## Checked and no defect found

- `arc_range_ok` (`.h:102-104`): `sz <= fsize && off <= fsize - sz` cannot wrap; used consistently at register (`.c:260`), lookup (`.c:334`) and lock (`.c:363`); zero-size tensor at EOF yields a one-past-end pointer (valid, C17 6.5.6p8), never dereferenced by the library.
- Sealed table atomics: release store at `.c:305`, acquire loads at `.c:276, 319, 453`. Readers touch `layers[]` only after observing `sealed == true`, so writes in `register_layer` happen-before every lookup (C17 5.1.2.4p11). `probe_mm_tsan 3` (4 threads x 200000 lookups after seal, including status and count reads) reports nothing. `status`/`os_error`/`locked_bytes` use relaxed atomics, which is right for diagnostics. Remaining contract: `register_layer` racing `seal` from different threads is a data race; the Python engine does not serialise `seal()` (`bridge.py:521`) against `register_layer`, but `index_layers` registers before any prefetch thread starts (`weight_streamer.py:186-220`), so the contract holds today.
- `_MappingGuard` logic: view count under lock, finalizer holds the bound method, close-on-last-view, double close guarded by swapping `_ctx`. VERIFIED `probe_py2.py`: numpy base chain keeps the ctypes owner alive through slices; after `close()` with a live view the mapping stays (`ctx alive True`), and it is released when the last slice dies. The one hole is D5.
- ctypes versus C prototypes: every configured `argtypes`/`restype` matches the C declaration (pointer, `int32_t`, `uint64_t`, `size_t`, `int` enum parameter). Struct sizes 184/11792 are asserted on both sides (`.h:63-64`, `bridge.py:66`) and I re-derived them (128+4+4+32+8+8; 16+64x184).
- ABI handling order: sidecar hash, `dlopen`, `arc_native_abi_version`, then signatures; a stale library missing a new export raises `AttributeError`, which is caught, and native falls back (`bridge.py:189`).
- Error paths: every failure path in `arc_mmap_open` (both OS branches) releases handles and `ctx` (TRACED; harness runs under ASan+LSan with no report); `arc_ze_init_ex` failure runs `arc_teardown` in reverse order tolerating partial init; `arc_find_alloc`/`track` release the lock on every path; table-full allocation is freed (`ze_harness.c:154-171`).
- `arc_bind` copies the symbol bytes with `memcpy` instead of casting `void*` to a function pointer, so it is `-Wpedantic` clean.
- Static tools, all clean: gcc `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wcast-qual -Wformat=2 -Wnull-dereference -fanalyzer`, mingw gcc with `-Wcast-function-type -fanalyzer`, `clang --analyze`, clang `-Wthread-safety`. Existing harnesses pass under ASan+UBSan: `mmap_harness` 1,000,078 checks 0 failures; `ze_harness` 66 checks 0 failures.
- Exports: all 28 `ARC_EXPORT` functions in the headers are exported by both the `.so` and the committed `.dll` (VERIFIED with `nm -D` and `objdump -p`).
- `arc_ze_alloc_usm` accepts `usm_type` as an enum; with GCC/mingw the enum is unsigned so a negative `int` from ctypes is rejected by `> ARC_USM_SHARED`. This relies on the compiler's enum representation (MSVC uses `int`), so keep the Python-side string map as the only entry point.

## Do the C harnesses exercise the risky paths?

Partly. Strengths: boundary tables and an independent oracle for structure types (`fake_ze_loader.c:41-60`), failure injection for every init call, alloc-table exhaustion, device-lost and timeout behaviour.
Gaps (all TRACED from the harness sources):
1. The "1,000,000-case fuzz" (`mmap_harness.c:71-95`) drives only the header inline `arc_range_ok` against a carry reference. `arc_mmap_register_layer`/`arc_mmap_get_tensor_ptr` are covered by a few hand cases, not fuzzed (unfuzzed: tensor_count, overlapping tensors, `n_dims`, name edge cases).
2. No multi-threaded test of the sealed table, even though the design depends on it. My TSan run (`probe_mm.c` mode 3) is the first; it should become `tests/native_c/mmap_tsan_harness.c` run with `-fsanitize=thread`.
3. No test that reads the full extent returned by `get_tensor_ptr` (the harness reads 16 bytes), no zero-size, duplicate-name, or replace-before-seal case, no truncated-file/SIGBUS expectation, no fd-0 case.
4. `fake_zeCommandListAppendMemoryCopy` ignores every argument (`fake_ze_loader.c`, near the end of the Level Zero section), and the harness fallback test uses full-size 64-byte copies (`ze_harness.c:195-200`), so D1 is invisible; neither ordering (D2) nor failure after `Close` (D10) is modelled.
5. Timeout: only 300 ms and `slow:600` are tested; the documented 1900 ms maximum (D3) is not.
6. Windows C paths (`CreateFileW`, `arc_utf8_to_wide_path`, `VirtualLock`, `LoadLibraryA`) are compile-checked only (`test_native_bounds.py:297-310`, `test_level_zero_bridge.py:198`); on Windows CI they run only through the Python tests, without sanitizers, and no test covers non-ASCII paths, paths over 260 characters (`\\?\` prefix at `.c:112-137`), or UNC paths (not prefixed: `utf8[0] == '\\'` is not a drive path). I could not run any Windows code here (no wine).

## Suggested red-first tests (not written; the task did not authorise repo edits)
1. `ze_harness`: host alloc 64 B, `FAKE_ZE_COPY_FAIL=1`, `arc_ze_dma_copy_async(dst, src, 4096)` must return `ARC_ZE_ERR_INVALID_ARG` or `ARC_ZE_ERR_API`, never copy (fails today under ASan).
2. `ze_harness`: `ARC_ZE_SYNC_TIMEOUT_MS=1900`, `never`: assert `waited <= 1900 + 20` (fails today at about 2055).
3. `mmap_harness`: close fd 0, open, close, assert fd 0 is closed (fails today).
4. `test_native_bounds.py`: `tensor_type=2**32` and `layer_idx=2**31` raise `NativeBoundsError`; record without `offset`/`size_bytes` raises.
5. `test_native_system_layer.py`: hook between lookup and `add_view` (as in `probe_py.py`) must not leave a view on an unmapped context.
6. `ze_harness`: `FAKE_ZE_FAIL=zeCommandQueueExecuteCommandLists` then a second `synchronize` or copy; assert a defined state (dead, or list reset).

## Priority order for fixes
D1, D3, D5, D4 first (memory safety, a stated requirement FR-007, lifetime). Then D7, D8, D10, D13 (cheap and mechanical). D6, D12 and D14 are decisions to record in `DECISION_LOG.md` rather than code changes.

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | library agent via main session | first version | independent review wave A | pending |
