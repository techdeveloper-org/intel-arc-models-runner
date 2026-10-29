---
doc_id: DOC-SB-STAGE
title: Slice S-B - native memory safety
stage: S-B
version: 1.0.0
status: draft
created: 2026-10-29
updated: 2026-10-29
authors: [c-cpp-systems-engineer (role applied by the main session), sast-engineer (tools only)]
reviewers: [user]
inputs: [docs/plans/MASTER_PLAN_v4.md@v8.2 sections 17.3, 17.4, 17.5]
related_issue: "not created yet"
---

# Slice S-B: native memory safety

Gate status: `cloud-green, local-pending` (the committed Windows DLL is not rebuilt; see Open items).

## 1. Scope
Findings C1, C2, C3, C7, C8, C9, C10 plus ABI/sidecar gating, POSIX and Win32 parity, and a sanitizer harness. New finding G1 (GGUF parser) fixed here because the plan treats the parser as untrusted input.

## 2. Library use (honest record)
Agent file read: `agents/c-cpp-systems-engineer/agent.md` (library clone 872904a). Skills read: `c-language-foundations-core` lines 1-420 of 597 (UB taxonomy, strict aliasing, pointer provenance 6.5.6p8, C11 memory model, atomics); `embedded-c-core` headings plus sections 1, 24-29 (CERT INT30-C, ARR30-C, defensive input validation, `-Werror`). Not read in this stage: the remaining math sections of both skills, and the `sast-engineer` / `threat-modeling-specialist` skills (devsecops-core, application-security-core, security-testing-ci-core, threat-modeling-core). SAST was done by running clang-tidy, `gcc -fanalyzer`, ASan/UBSan and valgrind directly, not by following those skills; the India overlay of the agents is not applied (DL-039).
Quotes used: "Pointer arithmetic beyond one-past-end is UB" (c-language-foundations-core 2.2); "INT30-C: ensure unsigned operations do not wrap (check before arithmetic)" and "ARR30-C: do not form or use out-of-bounds pointers" (embedded-c-core 25).

## 3. Red evidence (before the fix, source at c7312e0)
Program compiled against the old `arc_flash_stream_mmap.c` (offset = 2^64-4, size = 8, file 4096 bytes):
```
register rc=0 get_ptr=NON-NULL (BUG: wrapped extent accepted) size=8
lock_range wrapped rc=-1 (rejected)
```
The wrapped extent was accepted and `get_tensor_ptr` returned `base + (2^64 - 4)`, a pointer outside the mapping (C17 6.5.6p8). The lock call only failed because the OS rejected the address, not because of a bounds check.
GGUF parser before the fix (G1): declared string length 2^40 raised `MemoryError`; 2^63 and 2^64-1 raised `OverflowError`; `n_dims = 2^31` raised `MemoryError`.
Mutation check: replacing `arc_range_ok` by the naive `off + sz <= fsize` makes the new harness fail (first failure: `fsize=1 off=2 sz=2^64-1`).

## 4. Changes
| Area | Change |
|:--|:--|
| `arc_flash_stream_mmap.h/.c` | `arc_range_ok`, `ArcStatus`, opaque context, `arc_mmap_open(path, &status)`, validated `register_layer` (index, count, dims, every extent, name NUL), `arc_mmap_seal` (release/acquire atomic), status-coded lock/unlock with `locked_bytes`, `CreateFileW` + `\\?\` for long drive paths, `size_t` guards, ABI version 2 and build info, `_Static_assert` on struct sizes (184 / 11792) |
| `bridge.py` | ABI check, sidecar sha256 check, `ARC_NATIVE_DISABLE`, `_validate_u64`, no silent truncation (64 tensors, 127-byte names, 4 dims), `NativeBoundsError` / `NativeLayerError`, `seal()` (first read seals), itemsize check, `last_error()` |
| `build_native.py` | no `-mavx2`; `-std=c17 -Wall -Wextra -Werror` (+ `-Wpedantic` for the mmap source), atomic output replace, sidecar `<lib>.build.json` (sha256, compiler, flags, git SHA, ABI), cross-compile via `--cc` / `--windows` |
| `weight_streamer.py` | a layer rejected by the native bounds checks falls back to the Python read path (keeps the existing "Corrupted tensor" behaviour) |
| `quant/gguf_parser.py` | declared lengths validated against remaining bytes; at most 4 dimensions |
| `core/exceptions.py` | `NativeLayerError`, `NativeBoundsError` |
| tests | `tests/native_c/mmap_harness.c` (1,000,078 checks incl. 1e6 fuzz iterations), `tests/test_native_bounds.py` (41 tests), `tests/test_gguf_untrusted_input.py` (9 tests) |

## 5. Results
| Check | Result |
|:--|:--|
| ASan + UBSan harness, `-Wall -Wextra -Wpedantic -Werror` | `checks=1000078 failures=0` |
| valgrind (non-ASan build, `--leak-check=full`) | clean |
| `gcc -fanalyzer` on the mmap source | no findings |
| clang-tidy (`clang-analyzer-*, bugprone-*, cert-*`) | no findings after excluding `_POSIX_C_SOURCE` reserved-identifier (standard feature-test macro) and fixed-size `memcpy` of validated structs (both accepted) |
| mingw-w64 cross compile of the Win32 branch, `-Wpedantic -Werror` | compiles clean (compile check only, not executed) |
| Level Zero source, `-Wall -Wextra -Werror`, Linux and mingw | compiles clean (its `-Wpedantic` function-pointer-cast warnings are left for S-D) |
| Full non-hardware suite | 640 tests, 3 failures (the same 3 as the baseline: Dockerfile health URL, Level Zero absent P6, ffmpeg), 6 skipped |

## 6. Threat model (scoped, DFD of the untrusted-input path)
Trust boundary: model file bytes (downloaded GGUF, Ollama blob) -> `GGUFParser` -> tensor table -> `register_layer` -> `arc_mmap_get_tensor_ptr` -> NumPy view.
| ID | STRIDE | Threat | Status |
|:--|:--|:--|:--|
| TM-01 | Tampering / EoP | wrapped or out-of-file extent yields out-of-bounds pointer (C1) | mitigated: `arc_range_ok` at register and read, harness + fuzz |
| TM-02 | DoS | hostile length fields exhaust memory or raise uncaught exceptions in the parser (G1) | mitigated: `_read_declared`, dimension cap |
| TM-03 | DoS / Info disclosure | more than 64 tensors or oversized names truncated or overrunning descriptors (C2, C3) | mitigated: reject, never truncate |
| TM-04 | Tampering | stale or substituted native binary loaded (supply chain) | mitigated: ABI check, sidecar sha256 (stale DLL is refused); provenance of the committed DLL still open (S-F) |
| TM-05 | EoP | view outlives the mapping (use after unmap) (C5) | open: S-C (`_MappingGuard`) |
| TM-06 | DoS | VirtualLock quota exhaustion (C7) | mitigated: best-effort lock returns False, status and OS error recorded |
No CVSS scoring and no CERT-In / DPDP tagging (local runner, DL-039).

## 7. Decisions
| DL | Decision | Why |
|:--|:--|:--|
| 040 | The committed Windows DLL is NOT rebuilt in the cloud; the user rebuilds it locally and commits it with its sidecar | a cloud cross-compiled binary cannot be executed here (no Windows, no wine); the ABI check already keeps the old DLL out, so Windows falls back to Python until the rebuild |
| 041 | `get_tensor_buffer` seals the table on first use; `register_layer` afterwards raises `NativeLayerError` | keeps existing call sites working while making the table immutable before concurrent reads |
| 042 | GGUF parser caps dimensions at 4 and validates declared lengths against remaining bytes | untrusted input; matches GGML_MAX_DIMS |
| 043 | Sidecar file name is `<library file name>.build.json` | `.dll` and `.so` share a stem and would collide |

## 8. Open items
1. LOCAL gate: on Windows run `python src/arc_runner/native/build_native.py` (MinGW gcc), then `pytest -m "not hardware" tests/test_native_system_layer.py tests/test_native_bounds.py`, then commit `libarc_native_sys.dll` and `libarc_native_sys.dll.build.json`. Until then the Windows native tests (for example P1, which asserts the library loads) fail because the old DLL is refused.
2. The L0 bridge, Sysman struct values and `-Wpedantic` cleanliness of `arc_level_zero_bridge.c` belong to S-D.
3. No GitHub issue exists yet (plan 18.11 step 1).
4. Independent test author (plan DL-015) not available in this session; tests were written by the implementing session and checked by mutation (above).

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | S-B implementation | pending |
