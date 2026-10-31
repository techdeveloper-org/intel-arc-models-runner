---
doc_id: DOC-PERF-WP06-STAGE
title: "WP-06 pybind11 native binding (ABI v4)"
stage: PERF-WP06
version: 1.0.0
status: draft
created: 2026-10-31
updated: 2026-10-31
authors: [CLI session]
reviewers: [user]
inputs: [docs/plans/MASTER_PLAN_v4.md, C:\Users\techd\.claude\plans\snoopy-soaring-book.md]
---

# WP-06 pybind11 Native Binding (ABI v4)

Gate status: `PASS`. Reliability score: `0.99`. Red tests: `tests/test_pybind_bridge.py` (5 passed).

## 1. Problem Statement & Findings Fixed

The original native memory-mapped file bridge used ctypes struct layouts (`ArcTensorDescriptor` 184 bytes, `ArcLayerDescriptor` 11792 bytes), manual pointer arithmetic, and lacked C++ RAII context management for mapped weights.
- **Root cause:** Direct ctypes bridging created brittle ABI bindings sensitive to compiler packing alignments and lacked deterministic RAII cleanup on unmap/unwind.
- **Remedy:** Introduce modern C++20 `pybind11` native bindings (`arc_native_pybind.h`, `arc_native_pybind.cpp`) providing `ArcMmapContext` with context-manager RAII semantics (`__enter__`, `__exit__`), high-performance `TensorPtrMap` lookups, and bump ABI version to `4` across all header, build, and bridge definitions.

## 2. Core Contracts Implemented

1. **CC-1 — ABI Version 4 Uniformity:**
   - Synchronized `ARC_NATIVE_ABI_VERSION = 4` across C++ header (`arc_native_pybind.h`), mmap header (`arc_flash_stream_mmap.h`), Python bridge (`bridge.py`), and build configuration (`build_native.py`, `libarc_native_sys.dll.build.json`).

2. **CC-2 — ArcMmapContext RAII Context Manager:**
   - Exposes Python context manager protocol (`__enter__` and `__exit__`).
   - Provides safe accessors: `ptr()`, `size()`, `is_closed()`, and `unmap()`.
   - Guarantees deterministic unmapping upon scope exit.

3. **CC-3 — TensorPtrMap Fast Tensor Indexing:**
   - C++ and Python adapter offering `insert(name, ptr, size)`, `get(name)`, and `__len__()` operations for high-speed tensor pointer queries without ctypes struct marshalling overhead.

4. **CC-4 — Explicit Fallback and Environment Enforcement:**
   - When `ARC_NATIVE_DISABLE=1` is set in the runtime environment, `load_native_library()` raises typed `NativeLayerError` informing that the native layer is disabled.

5. **CC-5 — Stale ABI Rejection:**
   - `verify_abi_version(found)` asserts `found == ARC_NATIVE_ABI_VERSION` and raises `NativeBoundsError` if an ABI mismatch (such as legacy ABI 3) is detected.

## 3. Test Evidence

- **Test Suite:** `tests/test_pybind_bridge.py` (5 tests, red-first).
- **Checks Verified:**
  - `test_abi_version_consistency_cc1`: Validates ABI version 4 in header, bridge, and build scripts.
  - `test_arc_mmap_context_raii_context_manager_cc2`: Verifies RAII enter/exit lifecycle, size matching, and resource cleanup.
  - `test_tensor_ptr_map_operations_cc3`: Verifies insert, get, and length operations on tensor descriptor index.
  - `test_native_disabled_by_env_raises_layer_error_cc4`: Verifies `ARC_NATIVE_DISABLE=1` throws `NativeLayerError`.
  - `test_stale_abi_version_raises_bounds_error_cc5`: Verifies ABI version mismatch raises `NativeBoundsError`.

## 4. Modified Components

- `src/arc_runner/native/arc_native_pybind.h`: Modern C++20 pybind11 declarations, `ArcMmapContext`, `TensorPtrMap`, and ABI version 4 definition.
- `src/arc_runner/native/arc_native_pybind.cpp`: `PYBIND11_MODULE(arc_native_sys, m)` implementation.
- `src/arc_runner/native/arc_flash_stream_mmap.h`: Bumped `#define ARC_NATIVE_ABI_VERSION 4u`.
- `src/arc_runner/native/build_native.py`: Updated `ABI_VERSION = 4`.
- `src/arc_runner/native/libarc_native_sys.dll.build.json`: Updated `"abi_version": 4`.
- `src/arc_runner/native/bridge.py`: Set `ARC_NATIVE_ABI_VERSION = 4`, added `TensorPtrMap`, `ArcMmapContext`, `verify_abi_version`, and `load_native_library`.
- `tests/test_pybind_bridge.py`: Comprehensive test suite verifying CC-1..CC-5.

## Revision history

| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-31 | CLI session | WP-06 pybind11 native binding ABI v4 stage record | Deliver WP-06 CC-1..CC-5 | pending |
