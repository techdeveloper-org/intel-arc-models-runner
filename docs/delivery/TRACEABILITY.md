---
doc_id: DOC-TRACEABILITY
title: Finding to requirement to code to test traceability
stage: all
version: 1.2.0
status: draft
created: 2026-10-29
updated: 2026-10-31
authors: [main session, CLI session]
reviewers: [user]
---

# Traceability

Findings are defined in `docs/plans/MASTER_PLAN_v4.md` sections 2, 19.2 and 21.2. Status: closed, partial, open.

| Finding | Requirement | Code symbol | Test | Stage | Status |
|:--|:--|:--|:--|:--|:--|
| C1 wrapped extents | FR-001 | `arc_range_ok`, `arc_mmap_get_tensor_ptr` | `tests/native_c/mmap_harness.c`, `test_register_layer_out_of_file_extent_raises_bounds_error` | S-B | closed |
| C2 count/extent validation | FR-001 | `arc_validate_layer` | `test_register_layer_more_than_64_tensors_raises_instead_of_truncating` | S-B | closed |
| C3 ctypes wrap/truncation | FR-001 | `_validate_u64`, `_build_c_layer` | `test_validate_u64_*`, `test_register_layer_unrepresentable_integer_raises` | S-B | closed |
| C4 dtype | FR-001, NFR-002 | `dtype_for_tensor_type`, `_load_layer_arrays` | `test_dtype_for_tensor_type_maps_plain_and_quantized_types` | S-C | closed |
| C5 view lifetime | FR-001 | `_MappingGuard` | `test_close_with_live_view_defers_unmap_until_view_is_released`, `test_native_views_are_read_only` | S-C | closed |
| C6 slot races, shutdown | FR-006 | `_reserve_slot`, `_publish_load`, `close` | `test_acquire_concurrent_same_layer_loads_once`, `test_acquire_stress_*`, `test_close_waits_for_prefetch_and_later_acquire_raises` | S-C | closed (TLC run pending) |
| C7 lock codes, quota | FR-001 | `arc_mmap_lock_range`, `lock_range` | `test_lock_range_invalid_arguments_are_rejected` | S-B | partial (not wired into the streamer) |
| C8 UTF-8 / long paths | FR-001 | `arc_utf8_to_wide_path` | mingw compile check `test_windows_branch_compiles_clean_with_werror` | S-B | closed (not executed on Windows) |
| C9 size_t guards | FR-001 | `arc_fits_size_t` | harness | S-B | closed |
| C10 -mavx2 | NFR-005 | `build_native.py` | `test_compile_library_produces_loadable_abi3_library_with_matching_sidecar` | S-B | closed |
| X1 LRU on cyclic scan | FR-006 | `_slot_index_for` | `test_cyclic_layer_scan_over_resident_layers_hits_after_first_pass` | S-C | closed |
| X2 resident vs mapped memory | FR-006 | `get_resident_memory_bytes` | `test_get_resident_memory_bytes_reports_positive_value_on_linux` | S-C | closed |
| X3 GGUF dimension order | NFR-002 | `view_shape_for_dims` | `test_view_shape_for_dims_reverses_gguf_order` | S-C | closed |
| L1 constant power | FR-003 | `LevelZeroSysman.query_package_power` | `test_query_package_power_without_power_api_returns_none` | S-A | closed |
| L2 RAM as GPU memory | FR-003 | `query_gpu_memory` | `test_query_gpu_memory_reports_total_only_never_invented_free` | S-A | closed |
| L3 structure types | FR-003 | `arc_ze_types.h` | `ze_harness.c` (independent oracle) | S-D | closed |
| L4 NPU selected | FR-003 | `arc_select_gpu`, `_gpu_devices_of` | `test_arc_ze_init_selects_gpu_not_npu_when_npu_driver_comes_first` | S-D | closed (real driver run pending) |
| L5 ignored return codes | FR-003 | `arc_create_objects`, `arc_ze_init_ex` | `test_arc_ze_init_failing_driver_call_reports_api_error`, harness | S-D | closed |
| L6 queue priority | FR-003 | `arc_create_objects` | fake loader rejects non-NORMAL | S-D | closed |
| L7 memcpy fallback | FR-003 | `arc_ze_dma_copy_async` | `test_dma_copy_falls_back_to_memcpy_for_host_and_shared_pointers` | S-D | closed |
| L8 10 s wait vs TDR | FR-007 | `arc_ze_synchronize` | `test_synchronize_timeout_marks_context_dead_within_tdr_limit` | S-D | closed |
| L9 USM leak on destroy | FR-003 | `arc_teardown` | harness (live allocation count) | S-D | closed |
| L10 size guard | FR-003 | `arc_ze_alloc_usm` | `test_allocate_usm_oversize_and_invalid_sizes_return_none` | S-D | closed |
| L11 zesInit order | FR-003 | n/a (core memory API only) | n/a | S-D | documentation only |
| L12 USM path unused | FR-003 | n/a | n/a | S-D | open (kept, tested with a fake loader) |
| E1 fabricated fallbacks | FR-002 | `engine/telemetry.py` | `tests/test_honesty_no_fabrication.py` | S-A | closed |
| E2 context window | FR-004 | `_enforce_context_window` | `tests/test_engine_context_and_kv.py` | S-D | closed (speculative engine open) |
| E3 retry swallows errors | FR-004 | `_invoke_pipeline` | `test_invoke_pipeline_uses_single_list_call_form_and_propagates_errors` | S-D | closed |
| E4 u8 KV default | FR-004 | `_build_ov_properties` | `test_build_ov_properties_gpu_defaults_do_not_enable_u8_kv_cache` | S-D | closed (quality gate open) |
| E5 throughput definition | FR-002 | `decode_tokens_per_second` | `tests/test_telemetry_details.py` | S-D | closed as: documented counters only |
| E6 stream token counts | FR-002 | `telemetry_from_stream` | `test_generate_stream_with_tokens_labels_wallclock_estimate` | S-A | partial (labelled estimate) |
| H1 REFERENCE_SWEEP rows | FR-005 | live benchmark script, JSON | `test_benchmark_json_contains_no_reference_sweep_rows` | S-A | closed |
| H2 fair Ollama comparison | FR-005 | `docs/benchmarks/METHOD.md` | registered hypotheses | S-E | open (needs a hardware run) |
| H3 speculative constants | FR-002 | `HybridSpeculativeEngine` | `test_speculative_generate_never_invents_acceptance_or_speedup` | S-A | closed |
| H4 Ollama route durations | FR-002 | `_eval_timing_ns` | grep gate | S-A | closed |
| H5 synthetic heavy results | FR-005 | benchmark script/JSON/README | grep gate, README notice | S-A | closed |
| H6 README vs JSON | FR-005 | README | `render_bench_docs.py --check` | S-A | closed |
| H7 pseudo-token prefix probe | FR-002 | `match_prefix_with_real_tokens` | `test_tokenize_returns_pipeline_token_ids_and_none_when_unloaded_or_failing` | S-D | closed |
| H8 canned flash tokens unlabelled | FR-002 | `FlashStreamInferenceEngine` | `test_flash_stream_engine_telemetry_is_labelled_synthetic` | S-D | closed |
| G1 GGUF hostile lengths | NFR-001 | `_read_declared` | `tests/test_gguf_untrusted_input.py` | S-B | closed |
| B11 Student-t table | FR-005 | `student_t_critical_95` | `test_student_t_critical_95_matches_published_table` | S-E | closed |
| E2 speculative engine window | FR-002 | `ContextBudgetMixin` | `test_speculative_generate_prompt_plus_completion_over_context_raises_before_pipeline_call` | S-G | closed |
| E4 u8 KV gate | FR-005 | `benchmark/kv_quality` | `tests/test_kv_quality.py`; hardware: `test_kv_cache_u8_greedy_prefix_matches_default_precision` | S-G | harness done, run pending |
| E5 throughput definition | FR-002 | `telemetry._decode_rate`, METHOD.md | source read: `get_throughput` = 1000 / mean TPOT, decode-only (perf_metrics.cpp 2026.4.1.0) | S-G | closed |
| L12 USM/DMA unused | FR-001 | `DeviceLayerStager` | `tests/test_device_stager.py` | S-G | opt-in path done, no consumer |
| C7 `try_pin` wiring | FR-001 | `FlashStreamWeightManager._pin_layer` | `tests/test_flash_stream_ring.py` (pin tests) | S-G | closed |
| L1/L2 power and free memory | FR-002 | `LevelZeroSysman`, `PowerMeter` | `tests/test_sysman_telemetry.py`, `tests/test_power_meter.py` | S-G | measured when the driver exposes it |
| H9 benchmark publication | FR-010 | `scripts/validate_bench_json.py` line 27, `tests/test_bench_results.py::TestQuickBenchSchema`, `docs/benchmarks/results/llm_*.json`, `README.md` benchmark table | `TestQuickBenchSchema` (13 tests); `scripts/validate_bench_json.py --check`; manual: README table matches result file values | S-G | closed |

## Plan test-list audit (S-G)
Plan test names were replaced by `test_<function>_<scenario>_<expected>` names; every planned test maps to a repository test or is marked hardware:

| Plan test | Repository test |
|:--|:--|
| `test_arc_range_ok_uint64_wrap_rejected`, `test_register_layer_oob_extent_rejected` | `test_register_layer_out_of_file_extent_raises_bounds_error` |
| `test_register_80_layers_boundary_ok_and_256_plus_rejected` | `test_register_80_layers_ok_and_layer_index_at_or_above_limit_rejected` (added) |
| `test_tensor_crossing_4k_64k_boundary_reads_exact_bytes` | `test_get_tensor_buffer_crossing_page_and_allocation_boundaries_reads_exact_bytes` (added) |
| `test_virtuallock_quota_failure_falls_back_unpinned` | `test_lock_range_quota_failure_returns_false_and_logs_without_raising` (added), `test_pin_refused_by_os_keeps_streaming_and_stops_retrying` |
| `test_native_disabled_falls_back_byte_identical` | `test_disabled_native_layer_streams_byte_identical_weights` (added), `test_native_disabled_by_environment_returns_no_library` |
| `test_load_native_library_missing_symbol_returns_none` | `test_missing_abi_symbol_returns_no_library` |
| `test_get_tensor_buffer_f16_dtype_matches_fallback` | `test_native_and_python_paths_return_identical_tensor_bytes`, `test_dtype_for_tensor_type_maps_plain_and_quantized_types` |
| `test_close_with_live_view_defers_unmap` | `test_close_with_live_view_defers_unmap_until_view_is_released` |
| `test_acquire_concurrent_same_layer_single_load`, `test_concurrent_acquire_same_layer_loads_once` | `test_acquire_concurrent_same_layer_loads_once` |
| `test_allocate_usm_oversize_returns_none` | `test_allocate_usm_oversize_and_invalid_sizes_return_none` |
| `test_arc_ze_init_selects_gpu_not_npu` | `test_arc_ze_init_selects_gpu_not_npu_when_npu_driver_comes_first` (fake loader; real driver: `@hardware` P6 skips without a driver) |
| `test_generate_over_context_raises_before_pipeline_call` | `test_generate_prompt_over_context_raises_before_pipeline_call` |
| `test_generate_empty_prompt_returns_zero_tokens_and_resets_telemetry` | `test_generate_empty_prompt_returns_empty_text_and_resets_telemetry` |
| `test_generate_ttft_under_200ms_uses_perf_metrics` | `test_generate_ttft_uses_perf_metrics_and_is_reported_valid` (`@hardware`; no 200 ms threshold, the value goes to result files) |
| `test_kv_cache_u8_vs_f16_greedy_prefix_matches` | `test_kv_cache_u8_greedy_prefix_matches_default_precision` (`@hardware`, needs `ARC_TEST_MODEL_DIR`) |
| `test_query_gpu_memory_total_only_labelled_shared`, `test_query_package_power_none_without_domains`, `test_sysman_returns_none_without_driver` | `test_query_gpu_memory_system_memory_module_gives_no_free_value`, `test_query_package_power_without_package_domain_is_none_with_reason`, `test_sysman_without_gpu_has_no_devices_and_a_reason` |
| `test_telemetry_never_contains_synthetic_values` | `tests/test_honesty_no_fabrication.py` (whole file) |
| `test_truncated_gguf_rejected_subprocess` | `tests/test_gguf_untrusted_input.py` (in-process; no subprocess variant) |

## Revision history
| Version | Date | Author | What changed | Why | Commit |
|:--|:--|:--|:--|:--|:--|
| 1.0.0 | 2026-10-29 | main session | first version | traceability after slices S-A to S-E | pending |
| 1.1.0 | 2026-10-29 | main session | S-G findings and plan test-list audit | map every planned test to a repository test | pending |
| 1.2.0 | 2026-10-31 | CLI session | H9 benchmark publication row added (FR-010, issue #49) | traceability for quick_v1 benchmark protocol and TestQuickBenchSchema | next revision |
