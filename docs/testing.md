# 🧪 Complete Test Coverage & Verification Guide

[← Back to README](../README.md) · [Architecture Guide](architecture.md) · [Benchmarks Matrix](benchmarks/comparative_matrix.md) · [Roadmap](roadmap.md)

---

Every component in `intel-arc-models-runner` is governed by strict **Triad Testing** (Positive, Negative, and Edge-Case scenarios) derived from Claude Global Rules (`33-test-case-roadmap.md`, `35-positive-testing-standards.md`, `36-negative-testing-standards.md`, `37-edge-case-testing-standards.md`, `38-test-mocking-strategy.md`).

Measured on 2026-10-29 on a Linux x86-64 VM with no Intel GPU or NPU (command and result):

| Command | Result |
|:--|:--|
| `pytest --collect-only` | 1507 tests collected |
| `pytest -m "not hardware" --collect-only` | 1484 collected, 23 deselected (hardware-marked) |
| `pytest -m "not hardware"` | **1476 passed, 8 skipped, 0 failed** (about 67 s) |
| `pytest -m hardware` | **not run for this count**; the 23 hardware-marked tests (21 in `tests/integration/test_real_hardware.py`, 2 in `tests/test_plan_gap_tests.py`) are reported as not run, never as passed |

The 8 skipped tests: `test_dll_loader.py:61` (`os.add_dll_directory` not supported on Linux), six in `test_native_system_layer.py` (no Level Zero driver on this host) and `test_nextgen_orchestration_triad.py:268` (ffmpeg lists `h264_qsv` but no Quick Sync device is present). The suite is still growing, so the numbers drift; always quote the date and the command.

The `hardware` marker means the test is meant for Intel silicon, not that it cannot execute elsewhere: in an earlier measurement 15 of the 21 `test_real_hardware.py` tests ran on a CPU-only Linux VM (15 passed, 5 skipped, 1 failed with an OpenVINO RuntimeError because the GPU/NPU device was absent). Treat a hardware-marked pass on a machine without Intel GPU/NPU as not evidence of silicon behaviour.

---

## 📊 Test Module Matrix (generated from `pytest --collect-only -q`)

The counts below are collected tests per module, captured on 2026-10-29 together with the totals in the summary above. They change whenever tests are added; regenerate them with `python -m pytest --collect-only -q`. Modules under `tests/independent/` are the independent reviewers' tests.

| # | Test module | Collected | Coverage area |
|:--:|:---|:---:|:---|
| 1 | `tests/independent/test_ind_gguf_fuzz.py` | 33 | independent review tests: gguf fuzz |
| 2 | `tests/independent/test_ind_kv_quality.py` | 31 | independent review tests: kv quality |
| 3 | `tests/independent/test_ind_native_ranges.py` | 30 | independent review tests: native ranges |
| 4 | `tests/independent/test_ind_results.py` | 264 | independent review tests: results |
| 5 | `tests/independent/test_ind_ring_buffer.py` | 21 | independent review tests: ring buffer |
| 6 | `tests/independent/test_ind_stats.py` | 51 | independent review tests: stats |
| 7 | `tests/integration/test_real_hardware.py` | 21 | Real hardware integration (Arc Xe-LPG, AI Boost NPU, USM) |
| 8 | `tests/test_aot_cache.py` | 8 | Ahead-of-Time NPU compilation & warm-load speedup |
| 9 | `tests/test_audio_engine.py` | 32 | OpenVINO Whisper ASR & Kokoro TTS, VAD silence gating |
| 10 | `tests/test_batch_api.py` | 10 | Batch status API, preemption routes, 429 saturation |
| 11 | `tests/test_batch_pipeline.py` | 16 | Continuous batch pipeline, Little's Law, dynamic reclamation |
| 12 | `tests/test_bench_results.py` | 26 | result schema and generated tables |
| 13 | `tests/test_benchmark.py` | 13 | TTFT/TPOT measurement, warmup discard, Student's t CI |
| 14 | `tests/test_benchmark_stats.py` | 28 | benchmark statistics and claim gates |
| 15 | `tests/test_continuous_batcher.py` | 9 | PagedBlockTable, RadixCache, Little's Law admission |
| 16 | `tests/test_device_stager.py` | 13 | Device layer stager (no production consumer) |
| 17 | `tests/test_devices.py` | 8 | Level Zero & OpenVINO hardware discovery & device prober |
| 18 | `tests/test_dll_loader.py` | 6 | Windows DriverStore DLL lazy loader & path discovery |
| 19 | `tests/test_docker_container_triad.py` | 11 | Ubuntu 24.04 multi-stage Dockerfile, Rule 29, DRI `/dev/dri` |
| 20 | `tests/test_docs_sync_scripts.py` | 10 | docs-sync and delivery-doc check scripts |
| 21 | `tests/test_embeddings.py` | 13 | NPU embeddings, L2 norm, mean pooling, cosine similarity |
| 22 | `tests/test_engine.py` | 28 | OpenVINO GenAI execution engine, streaming, KV-cache |
| 23 | `tests/test_engine_context_and_kv.py` | 13 | engine context and kv |
| 24 | `tests/test_flash_stream_ring.py` | 32 | ring-buffer protocol |
| 25 | `tests/test_flash_stream_triad.py` | 14 | Flash-Stream O(1) RAM streaming, NPU/GPU disaggregation, SwiGLU |
| 26 | `tests/test_gguf_parser.py` | 10 | Binary GGUF v1/v2/v3 parsing, tensor extraction |
| 27 | `tests/test_gguf_untrusted_input.py` | 20 | GGUF parser on hostile input |
| 28 | `tests/test_harness_common.py` | 5 | shared benchmark harness helpers |
| 29 | `tests/test_harness_protocol.py` | 25 | benchmark protocol and result-file labels |
| 30 | `tests/test_honesty_no_fabrication.py` | 15 | no invented telemetry values |
| 31 | `tests/test_hub.py` | 12 | Model Hub registry, downloader, manifest verification |
| 32 | `tests/test_hub_checksum.py` | 6 | download checksum handling |
| 33 | `tests/test_image_engine.py` | 25 | OpenVINO Text2Image (SD 1.5, LCM, SD-Turbo, SDXL), DPAS |
| 34 | `tests/test_kv_quality.py` | 12 | KV-cache quality helpers |
| 35 | `tests/test_level_zero_bridge.py` | 30 | Level Zero bridge against a fake loader |
| 36 | `tests/test_live_benchmark_scripts.py` | 22 | live benchmark scripts (no hardware) |
| 37 | `tests/test_lora.py` | 16 | Multi-LoRA manager, safetensors zero-copy, LRU cache |
| 38 | `tests/test_lora_api.py` | 10 | LoRA REST API endpoints, hot-swapping dispatch, isolation |
| 39 | `tests/test_multimodal_orchestrator.py` | 7 | End-to-end Studio DAG pipeline, heterogeneous scheduling |
| 40 | `tests/test_native_bounds.py` | 51 | native bounds checks and bridge/sidecar gating |
| 41 | `tests/test_native_system_layer.py` | 26 | native system layer (some tests skip without a Level Zero driver) |
| 42 | `tests/test_nextgen_orchestration_triad.py` | 42 | Milestones 14.1–14.7 (QSV, Alpha dissolve, Reactive DAG) |
| 43 | `tests/test_nonllm_runners.py` | 14 | non-LLM benchmark runners |
| 44 | `tests/test_ollama_importer.py` | 8 | Ollama model discovery & OpenVINO profile mapping |
| 45 | `tests/test_perf_metric_shapes.py` | 10 | OpenVINO perf-metric shapes |
| 46 | `tests/test_plan_gap_tests.py` | 9 | gap tests from the master plan |
| 47 | `tests/test_power_meter.py` | 9 | power meter (unavailable is reported, never invented) |
| 48 | `tests/test_pypi_packaging_triad.py` | 11 | Standalone wheel packaging, PEP 561, dual CLI entrypoints |
| 49 | `tests/test_real_models_summary.py` | 3 | real-model summary table rendering |
| 50 | `tests/test_review_fixes_native.py` | 11 | regression tests for native review findings |
| 51 | `tests/test_server.py` | 24 | FastAPI Ollama + OpenAI parity endpoints |
| 52 | `tests/test_server_security.py` | 39 | server input validation and security |
| 53 | `tests/test_slide_compositor_triad.py` | 38 | 16:9 HD Slide Compositor, Double-Slit, Bohr atom, Voronoi |
| 54 | `tests/test_speculative.py` | 20 | Dual-silicon speculative decoding (NPU draft + Arc verify) |
| 55 | `tests/test_streamer_review_fixes.py` | 5 | regression tests for weight-streamer review findings |
| 56 | `tests/test_studio_director.py` | 8 | Director prompt decomposition, regex AST auto-repair |
| 57 | `tests/test_studio_models.py` | 14 | Pydantic v2 schemas, telemetry serialization |
| 58 | `tests/test_studio_muxer.py` | 5 | Closed-loop pacing convergence, hold-frame padding |
| 59 | `tests/test_studio_routes.py` | 5 | `/v1/studio/generate` REST endpoint, SSE pipeline events |
| 60 | `tests/test_studio_ui_triad.py` | 33 | 3-tier Triad matrix for Tab 7 Studio Web UI |
| 61 | `tests/test_studio_workers.py` | 7 | AudioWorker (Kokoro TTS), VisualWorker (Compositor/DiT) |
| 62 | `tests/test_sysman_telemetry.py` | 16 | Sysman telemetry |
| 63 | `tests/test_telemetry_details.py` | 8 | telemetry builders |
| 64 | `tests/test_tier_failures.py` | 6 | benchmark tier failure handling |
| 65 | `tests/test_transpiler.py` | 18 | GGUF→OpenVINO transpiler, INT4 dequant, SQNR, XMX pack |
| 66 | `tests/test_transpiler_stream.py` | 14 | SSE streaming engine, O(1) RAM, SIMD quantization |
| 67 | `tests/test_transpiler_xml_escape.py` | 6 | transpiler XML escaping |
| 68 | `tests/test_ui_endpoints.py` | 24 | Web UI SPA, SSE stream endpoints, embeddings test, convert |
| 69 | `tests/test_video_engine.py` | 25 | OpenVINO Text2Video (LTX-Video, CogVideoX, AnimateDiff) |
| 70 | `tests/test_vision.py` | 9 | VLMPipeline vision engine, visual token calculation |
| 71 | `tests/test_workload_claim_gates.py` | 24 | workload claim gates |
| 72 | `tests/test_workloads.py` | 9 | workload generators |
| | **Total** | **1507** | 72 modules; 23 of these tests are hardware-marked (see above) |

---

## 🚀 Running Tests

### 1. Execute Full Test Suite
```bash
python -m pytest -v --tb=short
# Non-hardware subset used for the counts above:
python -m pytest -m "not hardware"
# → 1476 passed, 8 skipped (Linux, 2026-10-29, about 67 s)
```

### 2. Run Physical Silicon Hardware Integration Tests
```bash
python -m pytest tests/integration/test_real_hardware.py -v
# Needs an Intel GPU/NPU host for meaningful results. This repository has no committed run log for these 21 tests,
# so no pass count on silicon is claimed here; on a CPU-only Linux VM some tests skip and one fails (see above).
```

### 3. Run Studio UI & Compositor Triad Tests
```bash
python -m pytest tests/test_studio_ui_triad.py tests/test_slide_compositor_triad.py -v
```

### 4. Run Next-Gen Orchestration Triad Tests (Milestones 14.1–14.7)
```bash
python -m pytest tests/test_nextgen_orchestration_triad.py -v
```

### 5. Run Silicon Discovery Self-Test CLI
```bash
arc-runner test-hardware --iterations 10
```


## Native layer, ring buffer and benchmark tests (v0.2.0)

| Area | Where | What it proves |
|:--|:--|:--|
| Bounds checks in C | `tests/native_c/mmap_harness.c` (run by `tests/test_native_bounds.py`) | boundary table plus 1,000,000-case fuzz of `arc_range_ok`, registration limits, sealing, lock/unlock validation under ASan and UBSan |
| Level Zero bridge | `tests/native_c/ze_harness.c`, `fake_ze_loader.c`, `tests/test_level_zero_bridge.py` | GPU (not NPU) selection, every failing driver call, wrong structure types caught by an independent oracle, allocation limits, bounded waits, dead context; no real driver needed |
| Bridge and fallback | `tests/test_native_bounds.py` | integer/count/name validation, ABI and sidecar gating, `ARC_NATIVE_DISABLE`, identical bytes on both paths |
| Ring buffer | `tests/test_flash_stream_ring.py`, `docs/formal/RingSlots.tla`, `scripts/check_ring_model.py` | one load per slot under contention, no deadlock on colliding layers, safe shutdown, deferred unmap, read-only views |
| Telemetry honesty | `tests/test_honesty_no_fabrication.py`, `tests/test_telemetry_details.py`, `tests/test_engine_context_and_kv.py` | no invented values, context window enforced, u8 KV cache opt-in |
| Benchmarks | `tests/test_benchmark_stats.py`, `tests/test_bench_results.py` | exact confidence intervals, claim gates reject projected/derived/fallback/noisy rows, schema validation, generated tables |

Red-first rule: a fix for a confirmed defect starts with a failing test shown failing on the old code. The manual `redfirst` CI job re-runs the new regression tests against the pre-fix baseline commit and requires them to fail there.

Commands: `python scripts/check_docs_sync.py` (code, docs, diagrams, versions), `python scripts/check_delivery_docs.py`, `python scripts/validate_bench_json.py`, `python scripts/render_bench_docs.py --check`.
