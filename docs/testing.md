# 🧪 Complete Test Coverage & Verification Guide

[← Back to README](../README.md) · [Architecture Guide](architecture.md) · [Benchmarks Matrix](benchmarks/comparative_matrix.md) · [Roadmap](roadmap.md)

---

Every component in `intel-arc-models-runner` is governed by strict **Triad Testing** (Positive, Negative, and Edge-Case scenarios) derived from Claude Global Rules (`33-test-case-roadmap.md`, `35-positive-testing-standards.md`, `36-negative-testing-standards.md`, `37-edge-case-testing-standards.md`, `38-test-mocking-strategy.md`).

As of version 0.2.0 the non-hardware suite collects **741 tests (733 passed, 8 skipped, 0 failures on the Linux CI image)**; tests that need real Intel silicon carry `@pytest.mark.hardware` and are excluded from that count: a skipped hardware test is reported as **not run**, never as passed.

---

## 📊 Comprehensive Test Module Matrix (569 Tests / 36 Modules)

| # | Test Module | Tests | Coverage Area | Invariant / Proof Referenced |
|:---:|:---|:---:|:---|:---|
| 1 | `test_devices.py` | 8 | Level Zero & OpenVINO hardware discovery & device prober | Level Zero API enumeration |
| 2 | `test_dll_loader.py` | 6 | Windows DriverStore DLL lazy loader & path discovery | Level Zero DLL loading |
| 3 | `test_engine.py` | 28 | OpenVINO GenAI execution engine, streaming, KV-cache | Proof M61 (Transformer roofline) |
| 4 | `test_speculative.py` | 20 | Dual-silicon speculative decoding (NPU draft + Arc verify) | Proof M6 & Leviathan equation |
| 5 | `test_aot_cache.py` | 8 | Ahead-of-Time NPU compilation & warm-load speedup | Proof M93 & M112 |
| 6 | `test_benchmark.py` | 12 | TTFT/TPOT measurement, warmup discard, Student's t CI | Proof M65 & M81 |
| 7 | `test_hub.py` | 12 | Model Hub registry, downloader, manifest verification | EMA bandwidth smoothing |
| 8 | `test_gguf_parser.py` | 10 | Binary GGUF v1/v2/v3 parsing, tensor extraction | GGUF container specification |
| 9 | `test_ollama_importer.py` | 8 | Ollama model discovery & OpenVINO profile mapping | Ollama manifest parsing |
| 10 | `test_server.py` | 24 | FastAPI Ollama + OpenAI parity endpoints | HTTP API contracts |
| 11 | `test_embeddings.py` | 13 | NPU embeddings, L2 norm, mean pooling, cosine similarity | Proof M1, M2, M14 |
| 12 | `test_vision.py` | 9 | VLMPipeline vision engine, visual token calculation | Proof M5 (Visual token roofline) |
| 13 | `test_transpiler.py` | 18 | GGUF→OpenVINO transpiler, INT4 dequant, SQNR, XMX pack | Proof M7, M8, M9, M21 |
| 14 | `test_transpiler_stream.py` | 14 | SSE streaming engine, O(1) RAM, SIMD quantization | Proof M15–M20, P1–P6 |
| 15 | `test_continuous_batcher.py` | 9 | PagedBlockTable, RadixCache, Little's Law admission | Proof M11, M12, M13, M30 |
| 16 | `test_batch_pipeline.py` | 16 | Continuous batch pipeline, Little's Law, dynamic reclamation | Proof M28–M35 |
| 17 | `test_batch_api.py` | 10 | Batch status API, preemption routes, 429 saturation | Little's Law saturation |
| 18 | `test_ui_endpoints.py` | 24 | Web UI SPA, SSE stream endpoints, embeddings test, convert | Web UI API contracts |
| 19 | `test_lora.py` | 16 | Multi-LoRA manager, safetensors zero-copy, LRU cache | Proof M24, M26 |
| 20 | `test_lora_api.py` | 10 | LoRA REST API endpoints, hot-swapping dispatch, isolation | Proof M23, M27 |
| 21 | `test_real_hardware.py` | 21 | Real hardware integration (Arc Xe-LPG, AI Boost NPU, USM) | Proof M36–M43 |
| 22 | `test_audio_engine.py` | 32 | OpenVINO Whisper ASR & Kokoro TTS, VAD silence gating | Proof M59–M72 |
| 23 | `test_image_engine.py` | 24 | OpenVINO Text2Image (SD 1.5, LCM, SD-Turbo, SDXL), DPAS | Proof M73–M93 |
| 24 | `test_video_engine.py` | 24 | OpenVINO Text2Video (LTX-Video, CogVideoX, AnimateDiff) | Proof M94–M114 |
| 25 | `test_multimodal_orchestrator.py` | 5 | End-to-end Studio DAG pipeline, heterogeneous scheduling | Multi-engine fork-join |
| 26 | `test_studio_director.py` | 8 | Director prompt decomposition, regex AST auto-repair | Dual-stage AST repair |
| 27 | `test_studio_models.py` | 14 | Pydantic v2 schemas, telemetry serialization | Pydantic v2 invariants |
| 28 | `test_studio_workers.py` | 7 | AudioWorker (Kokoro TTS), VisualWorker (Compositor/DiT) | USM zero-copy bridge |
| 29 | `test_studio_muxer.py` | 5 | Closed-loop pacing convergence, hold-frame padding | Audio-video sync convergence |
| 30 | `test_studio_routes.py` | 5 | `/v1/studio/generate` REST endpoint, SSE pipeline events | SSE streaming protocol |
| 31 | `test_studio_ui_triad.py` | 33 | 3-tier Triad matrix for Tab 7 Studio Web UI | Triad (Pos, Neg, Edge) |
| 32 | `test_slide_compositor_triad.py` | 38 | 16:9 HD Slide Compositor, Double-Slit, Bohr atom, Voronoi | Dynamic physics & typography |
| 33 | `test_nextgen_orchestration_triad.py` | 42 | Milestones 14.1–14.7 (QSV, Alpha dissolve, Reactive DAG) | Proofs M7–M12 |
| 34 | `test_pypi_packaging_triad.py` | 11 | Standalone wheel packaging, PEP 561, dual CLI entrypoints | Proof M115 (Wheel Deflate Invariant) |
| 35 | `test_docker_container_triad.py` | 11 | Ubuntu 24.04 multi-stage Dockerfile, Rule 29, DRI `/dev/dri` | Proofs M116–M120 |
| 36 | `test_flash_stream_triad.py` | 14 | Flash-Stream O(1) RAM streaming, NPU/GPU disaggregation, SwiGLU | Proofs M121–M134 |
| **Total** | **36 Modules** | **569** | **100% Green · 0 Failures · 0 Errors** | **Complete Invariant Verification** |

---

## 🚀 Running Tests

### 1. Execute Full Test Suite
```bash
python -m pytest -v --tb=short
# → 733 passed, 8 skipped in ~40s (Linux, 4 vCPU)
```

### 2. Run Physical Silicon Hardware Integration Tests
```bash
python -m pytest tests/integration/test_real_hardware.py -v
# → 21 passed in 4.82s on physical Intel Core Ultra silicon
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
