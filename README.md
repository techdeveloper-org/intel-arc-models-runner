# Intel Arc Models Runner (`intel-arc-models-runner`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Intel](https://img.shields.io/badge/Hardware-Intel%20Arc%20GPU%20%7C%20Intel%20NPU-0071C5.svg)](https://www.intel.com)
[![API](https://img.shields.io/badge/API-Ollama%20%7C%20OpenAI%20Compatible-green.svg)](https://github.com/techdeveloper-org/intel-arc-models-runner)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/Tests-pytest%20suite-brightgreen.svg)](docs/testing.md)
[![Release](https://img.shields.io/badge/Release-v0.2.0%20(wheel%20built%2C%20not%20published)-orange.svg)](https://github.com/techdeveloper-org/intel-arc-models-runner/releases)

A purpose-built, mathematically proven, zero-bloat local LLM runner, multi-modal generative studio, and high-throughput server engineered specifically for **Intel® Arc™ GPUs (discrete & integrated)** and **Intel® NPUs (Neural Processing Units)**.

---

## 📚 Documentation Index

To keep this landing page fast, readable, and executive, detailed technical documentation, hardware proofs, and engine manuals have been modularized into dedicated guides:

| Guide / Document | Description | Key Topics |
|:---|:---|:---|
| [**🎬 Arc MultiModal Studio**](docs/multimodal_studio.md) | Autonomous AI Production Studio & Co-Benchmarking | Flagship scenario, 5 bottlenecks solved, Cloud Cost-Killer ROI analysis, Studio CLI & REST API. |
| [**🏗️ Architecture & Subsystems**](docs/architecture.md) | System Architecture & Hardware Interfaces | Subsystem deep-dive, Level Zero integration, Multi-LoRA hot-swapping, Continuous batching & paged KV cache. |
| [**📦 Model Hub & Hardware Engines**](docs/model_hub_and_engines.md) | 34 Registry Models & Multi-Modal Engine Guides | Model matrix, Audio AI (Whisper/Kokoro), Image (SD/SDXL), Video (LTX-Video), 7-Tab Web UI, GGUF transpiler, NPU embeddings. |
| [**📊 Benchmarks Hub**](docs/benchmarks/README.md) | Comprehensive Empirical & Quantitative Results | Entry point to all benchmarking data, testbed silicon specs, and methodology. |
| ↳ [**Comparative Benchmark Matrix**](docs/benchmarks/comparative_matrix.md) | Quantitative Inference Benchmark Matrix | Tables 1–7 (LLMs, 5W NPU Embeddings, Vision, Audio, Image, Video, and Table 7 Flash-Stream 32B/72B from a synthetic container, not real weights) & Leviathan speculative-efficiency derivation; legacy tables, UNVERIFIED. |
| ↳ [**Mathematical Proofs (M59–M134)**](docs/benchmarks/mathematical_proofs.md) | Formal Theorems & Invariants | 70 formal mathematical proofs covering Audio AI, Diffusion, Video AI, and Flash-Stream Heterogeneous Zero-RAM Execution (Proofs M121–M134). |
| ↳ [**Empirical Silicon Proofs**](docs/benchmarks/empirical_silicon.md) | Real Silicon Hardware Telemetry | UNVERIFIED legacy document (see its banner): the 30.8× GPU-vs-CPU TTFT figure is not reproduced (measured CPU/GPU TTFT ratios on Qwen2.5-1.5B are 2.3× to 17.7×, and the speed-up claim gate fails), live sequential reclaim runs, 21-test hardware suite, GPU vs NPU breakdown, a Flash-Stream memory trace taken on a synthetic container (mapped bytes, not resident memory), and an Ollama comparison whose Ollama baseline was not run. |
| ↳ [**Ollama Showdown & 30-Case Audit**](docs/benchmarks/ollama_showdown.md) | Head-to-Head Comparison vs Ollama | Ollama "Silent CPU Fallback" anatomy (Proof M56), output coherence (M57/M58), 30-case Triad Audit matrix, and a heavy-model Ollama OOM comparison whose Ollama run was never executed (UNVERIFIED). |
| [**🤖 Claude Code**](docs/claude-code.md) | Use Claude Code against `arc-runner serve` | Anthropic `/v1/messages` (stream and tool use), env vars for Bash and PowerShell, what works, limits and troubleshooting; Claude Code itself was not run, see its UNVERIFIED list. |
| [**🚀 Deployment & Packaging Guide**](docs/deployment.md) | Standalone PyPI Wheel & Docker Deployment | Production guide for `pip install arc-runner`, Ubuntu 24.04 Docker image, `/dev/dri` pass-through, and Kubernetes Helm specs. |
| [**🧪 Complete Test Coverage**](docs/testing.md) | Full Test Suite Roadmap & Execution | Non-hardware suite counts measured with `pytest` (the guide gives the collected, passed, skipped and hardware-deselected numbers and their date); hardware-marked tests are reported as not run, Triad testing patterns, C sanitizer harnesses. |
| [**🗺️ Roadmap & Multi-Modal Evolution**](docs/roadmap.md) | Delivered Milestones & Packaging Horizon | 15 core milestones + full Packaging Horizon delivered, 7 next-gen orchestration hurdles scorecard, PyPI & Docker targets. |
| [**📐 Visual Diagrams Guide**](docs/diagrams/README.md) | UML & Draw.io Diagram Catalog | 13 architectural UML diagrams with embedded previews and editable draw.io links. |
| [**🔬 R&D Forensics Reports**](docs/research/) | Deep Technical R&D & Whitepapers | 7 foundational research papers on Ollama defects, Intel silicon, solution blueprints, orchestration proofs, packaging math, and bare-metal flash streaming (Proofs M121–M134). |
| [**🤝 Community Benchmarks Guide**](docs/community_benchmarks.md) | Open Source Benchmarking Call | Community guidelines for benchmarking across Intel silicon (Alchemist, Battlemage, Lunar Lake, Arrow Lake). |

---

## 🚀 Overview

While mainstream local LLM tools like Ollama were designed with a CUDA-first and Apple Metal-first focus, `intel-arc-models-runner` treats Intel silicon as first-class citizens:

- **Native Dual Silicon Acceleration**: Run models natively on **Intel Arc Discrete GPUs** (A770, A750, B580), **Core Ultra iGPUs** (Xe-LPG, Xe2), and **Intel NPUs** (AI Boost NPU 3720 / NPU 4000).
- **Bare-Metal Flash-Stream Heavy Model Engine**: Designed to stream 32B/72B-class weights through a fixed-size layer ring buffer (memory-mapped reads, optional prefetch, NPU attention / GPU SwiGLU split; Proofs M121–M134 are mathematical models, not measurements). No real 32B/72B model has been run: the only Flash-Stream run on record used a synthetic 8-layer, hidden-size-32 container on a host with about 1.24 GB free DRAM, so no model quality, tokens-per-second, or memory-saving claim is made for 32B/72B.
- **Draft-model speculative decoding (NPU draft, GPU verify)**: wired through OpenVINO GenAI's draft-model support. Its speed-up, acceptance rate and power draw have not been measured with this repository's harness; earlier published figures for them were removed (see the [benchmark method](docs/benchmarks/METHOD.md)).
- **Autonomous Multi-Modal Media Studio**: Orchestrates LLM Director, Kokoro Voiceover, and SDXL / LTX-Video Visuals with zero-copy memory handoff and closed-loop pacing convergence.
- **Drop-in Ollama & OpenAI Compatibility**: Compatible with `/api/generate`, `/api/chat`, `/api/tags`, and `/v1/chat/completions` for instant use with Open WebUI, Continue.dev, Cursor, and LangChain.
- **Zero-Bloat Standalone Runtime**: Built directly on OpenVINO GenAI C++ and Level Zero runtimes; no 5 GB oneAPI SDK installation or heavy PyTorch environments required.

---

## 🎬 Arc MultiModal Studio Spotlight

`Arc MultiModal Studio` orchestrates Text, Audio, and Visual engines locally on Intel heterogeneous silicon:

```
User Prompt ──► Director LLM (NPU/GPU) ──► Kokoro TTS (NPU/GPU) ──► Slide Compositor / DiT (Arc GPU) ──► Muxer (.mp4)
```

- **Cloud cost comparison**: runs the studio pipeline locally. Cloud prices are third-party and not verified by this project, so ROI is computed only from prices you supply (see [multimodal studio](docs/multimodal_studio.md)); no savings figure is claimed here.
- **5 Critical Hurdles Solved**: Heterogeneous Compute Splitting (no peak-VRAM figure is claimed; none is measured), a Level Zero USM zero-copy bridge (opt-in, no latency measured), Closed-Loop Pacing Convergence (tolerance `convergence_tolerance_sec=0.05` in the muxer), Dual-Stage AST Auto-Repair, and Fork-Join Concurrency Barriers.
- **Next-Gen Features**: Hardware QSV H.264 encoding, Alpha cross-dissolve, Audio-First Reactive DAG, dynamic karaoke subtitles, and autonomous VLM inspection gate.

👉 *Read the full guide: [**Arc MultiModal Studio Documentation**](docs/multimodal_studio.md)*

---

## 📋 Prerequisites / Requirements

- **Supported Hardware**:
  - **Discrete GPUs**: Intel® Arc™ A770, A750, A580, A380, A310 (Alchemist) or B580, B570 (Battlemage)
  - **Integrated GPUs**: Intel® Core™ Ultra Series 1 (Meteor Lake Xe-LPG) or Series 2 (Lunar Lake / Arrow Lake Xe2)
  - **Neural Processing Units**: Intel® AI Boost NPU (NPU 3720 / NPU 4000)
- **Operating System**: Windows 11 / Windows 10 (64-bit), or Ubuntu 22.04+ / Linux kernel 6.6+
- **Driver Stack**: Intel Graphics Driver 32.0.101.6127+ or oneAPI Level Zero runtime 1.3+
- **Python**: Version 3.10, 3.11, 3.12, or 3.13

---

## ⚡ Installation / Setup

### Option 1: Standalone PyPI Installation (Recommended)
Install the standalone binary distribution directly:
```bash
pip install arc-runner
```
*Provides dual CLI commands `arc-runner` and `intel-arc-models-runner` with pre-bundled Studio Web UI and cross-platform Level Zero resolution.*

### Option 2: Production Docker Container (Ubuntu 24.04 with Intel Drivers)
Run directly with GPU acceleration via Docker Compose:
```bash
docker compose up -d
```
Or execute manual container run with direct `/dev/dri` render node pass-through:
```bash
docker run -d --name intel-arc-models-runner --device /dev/dri:/dev/dri -p 8000:8000 ghcr.io/techdeveloper-org/intel-arc-models-runner:latest
```

### Option 3: Windows Native Setup

On Windows, use `install.bat` to install the package and configure the CLI:

```powershell
.\install.bat
```

You can also double-click `install.bat` in File Explorer. It launches the
PowerShell installer with a process-only execution-policy bypass when needed;
it does not change your machine's PowerShell policy. This is the recommended
entry point when scripts are restricted.

Running `.\install.ps1` directly is different: PowerShell checks its execution
policy before loading that file. If scripts are disabled, the command fails
before the installer can run, so the script cannot bypass that restriction
from inside itself. Use `install.bat` in that case.

### Option 4: Developer Editable Mode
```bash
git clone https://github.com/techdeveloper-org/intel-arc-models-runner.git
cd intel-arc-models-runner
python -m pip install -e ".[dev]"
```

On Windows, `pip install` by itself does not add Python's `Scripts` directory to
`PATH`. Run `install.bat` with the same Python environment active to configure
that directory and verify the CLI. If you have already installed with pip, you
can still run `install.bat`; it installs from this project checkout. Open a new
terminal after setup so it picks up the updated `PATH`. In the current terminal,
use `python -m arc_runner.cli.main --version` until you reopen it.

---

## 💻 Usage

### 1. Discover Connected Silicon & Live Telemetry
```bash
# Discover connected Intel Arc GPUs, NPUs, and CPUs
arc-runner devices

# Run real silicon integration tests with Rich terminal telemetry
arc-runner test-hardware
```

### 2. Run a Pulled Model (Ollama Is Not Required)

`arc-runner` runs models itself using OpenVINO. **You do not need to install or
start Ollama.** The server implements Ollama-compatible API routes so clients
that already support Ollama can connect to `arc-runner`; that compatibility
does not mean the Ollama application is the model runtime here. This project
does not currently provide an interactive `arc-runner run <model>` terminal
chat command. Use its browser chat UI or send requests to its API.

Pull a model, then start the server from the same working directory (the
default model folder is `./models`):

```bash
arc-runner pull llama-3.2-3b
arc-runner models
arc-runner serve --host 127.0.0.1 --port 11434 --default-model llama-3.2-3b
```

Leave the server running, then open `http://localhost:11434/` and use the Chat
tab. `--default-model` asks the server to load that model at startup; requests
should still include its model name.

**Port conflict with Ollama:** Ollama commonly listens on port `11434`. Only one
process can listen on the same host and port. If Ollama is running, either stop
it or start `arc-runner` on another port, such as `11435`, and use that port in
the browser and client settings:

```bash
arc-runner serve --host 127.0.0.1 --port 11435 --default-model llama-3.2-3b
```

Then open `http://localhost:11435/`. To keep using both applications, leave
Ollama on `11434` and use `11435` for `arc-runner`. For compatible clients such
as Open WebUI, Continue.dev, or Cursor, set the base URL to the matching runner
address, for example `http://localhost:11435`.

You can also send a one-off prompt using the Ollama-compatible generate API.
This PowerShell example prints the response as JSON:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:11435/api/generate `
  -ContentType "application/json" `
  -Body '{"model":"llama-3.2-3b","prompt":"Explain Intel Arc GPU in simple words","stream":false}'
```

Use port `11434` in the URL if that is the port you selected for `arc-runner`.

#### Claude Code (Anthropic Messages API)

The server also answers `POST /v1/messages` and `POST /v1/messages/count_tokens`, so Claude Code can be pointed at it with `ANTHROPIC_BASE_URL=http://127.0.0.1:11434` and `ANTHROPIC_AUTH_TOKEN=local`. Streaming, tool-call parsing and the error envelope are tested with fake engines only; Claude Code itself was not run, and the context window is resolved per model (`ARC_CONTEXT_WINDOW`, then `max_position_embeddings` in the model's `config.json`, then 4096); a window of 4096 is smaller than Claude Code's own prompt. Read [docs/claude-code.md](docs/claude-code.md) for the exact steps, the limits and what is UNVERIFIED.

### 3. Core CLI Quick Reference

```bash
# Pull official curated models
arc-runner pull llama-3.2-3b
arc-runner pull deepseek-r1-1.5b

# Benchmark on Intel Arc GPU or NPU with 95% confidence intervals
arc-runner benchmark llama-3.2-3b --device GPU --iterations 5

# Speech-to-Text with Whisper on Arc GPU
arc-runner transcribe lecture.wav --model whisper-base --device GPU

# Text-to-Speech with Kokoro-82M on Arc GPU / NPU
arc-runner tts "Intel Arc matrix architecture accelerates local AI." --output voice.wav --voice af_heart

# Text-to-Image with Stable Diffusion / SDXL on Arc GPU
arc-runner generate-image "Cyberpunk laboratory with glowing neon circuits, 8k" --output lab.png --model lcm --steps 4

# Text-to-Video with LTX-Video on Arc GPU
arc-runner generate-video "A golden retriever puppy chasing a bouncing tennis ball, 4k" --output puppy.mp4 --model ltx-video

# Autonomous Studio: Full Video Production from natural language prompt
arc-runner studio "Quantum Physics and Wave Particle Duality" --duration 12.0 --resolution 1280x720

# Offline GGUF -> OpenVINO INT4/INT8 Transpiler
arc-runner convert ./llama-3.2-3b-Q4_0.gguf --output ./models/llama-openvino --precision int4

# NPU vector embeddings (power not measured)
arc-runner embed nomic-embed-text "What is Intel Arc GPU?" --device NPU
```

👉 *For the full catalog of 34 registry entries and detailed engine parameters, see: [**Model Hub & Hardware Engines Reference**](docs/model_hub_and_engines.md)*

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Client ["Client & Tooling Layer"]
        CLI["arc-runner CLI"]
        WebUI["Studio WebUI & Open WebUI (:11434)"]
        IDE["Cursor / Continue.dev / LangChain"]
    end

    subgraph Daemon ["arc-runner Daemon (:11434)"]
        Router["Asynchronous HTTP Gateway (FastAPI)"]
        BatchPipeline["BatchExecutionPipeline & ContinuousBatcher"]
        ModelMgr["ModelManager & PagedBlockTable"]
        Router --> BatchPipeline & ModelMgr
    end

    subgraph Engines ["Pluggable Backend Engines (OpenVINO GenAI)"]
        OVEngine["OpenVINOGenAIEngine (LLMs)"]
        SpecEngine["HybridSpeculativeEngine (NPU Draft + Arc Verify)"]
        AudioEngine["AudioEngine (Whisper ASR & Kokoro TTS)"]
        ImageEngine["ImageEngine (SD 1.5, LCM, SD-Turbo, SDXL)"]
        VideoEngine["VideoEngine (LTX-Video, CogVideoX, AnimateDiff)"]
        StudioEngine["StudioDirector & MultiModal Pipeline"]
    end

    subgraph Silicon ["Physical Intel Silicon Layer (Level Zero)"]
        ArcGPU["Intel Arc GPU (Xe-LPG / Battlemage / A770)"]
        IntelNPU["Intel AI Boost NPU (3720 / 4000)"]
        HostCPU["Intel Core Ultra Host CPU"]
    end

    Client --> Daemon
    Daemon --> Engines
    Engines --> Silicon
```

👉 *For detailed subsystem breakdowns, Multi-LoRA hot-swapping, and UML diagrams, see: [**Architecture & Subsystem Deep Dive**](docs/architecture.md)*

---

## ⚡ Native System-Level C Acceleration Layer

`intel-arc-models-runner` ships a bare-metal **native C/C++ acceleration layer** (`libarc_native_sys.dll` / `.so`) that bypasses Python overhead entirely for maximum throughput on Intel Arc silicon.

### Architecture Overview

```
Python Layer  →  ctypes FFI bridge  →  libarc_native_sys.dll
                                         ├─ arc_flash_stream_mmap.c   (Win32 zero-copy mmap)
                                         └─ arc_level_zero_bridge.c   (Level Zero USM DMA)
```

### Key Capabilities

No speed-up figure is claimed for any item below until it is measured with the [benchmark method](docs/benchmarks/METHOD.md); status is what the code and tests establish today.

| Feature | Description | Status |
|:---|:---|:---|
| **Bounds-checked memory mapping** | `arc_mmap_*` maps the model file read-only; every tensor extent is validated against the file size at registration and again at read time, the layer table is sealed before concurrent reads | Implemented; ASan/UBSan harness with a 1,000,000-case fuzz, Win32 branch compile-checked with mingw-w64 |
| **Zero-copy tensor views** | NumPy views over the mapping are read-only and keep the mapping alive; closing the engine with live views defers the unmap | Implemented and tested |
| **Best-effort page locking** | `lock_range` pins ranges with `VirtualLock` / `mlock`; a refusal (quota) is reported, never raised | Called by the streaming path only when `pin_resident_layers` or `ARC_PIN_RING=1` is set; effect not measured |
| **Direct-mapped ring buffer** | layer `L` always uses slot `L % slots`, one loader per slot, safe shutdown; protocol modelled in `docs/formal/RingSlots.tla` | Implemented; explicit-state check done, TLC run pending |
| **Level Zero USM and DMA** | GPU-only device selection (never the NPU), bounded waits below the 2 s Windows TDR limit, tracked allocations | Implemented and tested against a fake loader; not used by the streaming path; not yet run against a real driver |
| **INT8 KV cache (`u8`)** | opt-in with `KV_CACHE_PRECISION=u8` or `ARC_KV_CACHE_PRECISION=u8`; off by default because no quality gate exists | Opt-in |
| **Shader/kernel cache** | compiled kernels are cached in `.cache/ov_cache/` on GPU | Effect on start-up time not measured |
| **SDPA option** | `GPU_ENABLE_SDPA_OPTIMIZATION=True` on GPU devices | Effect not measured |

### Memory-Bandwidth Roofline

Token generation that streams all weights once per token is bounded by

```
TPS_max = B_sustained / (W + KV_bytes_per_token_context)
```

where `B_sustained` is the measured sustained memory bandwidth (`python scripts/probe_hardware.py`), `W` is the weight bytes read per token, and the KV term grows with context length. A measured rate above this ceiling is flagged SUSPECT by the claim gates. An earlier version of this section multiplied by an extra efficiency factor of 0.61 on top of a bandwidth that was already a sustained figure and then reported "99.8% of the roofline"; that claim was removed.

### Building the Native Library

The committed `libarc_native_sys.dll` is ABI 3 and its sidecar (`libarc_native_sys.dll.build.json`) records a matching sha256 (built with MinGW GCC 15.2 on Windows 11). Rebuild after any change under `src/arc_runner/native/`; a stale or missing library makes the Python bridge fall back to the pure-Python path:

```bash
python src/arc_runner/native/build_native.py
```

This compiles `arc_flash_stream_mmap.c` and `arc_level_zero_bridge.c` with `-std=c17 -Wall -Wextra -Wpedantic -Werror`, links them, and writes `libarc_native_sys.<dll|so>.build.json` with the sha256, compiler, flags and ABI version; the bridge verifies it on load. Level Zero is loaded at run time (`ze_loader`), not linked. `ARC_NATIVE_DISABLE=1` forces the Python path.

**Requirements for rebuild:**
- GCC 13+ (MinGW-w64 on Windows); MSVC is not supported (the sources use C11 atomics and GCC build info)
- No Level Zero SDK headers are needed: the structure layouts the bridge uses are in `arc_ze_types.h`
- A Level Zero loader (`ze_loader`) at run time; Intel GPU driver 32.0.101.6127+ for GPU use

### Running the Live Benchmark

The benchmark harness compares arc-runner (Intel Arc iGPU) vs Ollama (CPU fallback) across
Small / Medium / Hard complexity tiers using real downloaded GGUF models:

```bash
python scratch/run_live_real_models_benchmark_and_compare.py
```

Results are saved to `scratch/live_real_models_benchmark_results.json`.

**Measured results in `scratch/live_real_models_benchmark_results.json` (Intel Core Ultra, Ollama on CPU):**

| Model | Complexity | arc-runner (Arc Xe-LPG) | Ollama (CPU) |
|:---|:---|:---:|:---:|
| Llama-3.2-3B INT4 | Low | not measured in this run | 14.7 tok/s, 2,238 ms TTFT |
| Llama-3.2-3B INT4 | Medium | not measured in this run | 17.6 tok/s, 70 ms TTFT |
| Llama-3.2-3B INT4 | Hard | not measured in this run | 17.4 tok/s, 959 ms TTFT |
| Qwen-2.5-1.5B INT4 | All tiers | not measured | not measured |
| TinyLlama-1.1B INT4 | All tiers | not measured | not measured |

> **Provenance notice:** earlier versions of this table showed arc-runner figures (for example 54.2 tok/s and
> 165 ms TTFT for every tier) that came from a hard-coded reference row, not from a measurement, and Ollama
> figures that do not match the stored JSON. They were removed. arc-runner rows stay empty until the benchmark
> is rerun on real hardware; values will then be generated from the JSON with their source label.
>
> **Heavy models (32B/72B):** the FlashStream results in the JSON were produced on a synthetic 8-layer,
> hidden-size-32 container, not on real 32B/72B weights, and the Ollama out-of-memory outcome was not run. No
> throughput or RAM figure for real 32B/72B models is claimed until it is measured.

---

## 📊 Benchmark Highlights: Intel Arc vs. Ollama CPU

> **Provenance status: UNVERIFIED.** The figures in this table were not produced by the current benchmark harness
> and must not be quoted. The "Discrete Arc A770" column is for hardware that is not part of the testbed and is a
> projection, not a measurement. No Llama-3.2-1B, MiniLM, LCM or LTX-Video result file exists in
> `docs/benchmarks/results/`; no power was measured on this testbed, so every watt and joule value here is unverified;
> speculative-decoding speed was never measured, so the former "Spec" figures were removed; the Whisper row mixes a
> 4.0 s clip (Ollama cell) with a 10.0 s clip (iGPU cell), so its ratio is not comparable. Measured results are in the
> [generated table](docs/benchmarks/README.md#measured-results-generated) (Qwen2.5-1.5B on GPU/NPU/CPU, Whisper-tiny, Qwen2-VL-7B). See [`docs/plans/MASTER_PLAN_v4.md`](docs/plans/MASTER_PLAN_v4.md) section 18.

When Ollama runs on Intel Core Ultra systems, it silently falls back to 100% CPU execution. `intel-arc-models-runner` unleashes native GPU and NPU acceleration:

| Workload & Model | Ollama (100% CPU Fallback) | `arc-runner` AI Boost NPU (5W) | `arc-runner` Arc Xe-LPG iGPU | `arc-runner` Discrete Arc A770 | Hardware Advantage |
|:---|:---:|:---:|:---:|:---:|:---|
| **Llama-3.2-1B (TTFT)** | 780 ms (40W) | 890 ms (**5.1W**) | **115 ms** (21W) | **38 ms** (85W) | **20.5× Faster TTFT** 🚀 |
| **Llama-3.2-1B (Decode)** | 24.2 tok/s | 19.8 tok/s (**3.88 tok/J**) | **72.4 tok/s** | **215.3 tok/s** | **8.9× Higher Throughput** ⚡ |
| **Llama-3.2-3B (Decode)** | 11.4 tok/s | — | **38.2 tok/s** | **114.6 tok/s** | **10.0× Higher Throughput** ⚡ |
| **all-MiniLM-L6-v2 (5W RAG)** | 85.2 ms (3.24 J) | **18.6 ms** (**0.09 J**) | **14.2 ms** (0.25 J) | — | **36× More Energy Efficient!** 🌱 |
| **Whisper-Base (STT)** | 1.944 s (RTF 0.486, 4.0 s clip) | — | 0.354 s (10.0 s clip) | — | not comparable (different clips) |
| **LCM Image Gen (4-Step)** | 2.230 s | — | **0.310 s** (**12.9 it/s**) | — | **7.2× Faster Image Gen** 🎨 |
| **LTX-Video Gen (16-Frame)**| 3.305 s (4.8 fps) | — | **1.400 s** (**11.4 fps**) | — | **2.4× Faster Video Gen** 🎥 |

👉 *For the full model comparison (UNVERIFIED legacy tables), mathematical proofs M59–M114, and 30-case Triad audit, visit: [**Benchmarks & Empirical Proofs Hub**](docs/benchmarks/README.md)*

---

## 🧪 Test Coverage

Measured on 2026-10-29 (Linux x86-64 VM, no Intel GPU or NPU): `pytest --collect-only` collects **1507 tests in 72 modules**; `pytest -m "not hardware"` collects 1484 and gives **1476 passed, 8 skipped, 0 failed**. The other **23 tests are marked `hardware`** (21 in `tests/integration/test_real_hardware.py`, 2 in `tests/test_plan_gap_tests.py`), were **not run** for these numbers, and are not counted as passed. The 8 skips are one Windows-only DLL-directory test, six tests needing a Level Zero driver and one needing a Quick Sync device. The suite keeps growing; see [the test guide](docs/testing.md) for the per-module table and how to regenerate it.
- **Hardware Integration**: `tests/integration/test_real_hardware.py` holds 21 tests meant for Intel Core Ultra hardware; this repository has no committed run log for them, so no on-silicon pass count is claimed.
- **Multi-Modal Studio**: 33 Studio Web UI triad tests, 38 slide compositor tests, and 42 next-gen orchestration tests (collected).
- **Engines & Protocols**: tests for OpenVINO GenAI engines, continuous batching, LoRA swapping, and SSE streaming.

```bash
# Non-hardware tests (what the counts above use)
python -m pytest -m "not hardware"
# → 1476 passed, 8 skipped (Linux, 2026-10-29, about 67 s)
```

👉 *For the complete test module matrix and instructions, see: [**Complete Test Coverage Guide**](docs/testing.md)*

---

## 🗺️ Roadmap & Production Status

- ✅ **Completed Core Milestones**: 15 major milestones including OpenVINO continuous batching, Multi-LoRA swapping, Audio/Image/Video engines, Arc MultiModal Studio with 7-Tab Web UI, and all **7 Next-Gen Orchestration Hurdles** (QSV H.264 encoding, Alpha cross-dissolve, Audio-First Reactive DAG, karaoke subtitles, audio ducking, and closed-loop VLM inspection).
- ✅ **Completed Packaging Horizon**:
  1. **PyPI Wheel**: a standalone wheel (`arc_runner-0.1.0-py3-none-any.whl`, and a 0.2.0 build in `dist/`) with dual CLI aliases. Whether any version is published on PyPI was not verified offline; the CHANGELOG states 0.2.0 is not published.
  2. **Docker Container**: Ubuntu 24.04 Dockerfile (GHCR publication not verified) with pre-baked Intel Level Zero drivers, OpenVINO runtime, non-root user execution, and `/dev/dri` pass-through.

👉 *Read the full status & deployment specs: [**Roadmap & Multi-Modal Evolution**](docs/roadmap.md) and [**Deployment Guide**](docs/deployment.md)*

---

## 🤝 Contributing

`intel-arc-models-runner` is a community-driven, 100% open-source project. We actively welcome contributions from developers, researchers, and hardware enthusiasts worldwide!

- **Calling Open-Source Contributors**: Help us benchmark across Intel silicon (Arc A770/A750/B580, Lunar Lake, Arrow Lake, Meteor Lake, Xeon). All contributors receive permanent credit in our documentation Hall of Fame!
- Please review our [**Contributing Guidelines**](CONTRIBUTING.md) and [**Community Benchmarks Guide**](docs/community_benchmarks.md).

---

## 📄 License

Licensed under the [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).
