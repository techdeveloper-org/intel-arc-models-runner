# 🤝 Open Source Community Acknowledgments & Benchmarking Guide

[← Back to README](../README.md) · [Contributing Guidelines](../CONTRIBUTING.md) · [Benchmarks Hub](benchmarks/README.md)

---

## 🤝 Open Source Community Acknowledgments

This project stands on the shoulders of the global open-source community. We express our deepest gratitude and credit to the following pioneers and foundational projects:

- **Intel Corporation**: For open-sourcing the revolutionary [**OpenVINO™ Toolkit**](https://github.com/openvinotoolkit/openvino) and [**OpenVINO GenAI**](https://github.com/openvinotoolkit/openvino.genai), as well as developing the [**oneAPI Level Zero Specification**](https://spec.oneapi.io/level-zero/latest/index.html) and driver stacks for Intel Arc Graphics and Intel AI Boost NPUs.
- **Georgi Gerganov & llama.cpp / ggml Contributors**: For pioneering quantized local model execution ([**llama.cpp**](https://github.com/ggerganov/llama.cpp)) and defining the GGUF standard that ignited the local AI revolution.
- **The Ollama Community**: For setting the gold standard in local developer ergonomics, intuitive CLI commands, and accessible REST API conventions ([**ollama/ollama**](https://github.com/ollama/ollama)).
- **FastAPI & Sebastián Ramírez (@tiangolo)**: For creating [**FastAPI**](https://github.com/tiangolo/fastapi), the premier high-throughput asynchronous web framework that powers our drop-in Ollama & OpenAI API gateway.
- **Hugging Face**: For democratizing open-weights AI architectures, tokenizers, and the OpenVINO model hub ecosystem ([**huggingface.co**](https://huggingface.co)).
- **Claude Global Library & Knowledge Graph Community**: For providing the multi-agent orchestration knowledge graph, domain taxonomies, and rigorous mathematical derivation frameworks.

---

## 🌟 Open to Contributions

`intel-arc-models-runner` is a community-driven, 100% open-source project and we actively welcome contributions from developers, researchers, and hardware enthusiasts worldwide!

### 📢 Calling Open-Source Contributors: Help Us Benchmark Across Intel Silicon!

> [!IMPORTANT]
> **We Need Your Hardware Telemetry!** While we have physically verified and benchmarked models (`qwen-2.5-1.5b`, `llama-3.2-1b`, `deepseek-r1-1.5b`, `all-minilm-l6-v2`) on our physical testbed (**Intel Core Ultra 5 125H** with Xe-LPG iGPU and AI Boost NPU), our mission is to build the industry's most exhaustive, empirically-backed open benchmark matrix across every tier of Intel computing silicon.

If you have access to any of the following hardware, **we need your help**:

| Hardware Category | Target Architectures & Models |
|:---|:---|
| **Discrete Arc GPUs (Alchemist)** | Intel® Arc™ **A770** (16GB & 8GB), **A750**, **A580**, **A380**, **A310** |
| **Next-Gen Discrete GPUs (Battlemage)** | Intel® Arc™ **B580**, **B570** (Xe2-HPG Architecture) |
| **Core Ultra Series 2 (Lunar Lake)** | Intel® Core™ Ultra 7 **258V** / **268V** / **256V** (Xe2-LPG iGPU + 48 TOPS NPU 4000) |
| **Core Ultra Series 2 (Arrow Lake)** | Intel® Core™ Ultra 9 **285K**, Ultra 7 **265K**, Ultra 5 **245K** |
| **Core Ultra Series 1 (Meteor Lake)** | Intel® Core™ Ultra 7 **155H**, Ultra 9 **185H**, Ultra 5 **135H** |
| **Workstation & Data Center Silicon** | Intel® Data Center GPU **Flex 170** / **Max 1100**, Intel® Xeon® Scalable with AMX |

---

## ⚡ How to Run Benchmarks & Contribute in 3 Steps

### 1. Verify Your Intel Hardware Discovery
```bash
arc-runner devices
```

### 2. Execute an Empirical Benchmark (Disk-Safe & Automatic Cleanup)
You don't need hundreds of gigabytes of disk space! Our sequential runner downloads, compiles, tests, logs telemetry, and automatically reclaims disk space ($< 5\text{ GB}$ peak storage):

```bash
# Benchmark an individual model:
python scratch/run_live_sequential_model_benchmarks.py --model qwen-2.5-1.5b

# Other workloads (each writes a schema-v1 result file):
python scratch/run_live_vlm_benchmarks.py --model qwen2-vl-7b
python scratch/run_live_audio_benchmarks.py --model whisper-tiny
python scratch/run_live_image_benchmarks.py --model stable-diffusion-v1-5 --download
python scratch/run_live_video_benchmarks.py --model ltx-video --download
```

### 3. Submit Your Telemetry
- Each runner writes a schema-v1 result file under `docs/benchmarks/results/` (validate it with `python scripts/validate_bench_json.py`; method in [METHOD.md](benchmarks/METHOD.md)).
- [**Open a GitHub Issue**](https://github.com/techdeveloper-org/intel-arc-models-runner/issues/new) titled `[Community Benchmark] <Your Silicon Model> - <Model Slug>` and paste your telemetry.
- Or open a Pull Request updating the benchmark matrix in [`docs/benchmarks/comparative_matrix.md`](benchmarks/comparative_matrix.md)!
- **All community benchmark contributors receive permanent credit in the project's Hall of Fame and documentation!**

---

## 🛠️ Other Ways You Can Contribute

1. **Quantized Models & Benchmarks**: Share NNCF INT4/INT8 benchmark profiles, custom calibration datasets, and accuracy evaluations.
2. **Frontend Integrations**: Help expand test suites and verification for Open WebUI, Continue.dev, Cursor, and LangChain.
3. **Core Optimization**: Assist in kernel tuning, Level Zero USM memory pool improvements, and dynamic $\gamma$ speculative decoding heuristics.

Please review our [Contributing Guidelines](../CONTRIBUTING.md) and submit Pull Requests directly to the `main` branch.
