# 📊 Benchmarks & Empirical Proofs Hub

[← Back to README](../../README.md) · [Architecture Guide](../architecture.md) · [Model Hub & Engines](../model_hub_and_engines.md)

---

Welcome to the benchmarking and empirical verification center of `intel-arc-models-runner`. Every metric in this documentation has been physically measured on real Intel silicon (**Intel® Core™ Ultra 5 125H** with Xe-LPG iGPU and AI Boost NPU 3720) or derived from foundational mathematical principles.

---

## 🧭 Documentation Index

| Document | Focus & Scope | Key Content |
|:---|:---|:---|
| [**1. Comparative Benchmark Matrix**](comparative_matrix.md) | Quantitative 25-Model Matrix | Tables 1–6 covering LLMs, 5W NPU Embeddings, Multimodal VLM, Audio AI (STT/TTS), Image Generation (SD/LCM/SDXL), Video AI (LTX-Video/CogVideoX), and Leviathan speculative efficiency proof. |
| [**2. Mathematical Proofs (M59–M114)**](mathematical_proofs.md) | Formal Mathematical Theorems | 56 formal proofs covering Audio AI (Proofs M59–M72), Text-to-Image Generation (Proofs M73–M93), and Video AI (Proofs M94–M114). |
| [**3. Empirical Silicon Proofs**](empirical_silicon.md) | Physical Hardware Telemetry | Testbed specs, Head-to-head Arc GPU vs Host CPU (30.8× TTFT), live sequential reclaim benchmark, real hardware integration tests, and standalone GPU vs NPU breakdown (Proofs M50–M55). |
| [**4. Ollama Showdown & 30-Case Audit**](ollama_showdown.md) | Empirical Showdown vs. Ollama | Ollama "Silent CPU Fallback" anatomy (Proof M56), head-to-head telemetry, Shannon Token Entropy (Proofs M57–M58), and complete 30-case Triad Audit verification matrix. |

---

## 🖥️ Physical Testbed Hardware Specification

- **Host CPU**: Intel® Core™ Ultra 5 125H (14 Cores / 18 Threads, Meteor Lake Hybrid Architecture)
- **Host GPU**: Intel® Arc™ Graphics (Xe-LPG integrated GPU, 7 Xe-cores, Level Zero DPAS XMX engines, Unified Shared Memory `GPU_USM_MEMORY`)
- **Host NPU**: Intel® AI Boost NPU 3720 (5W dedicated neural accelerator)
- **Host OS**: Microsoft Windows 11 Home (AMD64 build 26100)
- **Driver Stack**: Intel Graphics Driver 32.0.101.6127 / oneAPI Level Zero 1.3
- **Evaluated Models**: Curated 25-model catalog from Model Hub
