# 📊 Benchmarks & Empirical Proofs Hub

[← Back to README](../../README.md) · [Architecture Guide](../architecture.md) · [Model Hub & Engines](../model_hub_and_engines.md) · [Flash-Stream Research](../research/07_bare_metal_flash_stream_heterogeneous_orchestration.md)

---

Welcome to the benchmarking and empirical verification center of `intel-arc-models-runner`. Every metric in this documentation has been physically measured on real Intel silicon (**Intel® Core™ Ultra 5 125H** with Xe-LPG iGPU and AI Boost NPU 3720) or derived from foundational mathematical principles.

---

## 🧭 Documentation Index

| Document | Focus & Scope | Key Content |
|:---|:---|:---|
| [**1. Comparative Benchmark Matrix**](comparative_matrix.md) | Quantitative 25-Model Matrix & Heavy Models | Tables 1–6 covering LLMs, 5W NPU Embeddings, Multimodal VLM, Audio AI, Image Generation, Video AI; **Table 7** covering Heavy Models Zero-RAM Flash-Stream (32B & 72B on 16GB RAM) and Leviathan speculative proof. |
| [**2. Mathematical Proofs (M59–M134)**](mathematical_proofs.md) | Formal Mathematical Theorems | 76 formal proofs covering Audio AI (M59–M72), Text-to-Image (M73–M93), Video AI (M94–M114), and Flash-Stream Zero-RAM Heterogeneous Execution (**M121–M134**). |
| [**3. Empirical Silicon Proofs**](empirical_silicon.md) | Physical Hardware Telemetry | Testbed specs, Head-to-head Arc GPU vs Host CPU (30.8× TTFT), live sequential reclaim benchmark, hardware integration tests, Standalone GPU vs NPU breakdown (M50–M55), **Section 6: Flash-Stream 64-layer memory trace ($\Delta M \le 1\text{ MB}$)**, and **Section 7: Live Ollama CPU vs Arc GPU multi-tier showdown**. |
| [**4. Ollama Showdown & 30-Case Audit**](ollama_showdown.md) | Empirical Showdown vs. Ollama | Ollama "Silent CPU Fallback" anatomy (Proof M56), head-to-head telemetry, Shannon Token Entropy (M57–M58), 30-case Triad Audit, **Section 5: Heavy Models Ollama OOM Collapse vs Flash-Stream Zero-RAM**, and **Section 6: Side-by-side prompt responses**. |

---

## 🖥️ Physical Testbed Hardware Specification

- **Host CPU**: Intel® Core™ Ultra 5 125H (14 Cores / 18 Threads, Meteor Lake Hybrid Architecture)
- **Host GPU**: Intel® Arc™ Graphics (Xe-LPG integrated GPU, 7 Xe-cores, Level Zero DPAS XMX engines, Unified Shared Memory `GPU_USM_MEMORY`)
- **Host NPU**: Intel® AI Boost NPU 3720 (5W dedicated neural accelerator)
- **Host RAM**: 15.43 GB Total Physical RAM (Windows 11 AMD64 build 26100)
- **Driver Stack**: Intel Graphics Driver 32.0.101.6127 / oneAPI Level Zero 1.3
- **Evaluated Models**: Curated 25-model catalog from Model Hub + Qwen 32B/72B Flash-Stream architectures
