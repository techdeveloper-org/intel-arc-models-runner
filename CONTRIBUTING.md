# Contributing to Intel Arc Models Runner

Welcome! We are excited to collaborate with engineers, researchers, and open-source enthusiasts passionate about unlocking the full potential of Intel Arc GPUs and Intel AI Boost NPUs.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10 to 3.13
- Intel Graphics Driver (WHQL recommended)
- Intel NPU Driver (included in Intel Core Ultra platforms)

### 2. Local Setup
```bash
git clone https://github.com/techdeveloper-org/intel-arc-models-runner.git
cd intel-arc-models-runner
pip install -e ".[dev]"
```

### 3. Verify Hardware Discovery
```bash
python -m arc_runner.cli.main devices
```

---

## 🧪 Testing Guidelines

We adhere strictly to the **Test Triad** methodology (Positive, Negative, Edge-Case):
```bash
python -m pytest -v
```

Before submitting a Pull Request, ensure:
1. All unit tests pass cleanly.
2. Code strictly adheres to Level 2.2 Docstrings-Only (`12-docstrings-only.md`): all public and protected classes, functions, and methods have PEP-257 docstrings. Inline narrative comments are prohibited.
3. No breaking changes to existing Ollama or OpenAI API schemas.

---

## 🤝 Community Hardware Reporting & Benchmark Initiative

We are on a mission to build the definitive open-source empirical benchmark matrix for Intel AI hardware. While our primary testbed runs on an **Intel Core Ultra 5 125H** (Meteor Lake Arc Xe-LPG iGPU + AI Boost NPU 3720), we urgently need real-world benchmark data from the open-source community across the diverse spectrum of Intel silicon!

### 🎯 Hardware We Need Testing On
- **Arc Alchemist Discrete GPUs**: Intel® Arc™ **A770** (16GB & 8GB), **A750**, **A580**, **A380**, **A310**
- **Arc Battlemage Discrete GPUs (Xe2-HPG)**: Intel® Arc™ **B580**, **B570**
- **Core Ultra Series 2 (Lunar Lake)**: Core Ultra 7 **258V**, **268V**, **256V** (Xe2-LPG iGPU + 48 TOPS NPU 4000)
- **Core Ultra Series 2 (Arrow Lake)**: Core Ultra 9 **285K**, Core Ultra 7 **265K**, Core Ultra 5 **245K**
- **Core Ultra Series 1 (Meteor Lake)**: Core Ultra 7 **155H**, Core Ultra 9 **185H**, Core Ultra 5 **135H**
- **Intel Data Center / Workstation Silicon**: Intel® Data Center GPU **Flex 170** / **Max 1100**, Intel® Xeon® Scalable with AMX

---

### ⚡ How to Run Benchmarks & Share Results

#### Step 1: Verify Hardware Discovery
Check that OpenVINO detects your Intel GPU and/or NPU correctly:
```bash
python -m arc_runner.cli.main devices
# Or with CLI shortcut:
arc-runner devices
```

#### Step 2: Run Disk-Safe Empirical Benchmarks
You do not need hundreds of gigabytes of disk space to test models. Our sequential benchmark harness tests one model at a time, logs high-precision telemetry (TTFT, TPOT, throughput, memory), and automatically cleans up model weights ($< 5\text{ GB}$ peak storage overhead):

```bash
# Option A: Benchmark a specific model (e.g. Qwen 2.5 1.5B)
python scratch/run_live_sequential_model_benchmarks.py --model qwen-2.5-1.5b

# Option B: Benchmark an ultra-fast NPU embedding model (5W power envelope)
python scratch/run_live_sequential_model_benchmarks.py --model all-minilm-l6-v2

# Option C: Run the comprehensive 30-case Triad catalog audit
python scratch/run_full_catalog_benchmark_audit.py
```

#### Step 3: Submit Your Results
1. Locate the generated JSON telemetry at `scratch/live_physical_benchmark_results.json` (or copy the summary table printed to stdout).
2. [**Open a GitHub Issue**](https://github.com/techdeveloper-org/intel-arc-models-runner/issues/new) using the title format:
   ```text
   [Community Benchmark] <GPU or NPU Model> - <Model Slug>
   ```
3. Include your environment details:
   - **OS**: Windows 11 / Ubuntu 22.04 / 24.04 (Linux kernel version)
   - **Driver Version**: Intel Graphics Driver (e.g., 32.0.101.6127) / Level Zero version
   - **Memory**: System RAM / Dedicated VRAM
   - **Telemetry Output**: Paste the benchmark table or JSON snippet.
4. Alternatively, submit a Pull Request directly updating the empirical benchmark matrix in `README.md`!

### 🏆 Contributor Recognition & Hall of Fame
Every contributor who submits verified empirical benchmark data on physical Intel silicon will be:
- Permanently credited in the **Contributors & Acknowledgements** section of `README.md`.
- Listed as a verified hardware testing partner in release notes and documentation.

Thank you for building the future of local AI on Intel silicon with us!
