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

## 🤝 Community Hardware Reporting

Have an Intel Arc dGPU (A770, A750, A580, A380, B580) or an Intel Core Ultra machine (Meteor Lake, Lunar Lake, Arrow Lake)?

Submit your hardware profile by running:
```bash
arc-runner devices --json
```
and posting the output in a new issue titled `[Hardware Profile] <Your GPU / NPU Model>`.

Thank you for building the future of local AI on Intel silicon with us!
