# Persistent Silicon Telemetry & Automated Bottleneck Diagnostic Guide

The `arc-runner` telemetry subsystem provides closed-loop hardware observability and automated performance diagnosis across Intel Arc GPUs (Xe-HPG, Xe-LPG) and Intel NPUs.

---

## 1. Running Telemetry & Benchmarking

```bash
# Execute benchmark and record high-resolution telemetry
arc-runner benchmark qwen2.5-3b-int4 --device GPU --iterations 10 --tokens 64

# Automatically diagnose hardware bottlenecks from a generated report
arc-runner benchmark --analyze benchmarks/reports/silicon_benchmark_qwen2.5-3b_latest.json
```

---

## 2. Telemetry Artifacts Generated

Every benchmark execution produces three synchronized, persistent artifacts under `benchmarks/reports/`:

1. **`*.json` (Machine-Readable Payload)**:
   Contains full execution telemetry including TTFT, min/max/median/mean/P95/P99 token generation rates, jitter standard deviation ($\sigma$), effective memory bandwidth (GB/s), bus saturation percentage, P-core residency %, and GC pause counts.
2. **`*.md` (Executive Markdown Report)**:
   Formatted markdown summary complete with performance metric tables and Roofline bus saturation positioning.
3. **`*.log` (High-Resolution Event Log)**:
   Microsecond time-stamped chronological audit log tracing prefill, per-token latency steps, and core affinity state.

---

## 3. Bottleneck Diagnostic Heuristics & Rules

`SiliconBottleneckAnalyzer` evaluates telemetry data against five concrete hardware rules:

| Rule | Detection Condition | Hardware Root Cause | Automated Tuning Prescription |
| :--- | :--- | :--- | :--- |
| **Roofline Saturation** | Bus saturation $> 70\%$ and intensity $< 1.2\text{ FLOPs/byte}$ | Memory-bandwidth bound on Intel Arc 112 GB/s bus | Quantize to Symmetric INT4 Group 128 and use Level Zero USM page-locked allocations. |
| **Thread Director Core Hopping** | $\sigma > 1.5\text{ tok/s}$ and P-Core residency $< 98\%$ | Windows scheduler moving feeder thread to LP-E cores | Activate `ThreadAffinityGuard` with mask `0x00FF` and `THREAD_PRIORITY_HIGHEST`. |
| **Cyclic GC Halts** | GC freeze count $> 0$ and peak latency $> 3 \times$ median | Python cyclic garbage collector sweeping mid-generation | Wrap generation loop with `GCBurstGate` (`gc.disable()` during active decode). |
| **Thermal / PL1 Throttling** | GPU frequency drops $> 15\%$ after initial burst | Intel SoC dropping from PL2 (64W) to PL1 (28W) | Lock sustained GPU equilibrium clock frequency via Level Zero Sysman to 1.85 GHz. |
| **Quantization Noise Hotspot** | Layer SQNR $< 20.0\text{ dB}$ (INT4) or $< 34.0\text{ dB}$ (INT8) | Non-native k-quant scale distortion | Retain sensitive linear projection layers in FP16 or switch to INT8. |

---

## 4. Sample Diagnostic Output

```
================================================================================
SILICON BOTTLENECK DIAGNOSIS: qwen2.5:3b on GPU
Status: OPTIMAL | Performance Health Score: 98.5 / 100
Roofline Classification: MEMORY_BANDWIDTH_BOUND
================================================================================
Detected 1 Bottlenecks:
  1. [!] Memory bus saturation reached 74.2% (83.1 GB/s of 112 GB/s peak).

Actionable System Tuning Prescriptions:
  1. [->] Weights are memory-bus bound. Apply Symmetric INT4 Group 128 quantization
         or enable static Level Zero USM page-locked buffers to maximize DMA bus saturation.
================================================================================
```
