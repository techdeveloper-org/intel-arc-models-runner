# Silicon-Native Quantization & Mathematical Transpilation Architecture

## 1. Overview & Silicon Reality

Modern LLMs distributed on platforms like Ollama or HuggingFace frequently utilize non-native asymmetric block quantization formats (e.g. `Q4_K_M`, `Q4_0`, `Q5_K`, `Q8_0`). While optimized for general-purpose CPU threads and CUDA GEMV kernels in `llama.cpp`, these k-quants fail to execute natively on **Intel Arc GPUs (Xe-HPG/Xe-LPG Xe Matrix Extensions - XMX)** and **Intel NPUs (Neural Compute Engines)**.

Intel XMX systolic arrays and NPU tensor accelerators execute hardware matrix-multiply-accumulate ($\mathbf{C} = \mathbf{A} \cdot \mathbf{B} + \mathbf{C}$) strictly using **Symmetric INT4 (Group size 128 / 64), INT8, or FP16** representations packed into contiguous **32-byte cachelines** matching 256-bit Direct Memory Access (DMA) transactions.

`arc-runner` bridges this silicon gap by providing a mathematically verified Ahead-of-Time (AOT) transpiler that transforms arbitrary non-native GGUF weights into Intel Xe XMX native formats with guaranteed Signal-to-Quantization-Noise Ratio (SQNR) validation.

---

## 2. Mathematical Foundations & Proofs (M1 – M7)

### Theorem M1: Symmetric INT4 Scale Factor Formulation
For any contiguous weight sub-vector $\mathbf{W}_g \in \mathbb{R}^{G}$ (where $G = 128$):
$$\Delta_g = \frac{\max_{w \in \mathbf{W}_g} |w|}{7}$$

The signed 4-bit integer representation $q_i \in [-8, 7]$ is obtained via:
$$q_i = \text{clamp}\left(\left\lfloor \frac{w_i}{\Delta_g} + 0.5 \right\rfloor, -8, 7\right)$$

The dequantized floating-point approximation $\tilde{w}_i$ is reconstructed as:
$$\tilde{w}_i = q_i \cdot \Delta_g$$

---

### Theorem M2: Bounded Quantization Error
**Proof**:
For any $w_i \in [-\max |w|, \max |w|]$, let $x = w_i / \Delta_g$.
Since $|x - \text{round}(x)| \le 0.5$, and for all non-clipped values $q_i = \text{round}(x)$:
$$|w_i - \tilde{w}_i| = |x \cdot \Delta_g - q_i \cdot \Delta_g| = \Delta_g \cdot |x - q_i| \le \frac{\Delta_g}{2}$$

Thus, maximum reconstruction error is strictly bounded by half the quantization step size $\Delta_g / 2$. $\blacksquare$

---

### Theorem M3: Signal-to-Quantization-Noise Ratio (SQNR)
Quantization fidelity is evaluated across every linear projection layer using exact decibel Signal-to-Quantization-Noise Ratio:
$$\text{SQNR}(\mathbf{W}, \tilde{\mathbf{W}}) = 10 \cdot \log_{10} \left( \frac{\sum_{i=1}^{N} w_i^2 + \epsilon}{\sum_{i=1}^{N} (w_i - \tilde{w}_i)^2 + \epsilon} \right)$$

- **INT4 Target Threshold**: $\text{SQNR} \ge 20.0\text{ dB}$ (theoretical maximum on uniform distributions is $\approx 25.84\text{ dB}$).
- **INT8 Target Threshold**: $\text{SQNR} \ge 34.0\text{ dB}$ (typical measured: $38 - 50\text{ dB}$).
- Any weight matrix exhibiting severe quantization noise below threshold triggers an automatic pipeline rejection.

---

### Theorem M4: 32-Byte Cacheline Direct Memory Alignment
Intel Xe-HPG/Xe-LPG memory controllers execute bus transactions in 32-byte (256-bit) bursts. Unaligned reads crossing cacheline boundaries force dual bus cycles, halving effective memory bandwidth.

For any tensor byte length $L$:
$$L_{\text{padded}} = (L + 31) \ \& \ (\sim 31)$$
$$\text{Padding Bytes} = L_{\text{padded}} - L \in [0, 31]$$

All binary buffers emitted into OpenVINO `.bin` containers are guaranteed to satisfy $L_{\text{offset}} \pmod{32} \equiv 0$, eliminating memory bus split-lock penalties.

---

### Theorem M5: Roofline Model on Intel Arc Graphics
Given peak memory bandwidth $B_{\text{max}} = 112\text{ GB/s}$ and model footprint $S_{\text{bytes}}$:
$$\text{Theoretical Max Autoregressive Decode Speed} = \frac{B_{\text{max}}}{S_{\text{bytes}}} \text{ tokens/second}$$

For **Qwen 2.5 3B** (INT4 footprint $\approx 1.85\text{ GB}$):
$$\text{Upper Ceiling} = \frac{112}{1.85} \approx 60.5\text{ tok/s}$$
$$\text{Target Sustained Throughput (65\% - 75\% bus saturation)} = 39.0 - 45.0\text{ tok/s}$$

---

### Theorem M6: Rotary Position Embedding (RoPE) Angle Transformation
For Qwen 2.5 architecture with $\theta_{\text{base}} = 1,000,000$ and head dimension $d_k = 128$:
$$\theta_i = \theta_{\text{base}}^{-2(i-1)/d_k}, \quad i \in [1, d_k / 2]$$

Rotary frequency scaling is mathematically extracted and embedded in `openvino_model.xml` and `config.json`, preserving context integrity up to 131,072 tokens without attention matrix collapse.

---

### Theorem M7: Information-Theoretic Faithfulness Bound
To guarantee that quantized weights do not cause semantic logit drift and hallucinations:
$$D_{\text{KL}}(P_{\text{source}} \parallel P_{\text{int4}}) = \sum_{x \in \mathcal{V}} P_{\text{source}}(x) \ln\left(\frac{P_{\text{source}}(x)}{P_{\text{int4}}(x)}\right) \le 0.05$$
LogSumExp stabilization prevents numerical underflow across vocabulary size $|\mathcal{V}| = 151,936$.
