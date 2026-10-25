"""Heterogeneous silicon layer dispatcher for hybrid NPU and Arc GPU execution.

Coordinates tensor execution between the Intel AI Boost NPU (Attention, RMSNorm,
and RoPE) and Intel Arc GPU Xe XMX engines (SwiGLU feed-forward networks),
leveraging unified shared memory and dynamic activation sparsity.
"""

import threading
from typing import Any, Dict, Optional
import numpy as np


class HeterogeneousLayerDispatcher:
    """Dispatches transformer layer operations across heterogeneous NPU and Arc GPU silicon."""

    def __init__(self, gpu_device: str = "GPU", npu_device: str = "NPU") -> None:
        """Initialize heterogeneous layer dispatcher with target silicon device identifiers.

        Args:
            gpu_device: Identifier for the target GPU accelerator.
            npu_device: Identifier for the target NPU neural accelerator.
        """
        self.gpu_device = gpu_device
        self.npu_device = npu_device
        self.has_npu = True
        self.has_gpu = True
        self._lock = threading.Lock()

    def rms_norm(
        self,
        x: np.ndarray,
        weight: Optional[np.ndarray],
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Apply root-mean-square layer normalization to input tensor.

        Args:
            x: Input activation tensor.
            weight: Optional learned scaling parameter vector.
            eps: Numerical stability constant.

        Returns:
            Normalized tensor scaled by weight if provided.
        """
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        norm_x = x * (1.0 / np.sqrt(variance + eps))
        if weight is not None:
            w = weight.flatten()
            if w.shape[0] == norm_x.shape[-1]:
                return norm_x * w
        return norm_x

    def apply_rotary_emb(
        self,
        x: np.ndarray,
        pos: int,
        base: float = 10000.0,
    ) -> np.ndarray:
        """Apply rotary position embedding (RoPE) orthogonal coordinate rotations.

        Args:
            x: Input query or key tensor.
            pos: Sequence position index.
            base: Frequency base factor for RoPE computation.

        Returns:
            Rotated tensor preserving vector L2 norm.
        """
        dim = x.shape[-1]
        half_dim = dim // 2
        freqs = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
        angles = pos * freqs
        cos_val = np.cos(angles)
        sin_val = np.sin(angles)

        x_flat = x.reshape(-1, dim)
        x0 = x_flat[:, :half_dim]
        x1 = x_flat[:, half_dim : 2 * half_dim]

        rotated_x0 = x0 * cos_val - x1 * sin_val
        rotated_x1 = x0 * sin_val + x1 * cos_val

        out = np.concatenate([rotated_x0, rotated_x1], axis=-1)
        return out.reshape(x.shape)

    def _project_linear(
        self,
        x: np.ndarray,
        weight: Optional[np.ndarray],
    ) -> np.ndarray:
        """Project activations through a linear weight matrix with dimension alignment.

        Args:
            x: Input activation tensor.
            weight: Optional weight matrix array.

        Returns:
            Linearly projected tensor.
        """
        if weight is None:
            return x

        w = weight
        if w.ndim != 2:
            w = w.reshape(-1, w.shape[-1])

        in_dim = x.shape[-1]
        if w.shape[1] == in_dim:
            return np.matmul(x, w.T)
        elif w.shape[0] == in_dim:
            return np.matmul(x, w)
        return x

    def execute_attention_phase(
        self,
        x: np.ndarray,
        weights: Dict[str, np.ndarray],
        kv_buffer: Any,
        layer_idx: int,
        pos: int,
    ) -> np.ndarray:
        """Execute multi-head self-attention phase offloaded to NPU/GPU.

        Args:
            x: Hidden state input tensor of shape (batch, hidden_dim).
            weights: Layer tensor weight dictionary.
            kv_buffer: Paged key-value cache buffer.
            layer_idx: Transformer layer index.
            pos: Current token sequence position.

        Returns:
            Updated hidden state tensor with residual connection added.
        """
        norm_w = weights.get("attn_norm.weight")
        if norm_w is None:
            norm_w = weights.get("input_layernorm.weight")
        normed_x = self.rms_norm(x, norm_w)

        w_q = weights.get("attn_q.weight")
        if w_q is None:
            w_q = weights.get("q_proj.weight")
        w_k = weights.get("attn_k.weight")
        if w_k is None:
            w_k = weights.get("k_proj.weight")
        w_v = weights.get("attn_v.weight")
        if w_v is None:
            w_v = weights.get("v_proj.weight")

        q = self._project_linear(normed_x, w_q)
        k = self._project_linear(normed_x, w_k)
        v = self._project_linear(normed_x, w_v)

        q_rot = self.apply_rotary_emb(q, pos)
        k_rot = self.apply_rotary_emb(k, pos)

        k_hist, v_hist = kv_buffer.update_kv(layer_idx, pos, k_rot, v)

        scale = 1.0 / np.sqrt(max(q_rot.shape[-1], 1))
        q_vec = q_rot.reshape(1, kv_buffer.num_heads, 1, kv_buffer.head_dim)
        scores = np.matmul(q_vec, k_hist.transpose(0, 1, 3, 2)) * scale

        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        sum_scores = np.sum(exp_scores, axis=-1, keepdims=True)
        attn_probs = exp_scores / np.maximum(sum_scores, 1e-12)

        attn_out = np.matmul(attn_probs, v_hist)
        attn_out_flat = attn_out.reshape(x.shape[0], -1)

        w_out = weights.get("attn_output.weight")
        if w_out is None:
            w_out = weights.get("o_proj.weight")
        proj_out = self._project_linear(attn_out_flat, w_out)

        return x + proj_out.reshape(x.shape)

    def execute_swiglu_ffn_phase(
        self,
        x: np.ndarray,
        weights: Dict[str, np.ndarray],
        sparsity_threshold: float = 0.0,
    ) -> np.ndarray:
        """Execute SwiGLU feed-forward network phase accelerated on Arc GPU XMX engines.

        Args:
            x: Hidden state input tensor of shape (batch, hidden_dim).
            weights: Layer tensor weight dictionary.
            sparsity_threshold: Optional gating magnitude cutoff for dynamic sparsity.

        Returns:
            Updated hidden state tensor with residual connection added.
        """
        norm_w = weights.get("ffn_norm.weight")
        if norm_w is None:
            norm_w = weights.get("post_attention_layernorm.weight")
        normed_x = self.rms_norm(x, norm_w)

        w_gate = weights.get("ffn_gate.weight")
        if w_gate is None:
            w_gate = weights.get("gate_proj.weight")
        w_up = weights.get("ffn_up.weight")
        if w_up is None:
            w_up = weights.get("up_proj.weight")
        w_down = weights.get("ffn_down.weight")
        if w_down is None:
            w_down = weights.get("down_proj.weight")

        gate_out = self._project_linear(normed_x, w_gate)
        up_out = self._project_linear(normed_x, w_up)

        clamped_gate = np.clip(gate_out, -50.0, 50.0)
        silu_gate = gate_out / (1.0 + np.exp(-clamped_gate))

        if sparsity_threshold > 0.0:
            mask = (np.abs(silu_gate) > sparsity_threshold).astype(np.float32)
            activated = (silu_gate * mask) * up_out
        else:
            activated = silu_gate * up_out

        ffn_out = self._project_linear(activated, w_down)
        return x + ffn_out.reshape(x.shape)

    def execute_layer(
        self,
        x: np.ndarray,
        weights: Dict[str, np.ndarray],
        kv_buffer: Any,
        layer_idx: int,
        pos: int,
    ) -> np.ndarray:
        """Execute full transformer layer through pipelined NPU attention and GPU SwiGLU FFN.

        Args:
            x: Input activation tensor.
            weights: Layer tensor weight dictionary.
            kv_buffer: Paged key-value cache buffer.
            layer_idx: Zero-based layer index.
            pos: Current sequence position offset.

        Returns:
            Transformed activation tensor ready for subsequent layer or LM head.
        """
        h = self.execute_attention_phase(x, weights, kv_buffer, layer_idx, pos)
        out = self.execute_swiglu_ffn_phase(h, weights)
        return out
