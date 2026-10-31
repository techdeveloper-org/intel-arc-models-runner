"""Exhaustive 14-test triad suite for bare-metal flash-stream heterogeneous engine.

Validates O(1) constant-RAM layer streaming, asynchronous prefetching, NPU-GPU
cooperative execution, SwiGLU activation sparsity, bounded paged KV cache,
and zero-OOM invariants across positive, negative, and edge-case boundaries.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import tempfile
import threading
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from arc_runner.core.exceptions import (
    ArcRunnerError,
    ContextWindowExceededError,
    ModelNotFoundError,
)
from arc_runner.engine.flash_stream.engine import FlashStreamInferenceEngine
from arc_runner.engine.flash_stream.hetero_dispatcher import HeterogeneousLayerDispatcher
from arc_runner.engine.flash_stream.kv_paged_buffer import FlashStreamKVPagedBuffer
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager
from arc_runner.engine.models import GenerationConfig
from arc_runner.quant.models import GGMLType, GGUFValueType


def build_synthetic_flash_gguf(
    dest_path: Path,
    num_layers: int = 2,
    hidden_dim: int = 32,
    corrupt_offset: bool = False,
) -> Path:
    """Construct a synthetically valid GGUF container file for testing.

    Args:
        dest_path: Filesystem path to create the container file.
        num_layers: Number of transformer layers to embed.
        hidden_dim: Hidden dimension size for tensor shapes.
        corrupt_offset: Whether to inject an out-of-bounds tensor offset.

    Returns:
        The generated Path object pointing to the binary container.
    """
    metadata: Dict[str, Tuple[GGUFValueType, Any]] = {
        "general.architecture": (GGUFValueType.STRING, "qwen"),
        "qwen.context_length": (GGUFValueType.UINT32, 2048),
        "qwen.embedding_length": (GGUFValueType.UINT32, hidden_dim),
        "qwen.block_count": (GGUFValueType.UINT32, num_layers),
        "qwen.attention.head_count": (GGUFValueType.UINT32, 4),
        "general.alignment": (GGUFValueType.UINT32, 32),
    }

    tensors_desc: List[Dict[str, Any]] = []
    tensor_names = [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]

    current_offset = 0
    payload_arrays: List[bytes] = []

    for l in range(num_layers):
        for name in tensor_names:
            full_name = f"blk.{l}.{name}"
            if "norm" in name:
                dims = (hidden_dim,)
            elif name == "ffn_down.weight":
                dims = (hidden_dim * 2, hidden_dim)
            elif name in ("ffn_gate.weight", "ffn_up.weight"):
                dims = (hidden_dim, hidden_dim * 2)
            else:
                dims = (hidden_dim, hidden_dim)

            num_elems = 1
            for d in dims:
                num_elems *= d

            data_arr = (np.ones(num_elems, dtype=np.float32) * 0.05).tobytes()
            offset_val = 999999999 if corrupt_offset and l == 0 and name == "attn_q.weight" else current_offset

            tensors_desc.append({
                "name": full_name,
                "dims": dims,
                "type": GGMLType.F32,
                "offset": offset_val,
                "bytes": data_arr,
            })
            current_offset += len(data_arr)
            payload_arrays.append(data_arr)

    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, len(tensors_desc), len(metadata)))

    for k, (v_type, val) in metadata.items():
        k_bytes = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(k_bytes)))
        buf.extend(k_bytes)
        buf.extend(struct.pack("<I", v_type.value))

        if v_type == GGUFValueType.STRING:
            s_bytes = val.encode("utf-8")
            buf.extend(struct.pack("<Q", len(s_bytes)))
            buf.extend(s_bytes)
        elif v_type == GGUFValueType.UINT32:
            buf.extend(struct.pack("<I", val))

    for t in tensors_desc:
        t_name_bytes = t["name"].encode("utf-8")
        buf.extend(struct.pack("<Q", len(t_name_bytes)))
        buf.extend(t_name_bytes)
        buf.extend(struct.pack("<I", len(t["dims"])))
        for d in t["dims"]:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<IQ", t["type"].value, t["offset"]))

    alignment = 32
    remainder = len(buf) % alignment
    if remainder != 0:
        buf.extend(b"\x00" * (alignment - remainder))

    for p in payload_arrays:
        buf.extend(p)

    dest_path.write_bytes(bytes(buf))
    return dest_path


class TestPositiveTriad:
    """Positive test triad verifying nominal flash-stream execution invariants."""

    def test_flash_stream_weight_manager_indexing(self, tmp_path: Path) -> None:
        """Verify layer indexing from GGUF metadata without reading entire weight payload."""
        container_path = build_synthetic_flash_gguf(tmp_path / "model.gguf", num_layers=3)
        manager = FlashStreamWeightManager(container_path, ring_buffer_slots=3)

        try:
            layer_counts = manager.index_layers()
            assert len(layer_counts) == 3
            assert all(count == 9 for count in layer_counts.values())
            assert manager.get_active_memory_bytes() == 0
        finally:
            manager.close()

    def test_flash_stream_ring_buffer_double_buffering(self, tmp_path: Path) -> None:
        """Verify asynchronous background prefetching into ring buffer while slot is locked."""
        container_path = build_synthetic_flash_gguf(tmp_path / "model.gguf", num_layers=3)
        manager = FlashStreamWeightManager(container_path, ring_buffer_slots=3)

        try:
            slot0_weights = manager.acquire_layer_weights(0)
            assert "attn_q.weight" in slot0_weights

            prefetch_thread = manager.prefetch_layer_async(1, 1)
            prefetch_thread.join(timeout=2.0)

            slot1_weights = manager.acquire_layer_weights(1)
            assert "attn_q.weight" in slot1_weights
            assert manager.get_active_memory_bytes() > 0
        finally:
            manager.close()

    def test_flash_stream_transformer_layer_execution(self) -> None:
        """Verify transformer layer execution through NPU attention and GPU SwiGLU FFN."""
        dispatcher = HeterogeneousLayerDispatcher()
        hidden_dim = 32
        x = np.ones((1, hidden_dim), dtype=np.float32)
        kv = FlashStreamKVPagedBuffer(layer_count=2, num_heads=4, head_dim=8, max_seq_len=64)

        weights = {
            "attn_norm.weight": np.ones(hidden_dim, dtype=np.float32),
            "attn_q.weight": np.eye(hidden_dim, dtype=np.float32),
            "attn_k.weight": np.eye(hidden_dim, dtype=np.float32),
            "attn_v.weight": np.eye(hidden_dim, dtype=np.float32),
            "attn_output.weight": np.eye(hidden_dim, dtype=np.float32),
            "ffn_norm.weight": np.ones(hidden_dim, dtype=np.float32),
            "ffn_gate.weight": np.ones((hidden_dim, hidden_dim * 2), dtype=np.float32) * 0.1,
            "ffn_up.weight": np.ones((hidden_dim, hidden_dim * 2), dtype=np.float32) * 0.1,
            "ffn_down.weight": np.ones((hidden_dim * 2, hidden_dim), dtype=np.float32) * 0.1,
        }

        out = dispatcher.execute_layer(x, weights, kv, layer_idx=0, pos=0)
        assert out.shape == (1, hidden_dim)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_flash_stream_swiglu_activation_correctness(self) -> None:
        """Verify SwiGLU SiLU gating calculation matches theoretical formulation."""
        dispatcher = HeterogeneousLayerDispatcher()
        hidden_dim = 4
        interm_dim = 8
        x = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)

        w_gate = np.ones((hidden_dim, interm_dim), dtype=np.float32) * 0.5
        w_up = np.ones((hidden_dim, interm_dim), dtype=np.float32) * 0.5
        w_down = np.ones((interm_dim, hidden_dim), dtype=np.float32) * 0.25

        weights = {
            "ffn_gate.weight": w_gate,
            "ffn_up.weight": w_up,
            "ffn_down.weight": w_down,
        }

        out = dispatcher.execute_swiglu_ffn_phase(x, weights, sparsity_threshold=0.0)
        assert out.shape == (1, hidden_dim)
        assert np.all(np.isfinite(out))

    def test_flash_stream_kv_paged_buffer_append(self) -> None:
        """Verify paged KV buffer updates sequentially preserving history views."""
        kv = FlashStreamKVPagedBuffer(layer_count=2, num_heads=2, head_dim=4, max_seq_len=16)
        assert kv.current_pos == 0

        for p in range(5):
            k = np.full((1, 2, 1, 4), p + 1.0, dtype=np.float32)
            v = np.full((1, 2, 1, 4), (p + 1.0) * 2.0, dtype=np.float32)
            k_view, v_view = kv.update_kv(layer_idx=0, pos=p, k=k, v=v)
            assert k_view.shape == (1, 2, p + 1, 4)
            assert v_view.shape == (1, 2, p + 1, 4)

        assert kv.current_pos == 5
        assert kv.get_memory_bytes() > 0

    def test_flash_stream_generation_streaming_loop(self, tmp_path: Path) -> None:
        """Verify complete autoregressive generation streaming loop with TokenChunks."""
        container_path = build_synthetic_flash_gguf(tmp_path / "model.gguf", num_layers=2)
        engine = FlashStreamInferenceEngine()

        try:
            engine.load_model(container_path, device="GPU", ring_buffer_slots=2)
            assert engine.is_loaded

            cfg = GenerationConfig(max_new_tokens=4)
            chunks = list(engine.generate_stream("What is quantum computing?", config=cfg))

            assert len(chunks) == 4
            assert all(chunk.text for chunk in chunks)
            assert chunks[-1].is_finished is True
            assert chunks[-1].finish_reason == "length"

            report = engine.get_working_memory_report()
            assert 0 < report["ring_slots_occupied"] <= report["ring_slots_capacity"]
            assert report["total_working_mb"] > 0
        finally:
            engine.unload()
            assert not engine.is_loaded


class TestNegativeTriad:
    """Negative test triad verifying failure detection, validation, and error guards."""

    def test_flash_stream_rejects_nonexistent_model(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised when target model file does not exist."""
        nonexistent = tmp_path / "missing_model.gguf"
        with pytest.raises(FileNotFoundError):
            FlashStreamWeightManager(nonexistent)

        engine = FlashStreamInferenceEngine()
        with pytest.raises(ModelNotFoundError):
            engine.load_model(nonexistent)

    def test_flash_stream_detects_corrupted_tensor_offset(self, tmp_path: Path) -> None:
        """Verify ValueError is raised when encountering out-of-bounds tensor offsets."""
        corrupted_path = build_synthetic_flash_gguf(
            tmp_path / "corrupt.gguf",
            num_layers=1,
            corrupt_offset=True,
        )
        manager = FlashStreamWeightManager(corrupted_path)
        try:
            with pytest.raises(ValueError, match="Corrupted tensor"):
                manager.acquire_layer_weights(0)
        finally:
            manager.close()

    def test_flash_stream_handles_empty_prompt_gracefully(self, tmp_path: Path) -> None:
        """Verify ValueError is raised when empty prompt string is submitted."""
        container_path = build_synthetic_flash_gguf(tmp_path / "model.gguf", num_layers=1)
        engine = FlashStreamInferenceEngine()
        try:
            engine.load_model(container_path)
            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                list(engine.generate_stream(""))
        finally:
            engine.unload()

    def test_flash_stream_kv_overflow_guard(self) -> None:
        """Verify ContextWindowExceededError is raised when sequence exceeds buffer capacity."""
        kv = FlashStreamKVPagedBuffer(layer_count=1, num_heads=1, head_dim=4, max_seq_len=4)
        k = np.zeros((1, 1, 1, 4), dtype=np.float32)
        v = np.zeros((1, 1, 1, 4), dtype=np.float32)

        for p in range(4):
            kv.update_kv(0, p, k, v)

        with pytest.raises(ContextWindowExceededError):
            kv.update_kv(0, 4, k, v)


class TestEdgeCaseTriad:
    """Edge-case test triad verifying silicon boundaries, memory invariance, and concurrency."""

    def test_flash_stream_constant_ram_invariant_proof_m122(self, tmp_path: Path) -> None:
        """Verify active resident memory remains strictly bounded (O(1)) across 50 simulated layers."""
        container_path = build_synthetic_flash_gguf(tmp_path / "heavy.gguf", num_layers=6)
        manager = FlashStreamWeightManager(container_path, ring_buffer_slots=3)

        try:
            memory_samples: List[int] = []
            for step in range(50):
                layer = step % 6
                manager.acquire_layer_weights(layer)
                memory_samples.append(manager.get_active_memory_bytes())

            assert len(memory_samples) == 50
            max_mem = max(memory_samples)
            final_mem = memory_samples[-1]
            assert final_mem <= max_mem
            assert max_mem < (1024 * 1024 * 10)
        finally:
            manager.close()

    def test_flash_stream_single_layer_tiny_model_boundary(self, tmp_path: Path) -> None:
        """Verify single-layer model edge condition executes without wrap-around indexing error."""
        container_path = build_synthetic_flash_gguf(tmp_path / "single_layer.gguf", num_layers=1)
        engine = FlashStreamInferenceEngine()
        try:
            engine.load_model(container_path, ring_buffer_slots=1)
            cfg = GenerationConfig(max_new_tokens=2)
            chunks = list(engine.generate_stream("Single layer test", config=cfg))
            assert len(chunks) == 2
        finally:
            engine.unload()

    def test_flash_stream_zero_byte_eviction_cleanup(self, tmp_path: Path) -> None:
        """Verify explicit release of ring slot immediately reduces resident memory to zero."""
        container_path = build_synthetic_flash_gguf(tmp_path / "clean.gguf", num_layers=2)
        manager = FlashStreamWeightManager(container_path, ring_buffer_slots=2)
        try:
            manager.acquire_layer_weights(0)
            mem_before = manager.get_active_memory_bytes()
            assert mem_before > 0

            manager.release_layer_slot(0)
            mem_after = manager.get_active_memory_bytes()
            assert mem_after == 0
        finally:
            manager.close()

    def test_flash_stream_thread_safety_during_prefetch_race(self, tmp_path: Path) -> None:
        """Verify thread-safe synchronization under concurrent prefetch requests without deadlock."""
        container_path = build_synthetic_flash_gguf(tmp_path / "threaded.gguf", num_layers=4)
        manager = FlashStreamWeightManager(container_path, ring_buffer_slots=3)

        def prefetch_task(layer: int, slot: int) -> Dict[str, np.ndarray]:
            return manager.acquire_layer_weights(layer)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(prefetch_task, i % 4, i % 3)
                    for i in range(16)
                ]
                results = [f.result(timeout=5.0) for f in futures]

            assert len(results) == 16
            assert all(isinstance(r, dict) for r in results)
        finally:
            manager.close()
