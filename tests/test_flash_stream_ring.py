"""Slice S-C tests: direct-mapped ring slots, single-load guarantee, safe shutdown, and view lifetime."""

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager, SlotState
from arc_runner.native import bridge
from arc_runner.native.bridge import NativeMemoryEngine, dtype_for_tensor_type, view_shape_for_dims
from arc_runner.quant.models import GGMLType, GGUFTensorInfo
from tests.test_flash_stream_triad import build_synthetic_flash_gguf


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Build a 6-layer synthetic GGUF container."""
    return build_synthetic_flash_gguf(tmp_path / "ring.gguf", num_layers=6, hidden_dim=32)


class _LoadProbe:
    """Wraps a manager's layer loader to count loads and in-flight concurrency per slot."""

    def __init__(self, manager: FlashStreamWeightManager, delay: float = 0.0) -> None:
        self.manager = manager
        self.delay = delay
        self.loads: List[int] = []
        self.max_in_flight_per_slot: Dict[int, int] = {}
        self._in_flight: Dict[int, int] = {}
        self._lock = threading.Lock()
        self._original = manager._load_layer_arrays
        manager._load_layer_arrays = self._probe  # type: ignore[method-assign]

    def _probe(self, layer_idx: int) -> Dict[str, np.ndarray]:
        slot = self.manager._slot_index_for(layer_idx)
        with self._lock:
            self.loads.append(layer_idx)
            self._in_flight[slot] = self._in_flight.get(slot, 0) + 1
            self.max_in_flight_per_slot[slot] = max(self.max_in_flight_per_slot.get(slot, 0), self._in_flight[slot])
        try:
            time.sleep(self.delay)
            return self._original(layer_idx)
        finally:
            with self._lock:
                self._in_flight[slot] -= 1


def test_slot_index_for_layer_is_direct_mapped_modulo_slots(model_path: Path) -> None:
    """X1: layer L always maps to slot L % N."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        assert [manager._slot_index_for(layer) for layer in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_acquire_concurrent_same_layer_loads_once(model_path: Path) -> None:
    """C6: eight threads asking for one layer trigger exactly one load and share the result."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        probe = _LoadProbe(manager, delay=0.05)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: manager.acquire_layer_weights(2), range(8)))
        assert probe.loads == [2]
        assert all(result is results[0] for result in results)


def test_acquire_stress_never_loads_one_slot_concurrently_and_never_deadlocks(model_path: Path) -> None:
    """C6: 8 threads x 150 acquires over 6 layers and 3 slots finish with at most one in-flight load per slot."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        probe = _LoadProbe(manager, delay=0.0005)

        def worker(seed: int) -> int:
            done = 0
            for step in range(150):
                weights = manager.acquire_layer_weights((seed + step) % 6)
                assert weights
                done += 1
            return done

        with ThreadPoolExecutor(max_workers=8) as pool:
            totals = [future.result(timeout=60) for future in [pool.submit(worker, seed) for seed in range(8)]]
        assert totals == [150] * 8
        assert max(probe.max_in_flight_per_slot.values()) == 1


def test_cyclic_layer_scan_over_resident_layers_hits_after_first_pass(model_path: Path) -> None:
    """X1: scanning layers 0..2 repeatedly in 3 slots loads each layer once (LRU would also, a 4-layer scan would not)."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        probe = _LoadProbe(manager)
        for _ in range(5):
            for layer in range(3):
                manager.acquire_layer_weights(layer)
        assert probe.loads == [0, 1, 2]


def test_acquire_colliding_layers_replace_each_other_without_deadlock(model_path: Path) -> None:
    """Layers 0 and 3 share slot 0; alternating them reloads each time and never blocks (no pin counts)."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        probe = _LoadProbe(manager)
        first = manager.acquire_layer_weights(0)
        manager.acquire_layer_weights(3)
        again = manager.acquire_layer_weights(0)
        assert probe.loads == [0, 3, 0]
        assert first.keys() == again.keys()


def test_failed_load_resets_slot_and_next_acquire_retries(model_path: Path) -> None:
    """A failing load raises to its loader only, leaves the slot EMPTY, and a retry succeeds."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        original = manager._load_layer_arrays
        calls = {"count": 0}

        def flaky(layer_idx: int) -> Dict[str, np.ndarray]:
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("simulated read failure")
            return original(layer_idx)

        manager._load_layer_arrays = flaky  # type: ignore[method-assign]
        with pytest.raises(ValueError):
            manager.acquire_layer_weights(1)
        assert manager._slots[1].state is SlotState.EMPTY
        assert manager.acquire_layer_weights(1)
        assert manager._slots[1].state is SlotState.READY


def test_close_waits_for_prefetch_and_later_acquire_raises(model_path: Path) -> None:
    """C6: close() joins prefetch threads without crashing them, then acquire raises ArcRunnerError."""
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3)
    _LoadProbe(manager, delay=0.2)
    thread = manager.prefetch_layer_async(1, 1)
    started = time.monotonic()
    manager.close()
    assert time.monotonic() - started < 5.0
    assert not thread.is_alive()
    with pytest.raises(ArcRunnerError):
        manager.acquire_layer_weights(0)
    manager.close()


def test_acquire_wait_timeout_raises_timeout_error(model_path: Path) -> None:
    """A load that never finishes makes waiters raise TimeoutError instead of hanging forever."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        manager._slot_wait_timeout_s = 0.2
        release = threading.Event()
        original = manager._load_layer_arrays

        def blocked(layer_idx: int) -> Dict[str, np.ndarray]:
            release.wait(5.0)
            return original(layer_idx)

        manager._load_layer_arrays = blocked  # type: ignore[method-assign]
        loader = threading.Thread(target=manager.acquire_layer_weights, args=(0,), daemon=True)
        loader.start()
        time.sleep(0.05)
        with pytest.raises(TimeoutError):
            manager.acquire_layer_weights(0)
        release.set()
        loader.join(5.0)


def test_release_layer_slot_clears_resident_layer(model_path: Path) -> None:
    """Releasing a slot drops its arrays and marks it EMPTY."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        manager.acquire_layer_weights(1)
        assert manager.get_occupied_slot_count() == 1
        manager.release_layer_slot(1)
        assert manager.get_occupied_slot_count() == 0
        assert manager.get_active_memory_bytes() == 0


def test_get_resident_memory_bytes_reports_positive_value_on_linux(model_path: Path) -> None:
    """X2: process resident memory is reported separately from mapped bytes."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=3) as manager:
        resident = manager.get_resident_memory_bytes()
        assert resident is None or resident > 0


def test_dtype_for_tensor_type_maps_plain_and_quantized_types() -> None:
    """C4: plain types keep their dtype, bf16 is raw uint16 bits, quantized types are raw bytes."""
    assert dtype_for_tensor_type(GGMLType.F32) == np.float32
    assert dtype_for_tensor_type(GGMLType.F16) == np.float16
    assert dtype_for_tensor_type(GGMLType.BF16) == np.uint16
    assert dtype_for_tensor_type(GGMLType.I8) == np.int8
    for quantized in (GGMLType.Q4_0, GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.IQ4_XS):
        assert dtype_for_tensor_type(quantized) == np.uint8


def test_view_shape_for_dims_reverses_gguf_order() -> None:
    """X3: GGUF ne[0] is the fastest dimension, so the row-major shape is reversed."""
    assert view_shape_for_dims((4, 3), 12) == (3, 4)
    assert view_shape_for_dims((4, 3), 11) is None
    assert view_shape_for_dims((), 4) is None


def test_two_dimensional_tensor_uses_reversed_dimensions_in_both_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """X3: native and Python paths agree on shape for a 2-D tensor."""
    model = build_synthetic_flash_gguf(tmp_path / "two_d.gguf", num_layers=1, hidden_dim=32)
    shapes: Dict[bool, Dict[str, tuple]] = {}
    for disable in (False, True):
        if disable:
            monkeypatch.setenv("ARC_NATIVE_DISABLE", "1")
        else:
            monkeypatch.delenv("ARC_NATIVE_DISABLE", raising=False)
        with FlashStreamWeightManager(model, ring_buffer_slots=1) as manager:
            shapes[disable] = {name: array.shape for name, array in manager.acquire_layer_weights(0).items()}
    assert shapes[False] == shapes[True]


@pytest.mark.parametrize(
    "tensor_type,elements,expected",
    [
        (GGMLType.F32, 64, 256),
        (GGMLType.F16, 64, 128),
        (GGMLType.Q4_0, 64, 36),
        (GGMLType.Q8_0, 64, 68),
        (GGMLType.Q4_K, 512, 288),
        (GGMLType.Q6_K, 256, 210),
    ],
)
def test_calculate_tensor_byte_size_known_types(model_path: Path, tensor_type: GGMLType, elements: int, expected: int) -> None:
    """Block layouts follow the GGML block sizes."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=1) as manager:
        info = GGUFTensorInfo("t", 1, (elements,), tensor_type, 0)
        assert manager._calculate_tensor_byte_size(info) == expected


def test_calculate_tensor_byte_size_unknown_quant_raises(model_path: Path) -> None:
    """C4: an unknown quantization layout raises instead of returning a guessed size."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=1) as manager:
        with pytest.raises(ValueError):
            manager._calculate_tensor_byte_size(GGUFTensorInfo("t", 1, (256,), GGMLType.IQ4_XS, 0))


requires_native = pytest.mark.skipif(not bridge._load_native_library(), reason="native library not built or not loadable")


@requires_native
def test_native_views_are_read_only(model_path: Path) -> None:
    """C5: zero-copy views over the read-only mapping cannot be written."""
    with FlashStreamWeightManager(model_path, ring_buffer_slots=2) as manager:
        arrays = manager.acquire_layer_weights(0)
        view = next(iter(arrays.values()))
        assert view.flags.writeable is False
        with pytest.raises(ValueError):
            view[...] = 0


@requires_native
def test_close_with_live_view_defers_unmap_until_view_is_released(tmp_path: Path) -> None:
    """C5: engine.close() with a live view keeps the mapping valid; it is unmapped when the view dies."""
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 16)
    engine = NativeMemoryEngine(path)
    engine.register_layer(0, [{"name": "w", "tensor_type": 0, "dimensions": [64], "offset": 0, "size_bytes": 64}])
    view = engine.get_tensor_buffer(0, "w", dtype=np.uint8)
    guard = engine._guard
    assert view is not None and guard is not None and guard.live_views == 1

    engine.close()
    assert engine.is_open is False
    assert guard._ctx is not None
    assert view.tolist() == list(range(64))

    del view
    gc.collect()
    assert guard.live_views == 0
    assert guard._ctx is None


@requires_native
def test_close_without_views_unmaps_immediately(tmp_path: Path) -> None:
    """With no live views, close() releases the native mapping at once."""
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(256))
    engine = NativeMemoryEngine(path)
    guard = engine._guard
    assert guard is not None and guard._ctx is not None
    engine.close()
    assert guard._ctx is None


def test_slot_state_names_match_formal_model() -> None:
    """The slot states used in code match the states of docs/formal/RingSlots.tla."""
    tla = (Path(__file__).resolve().parents[1] / "docs" / "formal" / "RingSlots.tla").read_text(encoding="utf-8")
    for state in SlotState:
        assert f'"{state.name}"' in tla


def test_ring_model_check_passes_for_small_configuration() -> None:
    """The explicit-state model of the ring protocol finds no violation (2 threads, 3 layers, 2 slots)."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_ring_model.py"
    spec = importlib.util.spec_from_file_location("check_ring_model", script)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = module.check(2, 3, 2, 2)
    assert stats["final_states"] > 0
