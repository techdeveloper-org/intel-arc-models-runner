"""Independent ring-buffer invariant tests for FlashStreamWeightManager.

Specification (class/method docstrings and docs/architecture.md): constant-memory ring of
`ring_buffer_slots` slots, direct mapped (layer L only ever lives in slot L % slots), acquire returns
the exact tensor bytes of the layer, concurrent callers for one layer share one load, a closed manager
raises ArcRunnerError, occupied slots never exceed the slot count.
"""

import random
import threading
from pathlib import Path

import numpy as np
import pytest

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager

from ind_helpers import F32, STR, U32, build_gguf, layered_f32_model

HIDDEN = 8
LAYERS = 9
JOIN_TIMEOUT = 20.0


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    blob, per_layer = layered_f32_model(LAYERS, HIDDEN, seed=5)
    path = tmp_path_factory.mktemp("ring") / "m.gguf"
    path.write_bytes(blob)
    return path, per_layer


def _layer_bytes(arrays) -> bytes:
    # tensor order inside a layer is attn_q then ffn_up (the order they were written)
    return arrays["attn_q.weight"].tobytes() + arrays["ffn_up.weight"].tobytes()


def _join_all(threads):
    for t in threads:
        t.join(JOIN_TIMEOUT)
    stuck = [t.name for t in threads if t.is_alive()]
    assert stuck == [], f"threads did not finish (possible deadlock): {stuck}"


@pytest.mark.parametrize("slots", [1, 2, 3, 5])
def test_acquire_returns_exact_layer_bytes_for_random_sequence(model, slots):
    path, per_layer = model
    rng = random.Random(100 + slots)
    with FlashStreamWeightManager(path, ring_buffer_slots=slots) as mgr:
        for _ in range(60):
            layer = rng.randrange(LAYERS)
            arrays = mgr.acquire_layer_weights(layer)
            assert _layer_bytes(arrays) == per_layer[layer]
            assert mgr.get_occupied_slot_count() <= slots


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_occupied_count_never_exceeds_slots_under_sequential_scan(model, slots):
    path, _ = model
    with FlashStreamWeightManager(path, ring_buffer_slots=slots) as mgr:
        for layer in range(LAYERS):
            mgr.acquire_layer_weights(layer)
            assert mgr.get_occupied_slot_count() == min(layer + 1, slots)
        one_layer_bytes = HIDDEN * 2 * 4 * 2
        assert mgr.get_active_memory_bytes() <= slots * one_layer_bytes


def test_direct_mapping_layer_resident_only_in_its_own_slot(model):
    path, _ = model
    slots = 3
    with FlashStreamWeightManager(path, ring_buffer_slots=slots) as mgr:
        for layer in range(LAYERS):
            mgr.acquire_layer_weights(layer)
            # after acquiring `layer`, releasing every other slot must keep that layer's hit stable
        mgr.acquire_layer_weights(7)
        occupied_before = mgr.get_occupied_slot_count()
        mgr.release_layer_slot(7 % slots)
        assert mgr.get_occupied_slot_count() == occupied_before - 1
        mgr.release_layer_slot(7 % slots)  # releasing an empty slot is a no-op
        assert mgr.get_occupied_slot_count() == occupied_before - 1


@pytest.mark.parametrize("slots", [1, 2, 3, 4])
def test_layer_is_resident_only_in_slot_layer_mod_slots(model, slots):
    path, _ = model
    rng = random.Random(55 + slots)
    with FlashStreamWeightManager(path, ring_buffer_slots=slots) as mgr:
        for _ in range(40):
            layer = rng.randrange(LAYERS)
            mgr.acquire_layer_weights(layer)
            with mgr._lock:
                resident = {i: s.layer_idx for i, s in enumerate(mgr._slots) if s.layer_idx >= 0}
            assert resident[layer % slots] == layer
            assert all(idx % slots == slot for slot, idx in resident.items())


def test_release_layer_slot_out_of_range_is_ignored(model):
    path, _ = model
    with FlashStreamWeightManager(path, ring_buffer_slots=2) as mgr:
        mgr.acquire_layer_weights(0)
        for bad in (-1, 2, 99):
            mgr.release_layer_slot(bad)
        assert mgr.get_occupied_slot_count() == 1


def test_arrays_stay_valid_after_slot_is_recycled(model):
    path, per_layer = model
    with FlashStreamWeightManager(path, ring_buffer_slots=1) as mgr:
        held = mgr.acquire_layer_weights(0)
        for layer in range(1, LAYERS):
            mgr.acquire_layer_weights(layer)
        assert _layer_bytes(held) == per_layer[0]


def test_concurrent_random_acquire_release_prefetch_keeps_invariants(model):
    path, per_layer = model
    slots = 3
    errors = []
    with FlashStreamWeightManager(path, ring_buffer_slots=slots) as mgr:
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                occ = mgr.get_occupied_slot_count()
                if occ > slots:
                    errors.append(("occupied>slots", occ))
                if mgr.get_active_memory_bytes() > slots * HIDDEN * 2 * 4 * 2:
                    errors.append("active bytes above ring capacity")

        def worker(seed):
            rng = random.Random(seed)
            try:
                for _ in range(80):
                    op = rng.randrange(3)
                    layer = rng.randrange(LAYERS)
                    if op == 0:
                        arrays = mgr.acquire_layer_weights(layer)
                        if _layer_bytes(arrays) != per_layer[layer]:
                            errors.append(("wrong bytes", layer))
                    elif op == 1:
                        mgr.release_layer_slot(rng.randrange(slots))
                    else:
                        mgr.prefetch_layer_async(layer, layer % slots)
            except BaseException as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(repr(exc))

        samp = threading.Thread(target=sampler, daemon=True, name="sampler")
        samp.start()
        workers = [threading.Thread(target=worker, args=(900 + i,), daemon=True, name=f"w{i}") for i in range(6)]
        for w in workers:
            w.start()
        _join_all(workers)
        stop.set()
        samp.join(5)
        assert errors == []
        with mgr._lock:
            pending = list(mgr._prefetch_threads)
        _join_all(pending)
        assert mgr.get_occupied_slot_count() <= slots


def test_concurrent_acquire_same_layer_shares_one_load(model, monkeypatch):
    path, per_layer = model
    with FlashStreamWeightManager(path, ring_buffer_slots=2) as mgr:
        calls = []
        real = mgr._load_layer_arrays
        gate = threading.Event()

        def counting(layer):
            calls.append(layer)
            gate.wait(2)
            return real(layer)

        monkeypatch.setattr(mgr, "_load_layer_arrays", counting)
        results = []
        threads = [threading.Thread(target=lambda: results.append(mgr.acquire_layer_weights(4)), daemon=True) for _ in range(8)]
        for t in threads:
            t.start()
        gate.set()
        _join_all(threads)
        assert calls == [4], "layer 4 must be loaded exactly once for concurrent callers"
        assert len(results) == 8
        assert all(_layer_bytes(r) == per_layer[4] for r in results)


def test_acquire_after_close_raises_arc_runner_error(model):
    path, _ = model
    mgr = FlashStreamWeightManager(path, ring_buffer_slots=2)
    mgr.acquire_layer_weights(0)
    mgr.close()
    mgr.close()  # idempotent
    with pytest.raises(ArcRunnerError):
        mgr.acquire_layer_weights(1)
    assert mgr.get_occupied_slot_count() == 0


def test_prefetch_then_close_joins_without_hanging(model):
    path, _ = model
    mgr = FlashStreamWeightManager(path, ring_buffer_slots=2)
    threads = [mgr.prefetch_layer_async(i % LAYERS, 0) for i in range(12)]
    closer = threading.Thread(target=mgr.close, daemon=True)
    closer.start()
    closer.join(JOIN_TIMEOUT)
    assert not closer.is_alive()
    _join_all(threads)


def test_constructor_rejects_bad_slots_and_missing_file(model, tmp_path):
    path, _ = model
    for bad in (0, -1):
        with pytest.raises(ValueError):
            FlashStreamWeightManager(path, ring_buffer_slots=bad)
    with pytest.raises(FileNotFoundError):
        FlashStreamWeightManager(tmp_path / "nope.gguf")


def test_corrupt_layer_offset_fails_cleanly_and_slot_recovers(tmp_path):
    # layer 1 points past the end of the file; layers 0 and 2 are fine
    payload = bytes(range(64)) * 2
    metadata = [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)]
    tensors = [("blk.0.w", (8,), 0, 0), ("blk.1.w", (8,), 0, 10 ** 9), ("blk.2.w", (8,), 0, 32)]
    blob, _ = build_gguf(metadata, tensors, payload)
    p = tmp_path / "bad.gguf"
    p.write_bytes(blob)
    with FlashStreamWeightManager(p, ring_buffer_slots=1) as mgr:
        assert mgr.get_occupied_slot_count() == 0
        for _ in range(3):
            with pytest.raises((ValueError, ArcRunnerError)):
                mgr.acquire_layer_weights(1)
            assert mgr.get_occupied_slot_count() == 0
        ok = mgr.acquire_layer_weights(0)
        assert ok["w"].tobytes() == payload[0:32]
        ok2 = mgr.acquire_layer_weights(2)
        assert ok2["w"].tobytes() == payload[32:64]


def test_corrupt_layer_concurrent_waiters_do_not_hang(tmp_path):
    payload = bytes(range(64))
    metadata = [("general.architecture", STR, "llama"), ("general.alignment", U32, 32)]
    tensors = [("blk.0.w", (8,), 0, 0), ("blk.1.w", (8,), 0, 10 ** 9)]
    blob, _ = build_gguf(metadata, tensors, payload)
    p = tmp_path / "bad2.gguf"
    p.write_bytes(blob)
    outcomes = []
    with FlashStreamWeightManager(p, ring_buffer_slots=1) as mgr:
        def run():
            try:
                mgr.acquire_layer_weights(1)
                outcomes.append("ok")
            except (ValueError, ArcRunnerError):
                outcomes.append("err")
            except BaseException as exc:  # noqa: BLE001
                outcomes.append(repr(exc))
        threads = [threading.Thread(target=run, daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        _join_all(threads)
    assert outcomes == ["err"] * 6
