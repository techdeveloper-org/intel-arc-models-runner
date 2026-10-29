"""Regression tests for the ring-buffer defects found by the TLA+ review (R05 S1-S4)."""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

from arc_runner.engine.flash_stream.device_stager import DeviceLayerStager
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager, SlotState
from tests.test_device_stager import _FakeBridge
from tests.test_flash_stream_triad import build_synthetic_flash_gguf


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Six-layer synthetic container."""
    return build_synthetic_flash_gguf(tmp_path / "r5.gguf", num_layers=6, hidden_dim=32)


def _arrays() -> dict:
    return {"w": np.arange(8, dtype=np.float32)}


def test_flush_releases_exactly_the_entries_it_drained_and_keeps_late_arrivals(model_path: Path) -> None:
    """S1: an eviction queued while a flush is running is released by a later flush, never consumed unreleased."""
    bridge = _FakeBridge()
    stager = DeviceLayerStager(bridge)
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3, device_stager=stager)
    stager.stage_layer(0, _arrays(), generation=1)
    stager.stage_layer(3, _arrays(), generation=2)
    manager._pending_unpin.append((0, 1))
    real_release = stager.release_layer

    def release_and_enqueue(layer: int, generation: Any = None) -> None:
        if layer == 0:
            manager._pending_unpin.append((3, 2))
        real_release(layer, generation)

    stager.release_layer = release_and_enqueue  # type: ignore[method-assign]
    manager._flush_unpins()
    assert stager.get(0) is None
    assert manager._pending_unpin == [(3, 2)], "late arrival must stay queued, not be consumed without a release"
    manager._flush_unpins()
    assert stager.get(3) is None
    manager.close()


def test_stale_eviction_does_not_free_a_buffer_staged_for_a_newer_generation(model_path: Path) -> None:
    """S2: releasing generation 1 of a layer must not free the buffer staged for generation 2."""
    stager = DeviceLayerStager(_FakeBridge())
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3, device_stager=stager)
    stager.stage_layer(0, _arrays(), generation=2)
    manager._pending_unpin.append((0, 1))
    manager._flush_unpins()
    assert stager.get(0) is not None
    manager._pending_unpin.append((0, 2))
    manager._flush_unpins()
    assert stager.get(0) is None
    manager.close()


def test_flush_failure_does_not_leave_the_slot_pending(model_path: Path) -> None:
    """S3: an exception while flushing evictions resets the slot so later acquires still work."""
    stager = DeviceLayerStager(_FakeBridge())
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3, device_stager=stager)
    manager.acquire_layer_weights(0)
    boom = {"on": True}
    real_release = stager.release_layer

    def failing_release(layer: int, generation: Any = None) -> None:
        if boom["on"]:
            raise RuntimeError("stager fault")
        real_release(layer, generation)

    stager.release_layer = failing_release  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        manager.acquire_layer_weights(3)
    assert manager._slots[0].state is not SlotState.PENDING
    boom["on"] = False
    manager._slot_wait_timeout_s = 0.5
    assert manager.acquire_layer_weights(3)
    manager.close()


def test_publish_after_close_discards_the_load(model_path: Path) -> None:
    """S4: a loader finishing after close() must not leave a READY slot or hold arrays in a closed manager."""
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3)
    action, _, _ = manager._reserve_slot(1)
    assert action == "load"
    manager._closing.set()
    manager._publish_load(1, _arrays())
    assert manager._slots[1].state is SlotState.EMPTY
    assert manager._slots[1].arrays == {} and manager.get_occupied_slot_count() == 0
    manager.close()


def test_staged_and_pinned_layers_are_always_resident_after_a_threaded_workout(model_path: Path) -> None:
    """S1/S2 invariant under real threads: at quiescence nothing is staged or pinned that is not resident."""
    import threading

    stager = DeviceLayerStager(_FakeBridge())
    manager = FlashStreamWeightManager(model_path, ring_buffer_slots=3, pin_resident_layers=False, device_stager=stager)
    errors: List[BaseException] = []

    def worker(seed: int) -> None:
        import random

        rng = random.Random(seed)
        try:
            for _ in range(60):
                layer = rng.randrange(6)
                if rng.random() < 0.3:
                    manager.release_layer_slot(layer % 3)
                else:
                    manager.acquire_layer_weights(layer)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    manager._flush_unpins()
    resident = {slot.layer_idx for slot in manager._slots if slot.state is SlotState.READY}
    staged = {layer for layer in range(6) if stager.get(layer) is not None}
    assert staged <= resident, f"staged {staged} not within resident {resident}"
    manager.close()
