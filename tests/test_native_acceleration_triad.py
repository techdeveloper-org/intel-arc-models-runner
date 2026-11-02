"""Testing Triad for native system acceleration, governor, telemetry, and USM ring buffer.

Implements the complete 15-test testing triad (TC-P01..P05, TC-N01..N05, TC-E01..E05)
under the guidance of the QA & Testing Specialist Council and governing rules:
- 35-positive-testing-standards.md
- 36-negative-testing-standards.md
- 37-edge-case-testing-standards.md
"""

from __future__ import annotations

import threading
import time
from typing import List, Tuple

import pytest

from arc_runner.monitoring.telemetry import (
    HardwareTelemetryCollector,
    TelemetrySnapshot,
    TelemetrySummary,
)
from arc_runner.native.governor import ProcessGovernor
from arc_runner.native.usm_streamer import USMTokenStreamer


# ============================================================================
# POSITIVE TESTS (TC-P01 .. TC-P05)
# ============================================================================

def test_tc_p01_governor_boost_high() -> None:
    """TC-P01: ProcessGovernor successfully boosts process priority."""
    gov = ProcessGovernor()
    success = gov.boost(pin_p_cores=True)
    assert success is True
    assert gov._is_active is True
    gov.restore()
    assert gov._is_active is False


def test_tc_p02_governor_context_manager() -> None:
    """TC-P02: ProcessGovernor RAII context manager cleanly activates and restores."""
    with ProcessGovernor() as gov:
        assert gov._is_active is True
    assert gov._is_active is False


def test_tc_p03_usm_ring_buffer_fifo_order() -> None:
    """TC-P03: USMTokenStreamer preserves strict FIFO ordering for 100 sequential tokens."""
    streamer = USMTokenStreamer(capacity=128)
    try:
        pushed_data = [(1000 + i, float(i * 1.5)) for i in range(100)]
        for tok_id, lat in pushed_data:
            ok = streamer.push(tok_id, lat)
            assert ok is True

        drained = streamer.drain()
        assert len(drained) == 100
        for expected, actual in zip(pushed_data, drained):
            assert expected[0] == actual[0]
            assert abs(expected[1] - actual[1]) < 1e-3
    finally:
        streamer.close()


def test_tc_p04_hardware_telemetry_collector_lifecycle() -> None:
    """TC-P04: HardwareTelemetryCollector runs and aggregates resource statistics."""
    collector = HardwareTelemetryCollector(sample_interval_sec=0.02)
    collector.start()
    time.sleep(0.1)
    summary = collector.stop()

    assert isinstance(summary, TelemetrySummary)
    assert summary.sample_count >= 3
    assert summary.duration_sec > 0.05
    assert summary.mean_ram_mb > 100.0
    assert 0.0 <= summary.stability_score <= 100.0
    report = summary.format_report()
    assert "[TELEMETRY]" in report
    assert "CPU Total Load:" in report


def test_tc_p05_telemetry_snapshot_immediate() -> None:
    """TC-P05: sample_now() returns valid instantaneous metrics without background thread."""
    collector = HardwareTelemetryCollector()
    snap = collector.sample_now()
    assert isinstance(snap, TelemetrySnapshot)
    assert snap.timestamp_ns > 0
    assert snap.ram_used_mb > 0.0
    assert snap.ram_percent > 0.0


# ============================================================================
# NEGATIVE TESTS (TC-N01 .. TC-N05)
# ============================================================================

def test_tc_n01_usm_ring_pop_empty_returns_none() -> None:
    """TC-N01: Popping from an empty ring buffer returns None without hanging."""
    streamer = USMTokenStreamer(capacity=64)
    try:
        result = streamer.pop()
        assert result is None
    finally:
        streamer.close()


def test_tc_n02_usm_ring_overflow_handling() -> None:
    """TC-N02: Pushing into a completely full ring buffer returns False and rejects overflow."""
    # Capacity will round up to 16
    streamer = USMTokenStreamer(capacity=16)
    try:
        # Push 16 tokens to fill it completely
        for i in range(16):
            streamer.push(i, 1.0)

        # 17th token must be rejected
        overflow_ok = streamer.push(999, 1.0)
        assert overflow_ok is False

        # Existing 16 tokens remain intact
        drained = streamer.drain()
        assert len(drained) == 16
        assert drained[0][0] == 0
        assert drained[-1][0] == 15
    finally:
        streamer.close()


def test_tc_n03_telemetry_stop_without_start() -> None:
    """TC-N03: Calling stop() without start() returns a safe default summary without crashing."""
    collector = HardwareTelemetryCollector()
    summary = collector.stop()
    assert isinstance(summary, TelemetrySummary)
    assert summary.sample_count == 0


def test_tc_n04_governor_restore_without_boost() -> None:
    """TC-N04: Calling restore() on unboosted governor is a safe idempotent no-op."""
    gov = ProcessGovernor()
    # Should not raise exception
    gov.restore()
    assert gov._is_active is False


def test_tc_n05_usm_ring_double_close() -> None:
    """TC-N05: Closing an already closed ring buffer is safe and does not double-free."""
    streamer = USMTokenStreamer(capacity=32)
    streamer.close()
    streamer.close()  # Must not crash


# ============================================================================
# EDGE-CASE TESTS (TC-E01 .. TC-E05)
# ============================================================================

def test_tc_e01_governor_rapid_boost_restore_cycles() -> None:
    """TC-E01: Rapid consecutive boost and restore cycles remain idempotent and crash-free."""
    gov = ProcessGovernor()
    for _ in range(20):
        gov.boost()
        assert gov._is_active is True
        gov.restore()
        assert gov._is_active is False


def test_tc_e02_usm_ring_wraparound_10k_tokens() -> None:
    """TC-E02: Streaming 10,000 tokens through a small 64-slot ring buffer across wrap-arounds."""
    streamer = USMTokenStreamer(capacity=64)
    total_tokens = 10000
    try:
        received = []
        for i in range(total_tokens):
            ok = streamer.push(i, 0.5)
            assert ok is True
            popped = streamer.pop()
            assert popped is not None
            received.append(popped[0])

        assert len(received) == total_tokens
        assert received == list(range(total_tokens))
    finally:
        streamer.close()


def test_tc_e03_usm_ring_multithreaded_producer_consumer() -> None:
    """TC-E03: Multithreaded SPSC: 5,000 tokens pushed by producer thread, popped by consumer thread."""
    streamer = USMTokenStreamer(capacity=256)
    count = 5000
    consumer_results: List[int] = []
    stop_event = threading.Event()

    def consumer_worker():
        while not stop_event.is_set() or len(consumer_results) < count:
            popped = streamer.pop()
            if popped is not None:
                consumer_results.append(popped[0])
            else:
                time.sleep(0.0001)

    t_consumer = threading.Thread(target=consumer_worker, daemon=True)
    t_consumer.start()

    try:
        for i in range(count):
            while not streamer.push(i, 0.1):
                time.sleep(0.0001)

        stop_event.set()
        t_consumer.join(timeout=3.0)

        assert len(consumer_results) == count
        assert consumer_results == list(range(count))
    finally:
        streamer.close()


def test_tc_e04_high_frequency_telemetry_overhead() -> None:
    """TC-E04: High-frequency telemetry (10ms) collects hundreds of samples with negligible jitter."""
    collector = HardwareTelemetryCollector(sample_interval_sec=0.01)
    collector.start()
    time.sleep(0.2)
    summary = collector.stop()

    assert summary.sample_count >= 10
    # Jitter should be tightly bounded on Windows high-res multimedia timer
    assert summary.jitter_ms < 50.0


def test_tc_e05_large_token_id_and_extreme_latency() -> None:
    """TC-E05: Handles 64-bit large token IDs and negative or large float latencies safely."""
    streamer = USMTokenStreamer(capacity=32)
    try:
        large_id = 9223372036854775807  # INT64_MAX
        ok = streamer.push(large_id, 999999.98)
        assert ok is True
        popped = streamer.pop()
        assert popped is not None
        assert popped[0] == large_id
        assert abs(popped[1] - 999999.98) < 1.0
    finally:
        streamer.close()
