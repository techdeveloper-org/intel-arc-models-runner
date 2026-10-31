"""Unit and mathematical invariant tests for StreamTelemetryTracker.

Verifies the mathematical precision of Time-to-First-Token (TTFT),
Time-Per-Output-Token (TPOT), and rolling generation throughput (TPS)
as formulated by foundation-model-mathematics-expert.
"""

import math
import pytest
from typing import Dict, Any


class StreamTelemetryTracker:
    """Python reference mirror of the client-side StreamTelemetryTracker in app.js."""

    def __init__(self) -> None:
        self.t_dispatch = 0.0
        self.t_first_token = 0.0
        self.t_last_token = 0.0
        self.token_count = 0

    def start(self, timestamp: float) -> None:
        """Start tracking from a high-resolution monotonic dispatch timestamp."""
        self.t_dispatch = timestamp
        self.t_first_token = 0.0
        self.t_last_token = 0.0
        self.token_count = 0

    def on_token_chunk(self, count: int, timestamp: float) -> None:
        """Record the arrival of newly decoded token chunk at monotonic timestamp."""
        self.token_count += count
        if self.token_count == count:
            self.t_first_token = timestamp
        self.t_last_token = timestamp

    def get_ttft(self) -> float:
        """Compute Time-to-First-Token (TTFT) in milliseconds."""
        if not self.t_first_token or not self.t_dispatch:
            return 0.0
        return max(0.0, self.t_first_token - self.t_dispatch)

    def get_tpot(self) -> float:
        """Compute average Time-Per-Output-Token (TPOT) in milliseconds per token."""
        if self.token_count <= 1 or not self.t_first_token:
            return 0.0
        return max(0.0, (self.t_last_token - self.t_first_token) / (self.token_count - 1))

    def get_tps(self, current_time: float) -> float:
        """Compute instantaneous tokens-per-second (TPS) throughput."""
        if not self.t_dispatch or not self.token_count:
            return 0.0
        duration_sec = (current_time - self.t_dispatch) / 1000.0
        return self.token_count / duration_sec if duration_sec > 0 else 0.0


def test_telemetry_tracker_zero_state_safe() -> None:
    """Verify tracker returns 0.0 before any chunks arrive without throwing."""
    tracker = StreamTelemetryTracker()
    assert tracker.get_ttft() == 0.0
    assert tracker.get_tpot() == 0.0
    assert tracker.get_tps(100.0) == 0.0


def test_telemetry_tracker_ttft_calculation_positive() -> None:
    """Verify TTFT correctly measures delta between dispatch and chunk 0 arrival."""
    tracker = StreamTelemetryTracker()
    tracker.start(1000.0)
    tracker.on_token_chunk(1, 1250.0)
    assert tracker.get_ttft() == 250.0


def test_telemetry_tracker_tpot_and_tps_accuracy() -> None:
    """Verify multi-token sequence yields exact expected TPOT and TPS metrics."""
    tracker = StreamTelemetryTracker()
    tracker.start(1000.0)
    tracker.on_token_chunk(1, 1200.0)
    tracker.on_token_chunk(1, 1240.0)
    tracker.on_token_chunk(1, 1280.0)
    tracker.on_token_chunk(1, 1320.0)

    assert tracker.token_count == 4
    assert tracker.get_ttft() == 200.0
    expected_tpot = (1320.0 - 1200.0) / 3.0
    assert math.isclose(tracker.get_tpot(), expected_tpot, rel_tol=1e-5)
    expected_tps = 4.0 / ((1320.0 - 1000.0) / 1000.0)
    assert math.isclose(tracker.get_tps(1320.0), expected_tps, rel_tol=1e-5)


def test_telemetry_tracker_monotonic_negative_edge_case() -> None:
    """Verify clock skew or out-of-order timestamps clamp safely to zero."""
    tracker = StreamTelemetryTracker()
    tracker.start(2000.0)
    tracker.on_token_chunk(1, 1900.0)
    assert tracker.get_ttft() == 0.0
