"""Tests for ContinuousBatchingScheduler (WP-04 CC-1..CC-4).

Covers import and instantiation, step loop termination with progress guard,
thread-safe request enqueueing, concurrency cap updates, and /health status.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from arc_runner.core.constants import CB_MAX_STEPS_WITHOUT_PROGRESS, MAX_CONCURRENT_SEQUENCES
from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.engine.cb_scheduler import ContinuousBatchingScheduler
from arc_runner.server.app import create_app


class FakePipeline:
    """Mock ContinuousBatchingPipeline for deterministic testing."""

    def __init__(self, steps_to_finish: int = 5) -> None:
        """Initialize with remaining step count.

        Args:
            steps_to_finish: Steps before has_non_finished_requests returns False.
        """
        self.steps_remaining = steps_to_finish
        self.added: List[tuple[str, str]] = []

    def add_request(self, request_id: str, prompt: str, config: Optional[Any] = None) -> None:
        """Record added request.

        Args:
            request_id: Sequence ID.
            prompt: Text prompt.
            config: Generation config.
        """
        self.added.append((request_id, prompt))

    def step(self) -> None:
        """Decrement steps remaining."""
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def has_non_finished_requests(self) -> bool:
        """Check if work remains.

        Returns:
            bool: True if steps remain.
        """
        return self.steps_remaining > 0


class NonTerminatingPipeline:
    """Pipeline that never completes to test ceiling guard."""

    def add_request(self, request_id: str, prompt: str, config: Optional[Any] = None) -> None:
        """Record added request.

        Args:
            request_id: Sequence ID.
            prompt: Text prompt.
            config: Generation config.
        """
        pass

    def step(self) -> None:
        """Simulate a step with no progress."""
        pass

    def has_non_finished_requests(self) -> bool:
        """Always returns True.

        Returns:
            bool: True.
        """
        return True


def test_continuous_batching_scheduler_import_and_instantiation_cc1() -> None:
    """CC-1: ContinuousBatchingScheduler imports cleanly and initializes with default config."""
    scheduler = ContinuousBatchingScheduler()
    assert scheduler is not None
    assert scheduler.concurrency_cap == MAX_CONCURRENT_SEQUENCES
    assert scheduler.status() == {"active_sequences": 0, "queue_depth": 0}


def test_step_until_done_terminates_normally_cc2() -> None:
    """CC-2: step_until_done finishes when pipeline has no non-finished requests."""
    fake = FakePipeline(steps_to_finish=3)
    scheduler = ContinuousBatchingScheduler(pipeline=fake)
    scheduler.add_request("req-1", "test prompt")
    steps = scheduler.step_until_done(max_steps=10)
    assert steps == 3
    assert not fake.has_non_finished_requests()


def test_step_until_done_raises_on_step_ceiling_cc2() -> None:
    """CC-2: step_until_done raises ArcRunnerError when exceeding max steps."""
    pipe = NonTerminatingPipeline()
    scheduler = ContinuousBatchingScheduler(pipeline=pipe)
    scheduler.add_request("stuck", "prompt")
    with pytest.raises(ArcRunnerError) as exc_info:
        scheduler.step_until_done(max_steps=50)
    assert "CB_MAX_STEPS_EXCEEDED" in str(exc_info.value.error_code)


def test_add_request_thread_safety_cc3() -> None:
    """CC-3: add_request handles concurrent additions across threads without corruption."""
    scheduler = ContinuousBatchingScheduler()
    total_requests = 100

    def submit_req(idx: int) -> None:
        scheduler.add_request(f"req-{idx}", f"prompt {idx}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(submit_req, range(total_requests)))

    st = scheduler.status()
    assert st["active_sequences"] + st["queue_depth"] == total_requests


def test_set_concurrency_cap() -> None:
    """Validate concurrency cap modification and input guards."""
    scheduler = ContinuousBatchingScheduler()
    scheduler.set_concurrency_cap(4)
    assert scheduler.concurrency_cap == 4

    with pytest.raises(ValueError):
        scheduler.set_concurrency_cap(0)

    with pytest.raises(ValueError):
        scheduler.set_concurrency_cap(-2)


def test_health_route_includes_scheduler_status_cc4() -> None:
    """CC-4: /health endpoint includes scheduler_status dictionary."""
    scheduler = ContinuousBatchingScheduler()
    scheduler.add_request("active-1", "p1")
    app = create_app()
    app.state.cb_scheduler = scheduler

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "scheduler_status" in data
    assert data["scheduler_status"]["active_sequences"] >= 1
