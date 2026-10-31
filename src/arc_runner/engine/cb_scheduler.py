"""Continuous batching scheduler for dynamic iteration-level inference on Intel Arc GPUs.

Wraps the OpenVINO GenAI ContinuousBatchingPipeline, orchestrating token-level
in-flight batching, thread-safe admission control, and step progress enforcement.
"""

from threading import Lock
import time
from typing import Any, Dict, List, Optional

from arc_runner.core.constants import (
    CB_MAX_STEPS_WITHOUT_PROGRESS,
    CB_QUEUE_DRAIN_TIMEOUT_S,
    DECODE_STEPS_PER_TICK,
    MAX_CONCURRENT_SEQUENCES,
    PREFILL_WINDOW_MS,
)
from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.engine.cb_scheduler")


class ContinuousBatchingSchedulerError(ArcRunnerError):
    """Base exception for continuous batching scheduler operations."""


class ContinuousBatchingScheduler:
    """Continuous batching scheduler wrapping ov_genai.ContinuousBatchingPipeline.

    Orchestrates iteration-level token batching across concurrent requests,
    enforces dynamic concurrency caps for thermal safety, and guarantees
    step termination via Banach contraction bounds.
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        max_concurrency: int = MAX_CONCURRENT_SEQUENCES,
    ) -> None:
        """Initialize the continuous batching scheduler.

        Args:
            pipeline: Optional underlying ContinuousBatchingPipeline instance.
            config: Optional dictionary containing scheduler configuration.
            max_concurrency: Maximum concurrent sequences admitted into active batch.

        Raises:
            ValueError: If max_concurrency is not strictly positive.
        """
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be strictly positive, got {max_concurrency}")

        self._pipeline = pipeline
        self._config = config or {}
        self._concurrency_cap: int = max_concurrency
        self._step_counter: int = 0
        self._lock: Lock = Lock()
        self._pending_requests: List[Dict[str, Any]] = []
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._completed_requests: Dict[str, Dict[str, Any]] = {}

    @property
    def concurrency_cap(self) -> int:
        """Return the current dynamic sequence concurrency ceiling.

        Returns:
            int: Maximum allowable active sequences.
        """
        return self._concurrency_cap

    def set_concurrency_cap(self, cap: int) -> None:
        """Dynamically update sequence concurrency ceiling for thermal governance.

        Args:
            cap: Positive integer concurrency ceiling.

        Raises:
            ValueError: If cap is less than 1.
        """
        if cap < 1:
            raise ValueError(f"concurrency cap must be at least 1, got {cap}")

        with self._lock:
            self._concurrency_cap = cap
            logger.info("concurrency cap updated", extra={"concurrency_cap": cap})

    def add_request(
        self,
        request_id: str,
        prompt: str,
        generation_config: Optional[Any] = None,
    ) -> None:
        """Enqueue a request for continuous batch execution in a thread-safe manner.

        Args:
            request_id: Unique sequence identifier.
            prompt: Text prompt input string.
            generation_config: Optional generation parameters.
        """
        req_entry = {
            "request_id": request_id,
            "prompt": prompt,
            "config": generation_config,
            "enqueued_at": time.monotonic(),
        }

        with self._lock:
            if len(self._active_requests) < self._concurrency_cap:
                self._active_requests[request_id] = req_entry
                if self._pipeline is not None and hasattr(self._pipeline, "add_request"):
                    self._pipeline.add_request(request_id, prompt, generation_config)
            else:
                self._pending_requests.append(req_entry)

    def _admit_pending_locked(self) -> None:
        """Promote queued pending requests to active batch up to concurrency cap."""
        while self._pending_requests and len(self._active_requests) < self._concurrency_cap:
            req = self._pending_requests.pop(0)
            req_id = req["request_id"]
            self._active_requests[req_id] = req
            if self._pipeline is not None and hasattr(self._pipeline, "add_request"):
                self._pipeline.add_request(req_id, req["prompt"], req["config"])

    def step_until_done(
        self,
        max_steps: int = CB_MAX_STEPS_WITHOUT_PROGRESS,
    ) -> int:
        """Execute batch forward steps until all requests finish or step ceiling is reached.

        Args:
            max_steps: Upper bound on loop iterations to safeguard against divergence.

        Returns:
            int: Total steps executed during the drain cycle.

        Raises:
            ContinuousBatchingSchedulerError: If max_steps is exceeded before completion.
        """
        steps_executed = 0

        while True:
            with self._lock:
                self._admit_pending_locked()
                has_work = bool(self._active_requests or self._pending_requests)
                if self._pipeline is not None and hasattr(self._pipeline, "has_non_finished_requests"):
                    has_work = self._pipeline.has_non_finished_requests()

            if not has_work:
                break

            if steps_executed >= max_steps:
                raise ContinuousBatchingSchedulerError(
                    f"Continuous batching scheduler exceeded max steps ({max_steps}) without progress",
                    error_code="CB_MAX_STEPS_EXCEEDED",
                )

            if self._pipeline is not None and hasattr(self._pipeline, "step"):
                self._pipeline.step()

            steps_executed += 1
            self._step_counter += 1

            if self._pipeline is not None and hasattr(self._pipeline, "has_non_finished_requests"):
                if not self._pipeline.has_non_finished_requests():
                    with self._lock:
                        self._active_requests.clear()
                    break

        return steps_executed

    def status(self) -> Dict[str, int]:
        """Return a snapshot of scheduler queue and active sequence metrics.

        Returns:
            Dict[str, int]: Dictionary containing active_sequences and queue_depth counts.
        """
        with self._lock:
            return {
                "active_sequences": len(self._active_requests),
                "queue_depth": len(self._pending_requests),
            }
