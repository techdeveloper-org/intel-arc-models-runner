"""Asynchronous Continuous Batching Execution Pipeline and Multi-User Session Dispatcher.

Orchestrates concurrent requests through token-level iteration scheduling (Proof M29),
enforces Little's Law admission control (Proof M30), isolated virtual-to-physical block tables
(Proof M34), and Radix Tree prefix caching (Proof M31).
"""

import asyncio
from dataclasses import dataclass, field
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import uuid

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.engine.continuous_batcher import (
    AdmissionDeniedError,
    ContinuousBatcher,
    ContinuousBatcherError,
    SequenceState,
)
from arc_runner.engine.models import GenerationConfig, TokenChunk
from arc_runner.engine.paged_attention import PagedBlockTable, RadixPrefixCache

logger = get_logger("arc_runner.server.batch_pipeline")


class BatchPipelineError(ArcRunnerError):
    """Base exception for batch execution pipeline failures."""


@dataclass
class AsyncBatchRequest:
    """Internal tracking context for an asynchronous client generation stream."""

    seq_id: str
    prompt: str
    config: GenerationConfig
    queue: asyncio.Queue[Union[TokenChunk, None, Exception]]
    is_cancelled: bool = False
    created_at: float = field(default_factory=time.perf_counter)


class BatchExecutionPipeline:
    """Asynchronous pipeline coordinator connecting web endpoints to continuous batching engine."""

    def __init__(
        self,
        batcher: Optional[ContinuousBatcher] = None,
        prefix_cache: Optional[RadixPrefixCache] = None,
        tick_interval_s: float = 0.005,
        max_batch_size: int = 16,
    ) -> None:
        """Initialize the asynchronous batch execution pipeline.

        Args:
            batcher: Configured ContinuousBatcher instance. Defaults to new instance with 1024 blocks.
            prefix_cache: Optional RadixPrefixCache instance for prompt prefix deduplication.
            tick_interval_s: Sleep duration between scheduler iteration ticks in seconds.
            max_batch_size: Concurrency ceiling for simultaneous in-flight sequences.
        """
        self._batcher = batcher or ContinuousBatcher(max_batch_size=max_batch_size)
        self._prefix_cache = prefix_cache or RadixPrefixCache()
        self._tick_interval_s = max(0.001, tick_interval_s)

        self._active_requests: Dict[str, AsyncBatchRequest] = {}
        self._lock = threading.RLock()
        self._loop_task: Optional[asyncio.Task] = None
        self._is_running = False

    @property
    def batcher(self) -> ContinuousBatcher:
        """Return the underlying ContinuousBatcher instance."""
        return self._batcher

    @property
    def prefix_cache(self) -> RadixPrefixCache:
        """Return the active RadixPrefixCache instance."""
        return self._prefix_cache

    @property
    def is_running(self) -> bool:
        """Return True if background iteration task is currently active."""
        return self._is_running

    def start(self) -> None:
        """Start background scheduler loop task if not already executing."""
        with self._lock:
            if not self._is_running:
                self._is_running = True
                try:
                    loop = asyncio.get_running_loop()
                    self._loop_task = loop.create_task(self._run_loop())
                except RuntimeError:
                    pass

    async def stop(self) -> None:
        """Gracefully terminate background scheduler task and drain active sequences."""
        with self._lock:
            self._is_running = False
            if self._loop_task and not self._loop_task.done():
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
                self._loop_task = None

    async def _run_loop(self) -> None:
        """Background asynchronous scheduler loop processing batch iterations."""
        while self._is_running:
            try:
                self.tick_step()
                await asyncio.sleep(self._tick_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in batch pipeline scheduler iteration: {exc}")
                await asyncio.sleep(0.05)

    def tick_step(self) -> List[TokenChunk]:
        """Execute one continuous batching forward pass step and route chunks to client queues.

        Returns:
            List of emitted TokenChunk instances.
        """
        with self._lock:
            if not self._batcher.has_work():
                return []

            try:
                chunks = self._batcher.step()
            except Exception as exc:
                for req in list(self._active_requests.values()):
                    try:
                        req.queue.put_nowait(exc)
                    except asyncio.QueueFull:
                        pass
                self._active_requests.clear()
                raise

            for chunk in chunks:
                seq_id = chunk.seq_id
                if not seq_id:
                    continue

                req = self._active_requests.get(seq_id)
                if not req:
                    continue

                try:
                    req.queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    logger.warning(f"Client streaming queue full for sequence {seq_id}; dropping chunk.")

                if chunk.is_finished:
                    req.queue.put_nowait(None)
                    self._active_requests.pop(seq_id, None)

            return chunks

    async def enqueue_request(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        seq_id: Optional[str] = None,
    ) -> AsyncIterator[TokenChunk]:
        """Enqueue prompt for continuous batch generation and stream decoded tokens.

        Args:
            prompt: Text prompt input.
            config: Optional generation configuration.
            seq_id: Optional unique sequence identifier. Defaults to generated UUID.

        Yields:
            TokenChunk emissions as decoded.

        Raises:
            AdmissionDeniedError: If VRAM block capacity exceeds saturation threshold (Proof M30).
            ValueError: If prompt is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        target_seq_id = seq_id or f"req-{uuid.uuid4().hex[:12]}"
        gen_cfg = config or GenerationConfig()
        token_queue: asyncio.Queue[Union[TokenChunk, None, Exception]] = asyncio.Queue(maxsize=256)

        req = AsyncBatchRequest(
            seq_id=target_seq_id,
            prompt=prompt,
            config=gen_cfg,
            queue=token_queue,
        )

        with self._lock:
            self._batcher.add_request(target_seq_id, prompt, gen_cfg)
            self._active_requests[target_seq_id] = req

            if not self._is_running:
                self.start()

        is_completed = False
        try:
            while True:
                item = await token_queue.get()
                if item is None:
                    is_completed = True
                    break
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, TokenChunk):
                    yield item
                    if item.is_finished:
                        is_completed = True
                        break
        finally:
            if not is_completed:
                self.cancel_sequence(target_seq_id)

    def cancel_sequence(self, seq_id: str) -> bool:
        """Cancel an active or queued sequence and immediately reclaim physical memory blocks.

        Args:
            seq_id: Target sequence identifier.

        Returns:
            True if sequence was found and cancelled, False otherwise.
        """
        with self._lock:
            cancelled = self._batcher.cancel_request(seq_id)
            req = self._active_requests.pop(seq_id, None)
            if req:
                req.is_cancelled = True
                req.queue.put_nowait(None)
            return cancelled

    def preempt_sequence(self, seq_id: str) -> bool:
        """Preempt an active sequence and return its allocated blocks to the free pool.

        Args:
            seq_id: Sequence identifier to preempt.

        Returns:
            True if sequence was found in running batch and preempted, False otherwise.
        """
        with self._lock:
            return self._batcher.preempt_sequence(seq_id)

    def get_status(self) -> Dict[str, Any]:
        """Compile comprehensive pipeline telemetry metrics.

        Returns:
            Dictionary containing active sequences, block utilization, and prefix cache performance.
        """
        with self._lock:
            metrics = self._batcher.get_metrics()
            metrics["prefix_cache_hit_rate"] = round(self._prefix_cache.hit_rate * 100.0, 2)
            metrics["prefix_cache_hits"] = self._prefix_cache.hit_count
            metrics["prefix_cache_misses"] = self._prefix_cache.miss_count
            metrics["active_client_connections"] = len(self._active_requests)
            metrics["running_sequence_ids"] = self._batcher.get_running_sequence_ids()
            metrics["waiting_sequence_ids"] = self._batcher.get_waiting_sequence_ids()
            return metrics
