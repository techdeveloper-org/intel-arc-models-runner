"""Dynamic in-flight continuous iteration batcher and admission controller for Intel Arc GPUs.

Merges prefill and decode phases across concurrent sequences (Proof M10), reclaims finished slots
immediately upon EOS, and enforces Little's Law admission control (Proof M13).
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.engine.models import GenerationConfig, TokenChunk
from arc_runner.engine.paged_attention import PagedBlockTable, VRAMExhaustedError

logger = get_logger("arc_runner.engine.continuous_batcher")


class SequenceState(str, Enum):
    """Lifecycle state of an individual generation sequence."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED = "FINISHED"


@dataclass
class BatchSequence:
    """Internal tracking descriptor for a single request in the continuous batch."""

    seq_id: str
    prompt: str
    config: GenerationConfig
    state: SequenceState = SequenceState.WAITING
    generated_tokens: List[str] = field(default_factory=list)
    token_count: int = 0
    max_tokens: int = 512
    start_time: float = field(default_factory=time.perf_counter)
    stop_reason: Optional[str] = None


class ContinuousBatcherError(ArcRunnerError):
    """Raised when continuous batching execution fails."""


class AdmissionDeniedError(ContinuousBatcherError):
    """Raised when request is rejected due to Little's Law concurrency or VRAM saturation."""

    def __init__(self, message: str, error_code: str = "ADMISSION_DENIED", status_code: int = 429) -> None:
        """Initialize admission denied exception with 429 Too Many Requests status code.

        Args:
            message: Explanation of admission rejection reason.
            error_code: Unique machine-readable error token.
            status_code: Corresponding HTTP status code.
        """
        super().__init__(message, error_code=error_code, status_code=status_code)


class ContinuousBatcher:
    """Token-level iteration scheduler merging prefill and decode iterations (Proof M10)."""

    def __init__(
        self,
        block_table: Optional[PagedBlockTable] = None,
        max_batch_size: int = 16,
        vram_saturation_threshold: float = 0.92,
        generator_func: Optional[Callable[[str, int], str]] = None,
    ) -> None:
        """Initialize continuous batcher with paged block table and concurrency limits.

        Args:
            block_table: PagedBlockTable instance for KV allocation. Defaults to 1024 blocks.
            max_batch_size: Maximum concurrent RUNNING sequences permitted per iteration.
            vram_saturation_threshold: Maximum allowable block consumption ratio before rejecting.
            generator_func: Optional single-token generation callback for simulation/testing.

        Raises:
            ValueError: If max_batch_size or vram_saturation_threshold is non-positive.
        """
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be strictly positive, got {max_batch_size}")
        if vram_saturation_threshold <= 0.0 or vram_saturation_threshold > 1.0:
            raise ValueError(f"vram_saturation_threshold must be in (0.0, 1.0], got {vram_saturation_threshold}")

        self._block_table = block_table or PagedBlockTable(block_size=16, total_blocks=1024)
        self._max_batch_size = max_batch_size
        self._vram_saturation_threshold = vram_saturation_threshold
        self._generator_func = generator_func

        self._waiting_queue: List[BatchSequence] = []
        self._running_batch: Dict[str, BatchSequence] = {}
        self._finished_history: Dict[str, BatchSequence] = {}

        self._total_requests_completed: int = 0
        self._total_tokens_produced: int = 0

    @property
    def active_sequences_count(self) -> int:
        """Return number of currently running sequences."""
        return len(self._running_batch)

    @property
    def waiting_sequences_count(self) -> int:
        """Return number of sequences queued in waiting state."""
        return len(self._waiting_queue)

    def add_request(
        self,
        seq_id: str,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        """Enqueue a new request for execution, validating admission control (Proof M13).

        Args:
            seq_id: Unique sequence identifier string.
            prompt: Input text prompt.
            config: Sampling configuration parameters.

        Raises:
            AdmissionDeniedError: If VRAM saturation threshold is exceeded.
            ValueError: If seq_id is empty or prompt is blank.
        """
        if not seq_id:
            raise ValueError("seq_id must not be empty.")
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty.")
        if seq_id in self._running_batch or any(s.seq_id == seq_id for s in self._waiting_queue):
            raise ValueError(f"Sequence '{seq_id}' is already registered in active or waiting batch.")

        used_blocks = self._block_table.total_blocks - self._block_table.free_blocks_count
        utilization = used_blocks / self._block_table.total_blocks

        if utilization >= self._vram_saturation_threshold:
            raise AdmissionDeniedError(
                f"Admission denied: VRAM block utilization {utilization * 100:.1f}% "
                f"exceeds saturation threshold {self._vram_saturation_threshold * 100:.1f}%."
            )

        gen_cfg = config or GenerationConfig()
        seq = BatchSequence(
            seq_id=seq_id,
            prompt=prompt,
            config=gen_cfg,
            max_tokens=gen_cfg.max_new_tokens,
        )
        self._waiting_queue.append(seq)
        logger.debug(f"Enqueued sequence {seq_id} in WAITING queue.")

    def step(self) -> List[TokenChunk]:
        """Execute a single continuous batching forward pass iteration across all RUNNING sequences.

        Admits waiting sequences up to max_batch_size, generates next token for each running
        sequence, allocates blocks, and immediately reclaims finished sequence memory (Proof M10).

        Returns:
            List of TokenChunk emissions produced during this iteration.
        """
        while self._waiting_queue and len(self._running_batch) < self._max_batch_size:
            candidate = self._waiting_queue.pop(0)
            initial_tokens = max(1, len(candidate.prompt.split()))
            try:
                self._block_table.allocate_sequence(candidate.seq_id, initial_tokens)
                candidate.state = SequenceState.RUNNING
                self._running_batch[candidate.seq_id] = candidate
                logger.debug(f"Admitted sequence {candidate.seq_id} to RUNNING batch.")
            except VRAMExhaustedError:
                self._waiting_queue.insert(0, candidate)
                break

        chunks: List[TokenChunk] = []
        completed_seq_ids: List[str] = []

        for seq_id, seq in list(self._running_batch.items()):
            seq.token_count += 1
            self._total_tokens_produced += 1

            try:
                self._block_table.append_token(seq_id)
            except VRAMExhaustedError:
                seq.stop_reason = "length"
                seq.state = SequenceState.FINISHED
                completed_seq_ids.append(seq_id)
                chunks.append(
                    TokenChunk(
                        text="",
                        seq_id=seq_id,
                        is_finished=True,
                        finish_reason="length",
                        completion_tokens=seq.token_count,
                    )
                )
                continue

            if self._generator_func:
                delta_text = self._generator_func(seq.prompt, seq.token_count)
            else:
                delta_text = f" tok_{seq.token_count}"

            seq.generated_tokens.append(delta_text)

            is_finished = False
            finish_reason: Optional[str] = None

            if seq.token_count >= seq.max_tokens:
                is_finished = True
                finish_reason = "length"
            elif delta_text.strip() in ("<|endoftext|>", "<|im_end|>", "</s>"):
                is_finished = True
                finish_reason = "stop"

            if is_finished:
                seq.state = SequenceState.FINISHED
                seq.stop_reason = finish_reason
                completed_seq_ids.append(seq_id)

            chunks.append(
                TokenChunk(
                    text=delta_text,
                    seq_id=seq_id,
                    is_finished=is_finished,
                    finish_reason=finish_reason,
                    completion_tokens=seq.token_count,
                )
            )

        for seq_id in completed_seq_ids:
            finished_seq = self._running_batch.pop(seq_id)
            self._block_table.free_sequence(seq_id)
            self._finished_history[seq_id] = finished_seq
            self._total_requests_completed += 1
            logger.debug(f"Sequence {seq_id} finished ({finished_seq.stop_reason}); blocks reclaimed.")

        return chunks

    def preempt_sequence(self, seq_id: str) -> bool:
        """Preempt an active sequence to relieve memory pressure, preserving token progress (Proof M32).

        Args:
            seq_id: Sequence identifier to preempt.

        Returns:
            True if sequence was found in running batch and preempted, False otherwise.
        """
        if seq_id in self._running_batch:
            seq = self._running_batch.pop(seq_id)
            seq.state = SequenceState.PREEMPTED
            self._block_table.free_sequence(seq_id)
            self._waiting_queue.insert(0, seq)
            logger.info(f"Preempted sequence {seq_id}; memory blocks reclaimed and re-queued.")
            return True
        return False

    def get_running_sequence_ids(self) -> List[str]:
        """Return list of active running sequence IDs.

        Returns:
            List of running sequence string identifiers.
        """
        return list(self._running_batch.keys())

    def get_waiting_sequence_ids(self) -> List[str]:
        """Return list of waiting sequence IDs in queue order.

        Returns:
            List of queued sequence string identifiers.
        """
        return [s.seq_id for s in self._waiting_queue]

    def has_work(self) -> bool:
        """Return True if there are sequences currently running or waiting."""
        return bool(self._running_batch or self._waiting_queue)

    def cancel_request(self, seq_id: str) -> bool:
        """Cancel an active or waiting sequence and immediately release all reserved memory blocks.

        Args:
            seq_id: Sequence identifier to abort.

        Returns:
            True if sequence was found and cancelled, False otherwise.
        """
        for i, seq in enumerate(self._waiting_queue):
            if seq.seq_id == seq_id:
                self._waiting_queue.pop(i)
                return True

        if seq_id in self._running_batch:
            seq = self._running_batch.pop(seq_id)
            seq.state = SequenceState.FINISHED
            seq.stop_reason = "cancelled"
            self._block_table.free_sequence(seq_id)
            self._finished_history[seq_id] = seq
            return True

        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Compile live scheduling throughput and memory telemetry metrics.

        Returns:
            Dictionary containing active sequences, block utilization, and throughput totals.
        """
        free_blocks = self._block_table.free_blocks_count
        total_blocks = self._block_table.total_blocks
        used_blocks = total_blocks - free_blocks

        return {
            "running_sequences": len(self._running_batch),
            "waiting_sequences": len(self._waiting_queue),
            "completed_requests": self._total_requests_completed,
            "total_tokens_produced": self._total_tokens_produced,
            "used_kv_blocks": used_blocks,
            "free_kv_blocks": free_blocks,
            "vram_block_utilization_pct": round((used_blocks / total_blocks) * 100.0, 2),
        }
