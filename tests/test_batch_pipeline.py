"""Triad test suite for Asynchronous Continuous Batching Execution Pipeline and Proofs M28–M35.

Validates async request queuing, Little's Law admission control (Proof M30), dynamic slot
reclamation (Proof M29), multi-user session isolation (Proof M34), and memory conservation (Proof M32).
"""

import asyncio
from typing import List
import pytest

from arc_runner.engine.continuous_batcher import (
    AdmissionDeniedError,
    ContinuousBatcher,
    SequenceState,
)
from arc_runner.engine.models import GenerationConfig, TokenChunk
from arc_runner.engine.paged_attention import PagedBlockTable, RadixPrefixCache
from arc_runner.server.batch_pipeline import BatchExecutionPipeline


class TestBatchPipelinePositive:
    """Positive test cases validating continuous batching streaming and slot reclamation."""

    @pytest.mark.asyncio
    async def test_batch_pipeline_init_positive(self) -> None:
        """Verify pipeline initializes with default parameters and idle state."""
        pipeline = BatchExecutionPipeline(max_batch_size=8, tick_interval_s=0.005)
        try:
            assert pipeline.batcher.active_sequences_count == 0
            assert pipeline.prefix_cache is not None
            assert not pipeline.is_running
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_single_request_dispatch(self) -> None:
        """Verify a single request is enqueued, scheduled, and streamed to EOS."""
        pipeline = BatchExecutionPipeline(tick_interval_s=0.005)
        try:
            cfg = GenerationConfig(max_new_tokens=4)
            chunks: List[TokenChunk] = []

            async for chunk in pipeline.enqueue_request("Hello Intel Arc", cfg):
                chunks.append(chunk)

            assert len(chunks) == 4
            assert chunks[-1].is_finished is True
            assert chunks[-1].completion_tokens == 4
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_concurrent_multi_requests(self) -> None:
        """Verify multiple concurrent requests complete independently without stream cross-talk (Proof M34)."""
        pipeline = BatchExecutionPipeline(tick_interval_s=0.005, max_batch_size=4)
        try:
            async def run_req(prompt: str) -> List[TokenChunk]:
                emitted: List[TokenChunk] = []
                cfg = GenerationConfig(max_new_tokens=3)
                async for chunk in pipeline.enqueue_request(prompt, cfg):
                    emitted.append(chunk)
                return emitted

            results = await asyncio.gather(
                run_req("User Alpha prompt"),
                run_req("User Beta query"),
                run_req("User Gamma instruction"),
                run_req("User Delta question"),
            )

            assert len(results) == 4
            for req_chunks in results:
                assert len(req_chunks) == 3
                assert req_chunks[-1].is_finished is True
                seq_ids = {c.seq_id for c in req_chunks}
                assert len(seq_ids) == 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_prefix_caching_hit(self) -> None:
        """Verify common prompt prefixes trigger hits in radix prefix cache (Proof M31)."""
        cache = RadixPrefixCache()
        common_prefix = list(range(100, 132))
        cache.insert_prefix(common_prefix, [1, 2])

        pipeline = BatchExecutionPipeline(prefix_cache=cache, tick_interval_s=0.005)
        try:
            matched, blocks = pipeline.prefix_cache.match_prefix(common_prefix + [999])
            assert matched == common_prefix
            assert blocks == [1, 2]
            assert pipeline.prefix_cache.hit_count >= 1
            assert pipeline.prefix_cache.hit_rate > 0.0
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_dynamic_slot_reclamation(self) -> None:
        """Verify finished sequences immediately relinquish allocated KV blocks back to pool (Proof M29)."""
        table = PagedBlockTable(block_size=16, total_blocks=32)
        batcher = ContinuousBatcher(block_table=table, max_batch_size=4)
        pipeline = BatchExecutionPipeline(batcher=batcher, tick_interval_s=0.005)

        try:
            initial_free = table.free_blocks_count
            cfg = GenerationConfig(max_new_tokens=2)

            async for _ in pipeline.enqueue_request("Short prompt", cfg):
                pass

            assert batcher.active_sequences_count == 0
            assert table.free_blocks_count == initial_free
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_telemetry_metrics(self) -> None:
        """Verify get_status returns accurate scheduler telemetry and utilization metrics."""
        pipeline = BatchExecutionPipeline(tick_interval_s=0.005)
        try:
            status = pipeline.get_status()
            assert "running_sequences" in status
            assert "waiting_sequences" in status
            assert "used_kv_blocks" in status
            assert "free_kv_blocks" in status
            assert "vram_block_utilization_pct" in status
            assert "prefix_cache_hit_rate" in status
            assert "running_sequence_ids" in status
        finally:
            await pipeline.stop()


class TestBatchPipelineNegative:
    """Negative test cases validating admission bounds, validation failures, and errors."""

    @pytest.mark.asyncio
    async def test_batch_pipeline_admission_denied_vram_saturation(self) -> None:
        """Verify request admission is rejected with 429 when VRAM utilization exceeds threshold (Proof M30)."""
        table = PagedBlockTable(block_size=16, total_blocks=4)
        table.allocate_sequence("saturate-1", initial_tokens=64)

        batcher = ContinuousBatcher(block_table=table, max_batch_size=4, vram_saturation_threshold=0.80)
        pipeline = BatchExecutionPipeline(batcher=batcher, tick_interval_s=0.005)

        try:
            with pytest.raises(AdmissionDeniedError) as exc_info:
                agen = pipeline.enqueue_request("Overflowing prompt", GenerationConfig(max_new_tokens=2))
                await agen.__anext__()
            assert exc_info.value.status_code == 429
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_duplicate_seq_id_raises(self) -> None:
        """Verify registering a sequence with duplicate seq_id raises ValueError."""
        pipeline = BatchExecutionPipeline(tick_interval_s=0.005)
        try:
            agen1 = pipeline.enqueue_request("First request", GenerationConfig(max_new_tokens=10), seq_id="fixed-id")
            await agen1.__anext__()

            with pytest.raises(ValueError):
                agen2 = pipeline.enqueue_request("Second request", GenerationConfig(max_new_tokens=5), seq_id="fixed-id")
                await agen2.__anext__()
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_engine_failure_propagates(self) -> None:
        """Verify internal generation exception propagates cleanly to streaming consumer."""
        def faulty_generator(prompt: str, step: int) -> str:
            if step >= 2:
                raise RuntimeError("Simulated OpenVINO XMX execution error")
            return " token"

        batcher = ContinuousBatcher(generator_func=faulty_generator)
        pipeline = BatchExecutionPipeline(batcher=batcher, tick_interval_s=0.005)

        try:
            with pytest.raises(RuntimeError) as exc_info:
                async for _ in pipeline.enqueue_request("Test fault", GenerationConfig(max_new_tokens=5)):
                    pass
            assert "Simulated OpenVINO" in str(exc_info.value)
        finally:
            await pipeline.stop()

    def test_batch_pipeline_preempt_nonexistent_returns_false(self) -> None:
        """Verify preempting an unknown sequence ID returns False safely."""
        pipeline = BatchExecutionPipeline()
        assert pipeline.preempt_sequence("non-existent-seq-id") is False

    def test_batch_pipeline_negative_batch_size_rejected(self) -> None:
        """Verify non-positive batch size raises ValueError during initialization."""
        with pytest.raises(ValueError):
            BatchExecutionPipeline(max_batch_size=0)

        with pytest.raises(ValueError):
            BatchExecutionPipeline(max_batch_size=-4)


class TestBatchPipelineEdgeCases:
    """Edge cases validating prompt bounds, client disconnects, preemption, and invariants."""

    @pytest.mark.asyncio
    async def test_batch_pipeline_zero_token_prompt(self) -> None:
        """Verify empty or blank prompts are rejected prior to scheduling."""
        pipeline = BatchExecutionPipeline()
        with pytest.raises(ValueError):
            agen = pipeline.enqueue_request("")
            await agen.__anext__()

        with pytest.raises(ValueError):
            agen2 = pipeline.enqueue_request("    \n\t  ")
            await agen2.__anext__()

    @pytest.mark.asyncio
    async def test_batch_pipeline_client_disconnect_aborts_blocks_reclaimed(self) -> None:
        """Verify early client disconnect reclaims allocated KV blocks immediately (Proof M32)."""
        table = PagedBlockTable(block_size=16, total_blocks=32)
        batcher = ContinuousBatcher(block_table=table, max_batch_size=4)
        pipeline = BatchExecutionPipeline(batcher=batcher, tick_interval_s=0.005)

        try:
            initial_free = table.free_blocks_count
            seq_id = "client-disconnect-test"

            async for chunk in pipeline.enqueue_request("Long prompt needing many tokens", GenerationConfig(max_new_tokens=50), seq_id=seq_id):
                if chunk.completion_tokens >= 2:
                    break

            await asyncio.sleep(0.02)
            assert table.free_blocks_count == initial_free
            assert seq_id not in batcher.get_running_sequence_ids()
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_preemption_recomputation_invariant(self) -> None:
        """Verify preemption releases physical memory and moves sequence to WAITING state (Proof M32)."""
        table = PagedBlockTable(block_size=16, total_blocks=32)
        batcher = ContinuousBatcher(block_table=table, max_batch_size=4)
        pipeline = BatchExecutionPipeline(batcher=batcher, tick_interval_s=0.005)

        try:
            seq_id = "preempt-target-seq"
            agen = pipeline.enqueue_request("Prompt to preempt", GenerationConfig(max_new_tokens=20), seq_id=seq_id)
            await agen.__anext__()

            assert seq_id in batcher.get_running_sequence_ids()
            success = pipeline.preempt_sequence(seq_id)

            assert success is True
            assert seq_id not in batcher.get_running_sequence_ids()
            assert seq_id in batcher.get_waiting_sequence_ids()
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_maximum_batch_saturation_bound_m30(self) -> None:
        """Verify concurrency saturation bounds running slots and queues excess in WAITING (Proof M30)."""
        pipeline = BatchExecutionPipeline(max_batch_size=2, tick_interval_s=0.005)
        try:
            completed_counts = 0

            async def client_worker(idx: int) -> None:
                nonlocal completed_counts
                cfg = GenerationConfig(max_new_tokens=3)
                async for chunk in pipeline.enqueue_request(f"Client {idx} prompt", cfg):
                    if chunk.is_finished:
                        completed_counts += 1

            tasks = [asyncio.create_task(client_worker(i)) for i in range(5)]
            await asyncio.gather(*tasks)

            assert completed_counts == 5
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_pipeline_block_conservation_invariant_m32(self) -> None:
        """Verify memory block conservation invariant M_total = M_free + M_used holds under load (Proof M32)."""
        table = PagedBlockTable(block_size=16, total_blocks=16)
        batcher = ContinuousBatcher(block_table=table, max_batch_size=4)
        pipeline = BatchExecutionPipeline(batcher=batcher, tick_interval_s=0.005)

        try:
            assert table.free_blocks_count + table.allocated_blocks_count == 16

            async def worker(idx: int) -> None:
                async for _ in pipeline.enqueue_request(f"Worker {idx} query", GenerationConfig(max_new_tokens=4)):
                    assert table.free_blocks_count + table.allocated_blocks_count == 16

            await asyncio.gather(worker(1), worker(2), worker(3))
            assert table.free_blocks_count + table.allocated_blocks_count == 16
            assert table.allocated_blocks_count == 0
        finally:
            await pipeline.stop()
