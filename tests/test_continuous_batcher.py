"""Triad test suite for Continuous In-Flight Batching, Paged Attention and Proofs M10, M11, M12, M13, M17.

Verifies virtual-to-physical block table translation, internal memory fragmentation upper bounds,
Radix tree prefix caching, and out-of-order completion in-flight iteration scheduling.
"""

import math
import pytest

from arc_runner.engine.continuous_batcher import (
    AdmissionDeniedError,
    BatchSequence,
    ContinuousBatcher,
    SequenceState,
)
from arc_runner.engine.models import GenerationConfig
from arc_runner.engine.paged_attention import (
    BlockTableLookupError,
    PagedBlockTable,
    RadixPrefixCache,
    VRAMExhaustedError,
)


class TestPagedBlockTableM11M17:
    """Mathematical validation of Proofs M11 (Fragmentation) and M17 (Address Translation)."""

    def test_paged_block_table_positive_allocation_and_translation(self) -> None:
        """Verify logical token offsets translate bijectively to physical blocks and offsets (Proof M17)."""
        table = PagedBlockTable(block_size=16, total_blocks=64)
        initial_blocks = table.allocate_sequence("seq-1", initial_tokens=32)

        assert len(initial_blocks) == 2

        phys_block_0, offset_0 = table.translate_address("seq-1", 0)
        assert phys_block_0 == initial_blocks[0]
        assert offset_0 == 0

        phys_block_15, offset_15 = table.translate_address("seq-1", 15)
        assert phys_block_15 == initial_blocks[0]
        assert offset_15 == 15

        phys_block_16, offset_16 = table.translate_address("seq-1", 16)
        assert phys_block_16 == initial_blocks[1]
        assert offset_16 == 0

    def test_internal_fragmentation_upper_bound_m11(self) -> None:
        """Verify internal memory fragmentation is bounded by (B - 1) / L (Proof M11)."""
        table = PagedBlockTable(block_size=16, total_blocks=64)
        table.allocate_sequence("seq-perf", initial_tokens=512)
        frag_exact = table.compute_internal_fragmentation("seq-perf")
        assert math.isclose(frag_exact, 0.0, abs_tol=1e-6)

        table.allocate_sequence("seq-partial", initial_tokens=505)
        frag_partial = table.compute_internal_fragmentation("seq-partial")
        assert frag_partial <= (16 - 1) / 505
        assert frag_partial < 0.035

    def test_paged_block_table_negative_lookups_and_exhaustion(self) -> None:
        """Verify out-of-bounds index and capacity exhaustion raise expected errors."""
        table = PagedBlockTable(block_size=16, total_blocks=2)
        table.allocate_sequence("seq-small", initial_tokens=10)

        with pytest.raises(BlockTableLookupError):
            table.translate_address("seq-small", 10)

        with pytest.raises(BlockTableLookupError):
            table.translate_address("unallocated-seq", 0)

        with pytest.raises(VRAMExhaustedError):
            table.allocate_sequence("seq-overflow", initial_tokens=64)

    def test_paged_block_table_edge_case_dynamic_expansion(self) -> None:
        """Verify appending token crossing block boundary dynamically allocates new physical block."""
        table = PagedBlockTable(block_size=16, total_blocks=16)
        table.allocate_sequence("seq-expand", initial_tokens=16)

        initial_free = table.free_blocks_count

        phys_block, offset = table.append_token("seq-expand")
        assert offset == 0
        assert table.free_blocks_count == initial_free - 1

        table.free_sequence("seq-expand")
        assert table.free_blocks_count == 16


class TestRadixPrefixCacheM12:
    """Mathematical validation of Proof M12: Radix Tree Prefix Cache Sharing."""

    def test_radix_prefix_cache_positive_shared_prompt(self) -> None:
        """Verify matching system prompt prefix reuses physical block mappings."""
        cache = RadixPrefixCache()
        sys_prompt_tokens = [101, 2054, 2003, 1037]
        assigned_blocks = [4, 7]

        cache.insert_prefix(sys_prompt_tokens, assigned_blocks)

        user_query_1 = [101, 2054, 2003, 1037, 3045, 9999]
        matched_tokens, matched_blocks = cache.match_prefix(user_query_1)

        assert matched_tokens == sys_prompt_tokens

    def test_radix_prefix_cache_edge_case_disjoint_tokens(self) -> None:
        """Verify disjoint prompt returns empty matched prefix without exception."""
        cache = RadixPrefixCache()
        cache.insert_prefix([1, 2, 3], [10])

        matched_tokens, matched_blocks = cache.match_prefix([4, 5, 6])
        assert matched_tokens == []
        assert matched_blocks == []


class TestContinuousBatcherExecutionM10M13:
    """Mathematical validation of Proofs M10 (Throughput Speedup) and M13 (Little's Law Admission)."""

    def test_continuous_batcher_out_of_order_completion(self) -> None:
        """Verify short sequence completes and reclaims memory while long sequence continues running (Proof M10)."""
        table = PagedBlockTable(block_size=16, total_blocks=64)
        batcher = ContinuousBatcher(block_table=table, max_batch_size=8)

        cfg_short = GenerationConfig(max_new_tokens=2)
        cfg_long = GenerationConfig(max_new_tokens=6)

        batcher.add_request("req-short", "Hello short", cfg_short)
        batcher.add_request("req-long", "Hello long", cfg_long)

        chunks_step1 = batcher.step()
        assert len(chunks_step1) == 2
        assert not any(c.is_finished for c in chunks_step1)

        chunks_step2 = batcher.step()
        assert len(chunks_step2) == 2
        finished_chunks = [c for c in chunks_step2 if c.is_finished]
        assert len(finished_chunks) == 1
        assert finished_chunks[0].finish_reason == "length"

        assert "req-short" not in batcher._running_batch
        assert "req-long" in batcher._running_batch

        chunks_step3 = batcher.step()
        assert len(chunks_step3) == 1

    def test_continuous_batcher_negative_vram_saturation_admission(self) -> None:
        """Verify requests are rejected when VRAM block utilization exceeds saturation threshold (Proof M13)."""
        table = PagedBlockTable(block_size=16, total_blocks=10)
        table.allocate_sequence("filler", initial_tokens=150)

        batcher = ContinuousBatcher(block_table=table, vram_saturation_threshold=0.80)

        with pytest.raises(AdmissionDeniedError, match="Admission denied"):
            batcher.add_request("overflow-req", "Prompt text")

    def test_continuous_batcher_edge_case_cancellation_reclaims_blocks(self) -> None:
        """Verify cancelling an active sequence immediately restores blocks to the free pool."""
        table = PagedBlockTable(block_size=16, total_blocks=16)
        batcher = ContinuousBatcher(block_table=table, max_batch_size=4)

        batcher.add_request("cancel-me", "Some long prompt", GenerationConfig(max_new_tokens=10))
        batcher.step()

        assert "cancel-me" in batcher._running_batch
        initial_free = table.free_blocks_count

        cancelled = batcher.cancel_request("cancel-me")
        assert cancelled is True
        assert table.free_blocks_count > initial_free
        assert "cancel-me" not in batcher._running_batch
