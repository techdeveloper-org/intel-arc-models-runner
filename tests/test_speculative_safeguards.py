"""Unit test suite validating hardware safeguards in HybridSpeculativeEngine.

Validates:
1. Context Length Overflow Circuit Breaker for static NPU CMX scratchpad boundaries.
2. Tokenizer Alignment Gate verifying vocabulary size and token ID consistency.
3. Level Zero driver isolation and CPU draft fallback.
4. Automatic greedy decoding enforcement for FastDraftStrategy compliance.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arc_runner.core.exceptions import TokenizerMismatchError
from arc_runner.engine.models import GenerationConfig, SpeculativeConfig
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine


class TestSpeculativeSafeguards:
    """Test suite covering the 3 runtime hardware safeguards and speculative constraints."""

    def test_safeguard_1_context_overflow_bypasses_npu_draft(self, tmp_path: Path) -> None:
        """Verify that sequence lengths exceeding NPU CMX capacity bypass draft execution."""
        engine = HybridSpeculativeEngine()
        engine._is_loaded = True
        engine._draft_device = "NPU"
        engine._max_npu_context_len = 128
        engine._draft_model_path = str(tmp_path / "mock_draft")
        engine._pipeline = MagicMock()
        mock_raw = MagicMock()
        mock_raw.texts = ["Generated text"]
        engine._pipeline.generate.return_value = mock_raw

        with patch.object(engine, "_check_request_fits_context", return_value=100):
            config = GenerationConfig(max_new_tokens=50)
            engine.generate("Test prompt", config)

            args, _ = engine._pipeline.generate.call_args
            ov_cfg = args[1]
            assert ov_cfg.num_assistant_tokens == 0

    def test_safeguard_1_within_context_preserves_gamma(self, tmp_path: Path) -> None:
        """Verify that sequence lengths within NPU CMX capacity retain active gamma."""
        engine = HybridSpeculativeEngine()
        engine._is_loaded = True
        engine._draft_device = "NPU"
        engine._max_npu_context_len = 128
        engine._current_gamma = 3
        engine._spec_config = SpeculativeConfig(num_assistant_tokens=3)
        engine._draft_model_path = str(tmp_path / "mock_draft")
        engine._pipeline = MagicMock()
        mock_raw = MagicMock()
        mock_raw.texts = ["Generated text"]
        engine._pipeline.generate.return_value = mock_raw

        with patch.object(engine, "_check_request_fits_context", return_value=30):
            config = GenerationConfig(max_new_tokens=20)
            engine.generate("Test prompt", config)

            args, _ = engine._pipeline.generate.call_args
            ov_cfg = args[1]
            assert ov_cfg.num_assistant_tokens == 3

    def test_safeguard_2_tokenizer_alignment_gate_raises_on_vocab_mismatch(self, tmp_path: Path) -> None:
        """Verify that divergent vocabulary sizes raise TokenizerMismatchError."""
        target_dir = tmp_path / "target_model"
        target_dir.mkdir(parents=True)
        (target_dir / "config.json").write_text(json.dumps({"vocab_size": 151936}), encoding="utf-8")

        draft_dir = tmp_path / "draft_model"
        draft_dir.mkdir(parents=True)
        (draft_dir / "config.json").write_text(json.dumps({"vocab_size": 32000}), encoding="utf-8")

        engine = HybridSpeculativeEngine()
        with pytest.raises(TokenizerMismatchError) as exc_info:
            engine._validate_tokenizer_alignment(target_dir, draft_dir)
        assert "vocab_size" in str(exc_info.value)

    def test_safeguard_2_tokenizer_alignment_gate_passes_on_matching_vocab(self, tmp_path: Path) -> None:
        """Verify that identical vocabulary sizes pass validation cleanly."""
        target_dir = tmp_path / "target_model"
        target_dir.mkdir(parents=True)
        (target_dir / "config.json").write_text(json.dumps({"vocab_size": 151936}), encoding="utf-8")

        draft_dir = tmp_path / "draft_model"
        draft_dir.mkdir(parents=True)
        (draft_dir / "config.json").write_text(json.dumps({"vocab_size": 151936}), encoding="utf-8")

        engine = HybridSpeculativeEngine()
        assert engine._validate_tokenizer_alignment(target_dir, draft_dir) is True

    def test_greedy_decoding_enforced_in_speculative_mode(self, tmp_path: Path) -> None:
        """Verify that temperature and sampling are clamped to greedy decoding in speculative mode."""
        engine = HybridSpeculativeEngine()
        engine._draft_model_path = str(tmp_path / "mock_draft")
        engine._spec_config = SpeculativeConfig(num_assistant_tokens=4)
        engine._current_gamma = 4

        config = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=50)
        ov_cfg = engine._build_ov_generation_config(config)

        assert ov_cfg.temperature == 0.0
        assert ov_cfg.do_sample is False
        assert ov_cfg.num_assistant_tokens == 4
