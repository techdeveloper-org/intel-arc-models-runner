"""WP-14 engine-side token accounting tests.

Red-first gate: run before implementing changes to confirm all marked tests fail.
Covers: stop_strings validator, ModelConfigError, cross-chunk stop string matching,
stream token count from perf_metrics (not re-tokenization), telemetry is_estimate=False.
"""

import threading
import time
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest

from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.models import GenerationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ov_engine(prompt_token_count: int = 5, max_context: int = 4096):
    """Return a loaded OpenVINOGenAIEngine with the pipeline mocked.

    Args:
        prompt_token_count: Fixed token count the tokenizer returns for any prompt.
        max_context: Context window size for the KV manager.

    Returns:
        Configured engine instance with mocked pipeline.
    """
    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    pipeline = MagicMock()
    pipeline.get_tokenizer.return_value.encode.return_value = SimpleNamespace(
        input_ids=SimpleNamespace(shape=(1, prompt_token_count))
    )
    pipeline.generate.return_value = "placeholder"

    engine = OpenVINOGenAIEngine(kv_manager=KVCacheManager(max_context_length=max_context))
    engine._pipeline = pipeline
    engine._is_loaded = True
    engine._model_path = "test_model"
    engine._device = "CPU"
    return engine


# ---------------------------------------------------------------------------
# 1. GenerationConfig.stop_strings validator
# ---------------------------------------------------------------------------

class TestStopStringsValidator:
    """GenerationConfig must reject stop_strings containing empty or whitespace-only elements."""

    def test_stop_strings_empty_string_rejected_ok(self):
        """stop_strings=[""] must raise a validation error."""
        with pytest.raises(Exception):
            GenerationConfig(stop_strings=[""])

    def test_stop_strings_whitespace_only_rejected_ok(self):
        """stop_strings=["   "] must raise a validation error."""
        with pytest.raises(Exception):
            GenerationConfig(stop_strings=["   "])

    def test_stop_strings_nonempty_accepted_ok(self):
        """Non-empty stop strings are accepted without error."""
        cfg = GenerationConfig(stop_strings=["STOP", "</s>"])
        assert cfg.stop_strings == ["STOP", "</s>"]

    def test_stop_strings_empty_list_accepted_ok(self):
        """An empty stop_strings list is always accepted."""
        cfg = GenerationConfig(stop_strings=[])
        assert cfg.stop_strings == []


# ---------------------------------------------------------------------------
# 2. ModelConfigError
# ---------------------------------------------------------------------------

class TestModelConfigError:
    """ModelConfigError must exist as a typed ArcRunnerError subclass."""

    def test_model_config_error_importable_ok(self):
        """ModelConfigError is importable from arc_runner.core.exceptions."""
        from arc_runner.core.exceptions import ArcRunnerError, ModelConfigError

        assert issubclass(ModelConfigError, ArcRunnerError)

    def test_model_config_error_status_code_400_ok(self):
        """ModelConfigError maps to HTTP 400."""
        from arc_runner.core.exceptions import ModelConfigError

        err = ModelConfigError("missing num_hidden_layers")
        assert err.status_code == 400

    def test_model_config_error_error_code_ok(self):
        """ModelConfigError.error_code is 'MODEL_CONFIG_ERROR'."""
        from arc_runner.core.exceptions import ModelConfigError

        err = ModelConfigError("test")
        assert err.error_code == "MODEL_CONFIG_ERROR"


# ---------------------------------------------------------------------------
# 3. KV footprint with config.json geometry
# ---------------------------------------------------------------------------

class TestKvFootprintFromConfig:
    """KVCacheManager must use geometry from config.json when supplied."""

    def test_kv_footprint_with_qwen_geometry_smaller_than_default_ok(self):
        """Qwen2.5-1.5B geometry (2 KV heads) gives smaller footprint than the 8-head default."""
        from arc_runner.engine.model_config import KVGeometry

        qwen_geom = KVGeometry(num_layers=28, num_kv_heads=2, head_dim=64)
        default_mgr = KVCacheManager(max_context_length=4096)
        qwen_mgr = KVCacheManager(max_context_length=4096, geometry=qwen_geom)

        n_tokens = 100
        default_alloc = default_mgr.reserve_allocation(n_tokens, 100)
        qwen_alloc = qwen_mgr.reserve_allocation(n_tokens, 100)

        assert qwen_alloc.memory_footprint_mb < default_alloc.memory_footprint_mb
        assert qwen_alloc.footprint_from_model_config is True
        assert default_alloc.footprint_from_model_config is False

    def test_kv_footprint_default_uses_assumed_constants_ok(self):
        """Without geometry, reserve_allocation uses the 28/8/128 assumed constants."""
        from arc_runner.engine.kv_cache import (
            _ASSUMED_HEAD_DIM,
            _ASSUMED_KV_HEADS,
            _ASSUMED_LAYERS,
        )

        mgr = KVCacheManager(max_context_length=4096)
        alloc = mgr.reserve_allocation(100, 100)
        expected_mb = mgr.calculate_footprint_mb(200, _ASSUMED_LAYERS, _ASSUMED_KV_HEADS, _ASSUMED_HEAD_DIM, 2)
        assert abs(alloc.memory_footprint_mb - expected_mb) < 0.001
        assert alloc.footprint_from_model_config is False


# ---------------------------------------------------------------------------
# 4. telemetry_from_stream is_estimate parameter
# ---------------------------------------------------------------------------

class TestTelemetryFromStreamIsEstimate:
    """telemetry_from_stream must accept an is_estimate kwarg."""

    def test_telemetry_from_stream_is_estimate_false_ok(self):
        """Passing is_estimate=False results in EngineTelemetry.is_estimate=False."""
        from arc_runner.engine.telemetry import telemetry_from_stream

        t0 = time.monotonic()
        t1 = t0 + 0.5
        t_first = t0 + 0.1
        tel = telemetry_from_stream("m", "GPU", t0, t1, t_first, 5, token_count=5, is_estimate=False)
        assert tel.is_estimate is False

    def test_telemetry_from_stream_default_is_estimate_true_ok(self):
        """telemetry_from_stream without is_estimate defaults to True."""
        from arc_runner.engine.telemetry import telemetry_from_stream

        t0 = time.monotonic()
        t1 = t0 + 0.5
        t_first = t0 + 0.1
        tel = telemetry_from_stream("m", "GPU", t0, t1, t_first, 5)
        assert tel.is_estimate is True


# ---------------------------------------------------------------------------
# 5. Cross-chunk stop string matching
# ---------------------------------------------------------------------------

class TestCrossChunkStopStrings:
    """Stop strings spanning multiple streamer chunks must halt generation."""

    def test_stop_string_within_one_chunk_still_halts_ok(self):
        """Single-chunk stop string still works after the cross-chunk fix."""
        engine = _ov_engine()

        def fake_gen(prompt, ov_config, streamer):
            streamer("Hello")
            streamer(" STOP")
            streamer(" after_stop")
            pm = MagicMock()
            pm.get_num_generated_tokens.return_value = 3
            return SimpleNamespace(texts=["Hello STOP after_stop"], perf_metrics=pm)

        engine._pipeline.generate.side_effect = fake_gen
        cfg = GenerationConfig(stop_strings=["STOP"], max_new_tokens=50)
        chunks = list(engine.generate_stream("hi", cfg))
        all_text = "".join(c.text for c in chunks if c.text)
        assert "after_stop" not in all_text

    def test_stop_string_spanning_two_chunks_halts_ok(self):
        """Stop string 'ab' split across chunks 'a' and 'b' must halt generation.

        Current per-chunk matching misses this; cross-chunk accumulator is required.
        """
        engine = _ov_engine()

        def fake_gen(prompt, ov_config, streamer):
            streamer("pre")
            streamer("a")
            streamer("b")
            streamer("after_stop")
            pm = MagicMock()
            pm.get_num_generated_tokens.return_value = 4
            return SimpleNamespace(texts=["pre a b after_stop"], perf_metrics=pm)

        engine._pipeline.generate.side_effect = fake_gen
        cfg = GenerationConfig(stop_strings=["ab"], max_new_tokens=50)
        chunks = list(engine.generate_stream("hi", cfg))
        all_text = "".join(c.text for c in chunks if c.text)
        assert "after_stop" not in all_text, (
            "Text after cross-chunk stop string 'ab' must not appear in output"
        )


# ---------------------------------------------------------------------------
# 6. Stream token count from perf_metrics — no re-tokenization
# ---------------------------------------------------------------------------

class TestStreamTokenCountFromPerfMetrics:
    """generate_stream must not re-tokenize emitted text; use perf_metrics instead."""

    def test_encode_called_at_most_once_per_stream_request_ok(self):
        """tokenizer.encode is called at most once per request (context check), not twice."""
        engine = _ov_engine()
        tokenizer = engine._pipeline.get_tokenizer()
        baseline = tokenizer.encode.call_count

        def fake_gen(prompt, ov_config, streamer):
            streamer("tok1")
            streamer("tok2")
            pm = MagicMock()
            pm.get_num_generated_tokens.return_value = 2
            return SimpleNamespace(texts=["tok1 tok2"], perf_metrics=pm)

        engine._pipeline.generate.side_effect = fake_gen
        list(engine.generate_stream("test prompt", GenerationConfig()))
        calls = tokenizer.encode.call_count - baseline
        assert calls <= 1, (
            f"encode must be called at most once (context check); got {calls} calls — "
            "re-tokenizing emitted text must be replaced by perf_metrics.get_num_generated_tokens()"
        )

    def test_stream_telemetry_is_not_estimate_when_perf_metrics_ok(self):
        """When perf_metrics.get_num_generated_tokens() is available, is_estimate is False."""
        engine = _ov_engine()

        def fake_gen(prompt, ov_config, streamer):
            streamer("hello")
            streamer("world")
            pm = MagicMock()
            pm.get_num_generated_tokens.return_value = 2
            return SimpleNamespace(texts=["hello world"], perf_metrics=pm)

        engine._pipeline.generate.side_effect = fake_gen
        list(engine.generate_stream("test", GenerationConfig()))
        assert engine.get_telemetry().is_estimate is False, (
            "Stream telemetry must not be an estimate when perf_metrics are available"
        )

    def test_stream_telemetry_is_estimate_when_no_perf_metrics_ok(self):
        """When pipeline returns no perf_metrics, is_estimate stays True (wallclock fallback)."""
        engine = _ov_engine()

        def fake_gen(prompt, ov_config, streamer):
            streamer("hello")
            return "plain string without perf_metrics attribute"

        engine._pipeline.generate.side_effect = fake_gen
        list(engine.generate_stream("test", GenerationConfig()))
        assert engine.get_telemetry().is_estimate is True
