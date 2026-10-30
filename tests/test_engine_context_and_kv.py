"""Slice S-D tests: context window enforcement, single pipeline call form, and gated KV precision."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from arc_runner.core.exceptions import ContextWindowExceededError
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.models import GenerationConfig
from arc_runner.engine.openvino_engine import KV_PRECISION_ENV, OpenVINOGenAIEngine


def _engine(prompt_tokens: Any, max_context: int = 100) -> tuple:
    """Build a loaded engine whose tokenizer reports a fixed prompt length."""
    pipeline = mock.MagicMock()
    if isinstance(prompt_tokens, int):
        pipeline.get_tokenizer.return_value.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(shape=(1, prompt_tokens)))
    else:
        pipeline.get_tokenizer.side_effect = prompt_tokens
    pipeline.generate.return_value = "ok"
    engine = OpenVINOGenAIEngine(kv_manager=KVCacheManager(max_context_length=max_context))
    engine._pipeline = pipeline
    engine._is_loaded = True
    engine._model_path = "model"
    engine._device = "CPU"
    return engine, pipeline


def test_generate_prompt_over_context_raises_before_pipeline_call() -> None:
    """E2: a prompt longer than the window is rejected and the pipeline is never invoked."""
    engine, pipeline = _engine(101)
    with pytest.raises(ContextWindowExceededError):
        engine.generate("p", GenerationConfig(max_new_tokens=1))
    pipeline.generate.assert_not_called()


def test_generate_prompt_plus_completion_over_context_raises_before_pipeline_call() -> None:
    """E2: prompt + max_new_tokens beyond the window is rejected up front."""
    engine, pipeline = _engine(90)
    with pytest.raises(ContextWindowExceededError) as excinfo:
        engine.generate("p", GenerationConfig(max_new_tokens=11))
    assert excinfo.value.prompt_tokens == 101
    pipeline.generate.assert_not_called()


def test_generate_prompt_plus_completion_exactly_fitting_context_is_allowed() -> None:
    """E2: a request that fills the window exactly is accepted."""
    engine, pipeline = _engine(90)
    assert engine.generate("p", GenerationConfig(max_new_tokens=10)) == "ok"
    pipeline.generate.assert_called_once()
    assert engine.get_telemetry().prompt_tokens == 90


def test_generate_stream_over_context_raises_before_worker_starts() -> None:
    """E2: the streaming path enforces the same window."""
    engine, pipeline = _engine(95)
    with pytest.raises(ContextWindowExceededError):
        list(engine.generate_stream("p", GenerationConfig(max_new_tokens=20)))
    pipeline.generate.assert_not_called()


def test_generate_tokenizer_unavailable_skips_enforcement_and_reports_zero_prompt_tokens() -> None:
    """E2: if the tokenizer cannot be queried the request still runs and prompt_tokens stays 0."""
    engine, pipeline = _engine(RuntimeError("no tokenizer"))
    assert engine.generate("p", GenerationConfig(max_new_tokens=5000)) == "ok"
    assert engine.get_telemetry().prompt_tokens == 0


def test_generate_empty_prompt_returns_empty_text_and_resets_telemetry() -> None:
    """E6: an empty prompt returns immediately with zeroed telemetry for the loaded model."""
    engine, pipeline = _engine(1, max_context=4096)
    engine.generate("p")
    engine.generate("")
    telemetry = engine.get_telemetry()
    assert telemetry.model_id == "model"
    assert telemetry.total_tokens == 0 and telemetry.prompt_tokens == 0
    assert pipeline.generate.call_count == 1


def test_invoke_pipeline_uses_single_list_call_form_and_propagates_errors() -> None:
    """E3: one call form (list of prompts); a real failure is not retried or swallowed."""
    engine, pipeline = _engine(1)
    pipeline.generate.side_effect = RuntimeError("device lost")
    with pytest.raises(RuntimeError, match="device lost"):
        engine.generate("hello", GenerationConfig(max_new_tokens=4))
    assert pipeline.generate.call_count == 1
    assert pipeline.generate.call_args.args[0] == ["hello"]


def test_build_ov_properties_gpu_defaults_do_not_enable_u8_kv_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E4: the u8 KV cache is opt-in, not a default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KV_PRECISION_ENV, raising=False)
    engine = OpenVINOGenAIEngine()
    props = engine._build_ov_properties("GPU", {})
    assert "KV_CACHE_PRECISION" not in props
    assert props["PERFORMANCE_HINT"] == "LATENCY"
    assert props["GPU_ENABLE_SDPA_OPTIMIZATION"] is True
    assert engine._kv_cache_precision == ""


def test_build_ov_properties_env_opt_in_and_caller_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E4: ARC_KV_CACHE_PRECISION opts in; an explicit property still wins."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(KV_PRECISION_ENV, "u8")
    engine = OpenVINOGenAIEngine()
    assert engine._build_ov_properties("GPU", {})["KV_CACHE_PRECISION"] == "u8"
    assert engine._kv_cache_precision == "u8"
    assert engine._build_ov_properties("GPU", {"KV_CACHE_PRECISION": "f16"})["KV_CACHE_PRECISION"] == "f16"


def test_build_ov_properties_cpu_gets_no_gpu_only_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU-only SDPA and cache options are not applied to CPU or NPU."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(KV_PRECISION_ENV, raising=False)
    props = OpenVINOGenAIEngine()._build_ov_properties("CPU", {})
    assert set(props) == {"CACHE_DIR"}


def _speculative_engine(prompt_tokens: int, max_context: int = 100) -> tuple:
    """Build a loaded speculative engine whose tokenizer reports a fixed prompt length."""
    from arc_runner.engine.models import SpeculativeConfig
    from arc_runner.engine.speculative_engine import HybridSpeculativeEngine

    pipeline = mock.MagicMock()
    pipeline.get_tokenizer.return_value.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(shape=(1, prompt_tokens)))
    pipeline.generate.return_value = "ok"
    engine = HybridSpeculativeEngine(kv_cache_manager=KVCacheManager(max_context_length=max_context))
    engine._pipeline = pipeline
    engine._is_loaded = True
    engine._model_path = "model"
    engine._device = "GPU"
    engine._draft_model_path = "draft"
    engine._spec_config = SpeculativeConfig(draft_model_path="draft")
    return engine, pipeline


def test_speculative_generate_prompt_plus_completion_over_context_raises_before_pipeline_call() -> None:
    """E2: the speculative engine enforces the same window as the plain engine."""
    engine, pipeline = _speculative_engine(90)
    with pytest.raises(ContextWindowExceededError):
        engine.generate("p", GenerationConfig(max_new_tokens=11))
    pipeline.generate.assert_not_called()


def test_speculative_generate_exactly_fitting_context_is_allowed_and_records_prompt_tokens() -> None:
    """E2: a request filling the window exactly runs, and prompt_tokens is reported."""
    engine, pipeline = _speculative_engine(90)
    engine.generate("p", GenerationConfig(max_new_tokens=10))
    pipeline.generate.assert_called_once()
    assert engine.get_telemetry().prompt_tokens == 90


def test_speculative_generate_stream_over_context_raises_before_worker_starts() -> None:
    """E2: speculative streaming rejects an oversized request before any worker thread runs."""
    engine, pipeline = _speculative_engine(95)
    with pytest.raises(ContextWindowExceededError):
        list(engine.generate_stream("p", GenerationConfig(max_new_tokens=20)))
    pipeline.generate.assert_not_called()
