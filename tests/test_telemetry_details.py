"""Slice S-D tests: documented-counter telemetry, tokenizer ids, and labelling of the synthetic flash pipeline."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from arc_runner.engine.flash_stream.engine import FlashStreamInferenceEngine
from arc_runner.engine.models import GenerationConfig
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
from arc_runner.engine.telemetry import _decode_rate, telemetry_from_perf_metrics
from tests.test_flash_stream_triad import build_synthetic_flash_gguf


class _Metrics:
    """perf_metrics double exposing the documented OpenVINO counters."""

    def get_throughput(self) -> Any:
        return SimpleNamespace(mean=30.0)

    def get_ttft(self) -> Any:
        return SimpleNamespace(mean=100.0)

    def get_tpot(self) -> Any:
        return SimpleNamespace(mean=25.0)

    def get_num_generated_tokens(self) -> int:
        return 21

    def get_num_input_tokens(self) -> int:
        return 17

    def get_generate_duration(self) -> Any:
        return SimpleNamespace(mean=600.0)


def test_telemetry_from_perf_metrics_reports_prompt_tokens_tpot_and_derived_decode_rate() -> None:
    """E5: engine counters are copied; the decode rate is derived as (n-1)/(duration-TTFT)."""
    telemetry = telemetry_from_perf_metrics(_Metrics(), "m", "GPU", 0.6)
    assert telemetry.prompt_tokens == 17
    assert telemetry.tpot_ms == 25.0
    assert telemetry.tokens_per_second == 30.0
    assert telemetry.decode_tokens_per_second == 40.0


def test_decode_rate_returns_zero_when_not_derivable() -> None:
    """E5: fewer than two tokens or no decode time yields 0.0, never a guess."""
    assert _decode_rate(1, 500.0, 100.0) == 0.0
    assert _decode_rate(10, 100.0, 100.0) == 0.0
    assert _decode_rate(10, 50.0, 100.0) == 0.0


def test_telemetry_from_perf_metrics_tolerates_missing_optional_counters() -> None:
    """Optional counters absent from older runtimes leave their fields at zero."""

    class _Minimal(_Metrics):
        def get_tpot(self) -> Any:
            raise AttributeError

        def get_num_input_tokens(self) -> int:
            raise AttributeError

    telemetry = telemetry_from_perf_metrics(_Minimal(), "m", "GPU", 0.6)
    assert telemetry.prompt_tokens == 0 and telemetry.tpot_ms == 0.0
    assert telemetry.is_ttft_valid is True


def test_tokenize_returns_pipeline_token_ids_and_none_when_unloaded_or_failing() -> None:
    """H7: real token ids are exposed; no loaded model or a tokenizer error yields None, never invented ids."""
    engine = OpenVINOGenAIEngine()
    assert engine.tokenize("hello") is None
    pipeline = mock.MagicMock()
    pipeline.get_tokenizer.return_value.encode.return_value = SimpleNamespace(
        input_ids=SimpleNamespace(data=np.array([[5, 6, 7]], dtype=np.int64))
    )
    engine._pipeline = pipeline
    engine._is_loaded = True
    assert engine.tokenize("hello") == [5, 6, 7]
    pipeline.get_tokenizer.side_effect = RuntimeError("no tokenizer")
    assert engine.tokenize("hello") is None


def test_flash_stream_engine_telemetry_is_labelled_synthetic(tmp_path: Path) -> None:
    """H8: the flash-stream demonstrator emits canned tokens, so its telemetry says so."""
    model = build_synthetic_flash_gguf(tmp_path / "flash.gguf", num_layers=2)
    engine = FlashStreamInferenceEngine()
    try:
        engine.load_model(model)
        list(engine.generate_stream("hello", GenerationConfig(max_new_tokens=2)))
        telemetry = engine.get_telemetry()
        assert telemetry.metrics_source == "synthetic_pipeline"
        assert telemetry.is_estimate is True
    finally:
        engine.unload()


def test_telemetry_from_stream_uses_tokenizer_count_when_given() -> None:
    """E6: a tokenizer count replaces the chunk count; the rate is (tokens - 1) over the time after the first token."""
    from arc_runner.engine.telemetry import telemetry_from_stream

    t = telemetry_from_stream("m", "GPU", 0.0, 3.0, 1.0, chunk_count=3, token_count=21)
    assert t.total_tokens == 21
    assert t.tokens_per_second == 10.0
    assert t.is_estimate is True and t.metrics_source == "wallclock_stream"


def test_telemetry_from_stream_without_tokenizer_count_keeps_chunk_count() -> None:
    """E6: with no tokenizer the chunk count is used, as before, and stays labelled an estimate."""
    from arc_runner.engine.telemetry import telemetry_from_stream

    t = telemetry_from_stream("m", "GPU", 0.0, 3.0, 1.0, chunk_count=4)
    assert t.total_tokens == 4 and t.tokens_per_second == 2.0


def test_generate_stream_counts_tokens_with_the_pipeline_tokenizer() -> None:
    """E6: streamed total_tokens comes from perf_metrics when available, not re-tokenization.

    When the pipeline result carries perf_metrics.get_num_generated_tokens(), that value is
    used and is_estimate is set to False.  When absent, chunk_count is used as the fallback.
    """
    from types import SimpleNamespace

    from arc_runner.engine.models import GenerationConfig
    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    class _Pipe:
        def generate(self, *_a: Any, streamer: Any = None, **_k: Any) -> Any:
            for piece in ("ab", "cd", "ef"):
                streamer(piece)
            pm = SimpleNamespace(get_num_generated_tokens=lambda: 9)
            return SimpleNamespace(texts=["abcdef"], perf_metrics=pm)

        def get_tokenizer(self) -> Any:
            return SimpleNamespace(encode=lambda text: SimpleNamespace(
                input_ids=SimpleNamespace(shape=(1, len(text)), data=SimpleNamespace(flatten=lambda: SimpleNamespace(tolist=lambda: list(range(len(text))))))
            ))

    engine = OpenVINOGenAIEngine()
    engine._pipeline = _Pipe()
    engine._is_loaded = True
    engine._model_path = "m"
    engine._device = "CPU"
    list(engine.generate_stream("prompt", GenerationConfig(max_new_tokens=8)))
    assert engine.get_telemetry().total_tokens == 9
    assert engine.get_telemetry().is_estimate is False
