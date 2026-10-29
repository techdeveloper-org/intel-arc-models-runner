"""Slice S-A regression tests: telemetry and benchmark output must never contain invented values."""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest import mock

import pytest

from arc_runner.devices.sysman import LevelZeroSysman
from arc_runner.engine.models import GenerationConfig
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def _sysman_with_fake_gpu(total_bytes: int) -> LevelZeroSysman:
    """Build a LevelZeroSysman bound to a fake loader exposing one memory module."""
    sysman = object.__new__(LevelZeroSysman)
    lib = mock.MagicMock()

    def get_memory_properties(_dev: Any, count_ref: Any, props: Any) -> int:
        if props is None:
            count_ref._obj.value = 1
            return 0
        props[0].totalSize = total_bytes
        return 0

    lib.zeDeviceGetMemoryProperties.side_effect = get_memory_properties
    sysman._lib = lib
    sysman._initialized = True
    sysman._devices = [1]
    sysman._unavailable_reasons = {}
    return sysman


def _engine_with_pipeline(pipeline: Any) -> OpenVINOGenAIEngine:
    """Build an engine in the loaded state around a fake pipeline."""
    engine = OpenVINOGenAIEngine()
    engine._pipeline = pipeline
    engine._is_loaded = True
    engine._model_path = "fake-model"
    engine._device = "CPU"
    return engine


class _BrokenPerfMetrics:
    """perf_metrics object whose accessors all fail."""

    def get_throughput(self) -> Any:
        raise RuntimeError("no metrics")

    def get_ttft(self) -> Any:
        raise RuntimeError("no metrics")

    def get_num_generated_tokens(self) -> int:
        raise RuntimeError("no metrics")

    def get_generate_duration(self) -> Any:
        raise RuntimeError("no metrics")


class _GoodPerfMetrics:
    """perf_metrics object returning fixed, known values."""

    def get_throughput(self) -> Any:
        return SimpleNamespace(mean=40.0)

    def get_ttft(self) -> Any:
        return SimpleNamespace(mean=123.0)

    def get_num_generated_tokens(self) -> int:
        return 20

    def get_generate_duration(self) -> Any:
        return SimpleNamespace(mean=500.0)


class _FakePipeline:
    """Pipeline returning a canned result from generate()."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def generate(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._result


def test_query_package_power_without_power_api_returns_none() -> None:
    """L1: power must be None, never a constant."""
    sysman = _sysman_with_fake_gpu(8 * GIB)
    assert sysman.query_package_power(0) is None


def test_get_unavailable_reason_after_power_query_explains_none() -> None:
    """L1: a None power reading carries a human-readable reason."""
    sysman = _sysman_with_fake_gpu(8 * GIB)
    sysman.query_package_power(0)
    reason = sysman.get_unavailable_reason("package_power")
    assert reason


def test_query_gpu_memory_reports_total_only_never_invented_free() -> None:
    """L2: free memory must be None, not total//2 or system RAM."""
    sysman = _sysman_with_fake_gpu(8 * GIB)
    total_mb, free_mb = sysman.query_gpu_memory(0)
    assert total_mb == 8 * 1024
    assert free_mb is None


def test_query_gpu_memory_unavailable_free_has_reason() -> None:
    """L2: the missing free-memory reading carries a reason."""
    sysman = _sysman_with_fake_gpu(8 * GIB)
    sysman.query_gpu_memory(0)
    assert sysman.get_unavailable_reason("gpu_free_memory")


def test_sysman_module_has_no_system_ram_probe() -> None:
    """L2: system RAM must not be reported as GPU memory."""
    text = (REPO_ROOT / "src/arc_runner/devices/sysman.py").read_text(encoding="utf-8")
    assert "GlobalMemoryStatusEx" not in text
    assert "_MemoryStatusEx" not in text


def test_generate_with_broken_perf_metrics_reports_unavailable() -> None:
    """E1: failing perf_metrics must yield zeros and valid=False, not wall-clock guesses."""
    result = SimpleNamespace(texts=["hello world again"], perf_metrics=_BrokenPerfMetrics())
    engine = _engine_with_pipeline(_FakePipeline(result))
    text = engine.generate("prompt", GenerationConfig(max_new_tokens=8))
    telemetry = engine.get_telemetry()
    assert text == "hello world again"
    assert telemetry.metrics_source == "unavailable"
    assert telemetry.is_ttft_valid is False
    assert telemetry.is_tps_valid is False
    assert telemetry.ttft_ms == 0.0
    assert telemetry.tokens_per_second == 0.0
    assert telemetry.total_tokens == 0
    assert telemetry.total_latency_ms > 0.0


def test_generate_with_plain_string_result_reports_unavailable() -> None:
    """E1: a result without perf_metrics must not produce word-count token estimates."""
    engine = _engine_with_pipeline(_FakePipeline("one two three four"))
    engine.generate("prompt", GenerationConfig(max_new_tokens=8))
    telemetry = engine.get_telemetry()
    assert telemetry.metrics_source == "unavailable"
    assert telemetry.total_tokens == 0
    assert telemetry.ttft_ms == 0.0


def test_generate_with_real_perf_metrics_passes_values_through() -> None:
    """E1: real perf_metrics are reported unchanged and flagged valid."""
    result = SimpleNamespace(texts=["ok"], perf_metrics=_GoodPerfMetrics())
    engine = _engine_with_pipeline(_FakePipeline(result))
    engine.generate("prompt", GenerationConfig(max_new_tokens=8))
    telemetry = engine.get_telemetry()
    assert telemetry.metrics_source == "ov_perf_metrics"
    assert telemetry.is_ttft_valid is True
    assert telemetry.is_tps_valid is True
    assert telemetry.ttft_ms == 123.0
    assert telemetry.tokens_per_second == 40.0
    assert telemetry.total_tokens == 20


class _SilentStreamPipeline:
    """Pipeline whose streamer callback is never invoked."""

    def generate(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_generate_stream_without_tokens_reports_no_ttft() -> None:
    """E1: with no emitted token there is no TTFT; total*500 must not be invented."""
    engine = _engine_with_pipeline(_SilentStreamPipeline())
    chunks: List[Any] = list(engine.generate_stream("prompt", GenerationConfig(max_new_tokens=4)))
    telemetry = engine.get_telemetry()
    assert chunks[-1].is_finished
    assert telemetry.ttft_ms == 0.0
    assert telemetry.is_ttft_valid is False
    assert telemetry.metrics_source == "unavailable"


class _StreamingPipeline:
    """Pipeline that emits three chunks through the streamer callback."""

    def generate(self, *_args: Any, streamer: Any = None, **_kwargs: Any) -> None:
        for piece in ("a", "b", "c"):
            streamer(piece)


def test_generate_stream_with_tokens_labels_wallclock_estimate() -> None:
    """E6: stream metrics come from wall-clock callbacks and are labelled as estimates."""
    engine = _engine_with_pipeline(_StreamingPipeline())
    list(engine.generate_stream("prompt", GenerationConfig(max_new_tokens=4)))
    telemetry = engine.get_telemetry()
    assert telemetry.metrics_source == "wallclock_stream"
    assert telemetry.is_estimate is True
    assert telemetry.is_ttft_valid is True


def test_benchmark_json_contains_no_reference_sweep_rows() -> None:
    """H1: results JSON must not hold projected rows presented as results."""
    path = REPO_ROOT / "scratch" / "live_real_models_benchmark_results.json"
    assert "REFERENCE_SWEEP" not in path.read_text(encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


_FABRICATION_PATTERNS = [
    r"return 22\.0",
    r"elapsed_sec \* (200|500)",
    r"elapsed \* (200|500)",
    r"total_sec \* (200|400|500)",
    r"elapsed_sec \* 400",
    r"int\((approx_tokens|generated_count|draft_tokens) \* 0\.(8|72)\)",
    r"duration_ns \* 0\.(2|8)\b",
    r"hash\(word\) % 32000",
    r"acceptance_rate \* 0\.9",
    r"REFERENCE_SWEEP",
    r'"power_w":\s*22\.0',
    r"total_(sec|latency_ms) \* (850|0\.82|0\.06|0\.12)\b",
    r"power_watts\s*=\s*\d",
    r'"power_watts":\s*\d',
    r"package_power_w\s*=\s*\d",
    r"cloud_cost_saved\s*=\s*0\.",
    r"audio_rtf=0\.\d",
    r"base_step_ms\s*=",
    r"cpu_factor\s*=",
    r"target_it_s|target_rtf",
    r"len\((out|output|result)[a-z_]*\.split\(\)\)",
    r"warp_error\s*=\s*0\.\d",
    r"ssim_coherence\s*=\s*0\.\d",
]


@pytest.mark.parametrize("relative_dir", ["src", "scratch"])
def test_source_tree_has_no_fabricated_constants(relative_dir: str) -> None:
    """H1/E1/L1: known fabrication patterns must not exist in code or scripts."""
    offenders: List[str] = []
    for path in (REPO_ROOT / relative_dir).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _FABRICATION_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {pattern}")
    assert offenders == []


def _speculative_engine(pipeline: Any) -> Any:
    """Build a HybridSpeculativeEngine in the loaded state around a fake pipeline."""
    from arc_runner.engine.models import SpeculativeConfig
    from arc_runner.engine.speculative_engine import HybridSpeculativeEngine

    engine = HybridSpeculativeEngine()
    engine._pipeline = pipeline
    engine._is_loaded = True
    engine._model_path = "fake-model"
    engine._device = "GPU"
    engine._draft_model_path = "fake-draft"
    engine._draft_device = "NPU"
    engine._spec_config = SpeculativeConfig(draft_model_path="fake-draft")
    return engine


def test_speculative_generate_never_invents_acceptance_or_speedup() -> None:
    """E1: acceptance, draft counts and speedup must be zero/neutral when not measured."""
    engine = _speculative_engine(_FakePipeline("one two three four five six"))
    engine.generate("prompt")
    telemetry = engine.get_telemetry()
    assert telemetry.is_speculative is True
    assert telemetry.metrics_source == "unavailable"
    assert telemetry.total_tokens == 0
    assert telemetry.ttft_ms == 0.0
    assert telemetry.draft_tokens_total == 0
    assert telemetry.accepted_tokens_total == 0
    assert telemetry.acceptance_rate == 0.0
    assert telemetry.speculative_speedup == 1.0


def test_speculative_stream_without_tokens_reports_no_ttft() -> None:
    """E1: speculative streaming without a token has no TTFT and no acceptance data."""
    engine = _speculative_engine(_SilentStreamPipeline())
    list(engine.generate_stream("prompt", GenerationConfig(max_new_tokens=4)))
    telemetry = engine.get_telemetry()
    assert telemetry.ttft_ms == 0.0
    assert telemetry.is_ttft_valid is False
    assert telemetry.acceptance_rate == 0.0
    assert telemetry.speculative_speedup == 1.0
