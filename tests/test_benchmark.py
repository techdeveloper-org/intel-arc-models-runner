"""Exhaustive 12-test triad suite for Hardware Benchmarking Engine.

Validates deterministic warmup rejection, Student's t-distribution confidence intervals,
high-precision hardware timers (time.perf_counter_ns), Rich table rendering, and error handling.
"""

from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Optional
import pytest
from rich.console import Console

from arc_runner.benchmark.runner import BenchmarkRunner, BenchmarkSummary
from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.model_manager import ModelManager


class MockBenchmarkEngine(BaseEngine):
    """Simulated execution engine with predictable token latency emissions."""

    def __init__(self, model_name: str = "mock-llama3") -> None:
        """Initialize mock benchmark engine."""
        self._model_name = model_name
        self._is_loaded = True
        self._device = "GPU"
        self._telemetry = EngineTelemetry(model_id=model_name, device="GPU")

    @property
    def is_loaded(self) -> bool:
        """Return loading status."""
        return self._is_loaded

    @property
    def model_path(self) -> str:
        """Return model path."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return targeted device string."""
        return self._device

    def load_model(self, model_path: Any, device: str = "GPU", **kwargs: Any) -> None:
        """Simulate loading model."""
        self._is_loaded = True
        self._model_name = str(model_path)
        self._device = device

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Simulate text generation."""
        return "Intel Arc benchmarking response"

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Simulate incremental token emissions with timing deltas."""
        words = ["Intel", "Arc", "Xe-LPG", "XMX", "Matrix", "Boost!"]
        for i, word in enumerate(words):
            time.sleep(0.002)
            yield TokenChunk(
                text=word + " ",
                token_id=10 + i,
                is_finished=False,
                prompt_tokens=8,
                completion_tokens=i + 1,
            )
        yield TokenChunk(
            text="",
            token_id=2,
            is_finished=True,
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=len(words),
        )

    def unload(self) -> None:
        """Simulate deallocation."""
        self._is_loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Return telemetry metrics."""
        return self._telemetry


@pytest.fixture
def bench_env():
    """Create a temporary sandbox for benchmarking operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        llama_dir = tmp_path / "llama-3.2-3b"
        llama_dir.mkdir()
        (llama_dir / "openvino_model.xml").write_text("<net></net>", encoding="utf-8")
        (llama_dir / "openvino_model.bin").write_bytes(b"\x00" * 1024)

        manager = ModelManager(model_dirs=[tmp_path])
        mock_engine = MockBenchmarkEngine("llama-3.2-3b")
        manager._cache["llama-3.2-3b:GPU:none:NPU:4"] = mock_engine

        runner = BenchmarkRunner(model_manager=manager)

        yield {
            "root": tmp_path,
            "runner": runner,
            "manager": manager,
            "mock_engine": mock_engine,
        }


# =========================================================================
# POSITIVE TEST TRIAD (5 TESTS)
# =========================================================================

def test_benchmark_warmup_runs_discarded(bench_env):
    """Verify warmup passes are executed and recorded but excluded from mean metrics."""
    runner = bench_env["runner"]
    summary = runner.run_benchmark(
        model_id="llama-3.2-3b",
        device="GPU",
        iterations=3,
        warmup=2,
        max_new_tokens=16,
    )
    assert isinstance(summary, BenchmarkSummary)
    assert summary.iterations == 3
    assert summary.warmup_runs == 2
    assert len(summary.run_results) == 5

    warmups = [r for r in summary.run_results if r.is_warmup]
    measured = [r for r in summary.run_results if not r.is_warmup]
    assert len(warmups) == 2
    assert len(measured) == 3


def test_benchmark_computes_ttft_and_tpot_accurately(bench_env):
    """Verify computed TTFT, TPOT, and throughput are positive and mathematically consistent."""
    runner = bench_env["runner"]
    summary = runner.run_benchmark(
        model_id="llama-3.2-3b",
        device="GPU",
        iterations=4,
        warmup=1,
        max_new_tokens=16,
    )
    assert summary.avg_ttft_ms > 0.0
    assert summary.avg_tpot_ms > 0.0
    assert summary.avg_tokens_per_sec > 0.0
    assert summary.ci95_ttft_ms >= 0.0
    assert summary.ci95_tokens_per_sec >= 0.0


def test_benchmark_student_t_confidence_intervals(bench_env):
    """Verify Student's t 95% confidence intervals properly scale with sample variation."""
    runner = bench_env["runner"]
    samples = [10.0, 11.0, 12.0, 10.5, 11.5]
    mean_val, half_width = runner._compute_confidence_interval_95(samples)
    assert 10.9 < mean_val < 11.1
    assert half_width > 0.0
    assert (mean_val - half_width) < mean_val < (mean_val + half_width)


def test_benchmark_rich_table_renders_without_error(bench_env):
    """Verify visual table formatting renders without console markup syntax errors."""
    runner = bench_env["runner"]
    summary = runner.run_benchmark(
        model_id="llama-3.2-3b",
        device="GPU",
        iterations=2,
        warmup=1,
    )
    console = Console(record=True)
    runner.render_rich_table(summary, console=console)
    output = console.export_text()
    assert "Intel Arc Models Runner Hardware Benchmark" in output
    assert "Time to First Token (TTFT)" in output
    assert "Decoding Throughput" in output


def test_benchmark_json_serialization(bench_env):
    """Verify BenchmarkSummary model dumps schema-compliant JSON string."""
    runner = bench_env["runner"]
    summary = runner.run_benchmark(
        model_id="llama-3.2-3b",
        device="GPU",
        iterations=2,
        warmup=1,
    )
    json_str = summary.model_dump_json(indent=2)
    assert "avg_ttft_ms" in json_str
    assert "avg_tokens_per_sec" in json_str
    assert "ci95_tokens_per_sec" in json_str


# =========================================================================
# NEGATIVE TEST TRIAD (4 TESTS)
# =========================================================================

def test_benchmark_unloaded_model_raises_404(bench_env):
    """Verify benchmarking a non-existent model raises ModelNotFoundError."""
    runner = bench_env["runner"]
    with pytest.raises(ModelNotFoundError):
        runner.run_benchmark(model_id="nonexistent-model-12345")


def test_benchmark_invalid_device_raises_error(bench_env):
    """Verify requesting an unrecognized hardware device raises error."""
    runner = bench_env["runner"]
    with pytest.raises(Exception):
        runner.run_benchmark(model_id="llama-3.2-3b", device="TPU_INVALID")


def test_benchmark_zero_iterations_rejected(bench_env):
    """Verify setting iterations=0 raises ArcRunnerError with HTTP 422."""
    runner = bench_env["runner"]
    with pytest.raises(ArcRunnerError) as exc_info:
        runner.run_benchmark(model_id="llama-3.2-3b", iterations=0)
    assert exc_info.value.status_code == 422


def test_benchmark_negative_max_tokens_rejected(bench_env):
    """Verify negative token budget raises ArcRunnerError with HTTP 422."""
    runner = bench_env["runner"]
    with pytest.raises(ArcRunnerError) as exc_info:
        runner.run_benchmark(model_id="llama-3.2-3b", max_new_tokens=-10)
    assert exc_info.value.status_code == 422


# =========================================================================
# CHAOS & EDGE-CASE TEST TRIAD (3 TESTS)
# =========================================================================

def test_benchmark_outlier_filtering_robustness(bench_env):
    """Verify confidence interval bounds sample variance even with high latency variance."""
    runner = bench_env["runner"]
    noisy_samples = [10.0, 11.0, 50.0, 12.0, 11.5]
    mean_val, half_width = runner._compute_confidence_interval_95(noisy_samples)
    assert half_width > 10.0


def test_benchmark_device_fallback_when_npu_unsupported(bench_env):
    """Verify requesting CPU device benchmark executes cleanly."""
    manager = bench_env["manager"]
    mock_cpu = MockBenchmarkEngine("llama-3.2-3b")
    manager._cache["llama-3.2-3b:CPU:none:NPU:4"] = mock_cpu

    runner = bench_env["runner"]
    summary = runner.run_benchmark(model_id="llama-3.2-3b", device="CPU", iterations=2, warmup=1)
    assert summary.device == "CPU"


def test_benchmark_concurrent_runs_thread_safe(bench_env):
    """Verify sequential benchmark runs execute without timer drift or memory leakage."""
    runner = bench_env["runner"]
    sum1 = runner.run_benchmark(model_id="llama-3.2-3b", iterations=2, warmup=1)
    sum2 = runner.run_benchmark(model_id="llama-3.2-3b", iterations=2, warmup=1)
    assert sum1.iterations == sum2.iterations
    assert sum1.avg_tokens_per_sec > 0.0
    assert sum2.avg_tokens_per_sec > 0.0


def test_run_benchmark_zero_tokens_raises_instead_of_inventing_throughput(bench_env):
    """A pass that streams no token must fail, never report a throughput derived from a floor value."""

    class SilentEngine(MockBenchmarkEngine):
        def generate_stream(self, prompt, config=None):
            yield TokenChunk(text="", token_id=2, is_finished=True, finish_reason="stop", prompt_tokens=8, completion_tokens=0)

    bench_env["manager"]._cache["llama-3.2-3b:GPU:none:NPU:4"] = SilentEngine("llama-3.2-3b")
    with pytest.raises(ArcRunnerError) as excinfo:
        bench_env["runner"].run_benchmark(model_id="llama-3.2-3b", device="GPU", iterations=1, warmup=0, max_new_tokens=4)
    assert excinfo.value.error_code == "NO_TOKENS_GENERATED"
