"""Red-first tests for WP-02 02d: long_prompt TTFT support in BenchmarkRunner.

Verifies that run_benchmark accepts a long_prompt parameter and forwards it to
generate_stream so that TTFT measurements reflect long-context prefill timing.
"""

import inspect
import time
from pathlib import Path
import tempfile
from typing import Iterator, Optional

import pytest

from arc_runner.benchmark.runner import BenchmarkRunner, BenchmarkSummary
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.model_manager import ModelManager


class _RecordingEngine(BaseEngine):
    """Engine that records each prompt passed to generate_stream."""

    def __init__(self, model_name: str = "mock-model") -> None:
        """Initialise the recording engine."""
        self._model_name = model_name
        self._is_loaded = True
        self._device = "GPU"
        self._telemetry = EngineTelemetry(model_id=model_name, device="GPU")
        self.prompts_seen: list = []

    @property
    def is_loaded(self) -> bool:
        """Return load status."""
        return self._is_loaded

    @property
    def model_path(self) -> str:
        """Return model path."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return device identifier."""
        return self._device

    def load_model(self, model_path, device: str = "GPU", **kwargs) -> None:
        """Simulate model load."""
        self._is_loaded = True

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Simulate text generation."""
        return "response"

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Record the prompt, then yield minimal token stream."""
        self.prompts_seen.append(prompt)
        words = ["tok1", "tok2", "tok3"]
        for i, word in enumerate(words):
            time.sleep(0.001)
            yield TokenChunk(
                text=word + " ",
                token_id=10 + i,
                is_finished=False,
                prompt_tokens=len(prompt.split()),
                completion_tokens=i + 1,
            )
        yield TokenChunk(
            text="",
            token_id=2,
            is_finished=True,
            finish_reason="stop",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(words),
        )

    def unload(self) -> None:
        """Simulate deallocation."""
        self._is_loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Return telemetry object."""
        return self._telemetry


@pytest.fixture
def ttft_env():
    """Sandbox with a recording engine wired into BenchmarkRunner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").write_text("<net></net>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"\x00" * 64)

        manager = ModelManager(model_dirs=[tmp_path])
        engine = _RecordingEngine("test-model")
        manager._cache["test-model:GPU:none:NPU:4"] = engine

        runner = BenchmarkRunner(model_manager=manager)
        yield {"runner": runner, "engine": engine, "manager": manager}


class TestLongPromptParameterSignature:
    """run_benchmark must declare long_prompt in its signature."""

    def test_long_prompt_parameter_exists_on_run_benchmark(self):
        """long_prompt must be a named parameter of run_benchmark."""
        sig = inspect.signature(BenchmarkRunner.run_benchmark)
        assert "long_prompt" in sig.parameters, (
            "run_benchmark must declare a 'long_prompt' parameter"
        )

    def test_long_prompt_default_is_none(self):
        """long_prompt must default to None (optional)."""
        sig = inspect.signature(BenchmarkRunner.run_benchmark)
        param = sig.parameters["long_prompt"]
        assert param.default is None, (
            f"long_prompt default must be None, got {param.default!r}"
        )

    def test_long_prompt_annotation_is_optional_str(self):
        """long_prompt must be annotated Optional[str] or equivalent."""
        sig = inspect.signature(BenchmarkRunner.run_benchmark)
        param = sig.parameters["long_prompt"]
        annotation = param.annotation
        assert annotation is not inspect.Parameter.empty, (
            "long_prompt must have a type annotation"
        )
        annotation_str = str(annotation)
        assert "str" in annotation_str, (
            f"long_prompt annotation must reference str, got {annotation_str!r}"
        )


class TestLongPromptForwardedToEngine:
    """When long_prompt is supplied, generate_stream must receive it."""

    def test_long_prompt_is_forwarded_to_generate_stream(self, ttft_env):
        """Every generate_stream call must use long_prompt when it is provided."""
        runner = ttft_env["runner"]
        engine = ttft_env["engine"]

        long_text = "A " * 200
        runner.run_benchmark(
            model_id="test-model",
            device="GPU",
            iterations=2,
            warmup=1,
            max_new_tokens=4,
            long_prompt=long_text,
        )

        assert all(p == long_text for p in engine.prompts_seen), (
            "All generate_stream calls must use long_prompt when it is provided; "
            f"got prompts: {engine.prompts_seen!r}"
        )

    def test_long_prompt_none_falls_back_to_prompt(self, ttft_env):
        """When long_prompt is None, generate_stream must use prompt (or the default)."""
        runner = ttft_env["runner"]
        engine = ttft_env["engine"]

        custom_prompt = "Custom short prompt"
        runner.run_benchmark(
            model_id="test-model",
            device="GPU",
            iterations=2,
            warmup=1,
            max_new_tokens=4,
            prompt=custom_prompt,
            long_prompt=None,
        )

        assert all(p == custom_prompt for p in engine.prompts_seen), (
            "When long_prompt is None, generate_stream must use prompt; "
            f"got prompts: {engine.prompts_seen!r}"
        )

    def test_long_prompt_overrides_prompt_argument(self, ttft_env):
        """long_prompt takes precedence over prompt when both are supplied."""
        runner = ttft_env["runner"]
        engine = ttft_env["engine"]

        short_prompt = "short"
        long_text = "long " * 100

        runner.run_benchmark(
            model_id="test-model",
            device="GPU",
            iterations=1,
            warmup=0,
            max_new_tokens=4,
            prompt=short_prompt,
            long_prompt=long_text,
        )

        assert all(p == long_text for p in engine.prompts_seen), (
            "long_prompt must override prompt when both are supplied; "
            f"got prompts: {engine.prompts_seen!r}"
        )


class TestLongPromptSummaryValidity:
    """BenchmarkSummary produced with long_prompt must be well-formed."""

    def test_summary_returned_with_long_prompt_is_benchmark_summary(self, ttft_env):
        """run_benchmark must return a BenchmarkSummary when long_prompt is set."""
        runner = ttft_env["runner"]
        result = runner.run_benchmark(
            model_id="test-model",
            device="GPU",
            iterations=2,
            warmup=1,
            max_new_tokens=4,
            long_prompt="This is a long context. " * 50,
        )
        assert isinstance(result, BenchmarkSummary), (
            f"Expected BenchmarkSummary, got {type(result)}"
        )

    def test_ttft_is_positive_with_long_prompt(self, ttft_env):
        """avg_ttft_ms must be positive when long_prompt is used."""
        runner = ttft_env["runner"]
        result = runner.run_benchmark(
            model_id="test-model",
            device="GPU",
            iterations=2,
            warmup=1,
            max_new_tokens=4,
            long_prompt="Context token " * 80,
        )
        assert result.avg_ttft_ms > 0.0, (
            f"avg_ttft_ms must be positive, got {result.avg_ttft_ms}"
        )

    def test_iteration_count_unchanged_with_long_prompt(self, ttft_env):
        """iterations count in summary must match the requested value."""
        runner = ttft_env["runner"]
        result = runner.run_benchmark(
            model_id="test-model",
            device="GPU",
            iterations=3,
            warmup=1,
            max_new_tokens=4,
            long_prompt="Long prefix " * 60,
        )
        assert result.iterations == 3, (
            f"Expected iterations=3, got {result.iterations}"
        )
