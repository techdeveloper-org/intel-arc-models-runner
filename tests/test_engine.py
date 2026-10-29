"""Unit tests for inference engines, KV-cache manager, and token streaming.

Implements the 28-test Triad (Positive, Negative, Edge-Case) verifying engine lifecycle,
memory footprint formulas, error handling, and streaming backpressure in accordance with
Rule 41 and Rule 19 standards.
"""

import math
import os
import time
import unittest.mock as mock
from pathlib import Path
from typing import Iterator, List, Optional

import pytest
from pydantic import ValidationError

from arc_runner.core.exceptions import (
    ArcRunnerError,
    ContextWindowExceededError,
    DeviceNotFoundError,
    ModelLoadError,
    ModelNotFoundError,
)
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.models import (
    EngineTelemetry,
    GenerationConfig,
    KVAllocation,
    ModelPrecision,
    TokenChunk,
)
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine


class MockBaseEngine(BaseEngine):
    """Concrete mock implementation of BaseEngine for interface contract verification."""

    def __init__(self) -> None:
        """Initialize the mock engine."""
        self._loaded: bool = False
        self._telemetry = EngineTelemetry(model_id="mock_model", device="GPU")

    @property
    def is_loaded(self) -> bool:
        """Return mock loaded state."""
        return self._loaded

    def load_model(self, model_path: str, device: str = "GPU", **kwargs: object) -> None:
        """Simulate loading model."""
        self._loaded = True

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Simulate synchronous generation."""
        if not self._loaded:
            raise ArcRunnerError("Not loaded", "NOT_LOADED", 400)
        return f"Echo: {prompt}"

    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[TokenChunk]:
        """Simulate streaming generation."""
        if not self._loaded:
            raise ArcRunnerError("Not loaded", "NOT_LOADED", 400)
        yield TokenChunk(text="Hello", is_finished=False, completion_tokens=1)
        yield TokenChunk(text="", is_finished=True, finish_reason="stop", completion_tokens=1)

    def unload(self) -> None:
        """Simulate unloading model."""
        self._loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Return mock telemetry."""
        return self._telemetry


def test_generation_config_default_instantiation() -> None:
    """Verify default generation parameters match system specifications."""
    config = GenerationConfig()
    assert config.max_new_tokens == 512
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.top_k == 40
    assert config.repetition_penalty == 1.1
    assert config.stream is False
    assert config.stop_strings == []


def test_generation_config_custom_parameters() -> None:
    """Verify customization of sampling parameters."""
    config = GenerationConfig(
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.95,
        top_k=50,
        stop_strings=["\nUser:", "###"],
        stream=True,
    )
    assert config.max_new_tokens == 1024
    assert config.temperature == 0.2
    assert config.top_p == 0.95
    assert config.top_k == 50
    assert len(config.stop_strings) == 2
    assert config.stream is True


def test_generation_config_serialization_roundtrip() -> None:
    """Verify serialization to dictionary and restoration via model_dump."""
    config = GenerationConfig(max_new_tokens=256, temperature=1.2)
    dumped = config.model_dump()
    restored = GenerationConfig(**dumped)
    assert restored.max_new_tokens == 256
    assert restored.temperature == 1.2


def test_token_chunk_standard_delta() -> None:
    """Verify TokenChunk emission schema for incremental token deltas."""
    chunk = TokenChunk(text="intel", token_id=1420, is_finished=False, completion_tokens=1)
    assert chunk.text == "intel"
    assert chunk.token_id == 1420
    assert chunk.is_finished is False
    assert chunk.finish_reason is None
    assert chunk.completion_tokens == 1


def test_token_chunk_completion_reason_stop() -> None:
    """Verify TokenChunk schema on successful stop token completion."""
    chunk = TokenChunk(text="", is_finished=True, finish_reason="stop", prompt_tokens=128, completion_tokens=64)
    assert chunk.is_finished is True
    assert chunk.finish_reason == "stop"
    assert chunk.prompt_tokens == 128
    assert chunk.completion_tokens == 64


def test_token_chunk_completion_reason_length() -> None:
    """Verify TokenChunk schema when sequence hits max_new_tokens boundary."""
    chunk = TokenChunk(text="", is_finished=True, finish_reason="length", completion_tokens=512)
    assert chunk.is_finished is True
    assert chunk.finish_reason == "length"
    assert chunk.completion_tokens == 512


def test_kv_cache_headroom_valid_prompt() -> None:
    """Verify that headroom calculations accurately subtract prompt tokens from context window."""
    manager = KVCacheManager(max_context_length=4096)
    headroom = manager.check_headroom(512)
    assert headroom == 3584


def test_kv_cache_allocation_and_release() -> None:
    """Verify memory reservation, tracking, and subsequent deallocation lifecycle."""
    manager = KVCacheManager(max_context_length=4096)
    allocation = manager.reserve_allocation(prompt_tokens=256, max_new_tokens=512)

    assert isinstance(allocation, KVAllocation)
    assert allocation.prompt_tokens == 256
    assert allocation.reserved_new_tokens == 512
    assert allocation.total_tokens == 768
    assert allocation.is_active is True
    assert manager.active_allocated_tokens == 768

    manager.release_allocation(allocation.allocation_id)
    assert manager.active_allocated_tokens == 0


def test_kv_cache_memory_footprint_theoretical_match() -> None:
    """Assert that KV cache footprint formula matches theoretical bytes for Llama-3.2-3B."""
    manager = KVCacheManager()
    footprint = manager.calculate_footprint_mb(
        num_tokens=2048,
        num_layers=28,
        num_heads=8,
        head_dim=128,
        dtype_bytes=2,
    )
    expected_bytes = 2 * 28 * 8 * 128 * 2048 * 2
    expected_mb = expected_bytes / (1024.0 * 1024.0)
    assert math.isclose(footprint, expected_mb, rel_tol=1e-4)
    assert math.isclose(footprint, 224.0, rel_tol=1e-4)


def test_base_engine_context_manager_lifecycle() -> None:
    """Verify that context manager automatically handles unloading upon exit."""
    with MockBaseEngine() as engine:
        engine.load_model("fake_path")
        assert engine.is_loaded is True
    assert engine.is_loaded is False


def test_openvino_engine_mock_synchronous_generation(tmp_path: Path) -> None:
    """Verify synchronous generate() using a mocked LLMPipeline instance."""
    engine = OpenVINOGenAIEngine()
    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.return_value = "Hello from Intel Arc!"

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        result = engine.generate("Hello")

    assert result == "Hello from Intel Arc!"
    telemetry = engine.get_telemetry()
    assert telemetry.device == "GPU"
    assert telemetry.metrics_source == "unavailable"
    assert telemetry.total_tokens == 0
    assert telemetry.is_tps_valid is False


def test_openvino_engine_mock_streaming_generation(tmp_path: Path) -> None:
    """Verify streaming generation yields incremental chunks and completes with stop reason."""
    engine = OpenVINOGenAIEngine()

    def fake_generate(prompt: str, config: object = None, streamer: object = None) -> None:
        if callable(streamer):
            streamer("Intel ")
            streamer("Arc ")
            streamer("GPU")

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.side_effect = fake_generate

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        chunks = list(engine.generate_stream("Prompt"))

    text_emitted = "".join(c.text for c in chunks)
    assert text_emitted == "Intel Arc GPU"
    assert chunks[-1].is_finished is True
    assert chunks[-1].finish_reason == "stop"


def test_model_not_found_raises_model_not_found_error() -> None:
    """Verify that targeting a non-existent path raises ModelNotFoundError."""
    engine = OpenVINOGenAIEngine()
    with pytest.raises(ModelNotFoundError) as exc_info:
        engine.load_model("/path/that/does/not/exist", device="GPU")
    assert "not found" in str(exc_info.value).lower()
    assert exc_info.value.status_code == 404


def test_device_not_found_raises_device_not_found_error(tmp_path: Path) -> None:
    """Verify that requesting an unsupported device raises DeviceNotFoundError."""
    engine = OpenVINOGenAIEngine()
    with pytest.raises(DeviceNotFoundError) as exc_info:
        engine.load_model(str(tmp_path), device="TPU")
    assert "tpu" in str(exc_info.value).lower()
    assert exc_info.value.status_code == 400


def test_generate_before_load_raises_arc_runner_error() -> None:
    """Verify that calling generate() without loading a model raises ArcRunnerError."""
    engine = OpenVINOGenAIEngine()
    with pytest.raises(ArcRunnerError) as exc_info:
        engine.generate("Hello world")
    assert exc_info.value.error_code == "MODEL_NOT_LOADED"


def test_generate_stream_before_load_raises_arc_runner_error() -> None:
    """Verify that calling generate_stream() without loading a model raises ArcRunnerError."""
    engine = OpenVINOGenAIEngine()
    with pytest.raises(ArcRunnerError) as exc_info:
        list(engine.generate_stream("Hello world"))
    assert exc_info.value.error_code == "MODEL_NOT_LOADED"


def test_context_window_exceeded_raises_context_window_exceeded_error() -> None:
    """Verify that checking headroom for a prompt exceeding context window raises ContextWindowExceededError."""
    manager = KVCacheManager(max_context_length=2048)
    with pytest.raises(ContextWindowExceededError) as exc_info:
        manager.check_headroom(2049)
    assert exc_info.value.status_code == 400


def test_generation_config_negative_temperature_validation_error() -> None:
    """Verify that negative temperatures fail Pydantic validation."""
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=-0.5)


def test_generation_config_excessive_temperature_validation_error() -> None:
    """Verify that temperatures exceeding 2.0 fail Pydantic validation."""
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=2.5)


def test_generation_config_zero_or_negative_max_tokens_validation_error() -> None:
    """Verify that max_new_tokens <= 0 fails Pydantic validation."""
    with pytest.raises(ValidationError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValidationError):
        GenerationConfig(max_new_tokens=-10)


def test_generation_config_invalid_top_p_validation_error() -> None:
    """Verify that top_p <= 0.0 or > 1.0 fails Pydantic validation."""
    with pytest.raises(ValidationError):
        GenerationConfig(top_p=0.0)
    with pytest.raises(ValidationError):
        GenerationConfig(top_p=1.2)


def test_empty_prompt_handling_synchronous(tmp_path: Path) -> None:
    """Verify that passing an empty prompt returns empty string without calling C++ pipeline."""
    engine = OpenVINOGenAIEngine()
    mock_pipeline = mock.MagicMock()

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        result = engine.generate("")

    assert result == ""
    mock_pipeline.generate.assert_not_called()


def test_empty_prompt_handling_streaming(tmp_path: Path) -> None:
    """Verify that passing an empty prompt to generate_stream yields single finished chunk."""
    engine = OpenVINOGenAIEngine()
    mock_pipeline = mock.MagicMock()

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        chunks = list(engine.generate_stream(""))

    assert len(chunks) == 1
    assert chunks[0].is_finished is True
    assert chunks[0].finish_reason == "stop"
    mock_pipeline.generate.assert_not_called()


def test_max_new_tokens_one_boundary(tmp_path: Path) -> None:
    """Verify boundary condition where max_new_tokens is exactly 1."""
    engine = OpenVINOGenAIEngine()

    def fake_generate(prompt: str, config: object = None, streamer: object = None) -> None:
        if callable(streamer):
            streamer("A")

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.side_effect = fake_generate

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        chunks = list(engine.generate_stream("Test", GenerationConfig(max_new_tokens=1)))

    assert len(chunks) == 2
    assert chunks[0].text == "A"
    assert chunks[1].is_finished is True
    assert chunks[1].finish_reason == "length"


def test_stop_string_immediate_trigger(tmp_path: Path) -> None:
    """Verify that encountering a registered stop string terminates streaming immediately."""
    engine = OpenVINOGenAIEngine()

    def fake_generate(prompt: str, config: object = None, streamer: object = None) -> None:
        if callable(streamer):
            for token in ["Answer: 42", "\nUser:", "Should not be emitted"]:
                if streamer(token):
                    break

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.side_effect = fake_generate

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        cfg = GenerationConfig(stop_strings=["\nUser:"])
        chunks = list(engine.generate_stream("Prompt", cfg))

    text_emitted = "".join(c.text for c in chunks)
    assert "\nUser:" not in text_emitted
    assert "Should not be emitted" not in text_emitted
    assert "Answer: 42" in text_emitted


def test_cancellation_during_streaming(tmp_path: Path) -> None:
    """Verify that breaking out of the generator safely signals the streamer callback to abort."""
    engine = OpenVINOGenAIEngine()
    tokens_yielded_to_callback = 0

    def fake_generate(prompt: str, config: object = None, streamer: object = None) -> None:
        nonlocal tokens_yielded_to_callback
        if callable(streamer):
            for i in range(50):
                tokens_yielded_to_callback += 1
                time.sleep(0.01)
                should_abort = streamer(f"Token_{i}")
                if should_abort:
                    break

    mock_pipeline = mock.MagicMock()
    mock_pipeline.generate.side_effect = fake_generate

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="GPU")
        for idx, chunk in enumerate(engine.generate_stream("Prompt")):
            if idx == 1:
                break

    assert tokens_yielded_to_callback < 10


def test_engine_multiple_reload_no_leak(tmp_path: Path) -> None:
    """Verify that repeated load_model() and unload() cycles maintain clean state."""
    engine = OpenVINOGenAIEngine()
    mock_pipeline = mock.MagicMock()

    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        for _ in range(5):
            engine.load_model(str(tmp_path), device="GPU")
            assert engine.is_loaded is True
            engine.unload()
            assert engine.is_loaded is False


def test_extreme_temperature_greedy_decoding() -> None:
    """Verify that temperature=0.0 is permitted for deterministic greedy decoding."""
    config = GenerationConfig(temperature=0.0)
    assert config.temperature == 0.0
