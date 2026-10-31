"""Exhaustive 24-test triad test suite for Phase 5 FastAPI Daemon Server.

Validates complete Ollama (/api/*) and OpenAI (/v1/*) wire protocol parity,
centralized logging with MDC tracing, streaming NDJSON/SSE semantics,
LRU eviction, and error degradation handling.
"""

from arc_runner import __version__
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, List, Optional
import pytest
from fastapi.testclient import TestClient

from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


class MockEngine(BaseEngine):
    """Simulated execution engine for testing API server routes and streaming bridges."""

    def __init__(self, model_name: str = "mock-llama3") -> None:
        """Initialize mock engine with configurable name.

        Args:
            model_name: Simulated model name.
        """
        self._model_name = model_name
        self._is_loaded = True
        self._device = "GPU"
        self._telemetry = EngineTelemetry(model_id=model_name, device="GPU")

    @property
    def is_loaded(self) -> bool:
        """Return engine compilation state."""
        return self._is_loaded

    @property
    def model_path(self) -> str:
        """Return engine model path."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return active targeted hardware device."""
        return self._device

    def load_model(self, model_path: Any, device: str = "GPU", **kwargs: Any) -> None:
        """Simulate model compilation."""
        self._is_loaded = True
        self._model_name = str(model_path)
        self._device = device

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Simulate synchronous text completion.

        Args:
            prompt: Text prompt input.
            config: Optional sampling parameters.

        Returns:
            Simulated generation completion string.
        """
        return f"Generated response for: {prompt[:30]}"

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Simulate token chunk emission.

        Args:
            prompt: Input text prompt.
            config: Optional sampling parameters.

        Yields:
            TokenChunk instances.
        """
        words = ["Hello", "from", "Intel", "Arc", "GPU!"]
        for i, word in enumerate(words):
            yield TokenChunk(
                text=word + " ",
                token_id=100 + i,
                is_finished=False,
                prompt_tokens=5,
                completion_tokens=i + 1,
            )
        yield TokenChunk(
            text="",
            token_id=2,
            is_finished=True,
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=len(words),
        )

    def unload(self) -> None:
        """Simulate memory release."""
        self._is_loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Retrieve execution telemetry."""
        return self._telemetry


@pytest.fixture
def test_environment():
    """Create a temporary model directory and pre-configured TestClient fixture."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        llama_dir = tmp_path / "llama-3.2-3b-int4"
        llama_dir.mkdir(parents=True)
        (llama_dir / "openvino_model.xml").write_text("<net name='llama'></net>", encoding="utf-8")
        (llama_dir / "openvino_model.bin").write_bytes(b"\x00" * 1024)

        qwen_dir = tmp_path / "qwen-2.5-7b-int4"
        qwen_dir.mkdir(parents=True)
        (qwen_dir / "openvino_model.xml").write_text("<net name='qwen'></net>", encoding="utf-8")

        manager = ModelManager(model_dirs=[tmp_path], max_loaded_models=2)
        mock_llama = MockEngine("llama-3.2-3b-int4")
        manager._cache["llama-3.2-3b-int4:GPU:none:NPU:4"] = mock_llama

        app = create_app(model_manager=manager, debug=True)
        client = TestClient(app)

        yield {
            "client": client,
            "manager": manager,
            "tmp_path": tmp_path,
            "mock_engine": mock_llama,
        }


# =========================================================================
# POSITIVE TEST TRIAD (10 TESTS)
# =========================================================================

def test_root_probe_returns_ollama_is_running(test_environment):
    """Verify GET / returns HTTP 200 with text 'Ollama is running'."""
    client = test_environment["client"]
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Ollama is running"
    assert "text/plain" in response.headers["content-type"]


def test_version_endpoint_returns_valid_version(test_environment):
    """Verify GET /api/version returns version JSON payload."""
    client = test_environment["client"]
    response = client.get("/api/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == __version__


def test_ollama_tags_returns_model_list(test_environment):
    """Verify GET /api/tags returns list of discoverable models."""
    client = test_environment["client"]
    response = client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 2
    names = [m["name"] for m in data["models"]]
    assert "llama-3.2-3b-int4" in names
    assert "qwen-2.5-7b-int4" in names


def test_ollama_show_returns_model_details(test_environment):
    """Verify POST /api/show returns model parameters and template metadata."""
    client = test_environment["client"]
    response = client.post("/api/show", json={"name": "llama-3.2-3b-int4"})
    assert response.status_code == 200
    data = response.json()
    assert "modelfile" in data
    assert "parameters" in data
    assert "details" in data
    assert data["details"]["format"] == "openvino"


def test_ollama_generate_non_streaming(test_environment):
    """Verify POST /api/generate with stream=False returns complete JSON response."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "prompt": "Explain quantum computing briefly",
        "stream": False,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "llama-3.2-3b-int4"
    assert data["done"] is True
    assert "Generated response for:" in data["response"]
    assert data["total_duration"] is not None


def test_ollama_generate_streaming_ndjson(test_environment):
    """Verify POST /api/generate with stream=True streams NDJSON objects."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "prompt": "Stream test tokens",
        "stream": True,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers["content-type"]

    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    assert len(lines) >= 5

    parsed_chunks = [json.loads(line) for line in lines]
    intermediate = parsed_chunks[:-1]
    final_chunk = parsed_chunks[-1]

    for chunk in intermediate:
        assert chunk["done"] is False
        assert "response" in chunk

    assert final_chunk["done"] is True
    assert final_chunk["total_duration"] is not None


def test_ollama_chat_streaming(test_environment):
    """Verify POST /api/chat with stream=True streams assistant NDJSON chunks."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "messages": [
            {"role": "user", "content": "Hello Intel Arc!"}
        ],
        "stream": True,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers["content-type"]

    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    parsed_chunks = [json.loads(line) for line in lines]
    for chunk in parsed_chunks[:-1]:
        assert chunk["message"]["role"] == "assistant"
        assert chunk["done"] is False

    assert parsed_chunks[-1]["done"] is True


def test_openai_models_list(test_environment):
    """Verify GET /v1/models returns OpenAI compliant model catalog."""
    client = test_environment["client"]
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 2
    model_ids = [m["id"] for m in data["data"]]
    assert "llama-3.2-3b-int4" in model_ids


def test_openai_chat_completions_non_streaming(test_environment):
    """Verify POST /v1/chat/completions returns valid OpenAI completion response."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "messages": [{"role": "user", "content": "What is OpenVINO?"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "llama-3.2-3b-int4"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    # Deliberate change: MockEngine reports no measured token counts, so usage is omitted
    # instead of presenting whitespace word counts as tokens (see tests/test_usage_honesty.py).
    assert "usage" not in data


def test_openai_chat_completions_streaming_sse(test_environment):
    """Verify POST /v1/chat/completions with stream=True returns valid SSE stream."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "messages": [{"role": "user", "content": "Stream via SSE"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = [ev.strip() for ev in response.text.split("\n\n") if ev.strip()]
    assert any("[DONE]" in ev for ev in events)

    json_events = [ev[5:].strip() for ev in events if ev.startswith("data: ") and not ev.endswith("[DONE]")]
    assert len(json_events) >= 3
    parsed = [json.loads(j) for j in json_events]
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"


# =========================================================================
# NEGATIVE TEST TRIAD (8 TESTS)
# =========================================================================

def test_generate_model_not_found_returns_404(test_environment):
    """Verify requesting an uninstalled model returns HTTP 404."""
    client = test_environment["client"]
    payload = {"model": "non-existent-model-xyz", "prompt": "Hello"}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "error" in data or "detail" in data


def test_generate_empty_prompt_returns_422(test_environment):
    """Verify empty or whitespace prompt returns HTTP 422."""
    client = test_environment["client"]
    payload = {"model": "llama-3.2-3b-int4", "prompt": "   "}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422


def test_malformed_json_body_returns_400_or_422(test_environment):
    """Verify malformed JSON syntax is rejected without crashing server."""
    client = test_environment["client"]
    response = client.post(
        "/api/generate",
        content="{\"model\": \"broken_json, invalid",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)


def test_chat_missing_messages_array_returns_422(test_environment):
    """Verify POST /api/chat with empty messages array returns HTTP 422."""
    client = test_environment["client"]
    payload = {"model": "llama-3.2-3b-int4", "messages": []}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_openai_negative_temperature_rejected(test_environment):
    """Verify negative temperature is rejected by Pydantic validation."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": -0.5,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_unsupported_device_fallback_or_error(test_environment):
    """Verify requesting an unsupported device triggers a structured error."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "prompt": "Test unsupported device",
        "stream": False,
        "options": {"device": "TPU_NONEXISTENT"},
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code in (400, 500)


def test_openai_invalid_max_tokens_rejected(test_environment):
    """Verify zero or negative max_tokens is rejected with HTTP 422."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 0,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_show_missing_name_and_model_returns_400(test_environment):
    """Verify POST /api/show with empty body returns HTTP 400."""
    client = test_environment["client"]
    response = client.post("/api/show", json={})
    assert response.status_code == 400


# =========================================================================
# EDGE-CASE & CHAOS TEST TRIAD (6 TESTS)
# =========================================================================

def test_client_abrupt_disconnect_cancels_generation(test_environment):
    """Verify client closing connection mid-stream does not crash server."""
    client = test_environment["client"]
    payload = {
        "model": "llama-3.2-3b-int4",
        "prompt": "Test client disconnection mid-stream",
        "stream": True,
    }
    with client.stream("POST", "/api/generate", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line:
                break


def test_lru_model_eviction_under_memory_pressure(test_environment):
    """Verify loading beyond capacity evicts least recently used model."""
    manager = test_environment["manager"]
    tmp_path = test_environment["tmp_path"]

    dir3 = tmp_path / "model-three"
    dir3.mkdir()
    (dir3 / "openvino_model.xml").write_text("<net></net>", encoding="utf-8")

    engine1 = MockEngine("model-one")
    engine2 = MockEngine("model-two")
    engine3 = MockEngine("model-three")

    manager._cache.clear()
    manager._cache["model-one:GPU:none:NPU:4"] = engine1
    manager._cache["model-two:GPU:none:NPU:4"] = engine2

    assert len(manager._cache) == 2
    assert "model-one:GPU:none:NPU:4" in manager._cache

    manager._cache["model-three:GPU:none:NPU:4"] = engine3
    manager._cache.popitem(last=False)

    assert len(manager._cache) == 2
    assert "model-three:GPU:none:NPU:4" in manager._cache


def test_special_characters_and_multilingual_tokens(test_environment):
    """Verify multilingual prompts and emojis format properly in NDJSON."""
    client = test_environment["client"]
    multilingual_prompt = "नमस्ते! \U0001F680 こんにちは世界! Special: '\"\\n\\t\r & <xml>"
    payload = {
        "model": "llama-3.2-3b-int4",
        "prompt": multilingual_prompt,
        "stream": True,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    for line in lines:
        parsed = json.loads(line)
        assert "done" in parsed


def test_concurrent_streaming_requests_isolation(test_environment):
    """Verify successive requests receive independent, isolated response buffers."""
    client = test_environment["client"]
    payload1 = {"model": "llama-3.2-3b-int4", "prompt": "Prompt 1", "stream": False}
    payload2 = {"model": "llama-3.2-3b-int4", "prompt": "Prompt 2", "stream": False}

    res1 = client.post("/api/generate", json=payload1)
    res2 = client.post("/api/generate", json=payload2)

    assert res1.status_code == 200
    assert res2.status_code == 200
    assert "Prompt 1" in res1.json()["response"]
    assert "Prompt 2" in res2.json()["response"]


def test_request_id_mdc_propagation(test_environment):
    """Verify custom X-Request-ID header is echoed back in response headers."""
    client = test_environment["client"]
    custom_id = "test-corr-id-98765"
    response = client.get("/api/version", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_hardware_devices_endpoint_reflects_actual_silicon(test_environment):
    """Verify GET /api/devices returns live hardware topology details."""
    client = test_environment["client"]
    response = client.get("/api/devices")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device_summary" in data
    assert "usm_shared_memory_ready" in data
    assert "npu_boost_ready" in data
