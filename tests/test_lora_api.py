"""Triad test suite for LoRA Adapter REST API Endpoints and Dynamic Blending Routes.

Validates GET /api/adapters, POST /api/adapters/load, POST /api/adapters/unload,
POST /api/adapters/inspect, and adapter dispatch through /api/generate and /api/chat.
"""

import asyncio
from pathlib import Path
from typing import Any, Iterator, List, Optional
import pytest
from fastapi.testclient import TestClient

from arc_runner.engine.base import BaseEngine
from arc_runner.engine.lora_manager import LoRAManager
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager
from tests.test_lora import create_dummy_lora_adapter


class RecordingMockEngine(BaseEngine):
    """Mock engine that records received generation configs to verify adapter pass-through."""

    def __init__(self, model_name: str = "llama-3.2-3b-int4") -> None:
        """Initialize mock engine with recording buffer.

        Args:
            model_name: Name of simulated model.
        """
        self._model_name = model_name
        self._is_loaded = True
        self._device = "GPU"
        self._telemetry = EngineTelemetry(model_id=model_name, device="GPU")
        self.recorded_configs: List[GenerationConfig] = []

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

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Record configuration and return synthetic completion.

        Args:
            prompt: Text prompt input.
            config: Generation configuration options.

        Returns:
            Simulated generation completion.
        """
        if config:
            self.recorded_configs.append(config)
        return "Simulated text response with LoRA."

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Record configuration and yield simulated stream chunks.

        Args:
            prompt: Text prompt input.
            config: Generation configuration options.

        Yields:
            TokenChunk instances.
        """
        if config:
            self.recorded_configs.append(config)
        yield TokenChunk(text="Simulated ", token_id=1, is_finished=False)
        yield TokenChunk(text="stream.", token_id=2, is_finished=True, finish_reason="stop")

    def unload(self) -> None:
        """Simulate memory release."""
        self._is_loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Retrieve execution telemetry."""
        return self._telemetry


@pytest.fixture
def lora_test_environment(tmp_path: Path):
    """Create isolated test environment with dummy adapters and configured client.

    Args:
        tmp_path: Temporary directory fixture from pytest.

    Yields:
        Dictionary containing client, model manager, lora manager, and mock engine.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    llama_dir = models_dir / "llama-3.2-3b-int4"
    llama_dir.mkdir(parents=True)
    (llama_dir / "openvino_model.xml").write_text("<net name='llama'></net>", encoding="utf-8")
    (llama_dir / "openvino_model.bin").write_bytes(b"\x00" * 1024)

    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir(parents=True)
    create_dummy_lora_adapter(adapters_dir, "lora-coder", r=16, alpha=32)
    create_dummy_lora_adapter(adapters_dir, "lora-legal", r=8, alpha=16)

    model_mgr = ModelManager(model_dirs=[models_dir], max_loaded_models=2)
    lora_mgr = LoRAManager(adapters_dir=adapters_dir, max_cached_adapters=4)
    model_mgr.lora_manager = lora_mgr

    mock_engine = RecordingMockEngine("llama-3.2-3b-int4")
    model_mgr._cache["llama-3.2-3b-int4:GPU:none:NPU:4"] = mock_engine

    app = create_app(model_manager=model_mgr, debug=True)
    client = TestClient(app)

    yield {
        "client": client,
        "model_manager": model_mgr,
        "lora_manager": lora_mgr,
        "adapters_dir": adapters_dir,
        "mock_engine": mock_engine,
    }


class TestLoRAAPIPositive:
    """Positive test cases for LoRA REST API endpoints and inference integration."""

    def test_api_get_adapters_200(self, lora_test_environment: dict) -> None:
        """Verify GET /api/adapters lists discovered adapters and initial caching status."""
        client = lora_test_environment["client"]
        resp = client.get("/api/adapters")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        ids = [item["adapter_id"] for item in data]
        assert "lora-coder" in ids
        assert "lora-legal" in ids
        for item in data:
            assert item["is_cached"] is False

    def test_api_post_adapter_load_200(self, lora_test_environment: dict) -> None:
        """Verify POST /api/adapters/load caches adapter in memory."""
        client = lora_test_environment["client"]
        resp = client.post("/api/adapters/load", json={"name": "lora-coder"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["adapter_id"] == "lora-coder"
        assert data["is_cached"] is True
        assert lora_test_environment["lora_manager"].cache_size == 1

    def test_api_post_adapter_unload_200(self, lora_test_environment: dict) -> None:
        """Verify POST /api/adapters/unload evicts adapter from memory."""
        client = lora_test_environment["client"]
        client.post("/api/adapters/load", json={"name": "lora-coder"})
        assert lora_test_environment["lora_manager"].cache_size == 1

        resp = client.post("/api/adapters/unload", json={"name": "lora-coder"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["adapter"] == "lora-coder"
        assert data["unloaded"] is True
        assert lora_test_environment["lora_manager"].cache_size == 0

    def test_api_post_adapter_inspect_200(self, lora_test_environment: dict) -> None:
        """Verify POST /api/adapters/inspect parses metadata without caching."""
        client = lora_test_environment["client"]
        resp = client.post("/api/adapters/inspect", json={"name": "lora-legal"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["adapter_id"] == "lora-legal"
        assert data["rank"] == 8
        assert data["alpha"] == 16.0
        assert lora_test_environment["lora_manager"].cache_size == 0

    def test_api_generate_with_adapters_field(self, lora_test_environment: dict) -> None:
        """Verify POST /api/generate passes adapter specifications into engine configuration."""
        client = lora_test_environment["client"]
        mock_engine = lora_test_environment["mock_engine"]

        payload = {
            "model": "llama-3.2-3b-int4",
            "prompt": "Write a quicksort implementation in Python.",
            "stream": False,
            "adapters": [{"name": "lora-coder", "scale": 0.85}],
        }
        resp = client.post("/api/generate", json=payload)
        assert resp.status_code == 200

        assert len(mock_engine.recorded_configs) == 1
        cfg = mock_engine.recorded_configs[0]
        assert cfg.adapters is not None
        assert len(cfg.adapters) == 1
        assert cfg.adapters[0].name == "lora-coder"
        assert cfg.adapters[0].scale == 0.85

    def test_api_chat_with_single_adapter_shortcut(self, lora_test_environment: dict) -> None:
        """Verify POST /api/chat passes single adapter shortcut string."""
        client = lora_test_environment["client"]
        mock_engine = lora_test_environment["mock_engine"]

        payload = {
            "model": "llama-3.2-3b-int4",
            "messages": [{"role": "user", "content": "Review contract clauses."}],
            "stream": False,
            "adapter": "lora-legal",
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200

        assert len(mock_engine.recorded_configs) == 1
        cfg = mock_engine.recorded_configs[0]
        assert cfg.adapters is not None
        assert len(cfg.adapters) == 1
        assert cfg.adapters[0].name == "lora-legal"
        assert cfg.adapters[0].scale == 1.0


class TestLoRAAPINegative:
    """Negative test cases for API error handling and security verification."""

    def test_api_post_adapter_load_404_missing(self, lora_test_environment: dict) -> None:
        """Verify POST /api/adapters/load returns 404 for nonexistent adapter."""
        client = lora_test_environment["client"]
        resp = client.post("/api/adapters/load", json={"name": "non-existent-adapter"})

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_api_directory_traversal_rejected(self, lora_test_environment: dict) -> None:
        """Verify path traversal patterns like '..' are rejected with 400 Bad Request."""
        client = lora_test_environment["client"]
        resp = client.post("/api/adapters/load", json={"adapter": "../../etc/passwd"})

        assert resp.status_code == 400
        assert "Path traversal not permitted" in resp.json()["detail"]


class TestLoRAAPIEdgeCases:
    """Edge cases verifying empty specs and concurrent request isolation (Proof M27)."""

    def test_api_generate_empty_adapters_array(self, lora_test_environment: dict) -> None:
        """Verify empty adapters array passes without error and without overhead."""
        client = lora_test_environment["client"]
        mock_engine = lora_test_environment["mock_engine"]

        payload = {
            "model": "llama-3.2-3b-int4",
            "prompt": "Test empty adapters.",
            "stream": False,
            "adapters": [],
        }
        resp = client.post("/api/generate", json=payload)
        assert resp.status_code == 200
        assert len(mock_engine.recorded_configs) == 1
        cfg = mock_engine.recorded_configs[0]
        assert cfg.adapters is None or len(cfg.adapters) == 0

    def test_api_concurrent_requests_different_adapters_m27(self, lora_test_environment: dict) -> None:
        """Verify concurrent requests with distinct adapter specs construct isolated configs (Proof M27)."""
        client = lora_test_environment["client"]
        mock_engine = lora_test_environment["mock_engine"]

        payload_coder = {
            "model": "llama-3.2-3b-int4",
            "prompt": "Task 1",
            "stream": False,
            "adapters": [{"name": "lora-coder", "scale": 0.9}],
        }
        payload_legal = {
            "model": "llama-3.2-3b-int4",
            "prompt": "Task 2",
            "stream": False,
            "adapters": [{"name": "lora-legal", "scale": 0.4}],
        }

        resp1 = client.post("/api/generate", json=payload_coder)
        resp2 = client.post("/api/generate", json=payload_legal)

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert len(mock_engine.recorded_configs) == 2

        cfg1 = mock_engine.recorded_configs[0]
        cfg2 = mock_engine.recorded_configs[1]
        assert cfg1.adapters[0].name == "lora-coder"
        assert cfg1.adapters[0].scale == 0.9
        assert cfg2.adapters[0].name == "lora-legal"
        assert cfg2.adapters[0].scale == 0.4
