"""GET /health liveness route: exact body, no auth, no model access."""

from pathlib import Path
from typing import Any, List

import pytest
from fastapi.testclient import TestClient

from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


class _TouchRecorder(ModelManager):
    """ModelManager that records any attempt to load or list models."""

    def __init__(self, model_dirs: List[Path]) -> None:
        """Create the recorder.

        Args:
            model_dirs: Model directories passed to the base class.
        """
        super().__init__(model_dirs=list(model_dirs))
        self.touched: List[str] = []

    def get_or_load_engine(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call and fail the test path loudly.

        Raises:
            AssertionError: Always.
        """
        self.touched.append("get_or_load_engine")
        raise AssertionError("/health must not load a model")

    def list_available_models(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call and fail loudly.

        Raises:
            AssertionError: Always.
        """
        self.touched.append("list_available_models")
        raise AssertionError("/health must not scan models")


def _client(tmp_path: Path) -> "tuple[TestClient, _TouchRecorder]":
    """Build a client over a recording manager.

    Args:
        tmp_path: Temporary model directory.

    Returns:
        tuple: The client and the recorder.
    """
    manager = _TouchRecorder([tmp_path])
    return TestClient(create_app(model_manager=manager)), manager


def test_health_get_without_api_key_returns_ok_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ARC_API_KEY the route answers 200 and the exact body."""
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    client, manager = _client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"].startswith("application/json")
    assert manager.touched == []


def test_health_get_with_api_key_set_needs_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route is exempt from ApiKeyMiddleware while other routes still need the key."""
    monkeypatch.setenv("ARC_API_KEY", "secret-key")
    client, manager = _client(tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/devices").status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert manager.touched == []


def test_health_head_returns_200_without_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HEAD answers 200 with an empty body, with and without a key configured."""
    for key in (None, "secret-key"):
        if key is None:
            monkeypatch.delenv("ARC_API_KEY", raising=False)
        else:
            monkeypatch.setenv("ARC_API_KEY", key)
        client, _ = _client(tmp_path)
        response = client.head("/health")
        assert response.status_code == 200
        assert response.content == b""


def test_health_route_is_registered_once_and_not_shadowed(tmp_path: Path) -> None:
    """Exactly one route owns /health and the Ollama-style root is unchanged."""
    client, _ = _client(tmp_path)
    owners = [r for r in client.app.routes if getattr(r, "path", None) == "/health"]
    assert len(owners) == 1
    assert client.get("/health").json() == {"status": "ok"}
    root = client.get("/")
    assert root.status_code == 200
    assert "Ollama is running" in root.text


def test_health_post_is_rejected_with_405(tmp_path: Path) -> None:
    """Only GET and HEAD are served."""
    client, _ = _client(tmp_path)
    assert client.post("/health").status_code == 405
