"""Exhaustive 6-test triad suite for embedded Intel Arc Model Studio Web UI.

Validates single-page application delivery, browser vs CLI root dispatch,
static CSS/JS asset resolution, and local Ollama inspection endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


@pytest.fixture
def client():
    """Create a FastAPI TestClient configured for UI endpoint testing."""
    manager = ModelManager()
    app = create_app(model_manager=manager, debug=True)
    return TestClient(app)


def test_get_root_serves_html_when_browser_accepts(client: TestClient) -> None:
    """Verify GET / returns HTML dashboard when client Accept header specifies text/html."""
    resp = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Intel® Arc™ Model Studio" in resp.text
    assert "Silicon Cockpit" in resp.text


def test_get_root_serves_plaintext_when_cli(client: TestClient) -> None:
    """Verify GET / returns Ollama plain text contract for CLI and curl clients."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert resp.text == "Ollama is running"


def test_get_ui_endpoint_serves_dashboard(client: TestClient) -> None:
    """Verify GET /ui route returns HTML dashboard unconditionally."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "app-layout" in resp.text
    assert "Chat Playground" in resp.text


def test_get_static_assets(client: TestClient) -> None:
    """Verify static CSS stylesheet and JavaScript application bundle are served cleanly."""
    css_resp = client.get("/static/css/style.css")
    assert css_resp.status_code == 200
    assert "--intel-blue" in css_resp.text

    js_resp = client.get("/static/js/app.js")
    assert js_resp.status_code == 200
    assert "initTabNavigation" in js_resp.text


def test_api_ollama_local_endpoint(client: TestClient) -> None:
    """Verify GET /api/ollama/local endpoint returns structured JSON catalog."""
    resp = client.get("/api/ollama/local")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "count" in data
    assert isinstance(data["models"], list)


def test_api_ollama_inspect_missing_model_returns_404(client: TestClient) -> None:
    """Verify POST /api/ollama/inspect returns HTTP 404 when target model does not exist."""
    resp = client.post("/api/ollama/inspect", json={"model": "nonexistent-model-slug"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
