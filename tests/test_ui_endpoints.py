"""Exhaustive triad test suite for embedded Intel Arc Model Studio Web UI.

Validates single-page application delivery, browser vs CLI root dispatch,
static CSS/JS asset resolution, local Ollama inspection endpoints, GGUF
transpilation API, and NPU embedding similarity test endpoints.
"""

import json
from pathlib import Path
from unittest import mock
import struct

from fastapi.testclient import TestClient
import pytest

from arc_runner.engine.embedding_engine import EmbeddingEngine
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


@pytest.fixture
def client(tmp_path: Path):
    """Create a FastAPI TestClient whose models directory is the test's tmp_path."""
    manager = ModelManager(model_dirs=[tmp_path])
    app = create_app(model_manager=manager, debug=True)
    return TestClient(app)


@pytest.fixture
def client_with_embed(tmp_path: Path) -> TestClient:
    """Create a TestClient with a pre-cached mock EmbeddingEngine."""
    mock_engine = mock.MagicMock(spec=EmbeddingEngine)
    mock_engine.dimension = 768
    mock_engine.device = "NPU"
    mock_engine.embed_query.return_value = [0.5 / (768 ** 0.5)] * 768
    mock_engine.compute_similarity.return_value = 0.92

    mgr = ModelManager(model_dirs=[tmp_path])
    mgr._embedding_cache["nomic-embed-text:NPU"] = mock_engine

    app = create_app(model_manager=mgr, debug=True)
    return TestClient(app)


def _make_minimal_gguf(path: Path) -> None:
    """Write a valid minimal GGUF v3 container with one F32 tensor to disk."""
    from arc_runner.quant.models import GGMLType, GGUFValueType

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 1))

        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", GGUFValueType.STRING.value))
        val = b"llama"
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        t_name = b"token_embd.weight"
        f.write(struct.pack("<Q", len(t_name)))
        f.write(t_name)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<I", GGMLType.F32.value))
        f.write(struct.pack("<Q", 0))

        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(struct.pack("<16f", *([1.0] * 16)))


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


def test_ui_contains_embeddings_tab(client: TestClient) -> None:
    """Verify HTML dashboard includes NPU Embeddings Studio tab markup."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "NPU Embeddings" in resp.text
    assert "tab-embeddings" in resp.text


def test_ui_contains_multimodal_image_input(client: TestClient) -> None:
    """Verify HTML dashboard chat section contains image file input for multimodal vision."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "chat-image-input" in resp.text
    assert "chat-image-preview-container" in resp.text


def test_ui_js_contains_embeddings_studio_init(client: TestClient) -> None:
    """Verify app.js registers initEmbeddingsStudio function and tab title."""
    resp = client.get("/static/js/app.js")
    assert resp.status_code == 200
    assert "initEmbeddingsStudio" in resp.text
    assert "tab-embeddings" in resp.text


def test_ui_js_contains_gguf_convert_handler(client: TestClient) -> None:
    """Verify app.js contains GGUF convert button binding and /api/gguf/convert call."""
    resp = client.get("/static/js/app.js")
    assert resp.status_code == 200
    assert "btn-convert-gguf" in resp.text
    assert "/api/gguf/convert" in resp.text


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


def test_api_gguf_convert_missing_file_returns_404(client: TestClient, tmp_path: Path) -> None:
    """Verify POST /api/gguf/convert returns 404 when the source GGUF path does not exist."""
    resp = client.post(
        "/api/gguf/convert",
        json={"path": str(tmp_path / "missing.gguf"), "output_dir": str(tmp_path / "out")},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_api_gguf_convert_valid_container_success(
    client: TestClient, tmp_path: Path
) -> None:
    """Verify POST /api/gguf/convert transpiles a valid GGUF container to OpenVINO IR."""
    gguf_path = tmp_path / "test_model.gguf"
    _make_minimal_gguf(gguf_path)
    out_dir = tmp_path / "openvino_out"

    resp = client.post(
        "/api/gguf/convert",
        json={"path": str(gguf_path), "output_dir": str(out_dir), "precision": "int4"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "model_name" in data
    assert "xml_path" in data
    assert "bin_path" in data
    assert data["target_precision"].upper() == "INT4"
    assert data["duration_ms"] > 0.0


def test_api_embeddings_test_returns_similarity(
    client_with_embed: TestClient,
) -> None:
    """Verify POST /api/embeddings/test returns cosine similarity and vector metadata."""
    resp = client_with_embed.post(
        "/api/embeddings/test",
        json={
            "query_a": "Intel Arc GPU XMX matrix acceleration",
            "query_b": "Hardware AI acceleration matrix engines",
            "model": "nomic-embed-text",
            "device": "NPU",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "cosine_similarity" in data
    assert "dimension" in data
    assert "angular_distance" in data
    assert "duration_ms" in data
    assert "sample_vec_a" in data
    assert len(data["sample_vec_a"]) == 5
    assert data["model"] == "nomic-embed-text"
    assert data["device"] == "NPU"


def test_api_embeddings_test_model_not_found_returns_error(
    client: TestClient,
) -> None:
    """Verify POST /api/embeddings/test returns error when embedding model is missing."""
    resp = client.post(
        "/api/embeddings/test",
        json={
            "query_a": "hello",
            "query_b": "world",
            "model": "nonexistent-embed-model",
            "device": "NPU",
        },
    )
    assert resp.status_code in (404, 500)


def test_sse_convert_endpoint_200_stream_headers(client: TestClient, tmp_path: Path) -> None:
    """Verify GET /api/gguf/convert/stream returns 200 OK with text/event-stream content type."""
    gguf_path = tmp_path / "model.gguf"
    out_dir = tmp_path / "ov_out"
    _make_minimal_gguf(gguf_path)

    resp = client.get(f"/api/gguf/convert/stream?path={gguf_path}&output_dir={out_dir}")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_sse_convert_endpoint_wire_framing(client: TestClient, tmp_path: Path) -> None:
    """Verify SSE frames adhere to standard 'event: ...\\ndata: ...\\n\\n' wire framing."""
    gguf_path = tmp_path / "model.gguf"
    out_dir = tmp_path / "ov_out"
    _make_minimal_gguf(gguf_path)

    resp = client.get(f"/api/gguf/convert/stream?path={gguf_path}&output_dir={out_dir}")
    assert resp.status_code == 200
    text = resp.text
    assert "event: log\n" in text or "event: progress\n" in text
    assert "data: " in text


def test_sse_convert_endpoint_full_lifecycle(client: TestClient, tmp_path: Path) -> None:
    """Verify stream lifecycle contains log, progress, and complete events."""
    gguf_path = tmp_path / "model.gguf"
    out_dir = tmp_path / "ov_out"
    _make_minimal_gguf(gguf_path)

    resp = client.get(f"/api/gguf/convert/stream?path={gguf_path}&output_dir={out_dir}")
    assert resp.status_code == 200
    text = resp.text
    assert "event: log" in text
    assert "event: progress" in text
    assert "event: complete" in text


def test_sse_convert_endpoint_complete_payload_keys(client: TestClient, tmp_path: Path) -> None:
    """Verify event: complete payload contains all expected artifact metadata keys."""
    gguf_path = tmp_path / "model.gguf"
    out_dir = tmp_path / "ov_out"
    _make_minimal_gguf(gguf_path)

    resp = client.get(f"/api/gguf/convert/stream?path={gguf_path}&output_dir={out_dir}")
    assert resp.status_code == 200

    complete_data = None
    for block in resp.text.split("\n\n"):
        if "event: complete" in block:
            for line in block.split("\n"):
                if line.startswith("data: "):
                    complete_data = json.loads(line[6:])

    assert complete_data is not None
    assert "model_name" in complete_data
    assert "total_tensors" in complete_data
    assert "target_precision" in complete_data
    assert "xml_path" in complete_data
    assert "bin_path" in complete_data
    assert "sha256_hash" in complete_data
    assert "final_sqnr_db" in complete_data


def test_sse_convert_endpoint_missing_file_error_frame(client: TestClient, tmp_path: Path) -> None:
    """Verify non-existent GGUF path yields event: error frame with descriptive message."""
    missing_path = tmp_path / "nonexistent.gguf"
    out_dir = tmp_path / "ov_out"

    resp = client.get(f"/api/gguf/convert/stream?path={missing_path}&output_dir={out_dir}")
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "Source GGUF file does not exist" in resp.text


def test_sse_convert_endpoint_missing_required_params(client: TestClient) -> None:
    """Verify request missing required query parameters returns 422 Unprocessable Entity."""
    resp = client.get("/api/gguf/convert/stream")
    assert resp.status_code == 422


def test_sse_convert_endpoint_invalid_precision_error(client: TestClient, tmp_path: Path) -> None:
    """Verify unsupported target precision yields event: error frame."""
    gguf_path = tmp_path / "model.gguf"
    out_dir = tmp_path / "ov_out"
    _make_minimal_gguf(gguf_path)

    resp = client.get(f"/api/gguf/convert/stream?path={gguf_path}&output_dir={out_dir}&precision=int2")
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "Unsupported target precision" in resp.text


def test_sse_convert_endpoint_directory_traversal_protection(client: TestClient, tmp_path: Path) -> None:
    """Verify a source path outside the allowed roots is refused with 403."""
    out_dir = tmp_path / "ov_out"
    resp = client.get(f"/api/gguf/convert/stream?path=../../../../nonexistent.gguf&output_dir={out_dir}")
    assert resp.status_code == 403


def test_sse_convert_endpoint_special_characters_in_path(client: TestClient, tmp_path: Path) -> None:
    """Verify URL encoded paths with spaces and symbols are resolved without 500 error."""
    special_dir = tmp_path / "Models Dir (v1.0)"
    special_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = special_dir / "my model [llama].gguf"
    out_dir = tmp_path / "ov out"
    _make_minimal_gguf(gguf_path)

    resp = client.get(f"/api/gguf/convert/stream?path={gguf_path}&output_dir={out_dir}")
    assert resp.status_code == 200
    assert "event: complete" in resp.text


def test_ui_html_modal_contains_accessible_progress_attributes(client: TestClient) -> None:
    """Verify GET /ui template includes Phase 9 conversion controls and progress studio elements."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    html = resp.text

    assert 'id="convert-precision-select"' in html
    assert 'id="convert-device-select"' in html
    assert 'id="convert-output-dir-input"' in html
    assert 'id="gguf-convert-progress-card"' in html
    assert 'id="convert-progress-fill"' in html
    assert 'id="convert-progress-pct"' in html
    assert 'id="convert-terminal-box"' in html
    assert 'id="convert-completion-banner"' in html


