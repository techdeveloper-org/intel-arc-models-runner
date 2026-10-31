"""Server and hub hardening: path traversal, CORS, optional API key, GGUF path policy (R02 TM-01..05, R03 F1-F3)."""

import json
from pathlib import Path
from typing import Any, List

import pytest
from fastapi.testclient import TestClient

from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.core.path_policy import allowed_roots, ensure_path_allowed, resolve_inside, validate_model_name
from arc_runner.hub.manager import HubManager
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager

BAD_NAMES: List[str] = ["../victim", "..", ".", "/abs/victim", "a/../../victim", "C:\\victim", "a/b", "a\\b", "", "   ", "x\x00y", "~root", "-rf", "a" * 200]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_validate_model_name_rejects_traversal_and_odd_names(name: str) -> None:
    """Only flat, ordinary directory names are accepted as model names."""
    with pytest.raises(ArcRunnerError):
        validate_model_name(name)


@pytest.mark.parametrize("name", ["qwen-2.5-1.5b", "all-minilm-l6-v2", "Model_1.5", "a"])
def test_validate_model_name_accepts_ordinary_names(name: str) -> None:
    """Real alias shapes keep working."""
    assert validate_model_name(name) == name


def test_resolve_inside_stays_within_base_and_rejects_the_base_itself(tmp_path: Path) -> None:
    """The result is strictly below the base directory."""
    base = tmp_path / "models"
    base.mkdir()
    assert resolve_inside(base, "m1") == (base / "m1").resolve()
    for bad in ("..", "../x", str(tmp_path)):
        with pytest.raises(ArcRunnerError):
            resolve_inside(base, bad)


def test_resolve_inside_rejects_a_symlink_that_escapes_the_base(tmp_path: Path) -> None:
    """A link inside the models directory pointing outside must not be followed."""
    base = tmp_path / "models"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    try:
        (base / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted")
    with pytest.raises(ArcRunnerError):
        resolve_inside(base, "link")


def test_ensure_path_allowed_uses_roots_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A path is allowed only below a configured root; ARC_ALLOWED_PATHS adds roots."""
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "f.gguf").write_bytes(b"x")
    (other / "g.gguf").write_bytes(b"x")
    monkeypatch.delenv("ARC_ALLOWED_PATHS", raising=False)
    roots = allowed_roots([root])
    assert ensure_path_allowed(root / "f.gguf", roots) == (root / "f.gguf").resolve()
    with pytest.raises(ArcRunnerError):
        ensure_path_allowed(other / "g.gguf", roots)
    with pytest.raises(ArcRunnerError):
        ensure_path_allowed(root / ".." / "other" / "g.gguf", roots)
    monkeypatch.setenv("ARC_ALLOWED_PATHS", str(other))
    assert ensure_path_allowed(other / "g.gguf", allowed_roots([root])) == (other / "g.gguf").resolve()


@pytest.fixture
def hub(tmp_path: Path) -> HubManager:
    """Hub over a temporary models directory with a victim directory next to it."""
    (tmp_path / "victim").mkdir()
    (tmp_path / "victim" / "keep.txt").write_text("keep")
    return HubManager(base_models_dir=tmp_path / "models")


@pytest.mark.parametrize("name", ["../victim", "..", "/", "a/../../victim"])
def test_hub_delete_with_traversal_name_never_removes_anything_outside(hub: HubManager, tmp_path: Path, name: str) -> None:
    """F1/TM-02: deletion is confined to model directories inside the base directory."""
    with pytest.raises((ArcRunnerError, ModelNotFoundError)):
        hub.delete(name)
    assert (tmp_path / "victim" / "keep.txt").exists()
    assert (tmp_path / "models").exists()


def test_hub_delete_removes_a_normal_model_directory(hub: HubManager, tmp_path: Path) -> None:
    """The legitimate path still works."""
    model = tmp_path / "models" / "m1"
    model.mkdir()
    (model / "openvino_model.xml").write_text("x")
    assert hub.delete("m1") is True
    assert not model.exists()


@pytest.mark.parametrize("destination", ["../escaped", "/tmp/arc_escaped_copy", "a/b", ".."])
def test_hub_copy_refuses_destinations_outside_the_base(hub: HubManager, tmp_path: Path, destination: str) -> None:
    """TM-03: copy never writes outside the models directory."""
    (tmp_path / "models" / "src").mkdir()
    (tmp_path / "models" / "src" / "f").write_text("x")
    with pytest.raises(ArcRunnerError):
        hub.copy("src", destination)
    assert not (tmp_path / "escaped").exists()
    assert not Path("/tmp/arc_escaped_copy").exists()


def test_hub_resolve_rejects_traversal_in_the_model_name(hub: HubManager, tmp_path: Path) -> None:
    """A name like ../victim is not found, even though that directory exists."""
    with pytest.raises((ArcRunnerError, ModelNotFoundError)):
        hub.resolve_model_dir("../victim")


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str = "") -> TestClient:
    """Create the app over a temporary models directory."""
    if api_key:
        monkeypatch.setenv("ARC_API_KEY", api_key)
    else:
        monkeypatch.delenv("ARC_API_KEY", raising=False)
    monkeypatch.delenv("ARC_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ARC_ALLOWED_PATHS", raising=False)
    (tmp_path / "victim").mkdir(exist_ok=True)
    (tmp_path / "victim" / "keep.txt").write_text("keep")
    (tmp_path / "models").mkdir(exist_ok=True)
    return TestClient(create_app(model_manager=ModelManager(model_dirs=[tmp_path / "models"])))


def test_api_delete_with_traversal_returns_an_error_and_keeps_the_victim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F1 through the real HTTP route."""
    client = _client(tmp_path, monkeypatch)
    for name in ("../victim", "..", str(tmp_path / "victim")):
        response = client.request("DELETE", "/api/delete", json={"name": name})
        assert response.status_code in (400, 404), (name, response.status_code, response.text)
    assert (tmp_path / "victim" / "keep.txt").exists()


def test_api_copy_with_traversal_destination_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-03 through the real HTTP route."""
    client = _client(tmp_path, monkeypatch)
    (tmp_path / "models" / "src").mkdir()
    (tmp_path / "models" / "src" / "f").write_text("x")
    response = client.post("/api/copy", json={"source": "src", "destination": "../escaped"})
    assert response.status_code in (400, 409)
    assert not (tmp_path / "escaped").exists()


def test_cors_does_not_reflect_foreign_origins_or_allow_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-01: a web page on another site cannot call the local API; local origins still can."""
    client = _client(tmp_path, monkeypatch)
    foreign = client.options("/api/tags", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in {k.lower() for k in foreign.headers}
    local = client.options("/api/tags", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-credentials" not in {k.lower() for k in local.headers}


def test_cors_origins_can_be_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ARC_CORS_ORIGINS adds explicit origins."""
    monkeypatch.setenv("ARC_CORS_ORIGINS", "https://studio.example")
    (tmp_path / "models").mkdir()
    client = TestClient(create_app(model_manager=ModelManager(model_dirs=[tmp_path / "models"])))
    ok = client.options("/api/tags", headers={"Origin": "https://studio.example", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "https://studio.example"


def test_api_key_is_enforced_when_configured_and_not_otherwise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-01: with ARC_API_KEY set every API route needs it; the UI page and preflight stay open."""
    open_client = _client(tmp_path, monkeypatch)
    assert open_client.get("/api/tags").status_code == 200
    client = _client(tmp_path, monkeypatch, api_key="s3cret-key")
    assert client.get("/api/tags").status_code == 401
    assert client.get("/api/tags", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/tags", headers={"Authorization": "Bearer s3cret-key"}).status_code == 200
    assert client.get("/api/tags", headers={"X-API-Key": "s3cret-key"}).status_code == 200
    assert client.delete("/api/delete", headers={}).status_code in (401, 405, 422)
    assert client.options("/api/tags", headers={"Origin": "http://localhost", "Access-Control-Request-Method": "GET"}).status_code == 200


def test_inspect_rejects_paths_outside_the_allowed_roots_and_leaks_no_file_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-05: arbitrary files are not readable through inspect, and parse errors do not echo content."""
    client = _client(tmp_path, monkeypatch)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"TOPSECRETDATA")
    response = client.post("/api/ollama/inspect", json={"path": str(secret)})
    assert response.status_code in (403, 404) and "TOPS" not in response.text
    inside = tmp_path / "models" / "bad.gguf"
    inside.write_bytes(b"ABCDxxxxxxxxxxxxxxxxxxxx")
    response = client.post("/api/ollama/inspect", json={"path": str(inside)})
    assert response.status_code >= 400 and "ABCD" not in response.text


def test_convert_rejects_source_and_output_outside_the_allowed_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-04: convert reads only allowed sources and writes only below the models directory (POST and the GET stream)."""
    client = _client(tmp_path, monkeypatch)
    src = tmp_path / "outside.gguf"
    src.write_bytes(b"GGUF" + bytes(64))
    response = client.post("/api/gguf/convert", json={"path": str(src), "output_dir": str(tmp_path / "models" / "out")})
    assert response.status_code in (403, 404)
    inside = tmp_path / "models" / "in.gguf"
    inside.write_bytes(b"GGUF" + bytes(64))
    response = client.post("/api/gguf/convert", json={"path": str(inside), "output_dir": str(tmp_path / "escaped_out")})
    assert response.status_code == 403
    assert not (tmp_path / "escaped_out").exists()
    stream = client.get("/api/gguf/convert/stream", params={"path": str(inside), "output_dir": str(tmp_path / "escaped_out")})
    assert not (tmp_path / "escaped_out").exists()
    assert "error" in stream.text.lower() or stream.status_code == 403


def test_model_manager_does_not_accept_arbitrary_absolute_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F3: a directory outside the model directories is only usable if its root is explicitly allowed."""
    monkeypatch.delenv("ARC_ALLOWED_PATHS", raising=False)
    base = tmp_path / "models"
    base.mkdir()
    foreign = tmp_path / "foreign_model"
    foreign.mkdir()
    manager = ModelManager(model_dirs=[base])
    with pytest.raises(ModelNotFoundError):
        manager.resolve_model_path(str(foreign))
    (base / "ok").mkdir()
    assert manager.resolve_model_path("ok") == (base / "ok").resolve()
    monkeypatch.setenv("ARC_ALLOWED_PATHS", str(tmp_path))
    assert ModelManager(model_dirs=[base]).resolve_model_path(str(foreign)) == foreign.resolve()
