"""Server hardening: Host validation (DNS rebinding), browser CSRF guards, open paths, request-id hygiene, path leakage."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi import params
from fastapi.testclient import TestClient

from arc_runner.hub.manager import HubManager
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager
from tests.anthropic_support import MODEL, ScriptedEngine, body, make_app, make_client

MSG = "/v1/messages"
EVIL = "https://evil.example"


def _local_client(tmp_path: Path, engine: ScriptedEngine) -> TestClient:
    """TestClient whose default Host is localhost (the production-allowed name)."""
    return TestClient(make_app(tmp_path, engine), base_url="http://localhost")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Real app over a scripted engine; Host defaults to localhost so the Host check is exercised."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("ARC_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    return _local_client(tmp_path, ScriptedEngine())


def _get(client: TestClient, host: Optional[str], path: str = "/health", **kwargs: Any) -> Any:
    headers = dict(kwargs.pop("headers", {}))
    if host is not None:
        headers["Host"] = host
    return client.get(path, headers=headers, **kwargs)


@pytest.mark.parametrize(
    "host",
    ["localhost", "localhost:11434", "LOCALHOST:11434", "LocalHost", "localhost.", "127.0.0.1", "127.0.0.1:11434", "[::1]", "[::1]:11434", "192.168.1.5:80", "[2001:db8::1]:8080", "10.0.0.7"],
)
def test_host_check_allows_localhost_and_ip_literals_with_or_without_port(client: TestClient, host: str) -> None:
    """Loopback names and any IP literal are accepted, case-insensitively, port ignored."""
    assert _get(client, host).status_code == 200


@pytest.mark.parametrize(
    "host",
    ["evil.example", "evil.example:80", "evil.example:11434", "localhost.evil.example", "127.0.0.1.evil.example", "", "localhost:abc", "localhost:", "[::1", "::1", "a b", "user@localhost", "localhost/x", "localhost:99999999", "0x7f.1", "[::1%eth0]"],
)
def test_host_check_rejects_other_hosts_with_421_and_does_not_echo_the_header(client: TestClient, host: str) -> None:
    """Anything else gets 421 Misdirected Request in the standard error shape, without echoing the header."""
    response = _get(client, host)
    assert response.status_code == 421, response.text
    assert response.json() == {"error": "Host header not allowed", "code": "HOST_NOT_ALLOWED"}


def test_host_check_rejects_a_bad_host_on_every_route_including_health_and_preflight(client: TestClient) -> None:
    """/health, API routes, the Claude Code probe and CORS preflights are all covered."""
    assert _get(client, "evil.example", "/health").status_code == 421
    assert _get(client, "evil.example", "/api/tags").status_code == 421
    assert client.head("/api/hello", headers={"Host": "evil.example"}).status_code == 421
    pre = client.options(MSG, headers={"Host": "evil.example", "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 421


def test_host_check_rejects_multiple_host_headers(client: TestClient) -> None:
    """Two Host headers are ambiguous and refused even when both are individually allowed."""
    response = client.get("/health", headers=[("host", "localhost"), ("host", "evil.example")])
    assert response.status_code == 421
    response = client.get("/health", headers=[("host", "localhost"), ("host", "127.0.0.1")])
    assert response.status_code == 421


def test_host_check_error_for_anthropic_paths_uses_the_anthropic_envelope(client: TestClient) -> None:
    """/v1/messages errors keep the Anthropic shape and the request-id header."""
    response = client.post(MSG, json=body(), headers={"Host": "evil.example"})
    assert response.status_code == 421
    data = response.json()
    assert data["type"] == "error" and data["error"]["type"] == "invalid_request_error"
    assert data["request_id"] == response.headers["request-id"]
    assert "evil.example" not in response.text


def test_allowed_hosts_env_adds_names_case_insensitively_and_ignores_invalid_entries(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """ARC_ALLOWED_HOSTS is a comma list of bare names; whitespace is trimmed, junk entries are ignored."""
    monkeypatch.setenv("ARC_ALLOWED_HOSTS", "  myhost , Other.Example ,bad entry,*,http://scheme.example,withport:80,,-bad-,a/b")
    for ok in ("myhost", "MYHOST:8000", "other.example", "Other.Example:11434"):
        assert _get(client, ok).status_code == 200, ok
    for bad in ("bad", "entry", "scheme.example", "withport", "withport:80", "evil.example", "-bad-", "a", "x.example"):
        assert _get(client, bad).status_code == 421, bad


def test_allowed_hosts_env_empty_or_unset_keeps_only_the_defaults(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty variable adds nothing; testserver is not allowed in production configuration."""
    monkeypatch.setenv("ARC_ALLOWED_HOSTS", "")
    assert _get(client, "testserver").status_code == 421
    assert _get(client, "localhost").status_code == 200


def test_testclient_default_host_is_allowed_only_through_the_test_environment_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the hook, the plain TestClient host is refused; with it, accepted (production behaviour unchanged)."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    plain = TestClient(create_app(model_manager=ModelManager(model_dirs=[tmp_path])))
    assert plain.get("/health").status_code == 421
    monkeypatch.setenv("ARC_ALLOWED_HOSTS", "testserver")
    assert plain.get("/health").status_code == 200


# ---------------------------------------------------------------- H3: CSRF


def _post_raw(client: TestClient, path: str, content_type: Optional[str], origin: Optional[str], payload: Optional[Dict[str, Any]] = None, **extra: str) -> Any:
    headers: Dict[str, str] = {"Host": "localhost:11434", **extra}
    if content_type is not None:
        headers["Content-Type"] = content_type
    if origin is not None:
        headers["Origin"] = origin
    data = json.dumps(payload if payload is not None else body())
    return client.post(path, content=data, headers=headers)


def test_cross_origin_text_plain_post_to_messages_is_rejected_and_never_reaches_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The proven attack: text/plain from an evil page must not run inference."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    engine = ScriptedEngine()
    client = _local_client(tmp_path, engine)
    response = _post_raw(client, MSG, "text/plain", EVIL)
    assert response.status_code in (403, 415)
    assert engine.prompts == []


def test_cross_origin_text_plain_post_to_pull_is_rejected(client: TestClient) -> None:
    """Other state-changing routes are protected the same way."""
    response = _post_raw(client, "/api/pull", "text/plain", EVIL, {"name": "llama-3.2-3b", "stream": False})
    assert response.status_code in (403, 415)


@pytest.mark.parametrize("origin", [EVIL, "null", "http://localhost.evil.example", "http://evil.example:11434", "file://"])
def test_foreign_origin_gets_403_even_with_a_json_content_type(client: TestClient, origin: str) -> None:
    """An Origin outside the local policy is refused for state-changing requests."""
    response = _post_raw(client, MSG, "application/json", origin)
    assert response.status_code == 403, response.text
    assert response.json()["error"]["type"] == "permission_error"


def test_foreign_origin_is_also_refused_on_get_routes_including_the_state_changing_convert_stream(client: TestClient) -> None:
    """EventSource/fetch GETs from another site are refused, the convert stream in particular."""
    stream = client.get("/api/gguf/convert/stream", params={"path": "x.gguf", "output_dir": "y"}, headers={"Host": "localhost", "Origin": EVIL})
    assert stream.status_code == 403
    assert client.get("/api/tags", headers={"Host": "localhost", "Origin": EVIL}).status_code == 403


def test_local_origin_with_json_is_accepted_and_no_origin_is_accepted(client: TestClient) -> None:
    """Local web apps on any port, and curl/SDK/Claude Code (no Origin), keep working."""
    assert _post_raw(client, MSG, "application/json", "http://localhost:3000").status_code == 200
    assert _post_raw(client, MSG, "application/json", "http://127.0.0.1:5173").status_code == 200
    assert _post_raw(client, MSG, "application/json", "http://[::1]:8080").status_code == 200
    assert _post_raw(client, MSG, "application/json", None).status_code == 200


def test_arc_cors_origins_entries_are_allowed_origins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same list that opens CORS opens the Origin check."""
    monkeypatch.setenv("ARC_CORS_ORIGINS", "https://chat.example, https://other.example")
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    client = _local_client(tmp_path, ScriptedEngine())
    assert _post_raw(client, MSG, "application/json", "https://chat.example").status_code == 200
    assert _post_raw(client, MSG, "application/json", "https://notlisted.example").status_code == 403


def test_same_origin_requests_pass_when_the_origin_equals_the_validated_host(client: TestClient) -> None:
    """The bundled UI served from a LAN address posts with Origin equal to its own Host."""
    ok = client.post(MSG, content=json.dumps(body()), headers={"Host": "192.168.1.5:11434", "Origin": "http://192.168.1.5:11434", "Content-Type": "application/json"})
    assert ok.status_code == 200
    other = client.post(MSG, content=json.dumps(body()), headers={"Host": "192.168.1.5:11434", "Origin": "http://192.168.1.6:11434", "Content-Type": "application/json"})
    assert other.status_code == 403


def test_cross_site_fetch_metadata_without_origin_is_refused_for_state_changing_requests(client: TestClient) -> None:
    """An <img> or navigation GET carries no Origin but Sec-Fetch-Site: cross-site; the convert stream and POSTs refuse it."""
    headers = {"Host": "localhost", "Sec-Fetch-Site": "cross-site"}
    assert client.get("/api/gguf/convert/stream", params={"path": "x.gguf", "output_dir": "y"}, headers=headers).status_code == 403
    assert client.post(MSG, json=body(), headers=headers).status_code == 403
    assert client.get("/ui", headers=headers).status_code == 200


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", None, "application/jsonx", "json"])
def test_non_json_content_type_with_a_body_gets_415(client: TestClient, content_type: Optional[str]) -> None:
    """A body on a mutating method must be declared application/json."""
    for path, payload in ((MSG, body()), ("/api/pull", {"name": "llama-3.2-3b", "stream": False})):
        response = _post_raw(client, path, content_type, None, payload)
        assert response.status_code == 415, (path, response.text)


def test_non_json_content_type_on_messages_uses_the_anthropic_envelope(client: TestClient) -> None:
    """The 415 on /v1/messages is an invalid_request_error envelope with a request id."""
    response = _post_raw(client, MSG, "text/plain", None)
    assert response.status_code == 415
    assert response.json()["type"] == "error" and response.headers["request-id"] == response.json()["request_id"]


@pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8", "APPLICATION/JSON", "Application/Json;charset=UTF-8"])
def test_json_content_type_variants_are_accepted(client: TestClient, content_type: str) -> None:
    """Parameters and case do not matter."""
    assert _post_raw(client, MSG, content_type, None).status_code == 200


def test_bodyless_mutating_request_is_not_rejected_for_its_content_type(client: TestClient) -> None:
    """DELETE without a body reaches FastAPI validation (422) rather than failing with 415."""
    response = client.delete("/api/delete", headers={"Host": "localhost"})
    assert response.status_code != 415


def test_no_route_takes_form_or_file_bodies_so_the_content_type_exemption_list_is_empty() -> None:
    """If a route ever accepts multipart or form data it must be added to NON_JSON_BODY_PATHS (this test then fails)."""
    from arc_runner.server.security import NON_JSON_BODY_PATHS

    app = create_app(model_manager=ModelManager(model_dirs=[]))
    needs_exemption: List[str] = []
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant and any(isinstance(p.field_info, (params.Form, params.File)) for p in dependant.body_params):
            needs_exemption.append(route.path)
    assert sorted(needs_exemption) == sorted(NON_JSON_BODY_PATHS)


def test_preflight_behaviour_is_unchanged_for_local_and_foreign_origins(client: TestClient) -> None:
    """CORS still answers preflights itself: local origin allowed, foreign refused by CORS (400), no Content-Type needed."""
    ok = client.options(MSG, headers={"Host": "localhost", "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    bad = client.options(MSG, headers={"Host": "localhost", "Origin": EVIL, "Access-Control-Request-Method": "POST"})
    assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers


def test_claude_code_style_requests_pass_all_checks(client: TestClient) -> None:
    """HEAD /api/hello and a JSON POST with Host 127.0.0.1:11434, no Origin, SDK-style headers work."""
    headers = {"Host": "127.0.0.1:11434", "anthropic-version": "2023-06-01", "x-api-key": "k"}
    assert client.head("/api/hello", headers=headers).status_code == 200
    assert client.post(MSG, json=body(), headers=headers).status_code == 200


# ---------------------------------------------------------------- L1: tool-call parser depth


def test_parse_tool_calls_deeply_nested_json_returns_none_instead_of_raising() -> None:
    """Hostile model output with 200000 nested brackets (RecursionError in json) becomes plain text."""
    from arc_runner.server.anthropic_convert import parse_tool_calls

    for opener in ("[", '{"name": "get_weather", "arguments": ' + '{"a": ' * 100000):
        text = "<tool_call>" + opener * (200000 if opener == "[" else 1) + "</tool_call>"
        assert parse_tool_calls(text, {"get_weather"}) is None


def test_messages_blocking_and_stream_answer_plain_text_for_depth_bomb_tool_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route returns 200 with the whole output as text (all-or-nothing), blocking and streaming."""
    from tests.anthropic_support import TOOL_WEATHER, parse_sse

    bomb = "<tool_call>" + "[" * 200000 + "</tool_call>"
    client = make_client(tmp_path, ScriptedEngine(pieces=[bomb]))
    response = client.post(MSG, json=body(tools=[TOOL_WEATHER]))
    assert response.status_code == 200, response.text
    content = response.json()["content"]
    assert [block["type"] for block in content] == ["text"] and content[0]["text"] == bomb
    streamed = client.post(MSG, json=body(tools=[TOOL_WEATHER], stream=True))
    names = [name for name, _ in parse_sse(streamed.text)]
    assert streamed.status_code == 200 and "error" not in names and names[-1] == "message_stop"


# ---------------------------------------------------------------- L3: open paths with an API key


@pytest.fixture
def keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App with ARC_API_KEY set (debug on so /docs exists)."""
    monkeypatch.setenv("ARC_API_KEY", "secret-key")
    return _local_client(tmp_path, ScriptedEngine())


@pytest.mark.parametrize("path", ["/", "/health", "/docs", "/openapi.json", "/static/css/style.css", "/static/js/app.js"])
def test_api_key_open_paths_need_no_key(keyed: TestClient, path: str) -> None:
    """Exactly the documented paths are exempt."""
    assert keyed.get(path, headers={"Host": "localhost"}).status_code == 200, path


def test_api_key_exempts_the_claude_code_probe_for_head_and_get_only(keyed: TestClient) -> None:
    """HEAD and GET /api/hello are open; other methods are not."""
    assert keyed.head("/api/hello").status_code == 200
    assert keyed.get("/api/hello").status_code == 200
    assert keyed.post("/api/hello", json={}).status_code == 401


@pytest.mark.parametrize("path", ["/staticfoo", "/static", "/ui", "/api/tags", "/v1/models", "/healthz", "/docsx", "/openapi.json.bak", "/api/hello/x"])
def test_api_key_protects_everything_else_including_lookalike_prefixes(keyed: TestClient, path: str) -> None:
    """Lookalike paths and the UI page need the key."""
    assert keyed.get(path).status_code == 401, path


def test_api_key_options_requests_pass_the_key_check(keyed: TestClient) -> None:
    """Preflights cannot carry the key, so OPTIONS is never answered 401."""
    assert keyed.options("/api/tags").status_code != 401


# ---------------------------------------------------------------- L4: X-Request-ID


@pytest.mark.parametrize("supplied", ["a" * 65, "has space", "<script>x</script>", "caf\u00e9", "semi;colon", "a" * 5000])
def test_request_id_header_that_is_too_long_or_has_odd_characters_is_replaced(client: TestClient, supplied: str) -> None:
    """Untrusted ids outside [A-Za-z0-9._-]{1,64} are never echoed or logged; a fresh id is issued."""
    response = client.get("/health", headers={"X-Request-ID": supplied.encode("utf-8")})
    echoed = response.headers["x-request-id"]
    assert echoed != supplied and echoed.startswith("req-") and len(echoed) <= 64


@pytest.mark.parametrize("supplied", ["abc-123", "A.b_c-9", "a" * 64])
def test_request_id_header_in_the_safe_form_is_kept(client: TestClient, supplied: str) -> None:
    """Well-formed correlation ids still propagate."""
    assert client.get("/health", headers={"X-Request-ID": supplied}).headers["x-request-id"] == supplied


# ---------------------------------------------------------------- M1: /api/pull path escape


class _SpyDownloader:
    """Downloader double that records calls and fails the test if the disk would be touched."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Path]] = []

    def download_stream(self, repo_id: str, destination_dir: Path, expected_size: Optional[int] = None) -> Any:
        self.calls.append((repo_id, destination_dir))
        return iter(())

    def download_sync(self, repo_id: str, destination_dir: Path) -> Path:
        self.calls.append((repo_id, destination_dir))
        return destination_dir


PULL_BAD_NAMES = ["../../x", "../x", "..", "/abs/x", "/etc/passwd", "C:\\x", "a/b", "a\\b", "x\x00y", "a" * 200, "~root", "org/model", ".hidden"]


def _snapshot(root: Path) -> List[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize("name", PULL_BAD_NAMES)
@pytest.mark.parametrize("stream", [True, False])
def test_hub_pull_rejects_unsafe_names_before_touching_the_disk(tmp_path: Path, name: str, stream: bool) -> None:
    """Typed 400 and no directory anywhere, for streaming and blocking pulls."""
    from arc_runner.core.exceptions import ArcRunnerError

    hub = HubManager(base_models_dir=tmp_path / "models")
    spy = _SpyDownloader()
    hub._downloader = spy  # type: ignore[assignment]
    before = _snapshot(tmp_path)
    with pytest.raises(ArcRunnerError) as info:
        hub.pull(name, stream=stream)
    assert info.value.status_code == 400 and info.value.error_code == "INVALID_MODEL_NAME"
    assert spy.calls == [] and _snapshot(tmp_path) == before


def test_hub_pull_accepts_curated_aliases_and_plain_repo_names_inside_the_models_dir(tmp_path: Path) -> None:
    """A tag is dropped like everywhere else and the target directory lies below the base."""
    hub = HubManager(base_models_dir=tmp_path / "models")
    spy = _SpyDownloader()
    hub._downloader = spy  # type: ignore[assignment]
    hub.pull("llama-3.2-3b:latest", stream=False)
    hub.pull("gpt2", stream=True)
    base = (tmp_path / "models").resolve()
    assert [repo for repo, _ in spy.calls] == ["OpenVINO/Llama-3.2-3B-Instruct-int4-ov", "gpt2"]
    assert all(dest.parent == base for _, dest in spy.calls)


@pytest.mark.parametrize("stream", [True, False])
def test_api_pull_with_traversal_name_returns_400_and_creates_nothing(tmp_path: Path, stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over HTTP: a typed 400 before any streaming starts, nothing created outside (or inside) the models dir."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    models = tmp_path / "models"
    models.mkdir()
    app = create_app(model_manager=ModelManager(model_dirs=[models]))
    app.state.hub_manager = HubManager(base_models_dir=models)
    spy = _SpyDownloader()
    app.state.hub_manager._downloader = spy
    client = TestClient(app)
    before = _snapshot(tmp_path)
    for name in ("../../arcpoc/x", "/tmp/arcpoc", "a/b", "x\u0000y"):
        response = client.post("/api/pull", json={"name": name, "stream": stream}, headers={"Host": "localhost"})
        assert response.status_code == 400, (name, response.text)
        assert response.json()["code"] == "INVALID_MODEL_NAME"
    assert spy.calls == [] and _snapshot(tmp_path) == before


def test_api_show_delete_copy_reject_unsafe_names_without_touching_the_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The other hub entry points already confine names; copy now answers a typed 400 for a bad destination."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    models = tmp_path / "models"
    (models / "src").mkdir(parents=True)
    (models / "src" / "f").write_text("x")
    client = TestClient(create_app(model_manager=ModelManager(model_dirs=[models])))
    before = _snapshot(tmp_path)
    h = {"Host": "localhost"}
    for name in ("../models", "..", "/abs", "a/b"):
        assert client.post("/api/show", json={"name": name}, headers=h).status_code in (400, 404)
        assert client.request("DELETE", "/api/delete", json={"name": name}, headers=h).status_code in (400, 404)
        copy = client.post("/api/copy", json={"source": "src", "destination": name}, headers=h)
        assert copy.status_code == 400, (name, copy.text)
    assert _snapshot(tmp_path) == before
    dup = client.post("/api/copy", json={"source": "src", "destination": "src"}, headers=h)
    assert dup.status_code == 409


# ---------------------------------------------------------------- M4: path leakage


def _bodies_leaking(tmp_path: Path, responses: List[Tuple[str, Any]]) -> List[str]:
    needles = {str(tmp_path), str(tmp_path.resolve()), tmp_path.as_posix()}
    return [label for label, r in responses if any(n in r.text for n in needles)]


def test_error_bodies_on_every_route_family_contain_no_filesystem_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Model-not-found on Ollama, OpenAI, Anthropic, embeddings, lora, ui routes: only names, never tmp_path."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    client = make_client(tmp_path, ScriptedEngine())
    h = {"Host": "localhost"}
    calls = [
        ("chat", client.post("/api/chat", json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]}, headers=h)),
        ("generate", client.post("/api/generate", json={"model": "nope", "prompt": "hi"}, headers=h)),
        ("show", client.post("/api/show", json={"name": "nope"}, headers=h)),
        ("delete", client.request("DELETE", "/api/delete", json={"name": "nope"}, headers=h)),
        ("copy", client.post("/api/copy", json={"source": "nope", "destination": "new"}, headers=h)),
        ("openai-chat", client.post("/v1/chat/completions", json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]}, headers=h)),
        ("openai-model", client.get("/v1/models/nope", headers=h)),
        ("embeddings", client.post("/api/embeddings", json={"model": "nope", "prompt": "hi"}, headers=h)),
        ("anthropic", client.post(MSG, json=body(model="nope"), headers=h)),
        ("lora-load", client.post("/api/adapters/load", json={"adapter": "nope"}, headers=h)),
        ("lora-inspect", client.post("/api/adapters/inspect", json={"path": "nope"}, headers=h)),
        ("ollama-inspect", client.post("/api/ollama/inspect", json={"model": "nope"}, headers=h)),
        ("aot", client.post("/api/aot/compile", json={"model_id": "nope", "device": "GPU"}, headers=h)),
    ]
    for label, response in calls:
        assert response.status_code >= 400, label
    assert _bodies_leaking(tmp_path, calls) == []
    not_found = dict(calls)["chat"]
    assert not_found.status_code == 404 and "nope" in not_found.text


class _ExplodingManager(ModelManager):
    """Manager whose engine loads fail with an exception that carries a filesystem path."""

    def __init__(self, leak: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leak = leak

    def get_or_load_engine(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"cannot open {self._leak}/openvino_model.xml")

    def get_or_load_embedding_engine(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"cannot open {self._leak}/emb.xml")


def test_500_bodies_hide_raw_exception_text_and_the_detail_is_logged_server_side(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Engine, embedding, pull, delete, lora, aot, convert and studio failures: stable message out, detail in the log."""
    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    manager = _ExplodingManager(str(tmp_path), model_dirs=[tmp_path])
    app = create_app(model_manager=manager)
    hub = HubManager(base_models_dir=tmp_path)
    caplog.set_level(logging.ERROR)
    logging.getLogger("arc_runner").addHandler(caplog.handler)
    monkeypatch.setattr(logging.getLogger("arc_runner"), "propagate", True)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError(f"[Errno 13] Permission denied: '{tmp_path}/m'")

    monkeypatch.setattr(hub, "pull", boom)
    monkeypatch.setattr(hub, "delete", boom)
    app.state.hub_manager = hub
    monkeypatch.setattr(manager.lora_manager, "load_adapter", boom)
    monkeypatch.setattr(manager.lora_manager, "parse_adapter_metadata", boom)
    client = TestClient(app)
    h = {"Host": "localhost"}
    chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    calls = [
        ("chat", client.post("/api/chat", json=chat, headers=h)),
        ("generate", client.post("/api/generate", json={"model": "m", "prompt": "x"}, headers=h)),
        ("openai", client.post("/v1/chat/completions", json=chat, headers=h)),
        ("embeddings", client.post("/api/embeddings", json={"model": "m", "prompt": "x"}, headers=h)),
        ("embed-batch", client.post("/api/embed", json={"model": "m", "input": ["x"]}, headers=h)),
        ("pull-sync", client.post("/api/pull", json={"name": "llama-3.2-3b", "stream": False}, headers=h)),
        ("delete", client.request("DELETE", "/api/delete", json={"name": "m"}, headers=h)),
        ("lora-load", client.post("/api/adapters/load", json={"adapter": "a"}, headers=h)),
        ("lora-inspect", client.post("/api/adapters/inspect", json={"path": "a"}, headers=h)),
        ("embeddings-test", client.post("/api/embeddings/test", json={"model": "m", "query_a": "a", "query_b": "b"}, headers=h)),
    ]
    for label, response in calls:
        assert response.status_code >= 400, (label, response.status_code)
    assert _bodies_leaking(tmp_path, calls) == []
    assert not any("Errno" in r.text or "cannot open" in r.text for _, r in calls)
    assert dict(calls)["chat"].json() == {"detail": "Failed to load engine"}
    detail_logged = [r for r in caplog.records if getattr(r, "error_detail", "").startswith("cannot open")]
    assert detail_logged and all(r.levelno >= logging.ERROR for r in detail_logged)


def test_model_load_error_response_hides_the_model_path_but_keeps_status_code_and_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typed ModelLoadError that escapes a route is answered with the device only; the path stays in the log."""
    from arc_runner.core.exceptions import ModelLoadError

    monkeypatch.delenv("ARC_ALLOWED_HOSTS", raising=False)
    app = create_app(model_manager=ModelManager(model_dirs=[tmp_path]))

    @app.get("/boom-load")
    async def boom_load() -> None:
        raise ModelLoadError("native said no", str(tmp_path / "m"), "NPU")

    response = TestClient(app, base_url="http://localhost").get("/boom-load")
    assert response.status_code == 500
    assert response.json() == {"error": "Failed to load model on NPU", "code": "MODEL_LOAD_FAILED"}
