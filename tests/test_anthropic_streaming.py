"""Streaming behaviour that needs real incremental I/O: pings, disconnects, concurrency and the uvicorn wire format."""

import asyncio
import json
import re
import socket
import threading
from pathlib import Path
from typing import Any, List, Tuple

import httpx
import pytest
import uvicorn

from arc_runner.server import routes_anthropic
from tests.anthropic_support import (
    CALL_PIECES,
    FIXTURES,
    ScriptedEngine,
    asgi_post,
    body,
    make_app,
    measured,
    normalize_ids,
    parse_sse,
)

MSG = "/v1/messages"


async def test_stream_emits_ping_events_while_the_engine_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """During a slow prefill the client gets message_start and then ping events, before any token exists."""
    monkeypatch.setattr(routes_anthropic, "STREAM_PING_INTERVAL_SECONDS", 0.01)
    gate = threading.Event()
    engine = ScriptedEngine(gate=gate, telemetry=measured(3, 2))
    call = await asgi_post(make_app(tmp_path, engine), MSG, body(stream=True))
    first = await call.next_body()
    assert call.status == 200 and call.headers["content-type"].startswith("text/event-stream")
    assert first.startswith(b"event: message_start\n")
    for _ in range(3):
        chunk = await call.next_body()
        assert chunk == b'event: ping\ndata: {"type": "ping"}\n\n'
    assert not gate.is_set()
    gate.set()
    rest = (await call.read_all()).decode()
    names = [n for n, _ in parse_sse(rest.replace('event: ping\ndata: {"type": "ping"}\n\n', ""))]
    assert names == ["content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    await call.task


async def test_stream_ping_never_splits_a_block_and_keeps_order_after_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pings only occur between events, so every chunk read is one complete event."""
    monkeypatch.setattr(routes_anthropic, "STREAM_PING_INTERVAL_SECONDS", 0.01)
    gate = threading.Event()
    call = await asgi_post(make_app(tmp_path, ScriptedEngine(gate=gate)), MSG, body(stream=True))
    await call.next_body()
    await call.next_body()
    gate.set()
    chunks: List[bytes] = []
    while (chunk := await call.next_body()) is not None:
        chunks.append(chunk)
    assert all(c.startswith(b"event: ") and c.endswith(b"\n\n") and c.count(b"\n\n") == 1 for c in chunks)
    await call.task


async def test_stream_client_disconnect_stops_the_engine_stream(tmp_path: Path) -> None:
    """When the client goes away mid-stream the engine generator is closed instead of running to max_tokens."""
    engine = ScriptedEngine(endless=True)
    call = await asgi_post(make_app(tmp_path, engine), MSG, body(stream=True, max_tokens=32768))
    assert (await call.next_body()).startswith(b"event: message_start")
    await call.next_body()
    call.disconnect()
    await asyncio.wait_for(call.task, 10)
    assert await asyncio.to_thread(engine.stream_closed.wait, 10)


async def test_stream_client_disconnect_while_silent_releases_the_engine_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A client that leaves during a long prefill does not wedge later requests once the engine returns."""
    monkeypatch.setattr(routes_anthropic, "STREAM_PING_INTERVAL_SECONDS", 0.01)
    gate = threading.Event()
    engine = ScriptedEngine(gate=gate)
    app = make_app(tmp_path, engine)
    call = await asgi_post(app, MSG, body(stream=True))
    await call.next_body()
    await call.next_body()
    call.disconnect()
    await asyncio.wait_for(call.task, 10)
    gate.set()
    assert await asyncio.to_thread(engine.stream_closed.wait, 10)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        assert (await client.post(MSG, json=body())).status_code == 200


def _marker_reply(prompt: str) -> List[str]:
    found = re.findall(r"<<(\w+)>>", prompt)
    return ["reply-", found[-1]]


async def test_concurrent_requests_do_not_cross_talk(tmp_path: Path) -> None:
    """Twenty mixed stream and blocking requests each receive only their own answer."""
    engine = ScriptedEngine(reply_fn=_marker_reply)
    app = make_app(tmp_path, engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost", timeout=30) as client:
        async def one(i: int) -> Tuple[int, str]:
            payload = body(messages=[{"role": "user", "content": f"<<id{i}>>"}], stream=bool(i % 2))
            response = await client.post(MSG, json=payload)
            if payload["stream"]:
                events = parse_sse(response.text)
                return i, "".join(e["delta"]["text"] for n, e in events if n == "content_block_delta")
            return i, response.json()["content"][0]["text"]

        results = await asyncio.gather(*[one(i) for i in range(20)])
    assert all(text == f"reply-id{i}" for i, text in results)


async def test_requests_to_one_engine_run_one_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline is not re-entrant: a second request waits until the first has released the engine."""
    monkeypatch.setattr(routes_anthropic, "STREAM_PING_INTERVAL_SECONDS", 0.01)
    gate = threading.Event()
    engine = ScriptedEngine(gate=gate)
    app = make_app(tmp_path, engine)
    first = await asgi_post(app, MSG, body(stream=True))
    await first.next_body()
    assert await asyncio.to_thread(engine.stream_started.wait, 10)
    second = await asgi_post(app, MSG, body())
    for _ in range(50):
        await asyncio.sleep(0)
    assert len(engine.prompts) == 1
    gate.set()
    await first.read_all()
    assert (await second.read_all()).startswith(b"{")
    assert len(engine.prompts) == 2
    await asyncio.gather(first.task, second.task)


# ---------------------------------------------------------------- real uvicorn, raw sockets

class _Server:
    """uvicorn serving the real app on an ephemeral port in a thread."""

    def __init__(self, app: Any) -> None:
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        started = threading.Event()

        def wait() -> None:
            while not self.server.started:
                threading.Event().wait(0.01)
            started.set()

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        assert started.wait(15), "uvicorn did not start"
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.should_exit = True
        self.thread.join(15)


def _raw_post(port: int, path: str, payload: Any, until: bytes = b"", on_match: Any = None) -> bytes:
    """POST over a raw socket and return every byte received until the server closes the connection."""
    data = json.dumps(payload).encode()
    request = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
               f"Connection: close\r\n\r\n").encode() + data
    received = b""
    with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
        sock.sendall(request)
        released = False
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                return received
            received += chunk
            if on_match is not None and not released and until in received:
                released = True
                on_match()


def _dechunk(raw: bytes) -> Tuple[str, dict, bytes]:
    head, _, rest = raw.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    headers = {k.lower(): v for k, v in (line.split(": ", 1) for line in lines[1:])}
    decoded = b""
    while rest:
        size_line, _, rest = rest.partition(b"\r\n")
        size = int(size_line, 16)
        if size == 0:
            break
        decoded += rest[:size]
        rest = rest[size + 2:]
    return lines[0], headers, decoded


def test_uvicorn_stream_wire_format_matches_golden_bytes(tmp_path: Path) -> None:
    """Over a real socket: HTTP 200, chunked text/event-stream, and the body equals the golden file byte for byte."""
    engine = ScriptedEngine(telemetry=measured(9, 2))
    request = json.loads((FIXTURES / "text_request.json").read_text(encoding="utf-8"))
    with _Server(make_app(tmp_path, engine)) as srv:
        raw = _raw_post(srv.port, "/v1/messages?beta=true", {**request, "stream": True})
    status, headers, decoded = _dechunk(raw)
    assert status == "HTTP/1.1 200 OK"
    assert headers["content-type"].startswith("text/event-stream") and headers["transfer-encoding"] == "chunked"
    assert "content-length" not in headers
    assert normalize_ids(decoded.decode()) == (FIXTURES / "text_stream.sse").read_text(encoding="utf-8")


def test_uvicorn_tool_stream_wire_format_matches_golden_bytes(tmp_path: Path) -> None:
    """The tool_use stream is identical on the wire."""
    engine = ScriptedEngine(pieces=CALL_PIECES, telemetry=measured(20, 12))
    request = json.loads((FIXTURES / "tool_request.json").read_text(encoding="utf-8"))
    with _Server(make_app(tmp_path, engine)) as srv:
        raw = _raw_post(srv.port, "/v1/messages?beta=true", {**request, "stream": True})
    _, _, decoded = _dechunk(raw)
    assert normalize_ids(decoded.decode()) == (FIXTURES / "tool_stream.sse").read_text(encoding="utf-8")


def test_uvicorn_stream_is_unbuffered_ping_arrives_before_the_engine_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bytes (message_start and a ping) reach the socket while the engine is still blocked, so nothing is buffered."""
    monkeypatch.setattr(routes_anthropic, "STREAM_PING_INTERVAL_SECONDS", 0.05)
    gate = threading.Event()
    engine = ScriptedEngine(gate=gate)
    seen_before_release: List[bytes] = []

    def release() -> None:
        seen_before_release.append(b"released")
        gate.set()

    with _Server(make_app(tmp_path, engine)) as srv:
        raw = _raw_post(srv.port, "/v1/messages", body(stream=True), until=b"event: ping", on_match=release)
    assert seen_before_release and b"event: message_start" in raw and b"event: ping" in raw
    assert raw.index(b"event: ping") < raw.index(b"event: content_block_start")
    _, _, decoded = _dechunk(raw)
    assert decoded.decode().rstrip().endswith('data: {"type": "message_stop"}')


def test_uvicorn_non_stream_and_error_wire_format(tmp_path: Path) -> None:
    """A blocking response has a content-length JSON body; an error is the Anthropic envelope with the right status line."""
    engine = ScriptedEngine(telemetry=measured(9, 2))
    request = json.loads((FIXTURES / "text_request.json").read_text(encoding="utf-8"))
    bad = dict(request)
    bad.pop("max_tokens")
    with _Server(make_app(tmp_path, engine)) as srv:
        ok_raw = _raw_post(srv.port, "/v1/messages", request)
        err_raw = _raw_post(srv.port, "/v1/messages", bad)
    head, _, payload = ok_raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 OK") and b"content-type: application/json" in head.lower()
    assert json.loads(normalize_ids(payload.decode())) == json.loads((FIXTURES / "text_response.json").read_text(encoding="utf-8"))
    err_head, _, err_body = err_raw.partition(b"\r\n\r\n")
    assert err_head.startswith(b"HTTP/1.1 400 Bad Request")
    assert json.loads(normalize_ids(err_body.decode())) == json.loads((FIXTURES / "error_response.json").read_text(encoding="utf-8"))
