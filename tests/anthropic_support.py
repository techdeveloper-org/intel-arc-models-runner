"""Shared test doubles and helpers for the Anthropic Messages route tests (no real model, no hardware)."""

import asyncio
import json
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from fastapi.testclient import TestClient

from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager
from tests.test_server import MockEngine

MODEL = "llama-3.2-3b-int4"
CACHE_KEY = f"{MODEL}:GPU:none:NPU:4"
FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"

TOOL_WEATHER = {
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
CALL_PIECES = [
    "I will check.\n",
    '<tool_call>{"name": "get_weather", ',
    '"arguments": {"city": "Pune"}}</tool_call>',
]


def measured(prompt: int, completion: int) -> EngineTelemetry:
    """Build telemetry that carries engine-measured counts."""
    return EngineTelemetry(model_id=MODEL, prompt_tokens=prompt, total_tokens=completion, metrics_source="ov_perf_metrics")


class ScriptedEngine(MockEngine):
    """MockEngine with scripted output pieces, a word-count fake tokenizer and failure or gating switches."""

    def __init__(
        self,
        pieces: Sequence[str] = ("Hello", " there"),
        finish: str = "stop",
        telemetry: Optional[EngineTelemetry] = None,
        error: Optional[Exception] = None,
        error_after: Optional[int] = None,
        gate: Optional[threading.Event] = None,
        reply_fn: Optional[Callable[[str], List[str]]] = None,
        endless: bool = False,
        tokenizer: bool = True,
    ) -> None:
        """Store the script.

        Args:
            pieces: Text pieces streamed one per chunk; joined for the blocking call.
            finish: finish_reason of the final stream chunk.
            telemetry: Telemetry returned after a run (unmeasured by default).
            error: Exception raised at the start of a run, or after error_after pieces.
            error_after: Number of pieces to emit before raising error.
            gate: When set, a run blocks until the event is set (a slow prefill).
            reply_fn: Maps the prompt to the pieces to emit (overrides pieces).
            endless: Emit pieces forever until the generator is closed.
            tokenizer: Whether tokenize() exists and works.
        """
        super().__init__(MODEL)
        self.pieces = list(pieces)
        self.finish = finish
        self._telemetry = telemetry or EngineTelemetry(model_id=MODEL)
        self.error = error
        self.error_after = error_after
        self.gate = gate
        self.reply_fn = reply_fn
        self.endless = endless
        self.has_tokenizer = tokenizer
        self.prompts: List[str] = []
        self.configs: List[GenerationConfig] = []
        self.stream_started = threading.Event()
        self.stream_closed = threading.Event()
        self.throttle = threading.Event()

    def _script(self, prompt: str) -> List[str]:
        """Return the pieces for this prompt."""
        return self.reply_fn(prompt) if self.reply_fn else self.pieces

    def tokenize(self, prompt: str) -> Optional[List[int]]:
        """Fake tokenizer: one token per whitespace-delimited word, or None when disabled."""
        if not self.has_tokenizer:
            return None
        return list(range(len(prompt.split())))

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Return the scripted text."""
        self.prompts.append(prompt)
        self.configs.append(config)
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.error is not None:
            raise self.error
        return "".join(self._script(prompt))

    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[TokenChunk]:
        """Yield the scripted pieces, honouring the failure and gating switches."""
        self.prompts.append(prompt)
        self.configs.append(config)
        self.stream_started.set()
        try:
            if self.gate is not None:
                self.gate.wait(timeout=10)
            if self.error is not None and self.error_after is None:
                raise self.error
            if self.endless:
                while True:
                    yield TokenChunk(text="x", is_finished=False)
                    self.throttle.wait(0.001)
            for index, piece in enumerate(self._script(prompt)):
                if self.error is not None and self.error_after == index:
                    raise self.error
                yield TokenChunk(text=piece, is_finished=False)
            if self.error is not None and self.error_after is not None and self.error_after >= len(self._script(prompt)):
                raise self.error
            yield TokenChunk(text="", is_finished=True, finish_reason=self.finish)
        finally:
            self.stream_closed.set()


def make_app(tmp_path: Path, engine: Any) -> Any:
    """Create the real app whose model cache holds the engine under MODEL."""
    model_dir = tmp_path / MODEL
    model_dir.mkdir(exist_ok=True)
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    manager = ModelManager(model_dirs=[tmp_path], max_loaded_models=2)
    manager._cache[CACHE_KEY] = engine
    return create_app(model_manager=manager, debug=True)


def make_client(tmp_path: Path, engine: Any) -> TestClient:
    """Create a TestClient over make_app."""
    return TestClient(make_app(tmp_path, engine))


def body(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal valid /v1/messages payload with overrides applied (a None value removes the key)."""
    payload: Dict[str, Any] = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "Say hi"}]}
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not None}


def parse_sse(raw: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse an SSE body into (event, data) pairs, asserting the exact framing."""
    assert raw.endswith("\n\n"), "stream must end with a blank line"
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in raw.split("\n\n")[:-1]:
        lines = block.split("\n")
        assert len(lines) == 2, f"event block must be exactly two lines: {block!r}"
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), block
        name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == name, (name, data)
        events.append((name, data))
    return events


def normalize_ids(text: str) -> str:
    """Replace generated ids with fixed placeholders so output can be compared with a golden file."""
    text = re.sub(r"msg_[0-9a-f]{24}", "msg_ID", text)
    text = re.sub(r"toolu_[0-9a-f]{24}", "toolu_ID", text)
    return re.sub(r"req_[0-9a-f]{24}", "req_ID", text)


class AsgiCall:
    """One in-flight ASGI HTTP call whose response body can be read chunk by chunk."""

    def __init__(self) -> None:
        """Create the queues."""
        self.sent: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.disconnected = asyncio.Event()
        self.status = 0
        self.headers: Dict[str, str] = {}
        self.task: Optional["asyncio.Task[None]"] = None

    async def next_body(self, timeout: float = 10.0) -> Optional[bytes]:
        """Return the next non-empty body chunk, or None when the response is complete."""
        while True:
            message = await asyncio.wait_for(self.sent.get(), timeout)
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.headers = {k.decode().lower(): v.decode() for k, v in message["headers"]}
                continue
            if message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    return chunk
                if not message.get("more_body", False):
                    return None

    async def read_all(self) -> bytes:
        """Read the remaining body."""
        out = b""
        while True:
            chunk = await self.next_body()
            if chunk is None:
                return out
            out += chunk

    def disconnect(self) -> None:
        """Signal that the client went away."""
        self.disconnected.set()


async def asgi_post(app: Any, path: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, query: str = "") -> AsgiCall:
    """Start a POST against the ASGI app and return the call handle without waiting for completion."""
    call = AsgiCall()
    raw = json.dumps(payload).encode("utf-8")
    sent_request = False

    async def receive() -> Dict[str, Any]:
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await call.disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Dict[str, Any]) -> None:
        if call.disconnected.is_set():
            raise OSError("client disconnected")
        await call.sent.put(message)

    header_list = [(b"host", b"testserver"), (b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
    for key, value in (headers or {}).items():
        header_list.append((key.lower().encode(), value.encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": query.encode(),
        "headers": header_list, "client": ("127.0.0.1", 50000), "server": ("testserver", 80), "root_path": "",
    }
    async def run() -> None:
        try:
            await app(scope, receive, send)
        except OSError:
            pass

    call.task = asyncio.create_task(run())
    return call
