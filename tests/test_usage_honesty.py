"""Token usage on the wire is engine-reported or absent, never a word count presented as tokens."""

import json
import unittest.mock as mock
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from arc_runner.engine.embedding_engine import EmbeddingEngine
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager
from tests.test_server import MockEngine

MODEL = "llama-3.2-3b-int4"
CACHE_KEY = f"{MODEL}:GPU:none:NPU:4"


class CountingEngine(MockEngine):
    """MockEngine whose telemetry and final stream chunk carry configurable measured counts."""

    def __init__(self, telemetry: EngineTelemetry, final_prompt_tokens: Optional[int]) -> None:
        super().__init__(MODEL)
        self._telemetry = telemetry
        self._final_prompt_tokens = final_prompt_tokens

    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[TokenChunk]:
        yield TokenChunk(text="a ", token_id=1, is_finished=False)
        yield TokenChunk(text="", token_id=2, is_finished=True, finish_reason="stop", prompt_tokens=self._final_prompt_tokens)


def _client(tmp_path: Path, engine: MockEngine) -> TestClient:
    model_dir = tmp_path / MODEL
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    manager = ModelManager(model_dirs=[tmp_path], max_loaded_models=2)
    manager._cache[CACHE_KEY] = engine
    return TestClient(create_app(model_manager=manager, debug=True))


def _measured(prompt: int, completion: int) -> EngineTelemetry:
    return EngineTelemetry(model_id=MODEL, prompt_tokens=prompt, total_tokens=completion, metrics_source="ov_perf_metrics")


CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "one two three four five"}], "stream": False}


def test_openai_chat_usage_engine_reports_counts_uses_engine_values(tmp_path: Path) -> None:
    """Counts come from telemetry, not from splitting the text on whitespace."""
    client = _client(tmp_path, CountingEngine(_measured(31, 17), None))
    usage = client.post("/v1/chat/completions", json=CHAT).json()["usage"]
    assert usage == {"prompt_tokens": 31, "completion_tokens": 17, "total_tokens": 48}


def test_openai_chat_usage_engine_reports_none_omits_usage(tmp_path: Path) -> None:
    """Unmeasured telemetry means no usage object at all."""
    client = _client(tmp_path, CountingEngine(EngineTelemetry(model_id=MODEL), None))
    body = client.post("/v1/chat/completions", json=CHAT).json()
    assert "usage" not in body


def test_openai_chat_usage_zero_completion_tokens_is_reported_when_measured(tmp_path: Path) -> None:
    """A measured zero is a real zero and is kept."""
    client = _client(tmp_path, CountingEngine(_measured(9, 0), None))
    usage = client.post("/v1/chat/completions", json=CHAT).json()["usage"]
    assert usage["completion_tokens"] == 0 and usage["total_tokens"] == 9


def test_openai_chat_stream_final_chunk_has_no_fabricated_usage(tmp_path: Path) -> None:
    """The SSE stream never carries a made-up usage block."""
    client = _client(tmp_path, CountingEngine(EngineTelemetry(model_id=MODEL), None))
    text = client.post("/v1/chat/completions", json={**CHAT, "stream": True}).text
    assert '"usage"' not in text


def test_ollama_generate_non_stream_unmeasured_prompt_count_is_omitted(tmp_path: Path) -> None:
    """prompt_eval_count is absent, not 0, when the engine did not measure it."""
    client = _client(tmp_path, CountingEngine(EngineTelemetry(model_id=MODEL), None))
    data = client.post("/api/generate", json={"model": MODEL, "prompt": "hi", "stream": False}).json()
    assert "prompt_eval_count" not in data


def test_ollama_generate_non_stream_measured_prompt_count_appears(tmp_path: Path) -> None:
    """A measured prompt token count is reported."""
    client = _client(tmp_path, CountingEngine(_measured(12, 3), None))
    data = client.post("/api/generate", json={"model": MODEL, "prompt": "hi", "stream": False}).json()
    assert data["prompt_eval_count"] == 12


def test_ollama_chat_non_stream_counts_follow_engine(tmp_path: Path) -> None:
    """Chat mirrors generate: measured count appears, unmeasured is omitted."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    measured = _client(tmp_path, CountingEngine(_measured(7, 2), None)).post("/api/chat", json=body).json()
    assert measured["prompt_eval_count"] == 7


def test_ollama_chat_non_stream_unmeasured_prompt_count_is_omitted(tmp_path: Path) -> None:
    """No measurement, no field."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    data = _client(tmp_path, CountingEngine(EngineTelemetry(model_id=MODEL), None)).post("/api/chat", json=body).json()
    assert "prompt_eval_count" not in data


@pytest.mark.parametrize("path,body", [
    ("/api/generate", {"model": MODEL, "prompt": "hi", "stream": True}),
    ("/api/chat", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True}),
])
def test_ollama_stream_final_chunk_prompt_count_follows_chunk(tmp_path: Path, path: str, body: dict) -> None:
    """The final NDJSON line carries prompt_eval_count only when the chunk reports it (zero is kept)."""
    for reported, expect_present in ((None, False), (0, True), (14, True)):
        sub = tmp_path / f"r{reported}"
        sub.mkdir()
        client = _client(sub, CountingEngine(EngineTelemetry(model_id=MODEL), reported))
        lines = [json.loads(x) for x in client.post(path, json=body).text.splitlines() if x.strip()]
        final = lines[-1]
        assert final["done"] is True
        assert ("prompt_eval_count" in final) is expect_present
        if expect_present:
            assert final["prompt_eval_count"] == reported


def _embedding_client(tmp_path: Path, reported: Optional[int]) -> TestClient:
    (tmp_path / "mock-embed").mkdir()
    (tmp_path / "mock-embed" / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    engine = mock.MagicMock(spec=EmbeddingEngine)
    engine.dimension = 4
    engine.last_prompt_tokens = reported
    engine.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    manager = ModelManager(model_dirs=[tmp_path])
    manager._embedding_cache["mock-embed:NPU"] = engine
    return TestClient(create_app(model_manager=manager))


def test_openai_embeddings_engine_reports_none_omits_usage(tmp_path: Path) -> None:
    """Whitespace word counts are not token counts, so usage is absent when unmeasured."""
    body = _embedding_client(tmp_path, None).post("/v1/embeddings", json={"model": "mock-embed", "input": ["a b c d"]}).json()
    assert "usage" not in body


def test_openai_embeddings_engine_reports_counts_usage_appears(tmp_path: Path) -> None:
    """Engine-reported prompt tokens are passed through."""
    body = _embedding_client(tmp_path, 6).post("/v1/embeddings", json={"model": "mock-embed", "input": ["a b c d"]}).json()
    assert body["usage"] == {"prompt_tokens": 6, "total_tokens": 6}


def test_openai_embeddings_zero_reported_tokens_is_kept(tmp_path: Path) -> None:
    """A measured zero (empty input) is a real zero."""
    body = _embedding_client(tmp_path, 0).post("/v1/embeddings", json={"model": "mock-embed", "input": [""]}).json()
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


def test_ollama_embed_prompt_eval_count_follows_engine(tmp_path: Path) -> None:
    """/api/embed omits prompt_eval_count when unmeasured and reports it when measured."""
    payload = {"model": "mock-embed", "input": ["a b c d"]}
    (tmp_path / "x").mkdir()
    assert "prompt_eval_count" not in _embedding_client(tmp_path / "x", None).post("/api/embed", json=payload).json()
    (tmp_path / "y").mkdir()
    assert _embedding_client(tmp_path / "y", 5).post("/api/embed", json=payload).json()["prompt_eval_count"] == 5
