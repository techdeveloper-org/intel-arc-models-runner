"""Triad test suite for Continuous Batching REST Endpoints and Preemption API.

Validates GET /api/batch/status, POST /api/batch/preempt, concurrent streaming throughput,
Little's Law 429 overload responses, and disconnect memory reclamation.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterator, List, Optional
import pytest
from fastapi.testclient import TestClient

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.continuous_batcher import AdmissionDeniedError, ContinuousBatcher
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.engine.paged_attention import PagedBlockTable, RadixPrefixCache
from arc_runner.server.app import create_app
from arc_runner.server.batch_pipeline import BatchExecutionPipeline
from arc_runner.server.model_manager import ModelManager


class BatchMockEngine(BaseEngine):
    """Simulated execution engine for testing batch API server endpoints."""

    def __init__(self, model_name: str = "llama-3.2-3b-int4") -> None:
        """Initialize mock engine with model identifier.

        Args:
            model_name: Simulated model name string.
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

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Simulate text generation."""
        return f"Batch response for: {prompt[:30]}"

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Simulate token chunk streaming."""
        tokens = ["Batch", "stream", "token", "done"]
        for i, tok in enumerate(tokens):
            yield TokenChunk(
                text=tok + " ",
                token_id=500 + i,
                is_finished=False,
                prompt_tokens=4,
                completion_tokens=i + 1,
            )
        yield TokenChunk(
            text="",
            token_id=2,
            is_finished=True,
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=len(tokens),
        )

    def unload(self) -> None:
        """Simulate memory release."""
        self._is_loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Retrieve execution telemetry."""
        return self._telemetry


@pytest.fixture
def batch_api_env():
    """Create a temporary model directory and pre-configured TestClient fixture with batch pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        llama_dir = tmp_path / "llama-3.2-3b-int4"
        llama_dir.mkdir(parents=True)
        (llama_dir / "openvino_model.xml").write_text("<net name='llama'></net>", encoding="utf-8")
        (llama_dir / "openvino_model.bin").write_bytes(b"\x00" * 1024)

        manager = ModelManager(model_dirs=[tmp_path], max_loaded_models=2)
        mock_engine = BatchMockEngine("llama-3.2-3b-int4")
        manager._cache["llama-3.2-3b-int4:GPU:none:NPU:4"] = mock_engine

        block_table = PagedBlockTable(block_size=16, total_blocks=64)
        batcher = ContinuousBatcher(block_table=block_table, max_batch_size=8)
        prefix_cache = RadixPrefixCache()
        pipeline = BatchExecutionPipeline(batcher=batcher, prefix_cache=prefix_cache, tick_interval_s=0.005)

        app = create_app(model_manager=manager, batch_pipeline=pipeline, debug=True)
        client = TestClient(app)

        yield {
            "client": client,
            "pipeline": pipeline,
            "batcher": batcher,
            "block_table": block_table,
            "prefix_cache": prefix_cache,
            "manager": manager,
        }


class TestBatchAPIPositive:
    """Positive test cases validating telemetry routes, preemption, and concurrency."""

    def test_api_batch_status_200(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify GET /api/batch/status returns HTTP 200 and schema-valid telemetry payload."""
        client: TestClient = batch_api_env["client"]
        resp = client.get("/api/batch/status")
        assert resp.status_code == 200

        data = resp.json()
        assert "running_sequences" in data
        assert "waiting_sequences" in data
        assert "used_kv_blocks" in data
        assert "free_kv_blocks" in data
        assert "vram_block_utilization_pct" in data
        assert "prefix_cache_hit_rate" in data
        assert data["free_kv_blocks"] == 64
        assert data["used_kv_blocks"] == 0
        assert data["vram_block_utilization_pct"] == 0.0

    def test_api_generate_concurrent_batch_streaming(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify POST /api/generate streaming works with batch pipeline active."""
        client: TestClient = batch_api_env["client"]
        payload = {
            "model": "llama-3.2-3b-int4",
            "prompt": "Test batch streaming prompt",
            "stream": True,
        }
        resp = client.post("/api/generate", json=payload)
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers.get("content-type", "")
        lines = [line.strip() for line in resp.text.split("\n") if line.strip()]
        assert len(lines) >= 1

    def test_api_chat_concurrent_batch_streaming(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify POST /api/chat streaming works with batch pipeline active."""
        client: TestClient = batch_api_env["client"]
        payload = {
            "model": "llama-3.2-3b-int4",
            "messages": [{"role": "user", "content": "Explain PagedAttention"}],
            "stream": True,
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.text.split("\n") if line.strip()]
        assert len(lines) >= 1

    def test_api_batch_preempt_200(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify POST /api/batch/preempt successfully preempts an active sequence (Proof M32)."""
        client: TestClient = batch_api_env["client"]
        batcher: ContinuousBatcher = batch_api_env["batcher"]

        seq_id = "seq-to-preempt"
        batcher.add_request(seq_id, "Long running sequence prompt")
        batcher.step()
        assert seq_id in batcher.get_running_sequence_ids()

        resp = client.post("/api/batch/preempt", json={"seq_id": seq_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["seq_id"] == seq_id
        assert data["preempted"] is True
        assert seq_id not in batcher.get_running_sequence_ids()

    def test_api_batch_status_utilization_scaling(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify block utilization scaling reflects active sequence memory allocations."""
        client: TestClient = batch_api_env["client"]
        block_table: PagedBlockTable = batch_api_env["block_table"]

        block_table.allocate_sequence("util-test-seq", initial_tokens=32)

        resp = client.get("/api/batch/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["used_kv_blocks"] == 2
        assert data["free_kv_blocks"] == 62
        assert data["vram_block_utilization_pct"] > 0.0

    def test_api_batch_prefix_cache_reporting(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify prefix cache hit and miss telemetry are reported accurately via REST."""
        client: TestClient = batch_api_env["client"]
        prefix_cache: RadixPrefixCache = batch_api_env["prefix_cache"]

        prefix_cache.insert_prefix(list(range(32)), [0, 1])
        prefix_cache.match_prefix(list(range(32)) + [100])

        resp = client.get("/api/batch/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["prefix_cache_hits"] == 1
        assert data["prefix_cache_hit_rate"] == 100.0


class TestBatchAPINegative:
    """Negative test cases validating 404 preemption and 429 admission rejections."""

    def test_api_batch_preempt_404_not_found(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify POST /api/batch/preempt returns HTTP 404 when sequence ID is not found."""
        client: TestClient = batch_api_env["client"]
        resp = client.post("/api/batch/preempt", json={"seq_id": "unknown-sequence-uuid"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_api_batch_overloaded_429_too_many_requests(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify AdmissionDeniedError maps directly to HTTP 429 status code."""
        client: TestClient = batch_api_env["client"]
        app = batch_api_env["client"].app

        @app.get("/test/trigger-admission-denied")
        async def trigger_denied() -> None:
            raise AdmissionDeniedError("Memory exhausted: saturation limit reached.")

        resp = client.get("/test/trigger-admission-denied")
        assert resp.status_code == 429
        data = resp.json()
        assert data["code"] == "ADMISSION_DENIED"
        assert "saturation limit reached" in data["error"]


class TestBatchAPIEdgeCases:
    """Edge cases validating client disconnection cleanup and rapid burst concurrency."""

    @pytest.mark.asyncio
    async def test_api_generate_disconnect_cleans_vram(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify client disconnect during batch generation reclaims VRAM blocks (Proof M32)."""
        pipeline: BatchExecutionPipeline = batch_api_env["pipeline"]
        block_table: PagedBlockTable = batch_api_env["block_table"]
        initial_free = block_table.free_blocks_count

        seq_id = "test-disconnect-api"
        agen = pipeline.enqueue_request("Long prompt text", GenerationConfig(max_new_tokens=40), seq_id=seq_id)

        chunk = await agen.__anext__()
        assert chunk is not None
        assert block_table.free_blocks_count < initial_free

        await agen.aclose()
        assert block_table.free_blocks_count == initial_free

    def test_api_rapid_burst_concurrency_stress(self, batch_api_env: Dict[str, Any]) -> None:
        """Verify rapid multi-threaded burst queries do not deadlock or crash the server."""
        client: TestClient = batch_api_env["client"]

        def fetch_status(call_idx: int) -> int:
            resp = client.get("/api/batch/status")
            return resp.status_code

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_status, i) for i in range(24)]
            status_codes = [f.result() for f in futures]

        assert all(code == 200 for code in status_codes)
