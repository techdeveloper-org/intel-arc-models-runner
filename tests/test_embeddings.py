"""Triad test suite for Intel NPU/GPU vector embeddings and mathematical proofs (M1, M2, M3, M4, M14).

Verifies strict L2 unit sphere normalization, attention-masked mean pooling, cosine similarity
metrics, and REST API parity endpoints (/api/embeddings, /api/embed, /v1/embeddings).
"""

import math
from unittest import mock
import pytest
from fastapi.testclient import TestClient

from arc_runner.engine.embedding_engine import (
    DimensionMismatchError,
    EmbeddingEngine,
    EmptyMaskError,
)
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


class TestL2NormalizationM1:
    """Mathematical validation of Proof M1: Strict L2 Vector Normalization."""

    def test_l2_norm_positive_unit_sphere(self) -> None:
        """Verify normalized vector lies strictly on unit sphere with norm 1.0."""
        raw = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = EmbeddingEngine.l2_normalize(raw)

        norm = math.sqrt(sum(x * x for x in normalized))
        assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)

    def test_l2_norm_negative_empty_vector(self) -> None:
        """Verify empty vector raises ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize an empty vector"):
            EmbeddingEngine.l2_normalize([])

    def test_l2_norm_edge_case_subnormal_floats(self) -> None:
        """Verify subnormal tiny floating point values normalize safely without zero division."""
        tiny = [1e-25, 2e-25, 3e-25]
        normalized = EmbeddingEngine.l2_normalize(tiny)
        norm = math.sqrt(sum(x * x for x in normalized))
        assert not math.isnan(norm)
        assert not math.isinf(norm)
        assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)


class TestMaskedMeanPoolingM2:
    """Mathematical validation of Proof M2: Attention-Masked Mean Pooling."""

    def test_masked_mean_pooling_positive(self) -> None:
        """Verify padded tokens introduce zero contamination into pooled representation."""
        hidden_states = [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [99.0, 99.0, 99.0],
        ]
        attention_mask = [1, 1, 0]

        pooled = EmbeddingEngine.masked_mean_pooling(hidden_states, attention_mask)
        expected = [(1.0 + 3.0) / 2.0, (2.0 + 4.0) / 2.0, (3.0 + 5.0) / 2.0]

        for p, exp in zip(pooled, expected):
            assert math.isclose(p, exp, rel_tol=1e-5)

    def test_masked_mean_pooling_negative_empty_mask(self) -> None:
        """Verify zero valid tokens in mask raises EmptyMaskError."""
        hidden_states = [[1.0, 2.0], [3.0, 4.0]]
        attention_mask = [0, 0]

        with pytest.raises(EmptyMaskError):
            EmbeddingEngine.masked_mean_pooling(hidden_states, attention_mask)

    def test_masked_mean_pooling_edge_case_single_token(self) -> None:
        """Verify single valid token surrounded by pad tokens returns exact token representation."""
        hidden_states = [
            [42.0, -17.5],
            [100.0, 200.0],
        ]
        attention_mask = [1, 0]

        pooled = EmbeddingEngine.masked_mean_pooling(hidden_states, attention_mask)
        assert math.isclose(pooled[0], 42.0, rel_tol=1e-6)
        assert math.isclose(pooled[1], -17.5, rel_tol=1e-6)


class TestCosineSimilarityM14:
    """Mathematical validation of Proof M14: Cosine Distance Metric."""

    def test_cosine_similarity_positive_identical_and_orthogonal(self) -> None:
        """Verify identical vectors yield 1.0 and orthogonal vectors yield 0.0."""
        vec_a = [1.0, 2.0, 3.0]
        assert math.isclose(EmbeddingEngine.compute_similarity(vec_a, vec_a), 1.0, rel_tol=1e-6)

        vec_x = [1.0, 0.0]
        vec_y = [0.0, 1.0]
        assert math.isclose(EmbeddingEngine.compute_similarity(vec_x, vec_y), 0.0, abs_tol=1e-6)

    def test_cosine_similarity_negative_mismatched_dimensions(self) -> None:
        """Verify vectors of differing dimensionality raise DimensionMismatchError."""
        with pytest.raises(DimensionMismatchError):
            EmbeddingEngine.compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_cosine_similarity_edge_case_diametrically_opposed(self) -> None:
        """Verify opposite vectors produce -1.0."""
        vec_pos = [1.0, 2.0, 3.0]
        vec_neg = [-1.0, -2.0, -3.0]
        sim = EmbeddingEngine.compute_similarity(vec_pos, vec_neg)
        assert math.isclose(sim, -1.0, rel_tol=1e-6)


class TestEmbeddingEndpoints:
    """Integration test suite for embedding REST API parity endpoints."""

    @pytest.fixture
    def client(self, tmp_path: pytest.TempPathFactory) -> TestClient:
        """Construct test client with mocked embedding model."""
        model_dir = tmp_path / "mock-embed"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

        mock_engine = mock.MagicMock(spec=EmbeddingEngine)
        mock_engine.dimension = 384
        mock_engine.embed_query.return_value = [0.1] * 384
        mock_engine.embed_documents.side_effect = lambda texts: [[0.1] * 384 for _ in texts]

        mgr = ModelManager(model_dirs=[tmp_path])
        mgr._embedding_cache["mock-embed:NPU"] = mock_engine

        app = create_app(model_manager=mgr)
        return TestClient(app)

    def test_api_embeddings_legacy_success(self, client: TestClient) -> None:
        """Verify Ollama POST /api/embeddings legacy format response."""
        res = client.post("/api/embeddings", json={"model": "mock-embed", "prompt": "Hello world"})
        assert res.status_code == 200
        data = res.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 384

    def test_api_embed_v2_success(self, client: TestClient) -> None:
        """Verify Ollama POST /api/embed v2 multi-prompt format response."""
        res = client.post(
            "/api/embed",
            json={"model": "mock-embed", "input": ["Hello world", "Intel Arc"]},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["model"] == "mock-embed"
        assert len(data["embeddings"]) == 2
        assert "total_duration" in data

    def test_v1_embeddings_openai_success(self, client: TestClient) -> None:
        """Verify OpenAI POST /v1/embeddings format response."""
        res = client.post(
            "/v1/embeddings",
            json={"model": "mock-embed", "input": ["Document text"]},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["index"] == 0
        assert len(data["data"][0]["embedding"]) == 384

    def test_embeddings_negative_model_not_found(self, client: TestClient) -> None:
        """Verify 404 response on non-existent embedding model."""
        res = client.post("/api/embeddings", json={"model": "non-existent-model", "prompt": "Hello"})
        assert res.status_code == 404
