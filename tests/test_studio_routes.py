"""Triad unit tests for Arc MultiModal Studio FastAPI routes.

Validates REST and SSE generation endpoints, co-benchmarking triggers,
metadata retrieval, and error handling per Rule 41-testing-standards-python-fastapi.md.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M1-M5.
"""

from fastapi.testclient import TestClient

from arc_runner.server.app import create_app

client = TestClient(create_app())


class TestStudioRoutes:
    """Verification suite for Arc MultiModal Studio HTTP endpoints."""

    def test_post_generate_project_rest(self) -> None:
        """Verify POST /v1/studio/generate returns valid StudioArtifactPayload."""
        payload = {
            "prompt": "Explain quantum superposition in physics",
            "target_duration_sec": 4.0,
            "visual_modality": "IMAGE_SLIDE",
            "resolution": [128, 128],
            "fps": 8,
        }
        response = client.post("/v1/studio/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        assert "video_output_path" in data
        assert data["cloud_cost_saved_usd"] == 0.35
        assert len(data["script_scenes"]) >= 1

    def test_post_generate_project_sse_streaming(self) -> None:
        """Verify POST /v1/studio/generate?stream=true streams SSE event milestones."""
        payload = {
            "prompt": "Black hole event horizons",
            "target_duration_sec": 3.0,
            "resolution": [128, 128],
            "fps": 8,
        }
        response = client.post("/v1/studio/generate?stream=true", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body_text = response.text
        assert "data: " in body_text
        assert "INITIATED" in body_text or "COMPLETED" in body_text

    def test_post_run_co_benchmark(self) -> None:
        """Verify POST /v1/studio/benchmark triggers hardware co-benchmark report."""
        payload = {
            "prompt": "Superconductivity principles",
            "iterations": 1,
            "warmup_runs": 0,
            "strategy": "HETEROGENEOUS_PARALLEL",
        }
        response = client.post("/v1/studio/benchmark", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "benchmark_id" in data
        assert data["cloud_comparison_usd_per_run"] == 0.35
        assert data["iterations"] == 1

    def test_get_project_metadata_and_not_found(self) -> None:
        """Verify GET /v1/studio/projects/{id} retrieves existing project and 404s on unknown."""
        gen_res = client.post(
            "/v1/studio/generate",
            json={"prompt": "Neutron stars", "target_duration_sec": 3.0, "resolution": [128, 128], "fps": 8},
        )
        assert gen_res.status_code == 200
        project_id = gen_res.json()["project_id"]

        get_res = client.get(f"/v1/studio/projects/{project_id}")
        assert get_res.status_code == 200
        assert get_res.json()["project_id"] == project_id

        not_found_res = client.get("/v1/studio/projects/non_existent_project_999")
        assert not_found_res.status_code == 404

    def test_download_project_video_stream(self) -> None:
        """Verify GET /v1/studio/projects/{id}/video streams valid MP4 binary container."""
        gen_res = client.post(
            "/v1/studio/generate",
            json={"prompt": "Solar flares", "target_duration_sec": 2.0, "resolution": [128, 128], "fps": 8},
        )
        assert gen_res.status_code == 200
        project_id = gen_res.json()["project_id"]

        video_res = client.get(f"/v1/studio/projects/{project_id}/video")
        assert video_res.status_code == 200
        assert video_res.headers["content-type"] == "video/mp4"
        assert len(video_res.content) > 0
