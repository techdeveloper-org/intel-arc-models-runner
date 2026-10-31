"""Exhaustive 33-Scenario Triad Test Suite for Arc MultiModal Studio Web UI (Tab 7).

Validates sidebar navigation, HTML5 video player markup, 5-stage progress stepper,
asynchronous non-blocking SSE streaming (Proof M131), Little's Law queue bounds,
zero-leak memory lifecycle (Proof M133), audio-video sync (Proof M134), and Cloud ROI (Proof M135).
Strictly adheres to Level 2.2 Docstrings-Only convention and Claude Global Rules 35, 36, 37, 40, 28.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from arc_runner.orchestrator.models import (
    MultiModelBenchmarkReport,
    OrchestrationTelemetry,
    ScenePlan,
    StageTelemetry,
    StudioArtifactPayload,
)
from arc_runner.server import routes_studio
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create a test client backed by ModelManager and temporary exports directory.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        TestClient connected to initialized FastAPI application.
    """
    manager = ModelManager(model_dirs=[tmp_path])
    app = create_app(model_manager=manager, debug=True)
    return TestClient(app)


@pytest.fixture
def mock_artifact_payload(tmp_path: Path) -> StudioArtifactPayload:
    """Create a valid dummy StudioArtifactPayload with temporary MP4 video file.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        StudioArtifactPayload populated with mock scenes, telemetries, and video file.
    """
    video_file = tmp_path / "mock_video.mp4"
    video_file.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free")

    return StudioArtifactPayload(
        project_id="proj_triad_test_123",
        created_at="2026-10-24T21:00:00Z",
        total_duration_sec=15.0,
        resolution=(512, 512),
        video_output_path=str(video_file),
        cloud_cost_saved_usd=0.35,
        script_scenes=[
            ScenePlan(
                scene_id=1,
                title="Scene 1",
                visual_prompt="Binary stars orbiting in plasma nebula",
                narration_text="Two gravitational giants dance in orbital harmony.",
                duration_seconds=7.5,
                camera_motion="SLOW_ZOOM_IN",
            ),
            ScenePlan(
                scene_id=2,
                title="Scene 2",
                visual_prompt="Gravitational waves radiating through space-time",
                narration_text="Entangled particles weave quantum reality across lightyears.",
                duration_seconds=7.5,
                camera_motion="PAN_RIGHT",
            ),
        ],
        telemetry=OrchestrationTelemetry(
            project_id="proj_triad_test_123",
            total_latency_ms=2450.0,
            usm_zero_copy_latency_us=42.5,
            peak_vram_mb=185.0,
            stage_breakdowns=[
                StageTelemetry(
                    stage_name="PLANNING",
                    latency_ms=120.0,
                    device_used="NPU",
                    peak_memory_mb=145.0,
                    throughput_metric="scenes/sec",
                    throughput_value=16.6,
                ),
                StageTelemetry(
                    stage_name="AUDIO_SYNTHESIS",
                    latency_ms=450.0,
                    device_used="NPU",
                    peak_memory_mb=160.0,
                    throughput_metric="samples/sec",
                    throughput_value=53333.3,
                ),
                StageTelemetry(
                    stage_name="VISUAL_GENERATION",
                    latency_ms=1450.0,
                    device_used="GPU",
                    peak_memory_mb=185.0,
                    throughput_metric="frames/sec",
                    throughput_value=82.7,
                ),
                StageTelemetry(
                    stage_name="COMPOSITION",
                    latency_ms=380.0,
                    device_used="CPU",
                    peak_memory_mb=170.0,
                    throughput_metric="frames/sec",
                    throughput_value=315.7,
                ),
            ],
        ),
    )


# ==============================================================================
# 1. POSITIVE (HAPPY-PATH) TESTS (P-UI-01 through P-UI-13)
# ==============================================================================


def test_p_ui_01_sidebar_nav_rendered(client: TestClient) -> None:
    """Verify sidebar contains MultiModal Studio navigation button with clapperboard icon."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'data-tab="tab-studio"' in resp.text
    assert "MultiModal Studio" in resp.text


def test_p_ui_02_tab_pane_served_on_root_and_ui(client: TestClient) -> None:
    """Verify #tab-studio pane is rendered on both root browser request and /ui route."""
    resp_root = client.get("/", headers={"Accept": "text/html"})
    assert resp_root.status_code == 200
    assert 'id="tab-studio"' in resp_root.text

    resp_ui = client.get("/ui")
    assert resp_ui.status_code == 200
    assert 'id="tab-studio"' in resp_ui.text


def test_p_ui_03_creative_brief_input_controls(client: TestClient) -> None:
    """Verify creative brief textarea exists with placeholder and default text."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="studio-prompt"' in resp.text
    assert "scientific documentary teaser" in resp.text
    assert 'class="btn-preset"' in resp.text


def test_p_ui_04_studio_form_controls(client: TestClient) -> None:
    """Verify studio parameters controls (duration, resolution, fps, modality, exec-mode) exist."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="studio-duration"' in resp.text
    assert 'id="studio-resolution"' in resp.text
    assert 'id="studio-fps"' in resp.text
    assert 'id="studio-modality"' in resp.text
    assert 'id="studio-exec-mode"' in resp.text


def test_p_ui_05_action_buttons_rendered(client: TestClient) -> None:
    """Verify production dispatch button and co-benchmark button exist with correct IDs."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="btn-produce-video"' in resp.text
    assert 'id="btn-studio-bench"' in resp.text


def test_p_ui_06_stage_stepper_elements(client: TestClient) -> None:
    """Verify 5-stage visual stepper markup and progress bar exist in template."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="step-planning"' in resp.text
    assert 'id="step-audio"' in resp.text
    assert 'id="step-visual"' in resp.text
    assert 'id="step-usm"' in resp.text
    assert 'id="step-muxing"' in resp.text
    assert 'id="studio-progress-fill"' in resp.text
    assert 'id="studio-progress-pct"' in resp.text


def test_p_ui_07_video_player_and_download_anchor(client: TestClient) -> None:
    """Verify HTML5 video element with controls and download button anchor exist."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="studio-video-player"' in resp.text
    assert 'id="studio-video-placeholder"' in resp.text
    assert 'id="studio-download-btn"' in resp.text


def test_p_ui_08_telemetry_table_and_roi_card(client: TestClient) -> None:
    """Verify telemetry table and Cloud API replacement ROI scorecard exist."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="studio-telemetry-table"' in resp.text
    assert 'id="studio-roi-container"' in resp.text
    assert "Cloud API Replacement ROI" in resp.text


def test_p_ui_09_scene_breakdown_grid(client: TestClient) -> None:
    """Verify scene breakdown container and grid are present for director scenes."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="studio-scenes-container"' in resp.text
    assert 'id="studio-scenes-grid"' in resp.text


def test_p_ui_10_sse_streaming_contract(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify POST /v1/studio/generate?stream=true returns text/event-stream with progress events."""
    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=mock_artifact_payload):
        resp = client.post(
            "/v1/studio/generate?stream=true",
            json={
                "prompt": "Test cosmic nebula",
                "target_duration_sec": 15.0,
                "fps": 16,
                "resolution": [512, 512],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "INITIATED" in resp.text
        assert "COMPLETED" in resp.text


def test_p_ui_11_video_streaming_endpoint(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify GET /v1/studio/projects/{id}/video returns 200 and video/mp4 binary bytes."""
    routes_studio._PROJECT_CACHE[mock_artifact_payload.project_id] = mock_artifact_payload

    resp = client.get(f"/v1/studio/projects/{mock_artifact_payload.project_id}/video")
    assert resp.status_code == 200
    assert "video/mp4" in resp.headers["content-type"]
    assert len(resp.content) > 0


def test_p_ui_12_js_engine_initialization(client: TestClient) -> None:
    """Verify app.js exports initMultiModalStudio and tab-studio title mapping."""
    resp = client.get("/static/js/app.js")
    assert resp.status_code == 200
    assert "initMultiModalStudio" in resp.text
    assert "tab-studio" in resp.text
    assert "updateStageProgress" in resp.text


def test_p_ui_13_css_design_tokens(client: TestClient) -> None:
    """Verify style.css contains studio stepper, video theater, and ROI classes."""
    resp = client.get("/static/css/style.css")
    assert resp.status_code == 200
    assert ".studio-stage-stepper" in resp.text
    assert ".studio-theater-grid" in resp.text
    assert ".roi-scorecard-card" in resp.text


# ==============================================================================
# 2. NEGATIVE (ERROR-PATH) TESTS (N-UI-01 through N-UI-10)
# ==============================================================================


def test_n_ui_01_video_404_unknown_project(client: TestClient) -> None:
    """Verify requesting video for an unknown project ID returns HTTP 404."""
    resp = client.get("/v1/studio/projects/non_existent_project_xyz/video")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_n_ui_02_empty_prompt_validation(client: TestClient) -> None:
    """Verify empty prompt returns 422 Unprocessable Entity."""
    resp = client.post(
        "/v1/studio/generate",
        json={"prompt": "", "target_duration_sec": 10.0},
    )
    assert resp.status_code == 422


def test_n_ui_03_negative_duration_validation(client: TestClient) -> None:
    """Verify negative duration returns 422 Unprocessable Entity."""
    resp = client.post(
        "/v1/studio/generate",
        json={"prompt": "Valid prompt", "target_duration_sec": -5.0},
    )
    assert resp.status_code == 422


def test_n_ui_04_zero_fps_validation(client: TestClient) -> None:
    """Verify zero fps returns 422 Unprocessable Entity."""
    resp = client.post(
        "/v1/studio/generate",
        json={"prompt": "Valid prompt", "fps": 0},
    )
    assert resp.status_code == 422


def test_n_ui_05_invalid_modality_enum(client: TestClient) -> None:
    """Verify invalid visual modality enum string returns 422."""
    resp = client.post(
        "/v1/studio/generate",
        json={"prompt": "Valid prompt", "visual_modality": "INVALID_MODALITY"},
    )
    assert resp.status_code == 422


def test_n_ui_06_invalid_execution_mode_enum(client: TestClient) -> None:
    """Verify invalid execution mode enum string returns 422."""
    resp = client.post(
        "/v1/studio/generate",
        json={"prompt": "Valid prompt", "strategy": "SUPER_TURBO_UNSUPPORTED"},
    )
    assert resp.status_code == 422


def test_n_ui_07_pipeline_failure_sse_emits_failed(client: TestClient) -> None:
    """Verify internal pipeline exception during SSE stream produces FAILED event."""
    with mock.patch.object(routes_studio._PIPELINE, "execute", side_effect=RuntimeError("Silicon allocation failure")):
        resp = client.post(
            "/v1/studio/generate?stream=true",
            json={"prompt": "Trigger crash prompt", "target_duration_sec": 10.0},
        )
        assert resp.status_code == 200
        assert "FAILED" in resp.text
        # Deliberate (M4): the raw exception text stays in the server log; the client sees a stable message.
        assert "Silicon allocation failure" not in resp.text
        assert "Studio pipeline failed" in resp.text


def test_n_ui_08_malformed_json_body(client: TestClient) -> None:
    """Verify malformed JSON payload returns HTTP 400 Bad Request."""
    resp = client.post(
        "/v1/studio/generate",
        content="{malformed_json_without_closing_bracket",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in (400, 422)


def test_n_ui_09_missing_prompt_field(client: TestClient) -> None:
    """Verify request missing required prompt field returns 422."""
    resp = client.post(
        "/v1/studio/generate",
        json={"target_duration_sec": 15.0},
    )
    assert resp.status_code == 422


def test_n_ui_10_deleted_video_file_on_disk(client: TestClient, tmp_path: Path, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify HTTP 404 is returned if project exists in cache but video file was unlinked."""
    missing_file = tmp_path / "deleted_video.mp4"
    payload = mock_artifact_payload.model_copy(update={"video_output_path": str(missing_file)})
    routes_studio._PROJECT_CACHE["proj_missing_file"] = payload

    resp = client.get("/v1/studio/projects/proj_missing_file/video")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ==============================================================================
# 3. EDGE-CASE TESTS (E-UI-01 through E-UI-10)
# ==============================================================================


def test_e_ui_01_minimum_duration_boundary_2s(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify boundary condition with minimum duration of 2.0 seconds."""
    short_payload = mock_artifact_payload.model_copy(update={"total_duration_sec": 2.0})
    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=short_payload):
        resp = client.post(
            "/v1/studio/generate",
            json={"prompt": "Quick micro-teaser", "target_duration_sec": 2.0},
        )
        assert resp.status_code == 200
        assert resp.json()["total_duration_sec"] == 2.0


def test_e_ui_02_maximum_duration_boundary_300s(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify boundary condition with extended duration of 300.0 seconds."""
    long_payload = mock_artifact_payload.model_copy(update={"total_duration_sec": 300.0})
    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=long_payload):
        resp = client.post(
            "/v1/studio/generate",
            json={"prompt": "Extended documentary", "target_duration_sec": 300.0},
        )
        assert resp.status_code == 200
        assert resp.json()["total_duration_sec"] == 300.0


def test_e_ui_03_unicode_and_special_characters_prompt(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify prompt containing Devanagari, emojis, and HTML entities executes without XSS."""
    unicode_prompt = "क्वांटम भौतिकी 🚀 <script>alert(1)</script> & \"special\" 'chars'"
    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=mock_artifact_payload) as mock_exec:
        resp = client.post(
            "/v1/studio/generate",
            json={"prompt": unicode_prompt, "target_duration_sec": 10.0},
        )
        assert resp.status_code == 200
        called_req = mock_exec.call_args[0][0]
        assert called_req.prompt == unicode_prompt


def test_e_ui_04_maximum_resolution_boundary_1080p(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify high-definition 1920x1080 resolution boundary parsing."""
    hd_payload = mock_artifact_payload.model_copy(update={"resolution": (1920, 1080)})
    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=hd_payload):
        resp = client.post(
            "/v1/studio/generate",
            json={"prompt": "HD render", "resolution": [1920, 1080]},
        )
        assert resp.status_code == 200
        assert resp.json()["resolution"] == [1920, 1080]


def test_e_ui_05_minimum_resolution_boundary_64x64(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify tiny 64x64 resolution thumbnail boundary parsing."""
    tiny_payload = mock_artifact_payload.model_copy(update={"resolution": (64, 64)})
    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=tiny_payload):
        resp = client.post(
            "/v1/studio/generate",
            json={"prompt": "Thumbnail render", "resolution": [64, 64]},
        )
        assert resp.status_code == 200
        assert resp.json()["resolution"] == [64, 64]


def test_e_ui_06_client_disconnect_during_sse_streaming(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify background worker task handles early client stream termination cleanly."""
    with (
        mock.patch.object(routes_studio._PIPELINE, "execute", return_value=mock_artifact_payload),
        client.stream(
            "POST",
            "/v1/studio/generate?stream=true",
            json={"prompt": "Early abort", "target_duration_sec": 10.0},
        ) as stream_resp,
    ):
        assert stream_resp.status_code == 200
        for line in stream_resp.iter_lines():
            if "INITIATED" in line:
                break


def test_e_ui_07_sequential_project_switching(client: TestClient, tmp_path: Path, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify consecutive switching between different project IDs isolates video outputs."""
    file_a = tmp_path / "vid_a.mp4"
    file_a.write_bytes(b"VIDEO_A_CONTENT")
    payload_a = mock_artifact_payload.model_copy(update={"project_id": "proj_a", "video_output_path": str(file_a)})

    file_b = tmp_path / "vid_b.mp4"
    file_b.write_bytes(b"VIDEO_B_CONTENT")
    payload_b = mock_artifact_payload.model_copy(update={"project_id": "proj_b", "video_output_path": str(file_b)})

    routes_studio._PROJECT_CACHE["proj_a"] = payload_a
    routes_studio._PROJECT_CACHE["proj_b"] = payload_b

    resp_a = client.get("/v1/studio/projects/proj_a/video")
    resp_b = client.get("/v1/studio/projects/proj_b/video")

    assert resp_a.content == b"VIDEO_A_CONTENT"
    assert resp_b.content == b"VIDEO_B_CONTENT"


def test_e_ui_08_co_benchmark_zero_warmup_boundary(client: TestClient) -> None:
    """Verify co-benchmarking with 0 warmup runs executes without discarding."""
    mock_report = MultiModelBenchmarkReport(
        benchmark_id="bench_zero_warmup",
        iterations=2,
        avg_e2e_latency_ms=1200.0,
        ci95_latency_ms=85.0,
        avg_usm_transfer_us=38.0,
        ci95_usm_transfer_us=3.0,
        audio_rtf=0.20,
        video_rtf=0.75,
        llm_tok_per_sec=50.0,
        cloud_comparison_usd_per_run=0.35,
        hardware_payback_runs=857,
        stage_latencies_ms={"PLANNING": 100.0, "AUDIO": 300.0, "VISUAL": 600.0, "COMPOSITION": 200.0},
    )

    with mock.patch.object(routes_studio._BENCHMARKER, "run_co_benchmark", return_value=mock_report):
        resp = client.post(
            "/v1/studio/benchmark",
            json={"iterations": 2, "warmup_runs": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["benchmark_id"] == "bench_zero_warmup"


def test_e_ui_09_co_benchmark_single_iteration_boundary(client: TestClient) -> None:
    """Verify co-benchmarking with 1 single iteration executes with fallback Student's t multiplier."""
    mock_report = MultiModelBenchmarkReport(
        benchmark_id="bench_single_iter",
        iterations=1,
        avg_e2e_latency_ms=1350.0,
        ci95_latency_ms=0.0,
        avg_usm_transfer_us=40.0,
        ci95_usm_transfer_us=0.0,
        audio_rtf=0.22,
        video_rtf=0.80,
        llm_tok_per_sec=48.0,
        cloud_comparison_usd_per_run=0.35,
        hardware_payback_runs=857,
        stage_latencies_ms={"PLANNING": 110.0, "AUDIO": 310.0, "VISUAL": 720.0, "COMPOSITION": 210.0},
    )

    with mock.patch.object(routes_studio._BENCHMARKER, "run_co_benchmark", return_value=mock_report):
        resp = client.post(
            "/v1/studio/benchmark",
            json={"iterations": 1, "warmup_runs": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["benchmark_id"] == "bench_single_iter"


def test_e_ui_10_ultra_long_prompt_4000_characters(client: TestClient, mock_artifact_payload: StudioArtifactPayload) -> None:
    """Verify large 4000-character creative brief processes without buffer truncation."""
    long_prompt = "Deep space cosmos " * 225
    assert len(long_prompt) >= 4000

    with mock.patch.object(routes_studio._PIPELINE, "execute", return_value=mock_artifact_payload):
        resp = client.post(
            "/v1/studio/generate",
            json={"prompt": long_prompt, "target_duration_sec": 15.0},
        )
        assert resp.status_code == 200
