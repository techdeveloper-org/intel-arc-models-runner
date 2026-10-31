"""Test suite for Audio & Voice and Vision & Video media endpoints and UI elements.

Adheres strictly to Level 2.2 Docstrings-Only convention and Claude Global Rules
04, 12, 28, 35, 36, 37, 40.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create test client connected to initialized FastAPI application."""
    manager = ModelManager(model_dirs=[tmp_path])
    app = create_app(model_manager=manager, debug=True)
    return TestClient(app)


def test_p_media_01_ui_markup_contains_all_7_modality_tabs(client: TestClient) -> None:
    """Verify served single-page UI HTML contains all 7 dedicated navigation spaces."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    # 7 Tab Panes
    assert 'id="tab-cockpit"' in resp.text
    assert 'id="tab-chat"' in resp.text
    assert 'id="tab-audio"' in resp.text
    assert 'id="tab-vision"' in resp.text
    assert 'id="tab-ollama"' in resp.text
    assert 'id="tab-benchmark"' in resp.text
    assert 'id="tab-studio"' in resp.text


def test_p_media_02_silicon_cockpit_contains_circular_svg_gauges(client: TestClient) -> None:
    """Verify Silicon Cockpit includes GPU VRAM and NPU circular SVG radial gauges."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="gpu-vram-radial"' in resp.text
    assert 'id="gpu-vram-radial-text"' in resp.text
    assert 'id="npu-radial-gauge"' in resp.text
    assert 'id="npu-radial-text"' in resp.text


def test_p_media_03_audio_studio_submodules_present(client: TestClient) -> None:
    """Verify Audio Studio features Speech-to-Text and Text-to-Speech submodules."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="sub-audio-stt"' in resp.text
    assert 'id="sub-audio-tts"' in resp.text
    assert 'id="btn-transcribe-audio"' in resp.text
    assert 'id="btn-synthesize-tts"' in resp.text
    assert 'id="tts-audio-element"' in resp.text


def test_p_media_04_vision_studio_submodules_present(client: TestClient) -> None:
    """Verify Vision Studio features Text2Image and Text2Video submodules."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert 'id="sub-vision-img"' in resp.text
    assert 'id="sub-vision-vid"' in resp.text
    assert 'id="btn-generate-image"' in resp.text
    assert 'id="btn-generate-video"' in resp.text
    assert 'id="img-output-preview"' in resp.text
    assert 'id="vid-output-preview"' in resp.text


def test_n_media_05_audio_transcribe_missing_payload(client: TestClient) -> None:
    """Verify /api/audio/transcribe rejects requests with neither audio_path nor audio_base64."""
    resp = client.post("/api/audio/transcribe", json={"model_id": "whisper-base"})
    assert resp.status_code == 400
    assert "Either audio_path or audio_base64" in resp.json()["detail"]


def test_n_media_06_audio_transcribe_nonexistent_file(client: TestClient) -> None:
    """Verify /api/audio/transcribe returns 404 for missing local audio file path."""
    resp = client.post(
        "/api/audio/transcribe",
        json={"audio_path": "/nonexistent/sound/file.wav"},
    )
    assert resp.status_code == 404


def test_n_media_07_tts_empty_prompt_validation(client: TestClient) -> None:
    """Verify /api/audio/tts rejects 0-length prompt with 422 Unprocessable Entity."""
    resp = client.post("/api/audio/tts", json={"prompt": ""})
    assert resp.status_code == 422


def test_n_media_08_image_empty_prompt_validation(client: TestClient) -> None:
    """Verify /api/vision/generate-image rejects empty prompt with 422."""
    resp = client.post("/api/vision/generate-image", json={"prompt": ""})
    assert resp.status_code == 422


def test_n_media_09_video_empty_prompt_validation(client: TestClient) -> None:
    """Verify /api/vision/generate-video rejects empty prompt with 422."""
    resp = client.post("/api/vision/generate-video", json={"prompt": ""})
    assert resp.status_code == 422


def test_p_media_10_audio_tts_mock_stream(client: TestClient) -> None:
    """Verify /api/audio/tts returns streaming WAV with duration and RTF headers."""
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    with mock.patch("arc_runner.engine.audio_engine.AudioEngine.load_model"), \
         mock.patch("arc_runner.engine.audio_engine.AudioEngine.synthesize") as mock_synth, \
         mock.patch("arc_runner.engine.audio_engine.AudioEngine.save_audio_wav") as mock_save:

        class DummyResult:
            duration_s = 2.5
            sample_rate = 24000
            audio_samples = [0.0]

        mock_synth.return_value = DummyResult()
        mock_save.side_effect = lambda samples, buf, sample_rate: buf.write(dummy_wav)

        resp = client.post(
            "/api/audio/tts",
            json={"prompt": "Hello Intel Arc", "voice": "af_heart", "speed": 1.0},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert "X-Audio-Duration" in resp.headers
        assert "X-RTF" in resp.headers
        assert resp.content == dummy_wav


def test_p_media_11_vision_image_mock_stream(client: TestClient) -> None:
    """Verify /api/vision/generate-image returns streaming PNG image with telemetry."""
    dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    with mock.patch("arc_runner.engine.image_engine.ImageEngine.load_model"), \
         mock.patch("arc_runner.engine.image_engine.ImageEngine.generate") as mock_gen:

        class DummyBreakdown:
            iteration_rate_it_s = 14.2
            megapixels_per_sec = 0.35

        class DummyImageResult:
            image_bytes = dummy_png
            breakdown = DummyBreakdown()

        mock_gen.return_value = DummyImageResult()

        resp = client.post(
            "/api/vision/generate-image",
            json={"prompt": "A futuristic city on Intel Arc", "steps": 20},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["x-step-rate"] == "14.20"
        assert resp.content == dummy_png
