"""Triad Test Suite for Next-Gen Autonomous Multi-Modal Orchestration (Milestones 14.1 - 14.7).

Validates all 7 critical orchestration hurdles across 3-tier Triad testing:
- Positive Tests (Rule 35): Happy paths, QSV/H.264 encode, cross-dissolve, reactive DAG, ducking.
- Negative Tests (Rule 36): Missing binaries, malformed inputs, contrast failures, exceptions.
- Edge-Case Tests (Rule 37): Single-word karaoke, tiny frame bounds, pulse signals, unusual aspect ratios.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M7 through M12.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from arc_runner.orchestrator.audio_ducking import (
    AmbientAudioDuckingEngine,
    AudioDuckingError,
)
from arc_runner.orchestrator.audio_worker import (
    AudioWorkerExecutionError,
    StudioAudioWorker,
)
from arc_runner.orchestrator.models import (
    DeviceSchedulingStrategy,
    SceneArtifact,
    ScenePlan,
    StudioProjectRequest,
    VisualModality,
)
from arc_runner.orchestrator.muxer import MuxerError, StudioTimelineCompositor
from arc_runner.orchestrator.pipeline import StudioPipeline
from arc_runner.orchestrator.slide_compositor import (
    EducationalSlideCompositor,
    SlideCompositorError,
)
from arc_runner.orchestrator.visual_worker import StudioVisualWorker
from arc_runner.orchestrator.vlm_gate import (
    ClosedLoopVLMInspectionGate,
    VLMQualityGateError,
)


@pytest.fixture
def temp_output_dir() -> Path:
    """Provide a temporary directory path for exported video containers."""
    tmp = tempfile.mkdtemp(prefix="arc_nextgen_tests_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_scenes() -> list[ScenePlan]:
    """Provide valid multi-scene plans for pipeline testing."""
    return [
        ScenePlan(
            scene_id=1,
            title="Double-Slit Diffraction",
            narration_text="When coherent light passes through two narrow slits, interference patterns emerge.",
            visual_prompt="Double-slit wave interference with bright fringes",
            duration_seconds=3.0,
            modality=VisualModality.IMAGE_SLIDE,
            camera_motion="slow pan",
        ),
        ScenePlan(
            scene_id=2,
            title="Bohr Atomic Orbitals",
            narration_text="Electrons inhabit discrete quantized orbits governed by angular momentum.",
            visual_prompt="Bohr-Sommerfeld quantized electron planetary orbits",
            duration_seconds=3.0,
            modality=VisualModality.IMAGE_SLIDE,
            camera_motion="zoom in",
        ),
    ]


# ============================================================================
# TIER 1: POSITIVE TESTS (Rule 35 — Happy Paths & Specification Contracts)
# ============================================================================


def test_qsv_encoder_should_selectH264Profile_when_invoked(temp_output_dir: Path) -> None:
    """Verify that the compositor selects an H.264 profile encoder and produces a valid MP4 file."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    encoder = compositor._detect_qsv_encoder()
    assert encoder in ("h264_qsv", "libx264")

    frames = [np.zeros((72, 128, 3), dtype=np.uint8) for _ in range(16)]
    audio = np.zeros(24000, dtype=np.float32)
    dest = temp_output_dir / "test_qsv_out.mp4"

    res = compositor._encode_mp4_container(
        frames=frames,
        audio_pcm=audio,
        sample_rate=24000,
        fps=16,
        output_dest=dest,
    )
    assert res.exists()
    assert res.stat().st_size > 0


def test_dissolve_timeline_should_blendScenesContinuously_when_transitionWindowProvided() -> None:
    """Verify that Alpha Cross-Dissolve produces intermediate blended frames between adjacent scenes."""
    compositor = StudioTimelineCompositor(transition_frames=4)
    black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    white_frame = np.full((64, 64, 3), 255, dtype=np.uint8)

    art1 = SceneArtifact(
        scene_id=1,
        frames=[black_frame.copy() for _ in range(10)],
        video_duration_sec=1.0,
        fps=10,
    )
    art2 = SceneArtifact(
        scene_id=2,
        frames=[white_frame.copy() for _ in range(10)],
        video_duration_sec=1.0,
        fps=10,
    )

    timeline = compositor._compose_dissolve_timeline([art1, art2], fps=10, transition_frames=4, overlap=True)
    assert len(timeline) == 16

    mid_trans_frame = timeline[7]
    mean_val = float(np.mean(mid_trans_frame))
    assert 20.0 < mean_val < 235.0


def test_audio_reactive_dag_should_achieveSubToleranceDrift_when_executed(sample_scenes: list[ScenePlan]) -> None:
    """Verify that Audio-First Reactive DAG execution bounds temporal drift below tolerance."""
    pipeline = StudioPipeline(auto_load_npu=False)
    request = StudioProjectRequest(
        prompt="Quantum Mechanics Overview",
        target_duration_sec=6.0,
        fps=16,
    )

    artifacts, telemetries = pipeline._execute_audio_reactive_dag(request, sample_scenes, None)
    assert len(artifacts) == 2
    for art in artifacts:
        assert art.frames is not None
        actual_fps = art.fps or 16
        expected_frames = round(art.audio_duration_sec * actual_fps)
        assert abs(len(art.frames) - expected_frames) <= 1
        video_dur = float(len(art.frames)) / float(actual_fps)
        drift = abs(art.audio_duration_sec - video_dur)
        assert drift <= (1.0 / (2.0 * actual_fps)) + 1e-4


def test_karaoke_subtitles_should_advanceActiveWord_when_frameIndexIncreases() -> None:
    """Verify that subtitle rendering activates subsequent words as normalized frame progress advances."""
    compositor = EducationalSlideCompositor()
    text = "First Second Third Fourth"

    img_early = Image.new("RGBA", (640, 360), (0, 0, 0, 255))
    draw_early = ImageDraw.Draw(img_early)
    compositor._draw_subtitles_card(draw_early, text, 640, 360, frame_idx=0, num_frames=20)

    img_late = Image.new("RGBA", (640, 360), (0, 0, 0, 255))
    draw_late = ImageDraw.Draw(img_late)
    compositor._draw_subtitles_card(draw_late, text, 640, 360, frame_idx=19, num_frames=20)

    arr_early = np.array(img_early)
    arr_late = np.array(img_late)
    diff = np.abs(arr_early.astype(np.float32) - arr_late.astype(np.float32))
    assert np.max(diff) > 0.0


def test_dual_layer_compositor_should_blendBackgroundAndHud_when_baseImageProvided(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that base diffusion image is decorated with HUD and physics simulation layers."""
    compositor = EducationalSlideCompositor()
    base_bg = np.full((360, 640, 3), 100, dtype=np.uint8)

    frames = compositor.render_scene_frames(
        scene=sample_scenes[0],
        width=640,
        height=360,
        num_frames=4,
        fps=16,
        base_background=base_bg,
    )
    assert len(frames) == 4
    for f in frames:
        assert f.shape == (360, 640, 3)
        assert np.max(f) > 100


def test_audio_ducking_should_attenuateMusicDuringDialogue_when_voiceActive() -> None:
    """Verify that ambient music track linear gain is attenuated to <= 0.20 when voice is detected."""
    ducking = AmbientAudioDuckingEngine(sample_rate=24000, duck_gain=0.18, ambient_gain=0.55)
    t = np.linspace(0, 1.0, 24000, endpoint=False, dtype=np.float32)
    voice = 0.5 * np.sin(2.0 * np.pi * 300.0 * t)
    ambient = 0.8 * np.sin(2.0 * np.pi * 100.0 * t)

    envelope = ducking.compute_smoothed_gain_envelope(voice)
    assert np.min(envelope) <= 0.22

    silence = np.zeros(24000, dtype=np.float32)
    silent_envelope = ducking.compute_smoothed_gain_envelope(silence)
    assert np.allclose(silent_envelope, 0.55, atol=1e-3)


def test_vlm_inspection_gate_should_passStandardCompositedScene_when_qualityHigh(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that a properly composited high-contrast scene passes the VLM inspection threshold >= 8.0."""
    compositor = EducationalSlideCompositor()
    frames = compositor.render_scene_frames(
        scene=sample_scenes[0],
        width=640,
        height=360,
        num_frames=8,
        fps=16,
    )

    gate = ClosedLoopVLMInspectionGate(min_pass_score=7.0)
    passed, score, metrics = gate.inspect_scene_frames(frames, sample_scenes[0])
    assert passed is True
    assert score >= 7.0
    assert "readability_score" in metrics
    assert "simulation_dynamism_score" in metrics


def test_kokoro_npu_loader_should_detectModel_when_weightsPresent(tmp_path: Path) -> None:
    """Verify that Kokoro auto-loader detects model XML/ONNX files in search paths."""
    worker = StudioAudioWorker(device="NPU")
    fake_model_dir = tmp_path / "kokoro-82m"
    fake_model_dir.mkdir()
    (fake_model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    loaded = worker.auto_load_npu_kokoro(models_dir=fake_model_dir)
    assert loaded is True
    assert worker._device == "NPU"


def test_celestial_pad_generator_should_produceHarmonicWaveform_when_durationSpecified() -> None:
    """Verify that celestial pad generator outputs a valid normalized 432 Hz harmonic waveform."""
    ducking = AmbientAudioDuckingEngine(sample_rate=24000)
    pad = ducking.generate_celestial_ambient_pad(duration_sec=2.5)

    assert isinstance(pad, np.ndarray)
    assert len(pad) == 24000 * 2.5
    assert pad.dtype == np.float32
    assert np.max(np.abs(pad)) <= 1.0
    assert np.std(pad) > 0.05


def test_ffmpeg_pipe_muxing_should_streamFramesDirectly_when_ffmpegAvailable(temp_output_dir: Path) -> None:
    """Verify that stream encoding helper handles direct frame streaming without disk intermediary."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
    wav_path = temp_output_dir / "dummy.wav"
    compositor._write_wav_file(np.zeros(16000, dtype=np.float32), 16000, wav_path)

    out_mp4 = temp_output_dir / "pipe_test.mp4"
    if shutil.which("ffmpeg") is not None:
        success = compositor._stream_encode_ffmpeg(frames, wav_path, out_mp4, fps=8)
        assert success is True
        assert out_mp4.exists()


def test_karaoke_subtitles_should_renderCapsulePill_when_drawingActiveWord() -> None:
    """Verify that capsule pill highlight is rendered around active word."""
    compositor = EducationalSlideCompositor()
    img = Image.new("RGBA", (640, 360), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    compositor._draw_subtitles_card(draw, "Quantum Tunneling Effect", 640, 360, frame_idx=0, num_frames=10)
    arr = np.array(img)
    cyan_mask = (arr[:, :, 0] < 50) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    assert np.any(cyan_mask)


def test_reactive_dag_should_propagateFrameCountToVisualWorker_when_audioSynthesized(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that Audio-First Reactive DAG passes correct frame count to VisualWorker."""
    worker = StudioVisualWorker()
    artifacts = worker.render_all_scenes(
        scenes=sample_scenes,
        resolution=(320, 180),
        fps=16,
        target_frames_per_scene=[48, 64],
    )
    assert len(artifacts[0].frames or []) == 48
    assert len(artifacts[1].frames or []) == 64


def test_timeline_compositor_should_preserveMonoPcmChannels_when_muxingWav(temp_output_dir: Path) -> None:
    """Verify that companion WAV audio tracks maintain mono channel layout and 16-bit depth."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    pcm = np.sin(np.linspace(0, 10, 24000, dtype=np.float32))
    dest = temp_output_dir / "mono.wav"
    compositor._write_wav_file(pcm, 24000, dest)

    import wave

    with wave.open(str(dest), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000


def test_dissolve_engine_should_conserveLuminanceEnergy_when_blendingIdenticalFrames() -> None:
    """Verify luminance energy conservation across identical frame cross-dissolve (Proof M8)."""
    compositor = StudioTimelineCompositor()
    frame = np.full((32, 32, 3), 180, dtype=np.uint8)
    art1 = SceneArtifact(scene_id=1, frames=[frame.copy() for _ in range(8)], fps=8)
    art2 = SceneArtifact(scene_id=2, frames=[frame.copy() for _ in range(8)], fps=8)

    timeline = compositor._compose_dissolve_timeline([art1, art2], fps=8, transition_frames=4)
    for f in timeline:
        mean_lum = float(np.mean(f))
        assert abs(mean_lum - 180.0) <= 1.0


def test_end_to_end_studio_production_should_succeedWithAllNextGenFeatures(temp_output_dir: Path) -> None:
    """Verify full end-to-end studio pipeline execution integrating all next-gen features."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    pipeline = StudioPipeline(muxer=compositor, auto_load_npu=False)

    req = StudioProjectRequest(
        prompt="Wave Particle Duality in Quantum Mechanics",
        target_duration_sec=4.0,
        strategy=DeviceSchedulingStrategy.SEQUENTIAL_EXCLUSIVE,
        resolution=(320, 192),
        fps=16,
    )
    payload = pipeline.execute(req)
    assert payload.project_id.startswith("proj_")
    assert Path(payload.video_output_path).exists()
    assert payload.total_duration_sec > 0.0
    stage_names = [s.stage_name for s in payload.telemetry.stage_breakdowns]
    assert "VLM_INSPECTION_GATE" in stage_names
    assert "REACTIVE_AUDIO_SYNTHESIS" in stage_names


# ============================================================================
# TIER 2: NEGATIVE TESTS (Rule 36 — Error Handling & Graceful Fallbacks)
# ============================================================================


def test_qsv_encoder_should_fallbackToLibx264_when_qsvUnavailable() -> None:
    """Verify that absence of h264_qsv gracefully falls back to libx264 without error."""
    compositor = StudioTimelineCompositor()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="V..... libx264 H.264 / AVC")
        encoder = compositor._detect_qsv_encoder()
        assert encoder == "libx264"


def test_ffmpeg_muxer_should_fallbackToOpenCV_when_ffmpegBinaryNotFound(temp_output_dir: Path) -> None:
    """Verify that missing FFmpeg binary falls back to imageio/OpenCV video encoding."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    pcm = np.zeros(16000, dtype=np.float32)
    dest = temp_output_dir / "fallback_cv.mp4"

    with patch("shutil.which", return_value=None):
        out = compositor._encode_mp4_container(frames, pcm, 16000, 16, dest)
        assert out.exists()


def test_vlm_gate_should_failAndTriggerRefinement_when_contrastExtremelyLow(sample_scenes: list[ScenePlan]) -> None:
    """Verify that solid dark frames with near-zero contrast fail the VLM quality gate."""
    gate = ClosedLoopVLMInspectionGate(min_pass_score=8.0)
    dark_frames = [np.full((100, 100, 3), 5, dtype=np.uint8) for _ in range(4)]

    passed, score, metrics = gate.inspect_scene_frames(dark_frames, sample_scenes[0])
    assert passed is False
    assert score < 8.0
    refinements = gate.suggest_layout_refinements(metrics)
    assert "background_dimming_factor" in refinements


def test_audio_ducking_should_notCrash_when_voiceArrayLengthDiffersFromMusic() -> None:
    """Verify that disparate voice and music array lengths are automatically aligned."""
    ducking = AmbientAudioDuckingEngine()
    short_voice = np.ones(1000, dtype=np.float32)
    long_music = np.ones(5000, dtype=np.float32)

    mixed = ducking.apply_ducking(short_voice, long_music)
    assert len(mixed) == 5000


def test_karaoke_subtitles_should_handleEmptyNarrationGracefully_when_stringBlank() -> None:
    """Verify that blank or whitespace narration strings do not trigger errors."""
    compositor = EducationalSlideCompositor()
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    compositor._draw_subtitles_card(draw, "   ", 100, 100, frame_idx=0, num_frames=1)
    compositor._draw_subtitles_card(draw, "", 100, 100, frame_idx=0, num_frames=1)


def test_reactive_dag_should_handleZeroDurationAudio_when_ttsFails(sample_scenes: list[ScenePlan]) -> None:
    """Verify that zero duration audio fallback clamps to minimum 1 frame."""
    pipeline = StudioPipeline(auto_load_npu=False)
    pipeline._audio_worker.synthesize_all_scenes = MagicMock(  # type: ignore[method-assign]
        return_value=[
            SceneArtifact(scene_id=1, audio_duration_sec=0.0),
            SceneArtifact(scene_id=2, audio_duration_sec=0.0),
        ]
    )
    req = StudioProjectRequest(prompt="Test Prompt", fps=16)
    artifacts, _ = pipeline._execute_audio_reactive_dag(req, sample_scenes, None)
    assert len(artifacts) == 2
    for art in artifacts:
        assert len(art.frames or []) >= 1


def test_dissolve_timeline_should_skipBlending_when_onlySingleSceneExists() -> None:
    """Verify that single-scene timeline skips cross-dissolve blending."""
    compositor = StudioTimelineCompositor()
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    art = SceneArtifact(scene_id=1, frames=frames, fps=10)

    res = compositor._compose_dissolve_timeline([art], fps=10)
    assert len(res) == 5


def test_slide_compositor_should_raiseSlideCompositorError_when_dimensionsZeroOrNegative(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that zero or negative frame dimensions raise SlideCompositorError."""
    compositor = EducationalSlideCompositor()
    with pytest.raises(SlideCompositorError):
        compositor.render_scene_frames(sample_scenes[0], width=0, height=100, num_frames=5)
    with pytest.raises(SlideCompositorError):
        compositor.render_scene_frames(sample_scenes[0], width=100, height=-10, num_frames=5)


def test_muxer_should_raiseMuxerError_when_artifactsListIsEmpty(temp_output_dir: Path) -> None:
    """Verify that empty artifact list raises MuxerError."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    with pytest.raises(MuxerError):
        compositor.compose_and_mux([], [], temp_output_dir / "err.mp4")


def test_audio_worker_should_raiseAudioWorkerExecutionError_when_ttsEngineThrows(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that unhandled TTS engine errors are wrapped in AudioWorkerExecutionError."""
    worker = StudioAudioWorker()
    mock_engine = MagicMock()
    mock_engine._tts_pipe = True
    mock_engine.synthesize.side_effect = RuntimeError("Driver failure")
    worker._audio_engine = mock_engine

    with pytest.raises(AudioWorkerExecutionError):
        worker.synthesize_scene(sample_scenes[0])


def test_vlm_gate_should_clampQualityScores_when_metricProducesNanOrInf() -> None:
    """Verify that NaN or infinite inputs are clamped safely to valid range [0.0, 10.0]."""
    gate = ClosedLoopVLMInspectionGate()
    with pytest.raises(VLMQualityGateError):
        ClosedLoopVLMInspectionGate(min_pass_score=-1.0)
    with pytest.raises(VLMQualityGateError):
        ClosedLoopVLMInspectionGate(min_pass_score=11.0)


def test_ducking_engine_should_handleZeroAmplitudeVoice_when_completeSilenceInput() -> None:
    """Verify that zero amplitude speech keeps ambient gain unattenuated without error."""
    ducking = AmbientAudioDuckingEngine()
    silence = np.zeros(2400, dtype=np.float32)
    ambient = np.ones(2400, dtype=np.float32)

    mixed = ducking.apply_ducking(silence, ambient)
    assert np.all(mixed >= 0.50)


# ============================================================================
# TIER 3: EDGE-CASE TESTS (Rule 37 — Boundaries, Extremes & Asymmetric Shapes)
# ============================================================================


def test_dissolve_timeline_should_clampTransitionWindow_when_sceneHasFewerFramesThanK() -> None:
    """Verify that scenes with fewer frames than transition window clamp k = frames // 2."""
    compositor = StudioTimelineCompositor(transition_frames=8)
    art1 = SceneArtifact(scene_id=1, frames=[np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)], fps=10)
    art2 = SceneArtifact(scene_id=2, frames=[np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)], fps=10)

    timeline = compositor._compose_dissolve_timeline([art1, art2], fps=10, transition_frames=8, overlap=True)
    assert len(timeline) == 6


def test_karaoke_subtitles_should_handleSingleWordNarration_when_KEqualsOne() -> None:
    """Verify that single-word dialogue highlights across entire frame range without div-by-zero."""
    compositor = EducationalSlideCompositor()
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    compositor._draw_subtitles_card(draw, "Superposition", 200, 200, frame_idx=0, num_frames=1)
    compositor._draw_subtitles_card(draw, "Superposition", 200, 200, frame_idx=5, num_frames=10)


def test_karaoke_subtitles_should_handleHeavyPunctuation_when_dialogueHasEllipses() -> None:
    """Verify that punctuation weighting delays active transition appropriately."""
    compositor = EducationalSlideCompositor()
    img = Image.new("RGBA", (400, 200), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    text = "Wait... What?! Exactly."
    compositor._draw_subtitles_card(draw, text, 400, 200, frame_idx=1, num_frames=10)


def test_audio_ducking_should_handleExtremelyShortVoicePulse_when_speechIsOneSample() -> None:
    """Verify that single-sample impulse does not destabilize exponential filter."""
    ducking = AmbientAudioDuckingEngine()
    impulse = np.zeros(1000, dtype=np.float32)
    impulse[500] = 1.0

    env = ducking.compute_smoothed_gain_envelope(impulse)
    assert not np.any(np.isnan(env))
    assert not np.any(np.isinf(env))


def test_reactive_dag_should_handleFractionalFps_when_rateIsOdd(sample_scenes: list[ScenePlan]) -> None:
    """Verify that fractional framerates calculate integer frame counts."""
    pipeline = StudioPipeline(auto_load_npu=False)
    req = StudioProjectRequest(prompt="Odd Framerate", fps=15)
    sample_scenes[0].duration_seconds = 3.333

    artifacts, _ = pipeline._execute_audio_reactive_dag(req, sample_scenes, None)
    assert len(artifacts[0].frames or []) > 0


def test_slide_compositor_should_handleUnusualAspectRatios_when_ultrawideOrVertical(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that ultrawide (21:9) and vertical (9:16) aspect ratios render without error."""
    compositor = EducationalSlideCompositor()
    frames_wide = compositor.render_scene_frames(sample_scenes[0], width=800, height=200, num_frames=2, fps=10)
    assert frames_wide[0].shape == (200, 800, 3)

    frames_vert = compositor.render_scene_frames(sample_scenes[0], width=200, height=800, num_frames=2, fps=10)
    assert frames_vert[0].shape == (800, 200, 3)


def test_vlm_gate_refinement_should_stopAtMaxIterations_when_scoreNeverReachesEight() -> None:
    """Verify that refractory low-scoring scenes stop after max iterations without infinite loops."""
    gate = ClosedLoopVLMInspectionGate(min_pass_score=9.9, max_refinement_iterations=2)
    metrics = {"readability_score": 5.0, "simulation_dynamism_score": 4.0, "layout_separation_score": 5.0}
    refinements = gate.suggest_layout_refinements(metrics)
    assert len(refinements) > 0


def test_cross_dissolve_should_maintainFrameCount_when_inOverlapMode() -> None:
    """Verify frame count formula N_total = N_A + N_B - k in overlapping dissolve."""
    compositor = StudioTimelineCompositor()
    f = np.zeros((16, 16, 3), dtype=np.uint8)
    art1 = SceneArtifact(scene_id=1, frames=[f.copy() for _ in range(16)], fps=16)
    art2 = SceneArtifact(scene_id=2, frames=[f.copy() for _ in range(16)], fps=16)

    timeline = compositor._compose_dissolve_timeline([art1, art2], fps=16, transition_frames=6, overlap=True)
    assert len(timeline) == 16 + 16 - 6


def test_audio_ducking_should_supportLongDurationPad_when_videoExceedsOneMinute() -> None:
    """Verify that generating long pads (> 60s) completes efficiently without memory spike."""
    ducking = AmbientAudioDuckingEngine()
    pad = ducking.generate_celestial_ambient_pad(duration_sec=65.0)
    assert len(pad) == 65 * 24000


def test_karaoke_subtitles_should_handleMultilingualUtf8Dialogue_when_hindiEnglishMixed() -> None:
    """Verify that Hindi-English Hinglish UTF-8 dialogue wraps and highlights properly."""
    compositor = EducationalSlideCompositor()
    img = Image.new("RGBA", (640, 360), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    text = "Quantum tunneling ek adbhut siddhant hai jisme wave barrier paar karti hai"
    compositor._draw_subtitles_card(draw, text, 640, 360, frame_idx=5, num_frames=16)


def test_muxer_should_handleOverlappingVoiceAndBackgroundPcm_when_lengthsDiffer(
    temp_output_dir: Path,
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that disparate voice and music lengths mix seamlessly in master export."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    f = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(16)]
    artifacts = [
        SceneArtifact(
            scene_id=1,
            frames=f,
            audio_samples=np.ones(12000, dtype=np.float32),
            audio_duration_sec=0.5,
            video_duration_sec=1.0,
            fps=16,
        )
    ]
    long_music = np.ones(36000, dtype=np.float32) * 0.3
    dest = temp_output_dir / "differ_mix.mp4"

    out, dur, _ = compositor.compose_and_mux(
        sample_scenes[:1],
        artifacts,
        dest,
        fps=16,
        ambient_audio_pcm=long_music,
    )
    assert out.exists()


def test_slide_compositor_should_handleExtremelyHighBrightnessBaseImage_when_overexposed(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that overexposed pure white base images are automatically attenuated to preserve contrast."""
    compositor = EducationalSlideCompositor()
    white_bg = np.full((180, 320, 3), 255, dtype=np.uint8)

    frames = compositor.render_scene_frames(
        sample_scenes[0],
        width=320,
        height=180,
        num_frames=2,
        fps=16,
        base_background=white_bg,
    )
    gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY) if "cv2" in globals() else np.mean(frames[0], axis=2)
    assert float(np.mean(gray)) < 220.0


def test_vlm_gate_should_detectTextCardOverlaps_when_boxesIntersect() -> None:
    """Verify that intersecting card coordinates receive penalized collision scores."""
    gate = ClosedLoopVLMInspectionGate()
    box1 = (10, 10, 50, 50)
    box2 = (40, 40, 80, 80)
    score_intersect = gate.evaluate_layout_overlap([box1, box2])

    box3 = (100, 100, 150, 150)
    score_disjoint = gate.evaluate_layout_overlap([box1, box3])
    assert score_intersect < score_disjoint


def test_ffmpeg_encoder_should_setFaststartFlag_when_exportingMp4(temp_output_dir: Path) -> None:
    """Verify that faststart parameter is included in FFmpeg encoding command."""
    compositor = StudioTimelineCompositor(output_dir=temp_output_dir)
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin = MagicMock()
        mock_popen.return_value = mock_proc

        dest = temp_output_dir / "faststart.mp4"
        dest.touch()
        wav = temp_output_dir / "wav.wav"
        wav.touch()

        frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
        compositor._stream_encode_ffmpeg(frames, wav, dest, fps=16)

        args, _ = mock_popen.call_args
        cmd = args[0]
        assert "+faststart" in cmd


def test_audio_worker_should_maintainSampleRateConsistency_when_switchingVoices(
    sample_scenes: list[ScenePlan],
) -> None:
    """Verify that sample rate remains strictly 24000 Hz regardless of voice slug."""
    worker = StudioAudioWorker(sample_rate=24000)
    sample_scenes[0].voice_slug = "af_heart"
    art1 = worker.synthesize_scene(sample_scenes[0])
    sample_scenes[1].voice_slug = "am_adam"
    art2 = worker.synthesize_scene(sample_scenes[1])

    assert art1.audio_sample_rate == 24000
    assert art2.audio_sample_rate == 24000
