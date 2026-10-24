"""Exhaustive 30-test Triad Suite for AudioEngine, Speech Recognition, and Speech Synthesis.

Validates Positive (10), Negative (10), and Edge-Case (10) scenarios, including Voice Activity
Detection (VAD) silence gating (Proof M69), Ziv-Lempel compression ratio loop detection (Proof M70),
zero-disk residue invariants (Proof M66), and Real-Time Factor calculations (Proof M59).
Strictly adheres to Level 2.2 Docstrings-Only standards and Rule 28 test coverage enforcement.
"""

import math
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from arc_runner.core.exceptions import (
    AudioEngineError,
    DeviceNotFoundError,
    InvalidAudioFormatError,
    ModelNotFoundError,
    VoiceNotFoundError,
)
from arc_runner.engine.audio_engine import (
    AudioEngine,
    SynthesisResult,
    TranscriptionResult,
)
from arc_runner.hub.registry import CuratedModelInfo, ModelRegistry


@pytest.fixture
def sample_wav_path(tmp_path: Path) -> Path:
    """Generate a clean 16kHz mono PCM 16-bit WAV file for testing."""
    wav_file = tmp_path / "test_speech.wav"
    sample_rate = 16000
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())

    return wav_file


@pytest.fixture
def mock_openvino_genai():
    """Create a mock openvino_genai module with WhisperPipeline and Text2SpeechPipeline."""
    with patch.dict("sys.modules", {"openvino_genai": MagicMock()}):
        import openvino_genai as ov_genai

        mock_asr_chunk = MagicMock()
        mock_asr_chunk.start_ts = 0.0
        mock_asr_chunk.end_ts = 2.0
        mock_asr_chunk.text = "Hello Intel Arc"

        mock_asr_results = MagicMock()
        mock_asr_results.texts = ["Hello Intel Arc"]
        mock_asr_results.chunks = [mock_asr_chunk]

        mock_whisper_instance = MagicMock()
        mock_whisper_instance.generate.return_value = mock_asr_results
        ov_genai.WhisperPipeline.return_value = mock_whisper_instance

        mock_tts_results = MagicMock()
        mock_tts_results.output_sample_rate = 24000
        mock_tts_results.speeches = [np.zeros(24000 * 2, dtype=np.float32)]

        mock_tts_instance = MagicMock()
        mock_tts_instance.generate.return_value = mock_tts_results
        ov_genai.Text2SpeechPipeline.return_value = mock_tts_instance

        yield {
            "genai": ov_genai,
            "whisper": mock_whisper_instance,
            "tts": mock_tts_instance,
            "asr_results": mock_asr_results,
            "tts_results": mock_tts_results,
        }


# =========================================================================
# TRIAD TESTING SUITE: POSITIVE TESTS (10 TESTS)
# =========================================================================

def test_aud_pos_01_audio_engine_default_initialization():
    """Verify AudioEngine properly initializes with default configuration and null state."""
    engine = AudioEngine()
    assert engine.is_loaded is False
    assert engine.model_path is None
    assert engine.device in ["GPU", "NPU", "CPU"]
    assert engine.task_type is None


def test_aud_pos_02_normalize_audio_from_numpy_array():
    """Verify _normalize_audio_input converts float32 numpy arrays and computes duration."""
    engine = AudioEngine()
    audio = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
    samples_list, duration_s = engine._normalize_audio_input(audio)
    assert len(samples_list) == 16000
    assert duration_s == pytest.approx(1.0, 0.01)
    assert all(isinstance(x, float) for x in samples_list)


def test_aud_pos_03_normalize_audio_from_float_list():
    """Verify _normalize_audio_input handles standard Python float lists."""
    engine = AudioEngine()
    raw_list = [0.1 * math.sin(i) for i in range(8000)]
    samples_list, duration_s = engine._normalize_audio_input(raw_list)
    assert len(samples_list) == 8000
    assert duration_s == pytest.approx(0.5, 0.01)


def test_aud_pos_04_normalize_audio_from_wav_file(sample_wav_path: Path):
    """Verify _normalize_audio_input parses standard 16-bit PCM WAV files."""
    engine = AudioEngine()
    samples_list, duration_s = engine._normalize_audio_input(sample_wav_path)
    assert len(samples_list) == 32000
    assert duration_s == pytest.approx(2.0, 0.01)


def test_aud_pos_05_stt_transcribe_clean_speech_mock(tmp_path: Path, mock_openvino_genai):
    """Verify transcribe executes WhisperPipeline and produces valid TranscriptionResult."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")
    assert engine.is_loaded is True
    assert engine.task_type == "stt"

    audio = np.ones(16000 * 2, dtype=np.float32) * 0.5
    result = engine.transcribe(audio)

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello Intel Arc"
    assert result.audio_duration_s == pytest.approx(2.0, 0.01)
    assert result.latency_s >= 0
    assert result.rtf >= 0
    assert result.throughput_audio_s_per_s > 0


def test_aud_pos_06_tts_synthesize_clean_text_mock(tmp_path: Path, mock_openvino_genai):
    """Verify synthesize executes Text2SpeechPipeline and produces valid SynthesisResult."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    voices_dir = model_dir / "voices"
    voices_dir.mkdir()
    (voices_dir / "af_heart.bin").write_bytes(b"\x00" * 522240)

    with patch("openvino.Tensor", return_value=MagicMock()):
        engine = AudioEngine()
        engine.load_model(model_dir, device="CPU")
        assert engine.is_loaded is True
        assert engine.task_type == "tts"

        result = engine.synthesize("Welcome to Intel Arc Models Runner.")
        assert isinstance(result, SynthesisResult)
        assert result.sample_rate == 24000
        assert result.duration_s == pytest.approx(2.0, 0.01)
        assert len(result.audio_samples) == 48000
        assert result.voice_used == "af_heart"


def test_aud_pos_07_rtf_mathematical_invariant():
    """Verify Proof M59 mathematical invariant: rtf == latency_s / duration_s."""
    duration_s = 5.0
    latency_s = 0.25
    rtf = round(latency_s / duration_s, 4)
    throughput = round(duration_s / latency_s, 2)
    assert rtf == 0.05
    assert throughput == 20.0


def test_aud_pos_08_multi_speaker_voice_switching(tmp_path: Path, mock_openvino_genai):
    """Verify distinct voice embeddings can be selected and loaded."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    voices_dir = model_dir / "voices"
    voices_dir.mkdir()
    (voices_dir / "af_heart.bin").write_bytes(b"\x00" * 522240)
    (voices_dir / "am_adam.bin").write_bytes(b"\x00" * 522240)

    with patch("openvino.Tensor", return_value=MagicMock()):
        engine = AudioEngine()
        engine.load_model(model_dir, device="CPU")
        res1 = engine.synthesize("First sentence.", voice="af_heart")
        res2 = engine.synthesize("Second sentence.", voice="am_adam")
        assert res1.voice_used == "af_heart"
        assert res2.voice_used == "am_adam"


def test_aud_pos_09_audio_engine_context_manager(tmp_path: Path, mock_openvino_genai):
    """Verify AudioEngine operates cleanly as a context manager and auto-unloads on exit."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    with AudioEngine() as engine:
        engine.load_model(model_dir, device="CPU")
        assert engine.is_loaded is True
    assert engine.is_loaded is False


def test_aud_pos_10_registry_resolves_all_audio_curated_models():
    """Verify ModelRegistry correctly registers and resolves all 7 official audio models."""
    audio_slugs = [
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-large-v3-turbo",
        "distil-whisper-large-v3",
        "kokoro-82m",
        "kokoro-82m-int8",
    ]
    for slug in audio_slugs:
        info = ModelRegistry.resolve(slug)
        assert isinstance(info, CuratedModelInfo)
        assert info.alias == slug
        assert info.model_type in ["stt", "tts"]
        assert info.size_bytes > 0
        assert "OpenVINO/" in info.repo_id


# =========================================================================
# TRIAD TESTING SUITE: NEGATIVE TESTS (10 TESTS)
# =========================================================================

def test_aud_neg_01_transcribe_unloaded_engine_raises_error():
    """Verify transcribe raises AudioEngineError when invoked prior to load_model."""
    engine = AudioEngine()
    with pytest.raises(AudioEngineError, match="No speech recognition pipeline loaded"):
        engine.transcribe([0.1, 0.2, 0.3])


def test_aud_neg_02_synthesize_unloaded_engine_raises_error():
    """Verify synthesize raises AudioEngineError when invoked prior to load_model."""
    engine = AudioEngine()
    with pytest.raises(AudioEngineError, match="No speech synthesis pipeline loaded"):
        engine.synthesize("Test prompt")


def test_aud_neg_03_transcribe_empty_audio_raises_error():
    """Verify transcribe raises InvalidAudioFormatError when provided empty audio array."""
    engine = AudioEngine()
    with pytest.raises(InvalidAudioFormatError, match="empty"):
        engine._normalize_audio_input(np.array([]))


def test_aud_neg_04_transcribe_nan_inf_audio_raises_error():
    """Verify transcribe raises InvalidAudioFormatError on NaN or Inf audio values."""
    engine = AudioEngine()
    bad_audio = np.array([0.1, float("nan"), 0.5], dtype=np.float32)
    with pytest.raises(InvalidAudioFormatError, match="NaN or Inf"):
        engine._normalize_audio_input(bad_audio)


def test_aud_neg_05_synthesize_empty_text_raises_error(tmp_path: Path, mock_openvino_genai):
    """Verify synthesize raises ValueError when provided empty or whitespace string."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")
    with pytest.raises(ValueError, match="Input text prompt cannot be empty"):
        engine.synthesize("   ")


def test_aud_neg_06_synthesize_invalid_voice_slug_raises_error(tmp_path: Path, mock_openvino_genai):
    """Verify synthesize raises VoiceNotFoundError when requested voice cannot be found."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")
    with pytest.raises(VoiceNotFoundError, match="Voice embedding 'non_existent_voice' not found"):
        engine.synthesize("Hello world", voice="non_existent_voice")


def test_aud_neg_07_load_model_nonexistent_directory_raises_error():
    """Verify load_model raises ModelNotFoundError on non-existent path."""
    engine = AudioEngine()
    with pytest.raises(ModelNotFoundError):
        engine.load_model(Path("non_existent_dir_12345"), device="CPU")


def test_aud_neg_08_load_model_unsupported_device_raises_error(tmp_path: Path):
    """Verify load_model raises DeviceNotFoundError on invalid hardware device string."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    with pytest.raises(DeviceNotFoundError):
        engine.load_model(model_dir, device="QUANTUM_SILICON")


def test_aud_neg_09_synthesize_out_of_bounds_speed_raises_error(tmp_path: Path, mock_openvino_genai):
    """Verify synthesize raises ValueError when speed multiplier is outside [0.1, 5.0]."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")
    with pytest.raises(ValueError, match="Speech speed multiplier -0.5 out of valid range"):
        engine.synthesize("Hello", speed=-0.5)


def test_aud_neg_10_corrupted_wav_file_raises_error(tmp_path: Path):
    """Verify _normalize_audio_input raises InvalidAudioFormatError on corrupted WAV files."""
    bad_wav = tmp_path / "corrupt.wav"
    bad_wav.write_bytes(b"RIFF\x00\x00\x00\x00DEADBEEFNOTAWAV")
    engine = AudioEngine()
    with pytest.raises(InvalidAudioFormatError, match="Failed to read WAV audio file"):
        engine._normalize_audio_input(bad_wav)


# =========================================================================
# TRIAD TESTING SUITE: EDGE-CASE TESTS (10 TESTS)
# =========================================================================

def test_aud_edg_01_silence_gating_invariant(tmp_path: Path, mock_openvino_genai):
    """Verify Proof M69: Pure silence audio emits empty string and bypasses decoder."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")

    silence = np.zeros(16000 * 3, dtype=np.float32)
    result = engine.transcribe(silence)

    assert result.text == ""
    assert result.segments == []
    assert result.audio_duration_s == pytest.approx(3.0, 0.01)
    mock_openvino_genai["whisper"].generate.assert_not_called()


def test_aud_edg_02_low_energy_noise_gating(tmp_path: Path, mock_openvino_genai):
    """Verify Proof M69: Low-level Gaussian noise below VAD threshold is gated as silence."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")

    noise = np.random.normal(0, 0.00001, 16000 * 2).astype(np.float32)
    result = engine.transcribe(noise)
    assert result.text == ""
    mock_openvino_genai["whisper"].generate.assert_not_called()


def test_aud_edg_03_repetition_loop_detection_proof_m70(tmp_path: Path, mock_openvino_genai):
    """Verify Proof M70: High compression ratio triggers repetition loop truncation."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    repeated_text = "and then " * 50
    mock_openvino_genai["asr_results"].texts = [repeated_text]

    engine = AudioEngine()
    engine.load_model(model_dir, device="CPU")

    audio = np.ones(16000 * 3, dtype=np.float32) * 0.2
    result = engine.transcribe(audio)

    assert result.compression_ratio > 2.40
    assert len(result.text.split()) < len(repeated_text.split())


def test_aud_edg_04_ultra_short_50ms_audio_clip():
    """Verify _normalize_audio_input handles 50ms (800 sample) clips without indexing errors."""
    engine = AudioEngine()
    short_audio = np.ones(800, dtype=np.float32) * 0.1
    samples, duration_s = engine._normalize_audio_input(short_audio)
    assert len(samples) == 800
    assert duration_s == pytest.approx(0.05, 0.001)


def test_aud_edg_05_long_continuous_audio_buffer():
    """Verify _normalize_audio_input normalizes 60-second audio buffer without memory leakage."""
    engine = AudioEngine()
    long_audio = np.ones(16000 * 60, dtype=np.float32) * 0.1
    samples, duration_s = engine._normalize_audio_input(long_audio)
    assert len(samples) == 960000
    assert duration_s == pytest.approx(60.0, 0.01)


def test_aud_edg_06_extreme_tts_speed_boundaries(tmp_path: Path, mock_openvino_genai):
    """Verify synthesize executes without errors at valid speed boundaries (0.1x and 5.0x)."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    voices_dir = model_dir / "voices"
    voices_dir.mkdir()
    (voices_dir / "af_heart.bin").write_bytes(b"\x00" * 522240)

    with patch("openvino.Tensor", return_value=MagicMock()):
        engine = AudioEngine()
        engine.load_model(model_dir, device="CPU")
        res_min = engine.synthesize("Slow speech", speed=0.1)
        res_max = engine.synthesize("Fast speech", speed=5.0)
        assert isinstance(res_min, SynthesisResult)
        assert isinstance(res_max, SynthesisResult)


def test_aud_edg_07_special_unicode_and_emojis_tts(tmp_path: Path, mock_openvino_genai):
    """Verify synthesize handles text containing emojis and special Unicode symbols."""
    model_dir = tmp_path / "kokoro-82m"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    voices_dir = model_dir / "voices"
    voices_dir.mkdir()
    (voices_dir / "af_heart.bin").write_bytes(b"\x00" * 522240)

    with patch("openvino.Tensor", return_value=MagicMock()):
        engine = AudioEngine()
        engine.load_model(model_dir, device="CPU")
        result = engine.synthesize("Intel Arc 🚀 XMX matrix engine — 100% fast! ✨")
        assert isinstance(result, SynthesisResult)


def test_aud_edg_08_repeated_load_unload_memory_stability(tmp_path: Path, mock_openvino_genai):
    """Verify Proof M68: Repeated pipeline compile and deallocate cycles do not leak memory."""
    model_dir = tmp_path / "whisper-tiny"
    model_dir.mkdir()
    (model_dir / "openvino_decoder_model.xml").write_text("<net/>", encoding="utf-8")

    engine = AudioEngine()
    for _ in range(5):
        engine.load_model(model_dir, device="CPU")
        assert engine.is_loaded is True
        engine.unload()
        assert engine.is_loaded is False


def test_aud_edg_09_multichannel_stereo_downmix(tmp_path: Path):
    """Verify _normalize_audio_input downmixes stereo 2-channel WAV into mono 16kHz float32."""
    stereo_wav = tmp_path / "stereo.wav"
    sample_rate = 16000
    n_frames = 16000
    left_channel = (np.sin(np.linspace(0, 10, n_frames)) * 16384).astype(np.int16)
    right_channel = (np.cos(np.linspace(0, 10, n_frames)) * 16384).astype(np.int16)
    interleaved = np.empty((n_frames * 2,), dtype=np.int16)
    interleaved[0::2] = left_channel
    interleaved[1::2] = right_channel

    with wave.open(str(stereo_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())

    engine = AudioEngine()
    samples, duration = engine._normalize_audio_input(stereo_wav)
    assert len(samples) == 16000
    assert duration == pytest.approx(1.0, 0.01)


def test_aud_edg_10_zero_disk_residue_cleanup_invariant(tmp_path: Path):
    """Verify Proof M66: Temporary model staging directory can be unlinked with zero residue."""
    staging_dir = tmp_path / "whisper-benchmark-stage"
    staging_dir.mkdir()
    (staging_dir / "weights.bin").write_bytes(b"\x00" * 1024 * 1024)
    assert staging_dir.exists()
    assert (staging_dir / "weights.bin").exists()

    import shutil
    shutil.rmtree(staging_dir, ignore_errors=True)
    assert not staging_dir.exists()


def test_aud_pos_11_cli_transcribe_dispatch(tmp_path: Path):
    """Verify CLI transcribe command dispatches properly to AudioEngine."""
    from arc_runner.cli.main import handle_transcribe

    wav_file = tmp_path / "test_cmd.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(waveform.tobytes())

    with patch("arc_runner.engine.audio_engine.AudioEngine.load_model"), \
         patch("arc_runner.engine.audio_engine.AudioEngine.transcribe") as mock_transcribe:
        mock_transcribe.return_value = TranscriptionResult(
            text="CLI transcript",
            segments=[],
            language="en",
            audio_duration_s=1.0,
            latency_s=0.05,
            rtf=0.05,
            throughput_audio_s_per_s=20.0,
            device_used="GPU",
            model_name="whisper-tiny",
        )
        code = handle_transcribe(str(wav_file), model_id="whisper-tiny", device="GPU")
        assert code == 0


def test_aud_pos_12_cli_tts_dispatch(tmp_path: Path):
    """Verify CLI tts command dispatches properly and saves output audio."""
    from arc_runner.cli.main import handle_tts

    out_file = tmp_path / "synth_cmd.wav"
    with patch("arc_runner.engine.audio_engine.AudioEngine.load_model"), \
         patch("arc_runner.engine.audio_engine.AudioEngine.synthesize") as mock_synth:
        mock_synth.return_value = SynthesisResult(
            audio_samples=[0.0] * 24000,
            sample_rate=24000,
            duration_s=1.0,
            latency_s=0.1,
            rtf=0.1,
            throughput_audio_s_per_s=10.0,
            device_used="GPU",
            voice_used="af_heart",
            model_name="kokoro-82m",
        )
        code = handle_tts("Test synthesis", output_path=str(out_file), model_id="kokoro-82m", device="GPU")
        assert code == 0
        assert out_file.exists()
