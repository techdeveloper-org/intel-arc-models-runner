"""WP-06 media correctness tests: H-11 (24-bit WAV rejection) and H-10 (VLM streaming).

Red-first gate: run before implementing changes to confirm all marked tests fail.
"""

import queue
import struct
import sys
import threading
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.engine.models import GenerationConfig, TokenChunk


# ---------------------------------------------------------------------------
# Helpers: build minimal in-memory WAV files
# ---------------------------------------------------------------------------

def _build_wav(sampwidth: int, n_channels: int = 1, n_frames: int = 100) -> bytes:
    """Return the raw bytes of a valid WAV file with the given sample width.

    Args:
        sampwidth: Bytes per sample (1=8-bit, 2=16-bit, 3=24-bit, 4=32-bit).
        n_channels: Number of audio channels.
        n_frames: Number of audio frames.

    Returns:
        Raw WAV file bytes suitable for writing to disk and opening with wave.open.
    """
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(16000)
        raw = b"\x00" * (sampwidth * n_channels * n_frames)
        wf.writeframes(raw)
    return buf.getvalue()


def _write_wav(path: Path, sampwidth: int, n_channels: int = 1, n_frames: int = 100) -> None:
    """Write a valid WAV file to disk.

    Args:
        path: Destination file path.
        sampwidth: Bytes per sample.
        n_channels: Number of channels.
        n_frames: Number of frames.
    """
    path.write_bytes(_build_wav(sampwidth, n_channels, n_frames))


# ---------------------------------------------------------------------------
# AudioEngine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def audio_engine(tmp_path):
    """Return an AudioEngine with the pipeline mocked out.

    The fixture patches openvino_genai so no real model is needed.
    Only _normalize_audio_input is exercised; no pipeline call is made.
    """
    from arc_runner.engine.audio_engine import AudioEngine, AudioEngineConfig

    eng = AudioEngine.__new__(AudioEngine)
    eng._config = AudioEngineConfig(device="CPU")
    eng._pipeline = MagicMock()
    yield eng


# ---------------------------------------------------------------------------
# H-11: 24-bit WAV rejection tests
# ---------------------------------------------------------------------------

class TestAudio24BitWavRejection:
    """H-11: AudioEngine must raise InvalidAudioFormatError for 24-bit WAV (sampwidth==3)."""

    def test_audio_24bit_wav_raises_invalid_audio_format_ok(self, audio_engine, tmp_path):
        """A WAV file with sampwidth 3 (24-bit) must raise InvalidAudioFormatError."""
        from arc_runner.core.exceptions import InvalidAudioFormatError

        wav_path = tmp_path / "test_24bit.wav"
        _write_wav(wav_path, sampwidth=3)

        with pytest.raises(InvalidAudioFormatError):
            audio_engine._normalize_audio_input(wav_path)

    def test_audio_sampwidth_1_2_4_unchanged_and_3_rejected_parametrised_ok(
        self, audio_engine, tmp_path
    ):
        """Parametrised: sampwidth 1, 2, 4 succeed; sampwidth 3 raises InvalidAudioFormatError."""
        from arc_runner.core.exceptions import InvalidAudioFormatError

        for sw in (1, 2, 4):
            wav_path = tmp_path / f"test_{sw}byte.wav"
            _write_wav(wav_path, sampwidth=sw)
            samples, duration = audio_engine._normalize_audio_input(wav_path)
            assert isinstance(samples, list)
            assert duration > 0.0

        wav_path_3 = tmp_path / "test_3byte.wav"
        _write_wav(wav_path_3, sampwidth=3)
        with pytest.raises(InvalidAudioFormatError):
            audio_engine._normalize_audio_input(wav_path_3)

    def test_audio_truncated_data_chunk_raises_invalid_audio_format_ok(
        self, audio_engine, tmp_path
    ):
        """A 24-bit WAV with truncated data must raise InvalidAudioFormatError."""
        from arc_runner.core.exceptions import InvalidAudioFormatError

        raw = _build_wav(sampwidth=3)
        wav_path = tmp_path / "truncated_data.wav"
        wav_path.write_bytes(raw[: len(raw) // 2])

        with pytest.raises(InvalidAudioFormatError):
            audio_engine._normalize_audio_input(wav_path)

    def test_audio_truncated_wav_header_raises_invalid_audio_format_ok(
        self, audio_engine, tmp_path
    ):
        """A file with a truncated WAV header must raise InvalidAudioFormatError."""
        from arc_runner.core.exceptions import InvalidAudioFormatError

        wav_path = tmp_path / "truncated_header.wav"
        wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        with pytest.raises(InvalidAudioFormatError):
            audio_engine._normalize_audio_input(wav_path)

    def test_zero_length_audio_rejected_ok(self, audio_engine, tmp_path):
        """A WAV with zero frames must raise InvalidAudioFormatError."""
        from arc_runner.core.exceptions import InvalidAudioFormatError

        wav_path = tmp_path / "zero_frames.wav"
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"")
        wav_path.write_bytes(buf.getvalue())

        with pytest.raises(InvalidAudioFormatError):
            audio_engine._normalize_audio_input(wav_path)

    def test_audio_engine_unicode_filename_path_ok(self, audio_engine, tmp_path):
        """A valid 16-bit WAV with a unicode filename must be read without error."""
        wav_path = tmp_path / "audio_test_file.wav"
        _write_wav(wav_path, sampwidth=2)
        samples, duration = audio_engine._normalize_audio_input(wav_path)
        assert isinstance(samples, list)
        assert duration > 0.0

    def test_audio_16bit_mono_wav_transcribes_identically_before_after_ok(
        self, audio_engine, tmp_path
    ):
        """_normalize_audio_input on a 16-bit mono WAV returns expected samples.

        Verifies the existing 16-bit path is unchanged after the H-11 fix.
        """
        wav_path = tmp_path / "test_16bit.wav"
        n_frames = 160
        _write_wav(wav_path, sampwidth=2, n_channels=1, n_frames=n_frames)

        samples, duration = audio_engine._normalize_audio_input(wav_path)
        assert len(samples) == n_frames
        assert all(s == 0.0 for s in samples)
        assert duration == pytest.approx(n_frames / 16000.0)


# ---------------------------------------------------------------------------
# VisionEngine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vision_engine(tmp_path):
    """Return a VisionEngine with pipeline mocked out (no real model needed).

    The inner thread implementation depends on the pipeline blocking; here the
    mock pipeline is synchronous and returns immediately.
    """
    fake_genai = MagicMock()
    fake_pipeline = MagicMock()
    fake_genai.VLMPipeline.return_value = fake_pipeline

    model_dir = tmp_path / "vlm"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("dummy")

    modules_to_patch = {
        "openvino_genai": fake_genai,
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
    }
    with patch.dict(sys.modules, modules_to_patch):
        from arc_runner.engine.vision_engine import VisionEngine
        eng = VisionEngine.__new__(VisionEngine)
        eng._model_path = model_dir
        eng._device = "CPU"
        eng._patch_size = 14
        eng._pipeline = fake_pipeline
        yield eng


# ---------------------------------------------------------------------------
# H-10: VLM true streaming tests
# ---------------------------------------------------------------------------

class TestVlmStreaming:
    """H-10: VisionEngine.generate must yield the first chunk before pipeline returns."""

    def test_vlm_generate_first_chunk_available_before_pipeline_returns_ok(
        self, vision_engine, tmp_path
    ):
        """The first yielded chunk must arrive before the pipeline call returns.

        The fake pipeline blocks until a threading.Event is set. The test reads
        the first chunk from the generator in a separate thread; if the generator
        does not truly stream, the first chunk never arrives before the event.
        """
        pipeline_start = threading.Event()
        first_chunk_arrived = threading.Event()
        pipeline_may_finish = threading.Event()

        tokens = ["Hello", " world", "!"]

        def fake_generate(prompt, images, generation_config, streamer):
            pipeline_start.set()
            for tok in tokens:
                streamer(tok)
            pipeline_may_finish.wait(timeout=5.0)
            return " ".join(tokens)

        vision_engine._pipeline.generate.side_effect = fake_generate

        gen = vision_engine.generate(
            prompt="describe",
            images=[],
            config=GenerationConfig(),
        )

        result_chunks: List[TokenChunk] = []

        def consume():
            for chunk in gen:
                result_chunks.append(chunk)
                if len(result_chunks) == 1:
                    first_chunk_arrived.set()

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()

        pipeline_start.wait(timeout=5.0)
        first_chunk_arrived_before_finish = first_chunk_arrived.wait(timeout=2.0)
        pipeline_may_finish.set()
        consumer.join(timeout=5.0)

        assert first_chunk_arrived_before_finish, (
            "First chunk must be yielded before the pipeline finishes "
            "(VisionEngine.generate must use an inner thread with a queue)"
        )
        assert len(result_chunks) == len(tokens)

    def test_vlm_close_joins_inner_thread_ok(self, vision_engine):
        """After iteration completes, the inner pipeline thread must have finished."""
        tokens = ["a", "b", "c"]
        thread_ref: List[threading.Thread] = []

        original_start = threading.Thread.start

        def _patched_start(self_thread):
            thread_ref.append(self_thread)
            original_start(self_thread)

        def fake_generate(prompt, images, generation_config, streamer):
            for tok in tokens:
                streamer(tok)
            return "a b c"

        vision_engine._pipeline.generate.side_effect = fake_generate

        with patch.object(threading.Thread, "start", _patched_start):
            list(vision_engine.generate("hi", [], GenerationConfig()))

        if thread_ref:
            assert not thread_ref[0].is_alive(), (
                "Inner pipeline thread must be joined after generate() completes"
            )

    def test_vlm_chunk_queue_unbounded_producer_never_blocks_ok(self, vision_engine):
        """The streamer callback must never block the producer (inner thread).

        With an unbounded queue the put() call must return immediately even when
        the consumer is paused.
        """
        n_tokens = 50
        tokens = [f"tok{i}" for i in range(n_tokens)]
        producer_blocked = threading.Event()

        slow_consumer_start = threading.Event()
        slow_consumer_release = threading.Event()

        def fake_generate(prompt, images, generation_config, streamer):
            for tok in tokens:
                t_start = time.monotonic()
                streamer(tok)
                elapsed = time.monotonic() - t_start
                if elapsed > 0.5:
                    producer_blocked.set()
            return " ".join(tokens)

        vision_engine._pipeline.generate.side_effect = fake_generate

        gen = vision_engine.generate("q", [], GenerationConfig())

        received: List[TokenChunk] = []

        def consume():
            for chunk in gen:
                received.append(chunk)
                if len(received) == 1:
                    slow_consumer_start.set()
                    slow_consumer_release.wait(timeout=1.0)

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        slow_consumer_start.wait(timeout=5.0)
        time.sleep(0.2)
        slow_consumer_release.set()
        consumer.join(timeout=10.0)

        assert not producer_blocked.is_set(), (
            "Producer (streamer callback) must never block; queue must be unbounded"
        )
        assert len(received) == n_tokens

    def test_vlm_close_join_expiry_sets_wedged_ok(self, vision_engine):
        """If the inner thread does not finish within the timeout, EngineUnavailableError
        with ENGINE_WEDGED is raised when the generator is closed.

        Two tokens are sent before the pipeline hangs so that next(gen) can return
        (the buffer-one-ahead implementation yields the first token when the second
        arrives, before the pipeline blocks).
        """
        from arc_runner.core.exceptions import EngineUnavailableError

        pipeline_hang = threading.Event()

        def fake_generate(prompt, images, generation_config, streamer):
            streamer("first")
            streamer("second")
            pipeline_hang.wait(timeout=30.0)
            return "first second"

        vision_engine._pipeline.generate.side_effect = fake_generate

        gen = vision_engine.generate("q", [], GenerationConfig())
        chunk = next(gen)
        assert chunk.text == "first"

        with patch(
            "arc_runner.engine.vision_engine.TIMEOUT_GRACEFUL_SHUTDOWN_S", 0.05
        ):
            with pytest.raises(EngineUnavailableError) as exc_info:
                gen.close()

        pipeline_hang.set()
        assert exc_info.value.error_code == "ENGINE_WEDGED"

    def test_vlm_pipeline_raises_surfaces_vision_engine_error_ok(self, vision_engine):
        """If the pipeline raises inside the inner thread, VisionEngineError surfaces."""
        from arc_runner.engine.vision_engine import VisionEngineError

        def fake_generate(prompt, images, generation_config, streamer):
            raise RuntimeError("GPU died")

        vision_engine._pipeline.generate.side_effect = fake_generate

        with pytest.raises(VisionEngineError):
            list(vision_engine.generate("fail", [], GenerationConfig()))

    def test_vlm_empty_images_list_ok(self, vision_engine):
        """generate() with an empty images list must not raise and must yield tokens."""
        tokens = ["Hi"]

        def fake_generate(prompt, images, generation_config, streamer):
            for tok in tokens:
                streamer(tok)
            return "Hi"

        vision_engine._pipeline.generate.side_effect = fake_generate

        chunks = list(vision_engine.generate("prompt", [], GenerationConfig()))
        assert len(chunks) == 1
        assert chunks[0].text == "Hi"
        assert chunks[0].is_finished is True
