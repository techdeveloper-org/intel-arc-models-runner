"""Unified audio processing engine for Intel Arc GPU and Intel AI Boost NPU silicon.

Provides speech recognition (ASR via Whisper) and speech synthesis (TTS via Kokoro/StyleTTS2)
with Level Zero hardware acceleration, anti-hallucination VAD silence gating (Proof M69),
and auto-regressive repetition loop suppression (Proof M70).
Strictly adheres to Level 2.2 Docstrings-Only standards and Proofs M59 through M72.
"""

from __future__ import annotations

import gc
import math
import time
import wave
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Self

from arc_runner.core.constants import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
    TTS_OUTPUT_SAMPLE_RATE,
)
from arc_runner.core.exceptions import (
    AudioEngineError,
    AudioSynthesisError,
    AudioTranscriptionError,
    DeviceNotFoundError,
    InvalidAudioFormatError,
    ModelNotFoundError,
    VoiceNotFoundError,
)
from arc_runner.core.logging import get_logger
from arc_runner.devices.detector import DeviceDetector

logger = get_logger("arc_runner.engine.audio")


class AudioEngineConfig(BaseModel):
    """Runtime execution parameters and architectural limits for AudioEngine."""

    device: str = Field(default="GPU", description="Target accelerator: 'GPU', 'NPU', or 'CPU'.")
    sample_rate: int = Field(default=DEFAULT_AUDIO_SAMPLE_RATE, description="Expected ASR input sample rate in Hz.")
    tts_sample_rate: int = Field(default=TTS_OUTPUT_SAMPLE_RATE, description="Output sample rate for TTS in Hz.")
    default_tts_voice: str = Field(default=DEFAULT_TTS_VOICE, description="Default voice style slug for Kokoro/TTS.")
    default_tts_speed: float = Field(default=1.0, ge=0.1, le=5.0, description="Speech tempo speed multiplier.")
    language: str | None = Field(default=None, description="ISO language code or None for auto-detect.")
    return_timestamps: bool = Field(default=False, description="Whether to compute timestamped audio segments.")
    word_timestamps: bool = Field(default=False, description="Whether to compute word-level timestamps.")
    vad_energy_threshold: float = Field(default=1e-4, description="RMS energy threshold for silence gating.")
    vad_entropy_threshold: float = Field(default=0.05, description="Spectral entropy threshold for noise gating.")
    max_compression_ratio: float = Field(default=2.40, description="LZ77 compression ratio bound for loop detection.")


class TranscriptionSegment(BaseModel):
    """Temporal transcript segment with bounded time boundaries."""

    start_ts: float = Field(description="Segment start time offset in seconds.")
    end_ts: float = Field(description="Segment end time offset in seconds.")
    text: str = Field(description="Decoded speech text within this temporal slice.")


class TranscriptionResult(BaseModel):
    """Encapsulates decoded speech recognition transcript and physical telemetry."""

    text: str = Field(description="Complete aggregated decoded transcription.")
    segments: list[TranscriptionSegment] = Field(default_factory=list, description="Temporal phrase segments.")
    latency_s: float = Field(description="Wall-clock inference elapsed duration in seconds.")
    audio_duration_s: float = Field(description="Input audio duration in seconds.")
    rtf: float = Field(description="Real-Time Factor: latency_s / audio_duration_s.")
    throughput_audio_s_per_s: float = Field(description="Throughput in processed audio-seconds per wall-second.")
    device_used: str = Field(description="Physical silicon device executed on ('GPU', 'NPU', 'CPU').")
    model_name: str = Field(description="Identifier slug of the loaded ASR model.")
    compression_ratio: float = Field(default=1.0, description="Computed text Kolmogorov LZ77 compression ratio.")


class SynthesisResult(BaseModel):
    """Encapsulates generated text-to-speech audio samples and physical telemetry."""

    audio_samples: list[float] = Field(description="Normalized float32 PCM audio waveform samples [-1.0, 1.0].")
    sample_rate: int = Field(description="Sampling rate of the synthesized waveform in Hz.")
    duration_s: float = Field(description="Synthesized audio duration in seconds.")
    latency_s: float = Field(description="Wall-clock synthesis elapsed duration in seconds.")
    rtf: float = Field(description="Real-Time Factor: latency_s / duration_s.")
    throughput_audio_s_per_s: float = Field(description="Synthesized audio-seconds per wall-second.")
    device_used: str = Field(description="Physical silicon device executed on ('GPU', 'NPU', 'CPU').")
    voice_used: str = Field(description="Voice embedding style applied.")
    model_name: str = Field(description="Identifier slug of the loaded TTS model.")


class AudioEngine:
    """Unified OpenVINO GenAI execution engine for STT and TTS on Intel Silicon.

    Compiles Whisper and Kokoro OpenVINO IR graphs to Level Zero immediate command
    lists on Intel Arc GPUs, Intel AI Boost NPUs, or Intel CPUs.
    """

    def __init__(self, config: AudioEngineConfig | None = None) -> None:
        """Initialize the audio engine with runtime configuration.

        Args:
            config: Optional audio engine configuration. Defaults to AudioEngineConfig().
        """
        self._config = config or AudioEngineConfig()
        self._model_path: Path | None = None
        self._device: str = self._config.device.upper()
        self._task_type: str | None = None
        self._whisper_pipe: Any = None
        self._tts_pipe: Any = None
        self._speaker_cache: dict[str, Any] = {}
        self._topology = DeviceDetector().detect_topology()

    @property
    def is_loaded(self) -> bool:
        """Indicate whether an STT or TTS model pipeline is active in memory.

        Returns:
            bool: True if an underlying OpenVINO pipeline is active.
        """
        return self._whisper_pipe is not None or self._tts_pipe is not None

    @property
    def model_path(self) -> Path | None:
        """Return the filesystem path to the currently loaded model directory.

        Returns:
            Optional[Path]: Filesystem directory path or None.
        """
        return self._model_path

    @property
    def device(self) -> str:
        """Return the silicon device string currently active for inference.

        Returns:
            str: Device string ('GPU', 'NPU', 'CPU').
        """
        return self._device

    @property
    def task_type(self) -> str | None:
        """Return active model task category ('stt' or 'tts').

        Returns:
            Optional[str]: Active task type string or None.
        """
        return self._task_type

    def load_model(
        self,
        model_path: str | Path,
        device: str | None = None,
        task_type: str | None = None,
    ) -> None:
        """Compile and load an OpenVINO IR audio model onto designated Intel silicon.

        Args:
            model_path: Directory path containing the OpenVINO IR model files.
            device: Optional target silicon ('GPU', 'NPU', 'CPU'). Defaults to config.
            task_type: Optional task type ('stt' or 'tts'). Auto-detected if omitted.

        Raises:
            ModelNotFoundError: If the directory does not exist or lacks model files.
            DeviceNotFoundError: If the requested hardware device is unavailable.
            AudioEngineError: If compilation or Level Zero pipeline binding fails.
        """
        path = Path(model_path).resolve()
        if not path.exists() or not path.is_dir():
            raise ModelNotFoundError(str(model_path), str(path))

        target_dev = (device or self._config.device).upper()
        if target_dev == "GPU" and not self._topology.has_intel_arc:
            logger.warning("Intel Arc GPU not detected. Falling back to CPU.")
            target_dev = "CPU"
        elif target_dev == "NPU" and not self._topology.has_intel_npu:
            logger.warning("Intel NPU not detected. Falling back to GPU if present or CPU.")
            target_dev = "GPU" if self._topology.has_intel_arc else "CPU"
        elif target_dev not in ["GPU", "NPU", "CPU"]:
            raise DeviceNotFoundError(target_dev, ["GPU", "NPU", "CPU"])

        self.unload()

        detected_task = task_type
        if detected_task is None:
            if (path / "openvino_decoder_model.xml").exists() or (path / "openvino_encoder_model.xml").exists() or "whisper" in path.name.lower():
                detected_task = "stt"
            elif (path / "openvino_model.xml").exists() or "kokoro" in path.name.lower() or "speech" in path.name.lower():
                detected_task = "tts"
            else:
                detected_task = "stt"

        try:
            import openvino_genai as ov_genai

            t0 = time.perf_counter()
            if detected_task == "stt":
                self._whisper_pipe = ov_genai.WhisperPipeline(str(path), target_dev)
            else:
                self._tts_pipe = ov_genai.Text2SpeechPipeline(str(path), target_dev)

            elapsed = time.perf_counter() - t0
            self._model_path = path
            self._device = target_dev
            self._task_type = detected_task
            logger.info("Successfully loaded %s model on %s in %.2fs", detected_task.upper(), target_dev, elapsed)
        except Exception as exc:
            raise AudioEngineError(f"Failed to load OpenVINO audio pipeline from '{path}' on {target_dev}: {exc}") from exc

    def transcribe(
        self,
        audio: np.ndarray | list[float] | str | Path,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe speech audio into text using the active Whisper pipeline.

        Applies Voice Activity Detection (VAD) RMS energy and spectral entropy silence
        gating (Proof M69) to eliminate phantom hallucinations, and monitors output
        Kolmogorov compression ratio (Proof M70) to detect auto-regressive repetition loops.

        Args:
            audio: 16kHz audio samples as float array, float list, or path to WAV file.
            language: Optional ISO language code (e.g. 'en', 'hi') or None for auto.
            task: Operational mode: 'transcribe' or 'translate'.
            return_timestamps: Whether to extract timestamped phrase chunks.

        Returns:
            TranscriptionResult containing text, temporal segments, and RTF telemetry.

        Raises:
            AudioEngineError: If the engine is not loaded or loaded for TTS.
            InvalidAudioFormatError: If audio input cannot be normalized to 16kHz floats.
            AudioTranscriptionError: If OpenVINO inference pipeline execution fails.
        """
        if self._whisper_pipe is None:
            raise AudioEngineError("No speech recognition pipeline loaded. Call load_model() first.")

        samples_list, duration_s = self._normalize_audio_input(audio)

        samples_arr = np.array(samples_list, dtype=np.float32)
        if self._is_silence_or_noise(samples_arr):
            logger.debug("Acoustic silence or noise detected via Proof M69. Gating decoder to emit empty string.")
            return TranscriptionResult(
                text="",
                segments=[],
                latency_s=0.001,
                audio_duration_s=duration_s,
                rtf=round(0.001 / max(1e-6, duration_s), 4),
                throughput_audio_s_per_s=round(duration_s / 0.001, 2),
                device_used=self._device,
                model_name=self._model_path.name if self._model_path else "whisper",
                compression_ratio=1.0,
            )

        try:
            import openvino_genai as ov_genai

            if hasattr(self._whisper_pipe, "get_generation_config"):
                gen_config = self._whisper_pipe.get_generation_config()
            else:
                gen_config = ov_genai.WhisperGenerationConfig()

            if language:
                gen_config.language = f"<|{language.strip('<|>')}|>"
            gen_config.task = "transcribe" if task.lower() == "transcribe" else "translate"
            gen_config.return_timestamps = return_timestamps or self._config.return_timestamps

            t0 = time.perf_counter()
            asr_output = self._whisper_pipe.generate(samples_list, gen_config)
            elapsed_s = time.perf_counter() - t0

            combined_text = ""
            segments_out: list[TranscriptionSegment] = []

            if hasattr(asr_output, "texts") and asr_output.texts:
                combined_text = " ".join(t.strip() for t in asr_output.texts if t.strip())
            elif hasattr(asr_output, "chunks") and asr_output.chunks:
                for chunk in asr_output.chunks:
                    chunk_text = chunk.text.strip()
                    segments_out.append(TranscriptionSegment(
                        start_ts=round(chunk.start_ts, 3),
                        end_ts=round(chunk.end_ts, 3),
                        text=chunk_text,
                    ))
                combined_text = " ".join(s.text for s in segments_out if s.text)
            elif isinstance(asr_output, str):
                combined_text = asr_output.strip()

            comp_ratio = self._detect_repetition_loop(combined_text)
            if comp_ratio > self._config.max_compression_ratio:
                logger.warning(
                    "Repetition loop detected via Proof M70 (ratio=%.2f > %.2f). Truncating degenerate tokens.",
                    comp_ratio,
                    self._config.max_compression_ratio,
                )
                words = combined_text.split()
                seen_ngrams: set[str] = set()
                deduped_words: list[str] = []
                for w in words:
                    if w in seen_ngrams and len(deduped_words) > 10:
                        break
                    seen_ngrams.add(w)
                    deduped_words.append(w)
                combined_text = " ".join(deduped_words)

            rtf = round(elapsed_s / max(1e-6, duration_s), 6)
            throughput = round(duration_s / max(1e-6, elapsed_s), 2)

            return TranscriptionResult(
                text=combined_text,
                segments=segments_out,
                latency_s=round(elapsed_s, 4),
                audio_duration_s=round(duration_s, 3),
                rtf=rtf,
                throughput_audio_s_per_s=throughput,
                device_used=self._device,
                model_name=self._model_path.name if self._model_path else "whisper",
                compression_ratio=round(comp_ratio, 2),
            )
        except Exception as exc:
            raise AudioTranscriptionError(str(exc)) from exc

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize natural speech audio waveform from text using the active TTS pipeline.

        Args:
            text: Input text prompt to phonemize and synthesize into speech.
            voice: Speaker voice embedding slug (e.g. 'af_heart', 'am_adam').
            speed: Playback tempo speed multiplier between 0.1 and 5.0.

        Returns:
            SynthesisResult containing audio waveform, sample rate, and RTF telemetry.

        Raises:
            AudioEngineError: If the engine is not loaded or loaded for STT.
            ValueError: If input text is empty or speed is out of bounds.
            VoiceNotFoundError: If requested speaker voice embedding cannot be resolved.
            AudioSynthesisError: If OpenVINO synthesis pipeline execution fails.
        """
        if self._tts_pipe is None:
            raise AudioEngineError("No speech synthesis pipeline loaded. Call load_model() first.")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Input text prompt cannot be empty or whitespace.")

        if not (0.1 <= speed <= 5.0):
            raise ValueError(f"Speech speed multiplier {speed} out of valid range [0.1, 5.0].")

        target_voice = voice or self._config.default_tts_voice
        speaker_tensor = self._load_speaker_embedding(target_voice)

        try:
            import openvino_genai as ov_genai

            if hasattr(self._tts_pipe, "get_generation_config"):
                speech_config = self._tts_pipe.get_generation_config()
            else:
                speech_config = ov_genai.SpeechGenerationConfig()
            speech_config.speed = float(speed)

            t0 = time.perf_counter()
            tts_output = self._tts_pipe.generate(
                [clean_text],
                speaker_embedding=speaker_tensor,
                config=speech_config,
            )
            elapsed_s = time.perf_counter() - t0

            out_rate = getattr(tts_output, "output_sample_rate", self._config.tts_sample_rate)
            raw_speeches = getattr(tts_output, "speeches", [])

            if raw_speeches:
                first_speech = raw_speeches[0]
                if isinstance(first_speech, np.ndarray):
                    samples_out = first_speech.flatten().tolist()
                elif hasattr(first_speech, "data"):
                    samples_out = np.asarray(first_speech.data).flatten().tolist()
                elif isinstance(first_speech, Sequence):
                    samples_out = list(first_speech)
                else:
                    samples_out = [float(x) for x in first_speech]
            else:
                samples_out = []

            duration_s = round(len(samples_out) / max(1, out_rate), 3)
            rtf = round(elapsed_s / max(1e-6, duration_s), 6)
            throughput = round(duration_s / max(1e-6, elapsed_s), 2)

            return SynthesisResult(
                audio_samples=samples_out,
                sample_rate=out_rate,
                duration_s=duration_s,
                latency_s=round(elapsed_s, 4),
                rtf=rtf,
                throughput_audio_s_per_s=throughput,
                device_used=self._device,
                voice_used=target_voice,
                model_name=self._model_path.name if self._model_path else "tts",
            )
        except Exception as exc:
            raise AudioSynthesisError(str(exc)) from exc

    def _normalize_audio_input(
        self,
        audio: np.ndarray | list[float] | str | Path,
    ) -> tuple[list[float], float]:
        """Convert heterogeneous audio inputs into normalized 16kHz float32 list and duration.

        Args:
            audio: Heterogeneous audio input (numpy array, float list, or path to WAV file).

        Returns:
            Tuple of (normalized_samples_list, audio_duration_seconds).

        Raises:
            InvalidAudioFormatError: If the input audio cannot be parsed or is corrupted.
        """
        if isinstance(audio, (str, Path)):
            wav_path = Path(audio).resolve()
            if not wav_path.exists():
                raise InvalidAudioFormatError(f"Audio file '{wav_path}' does not exist.")
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_bytes = wf.readframes(n_frames)

                    if sampwidth == 3:
                        raise InvalidAudioFormatError(
                            f"24-bit WAV (3 bytes per sample) is not supported at"
                            f" '{wav_path}'; re-encode to 16-bit or 32-bit PCM."
                        )
                    if sampwidth == 2:
                        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                    elif sampwidth == 1:
                        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        raise InvalidAudioFormatError(
                            f"Unsupported WAV sample width {sampwidth} bytes per sample at '{wav_path}'."
                        )

                    if samples.size == 0:
                        raise InvalidAudioFormatError(
                            f"WAV file '{wav_path}' contains zero audio frames."
                        )

                    if n_channels > 1:
                        samples = samples.reshape(-1, n_channels).mean(axis=1)

                    if framerate != self._config.sample_rate:
                        target_len = int(len(samples) * self._config.sample_rate / framerate)
                        indices = np.linspace(0, len(samples) - 1, target_len)
                        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

                    samples_list = samples.tolist()
                    duration_s = len(samples_list) / self._config.sample_rate
                    return samples_list, duration_s
            except Exception as exc:
                raise InvalidAudioFormatError(f"Failed to read WAV audio file '{wav_path}': {exc}") from exc

        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                raise InvalidAudioFormatError("Input audio numpy array is empty.")
            if np.isnan(audio).any() or np.isinf(audio).any():
                raise InvalidAudioFormatError("Audio numpy array contains invalid NaN or Inf values.")
            flat_audio = audio.flatten().astype(np.float32)
            max_val = np.max(np.abs(flat_audio))
            if max_val > 1.0:
                flat_audio = flat_audio / max_val
            samples_list = flat_audio.tolist()
            duration_s = len(samples_list) / self._config.sample_rate
            return samples_list, duration_s

        if isinstance(audio, Sequence):
            if len(audio) == 0:
                raise InvalidAudioFormatError("Input audio sample sequence is empty.")
            samples_arr = np.array(audio, dtype=np.float32)
            if np.isnan(samples_arr).any() or np.isinf(samples_arr).any():
                raise InvalidAudioFormatError("Audio sample sequence contains NaN or Inf values.")
            max_val = np.max(np.abs(samples_arr))
            if max_val > 1.0:
                samples_arr = samples_arr / max_val
            samples_list = samples_arr.tolist()
            duration_s = len(samples_list) / self._config.sample_rate
            return samples_list, duration_s

        raise InvalidAudioFormatError(f"Unsupported audio type: {type(audio)}")

    def _is_silence_or_noise(self, samples: np.ndarray) -> bool:
        """Determine whether audio is pure acoustic silence or low-energy noise (Proof M69).

        Calculates Root-Mean-Square (RMS) energy and spectral entropy of the audio frame.

        Args:
            samples: Float32 normalized audio samples array.

        Returns:
            bool: True if segment is below VAD threshold and should bypass decoding.
        """
        if len(samples) == 0:
            return True

        rms_energy = float(np.sqrt(np.mean(samples ** 2)))
        if rms_energy < self._config.vad_energy_threshold:
            return True

        n_fft = min(512, len(samples))
        if n_fft < 32:
            return False

        windowed = samples[:n_fft] * np.hanning(n_fft)
        fft_vals = np.abs(np.fft.rfft(windowed)) ** 2
        sum_fft = np.sum(fft_vals)
        if sum_fft < 1e-12:
            return True

        psd_norm = fft_vals / sum_fft
        psd_norm = psd_norm[psd_norm > 1e-12]
        spectral_entropy = -float(np.sum(psd_norm * np.log2(psd_norm))) / math.log2(len(psd_norm))

        return bool(spectral_entropy < self._config.vad_entropy_threshold and rms_energy < 0.005)

    def _detect_repetition_loop(self, text: str) -> float:
        """Calculate Ziv-Lempel compression ratio of transcript to detect loops (Proof M70).

        Args:
            text: Decoded output transcription text string.

        Returns:
            float: Compression ratio r = len(text_bytes) / len(compressed_bytes).
        """
        text_bytes = text.strip().encode("utf-8")
        if len(text_bytes) < 30:
            return 1.0
        compressed = zlib.compress(text_bytes, level=9)
        return float(len(text_bytes)) / max(1.0, float(len(compressed)))

    def _load_speaker_embedding(self, voice_name: str) -> Any:
        """Load or retrieve cached OpenVINO tensor speaker embedding for Kokoro/TTS.

        Args:
            voice_name: Voice style slug (e.g. 'af_heart', 'am_adam').

        Returns:
            OpenVINO Tensor containing speaker style vector.

        Raises:
            VoiceNotFoundError: If the requested voice file cannot be found in the model directory.
        """
        if voice_name in self._speaker_cache:
            return self._speaker_cache[voice_name]

        if self._model_path is None:
            raise AudioEngineError("Model path is not initialized.")

        voice_candidates = [
            self._model_path / "voices" / f"{voice_name}.bin",
            self._model_path / f"{voice_name}.bin",
            self._model_path / "voices" / f"{self._config.default_tts_voice}.bin",
        ]

        found_path: Path | None = None
        for cand in voice_candidates:
            if cand.exists():
                found_path = cand
                break

        if found_path is None:
            available_voices = [p.stem for p in (self._model_path / "voices").glob("*.bin")] if (self._model_path / "voices").exists() else []
            raise VoiceNotFoundError(voice_name, available_voices)

        try:
            import openvino as ov

            arr = np.fromfile(str(found_path), dtype=np.float32)
            if arr.size == 510 * 256:
                arr = arr.reshape((510, 1, 256))
            elif arr.size == 512:
                arr = arr.reshape((1, 512))
            ov_tensor = ov.Tensor(arr)
            self._speaker_cache[voice_name] = ov_tensor
            return ov_tensor
        except Exception as exc:
            raise VoiceNotFoundError(f"Failed to load speaker embedding tensor from '{found_path}': {exc}") from exc

    @staticmethod
    def save_audio_wav(
        samples: Sequence[float],
        output_path: str | Path,
        sample_rate: int = TTS_OUTPUT_SAMPLE_RATE,
    ) -> Path:
        """Persist normalized float32 PCM audio samples to a 16-bit mono WAV file.

        Args:
            samples: Normalized floating point PCM samples [-1.0, 1.0].
            output_path: Destination filesystem path for output WAV.
            sample_rate: Sampling frequency in Hertz (default 24000).

        Returns:
            Resolved Path of the written audio file.
        """
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(samples, dtype=np.float32)
        clamped = np.clip(arr, -1.0, 1.0)
        pcm16 = (clamped * 32767.0).astype(np.int16)

        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        return dest

    def unload(self) -> None:
        """Unload active OpenVINO pipelines, clear caches, and trigger garbage collection."""
        self._whisper_pipe = None
        self._tts_pipe = None
        self._speaker_cache.clear()
        self._model_path = None
        self._task_type = None
        gc.collect()

    def __enter__(self) -> Self:
        """Enter runtime context manager.

        Returns:
            Self: Current engine instance.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit runtime context manager and release compiled accelerator buffers.

        Args:
            exc_type: Optional exception type.
            exc_val: Optional exception value.
            exc_tb: Optional exception traceback.
        """
        self.unload()
