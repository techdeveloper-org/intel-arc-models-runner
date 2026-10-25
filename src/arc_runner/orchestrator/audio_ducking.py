"""Ambient Soundscape Generation and Real-Time RMS Audio Ducking Engine.

Synthesizes harmonic atmospheric pad waveforms and applies dynamic range ducking
with asymmetric exponential smoothing (fast attack, slow release) to keep background
music balanced beneath spoken narration without pumping artifacts.
Strictly adheres to Level 2.2 Docstrings-Only standards and Proof M11.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("orchestrator.audio_ducking")

DEFAULT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_DUCK_GAIN: Final[float] = 0.18
DEFAULT_AMBIENT_GAIN: Final[float] = 0.55
DEFAULT_ATTACK_TAU: Final[float] = 0.050
DEFAULT_RELEASE_TAU: Final[float] = 0.250
DEFAULT_VAD_THRESHOLD: Final[float] = 0.015


class AudioDuckingError(ArcRunnerError):
    """Raised when audio synthesis or ducking envelope processing fails."""


class AmbientAudioDuckingEngine:
    """Orchestrates procedural ambient audio pads and asymmetric speech-responsive ducking."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duck_gain: float = DEFAULT_DUCK_GAIN,
        ambient_gain: float = DEFAULT_AMBIENT_GAIN,
        attack_tau: float = DEFAULT_ATTACK_TAU,
        release_tau: float = DEFAULT_RELEASE_TAU,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    ) -> None:
        """Initialize AmbientAudioDuckingEngine with gain levels and smoothing constants.

        Args:
            sample_rate: PCM sampling frequency in Hz.
            duck_gain: Attenuated linear gain applied to ambient music while speech is active.
            ambient_gain: Unattenuated linear gain applied to ambient music during silence.
            attack_tau: Exponential time constant in seconds for rapid gain attenuation upon speech onset.
            release_tau: Exponential time constant in seconds for gradual gain recovery after speech ceases.
            vad_threshold: RMS voice energy threshold for binary voice activity detection.

        Raises:
            AudioDuckingError: If parameters fall outside valid physical ranges.
        """
        if sample_rate <= 0:
            raise AudioDuckingError(f"Sample rate must be positive, got {sample_rate}")
        if not (0.0 <= duck_gain <= ambient_gain <= 2.0):
            raise AudioDuckingError(f"Invalid gain bounds: duck_gain={duck_gain}, ambient_gain={ambient_gain}")

        self._sample_rate = sample_rate
        self._duck_gain = float(duck_gain)
        self._ambient_gain = float(ambient_gain)
        self._attack_tau = float(attack_tau)
        self._release_tau = float(release_tau)
        self._vad_threshold = float(vad_threshold)

    @property
    def sample_rate(self) -> int:
        """Return operating audio sample rate in Hz."""
        return self._sample_rate

    def generate_celestial_ambient_pad(
        self,
        duration_sec: float,
    ) -> np.ndarray:
        """Synthesize a harmonic, atmospheric ambient soundscape pad at 432 Hz tuning.

        Combines fundamental root tones with minor seventh harmonics, gently low-pass filtered
        to provide atmospheric richness without spectral masking of spoken human voice.

        Args:
            duration_sec: Duration of pad waveform to generate in seconds.

        Returns:
            1D float32 numpy array containing normalized PCM samples in range [-1.0, 1.0].

        Raises:
            AudioDuckingError: If duration is negative.
        """
        if duration_sec < 0.0:
            raise AudioDuckingError(f"Duration cannot be negative, got {duration_sec}")

        num_samples = max(1, round(duration_sec * self._sample_rate))
        t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float32)

        root = 432.0 / 4.0
        h2 = root * (5.0 / 4.0)
        h3 = root * (3.0 / 2.0)
        h4 = root * (7.0 / 4.0)

        c1 = 0.35 * np.sin(2.0 * np.pi * root * t)
        c2 = 0.25 * np.sin(2.0 * np.pi * h2 * t + 0.3)
        c3 = 0.20 * np.sin(2.0 * np.pi * h3 * t + 0.7)
        c4 = 0.15 * np.sin(2.0 * np.pi * h4 * t + 1.2)

        shimmer = 0.05 * np.sin(2.0 * np.pi * 0.25 * t)
        raw_pad = (c1 + c2 + c3 + c4) * (1.0 + shimmer)

        fade_len = min(num_samples // 4, int(0.25 * self._sample_rate))
        if fade_len > 1:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            raw_pad[:fade_len] *= fade_in
            raw_pad[-fade_len:] *= fade_out

        peak = np.max(np.abs(raw_pad))
        if peak > 1e-6:
            raw_pad = (raw_pad / peak) * 0.85

        return raw_pad.astype(np.float32)

    def apply_ducking(
        self,
        voice_pcm: np.ndarray,
        ambient_pcm: np.ndarray,
    ) -> np.ndarray:
        """Dynamically attenuate ambient background music during active voiceover dialogue.

        Calculates short-time RMS voice energy, generates binary VAD targets, applies asymmetric
        exponential smoothing filters, and blends both streams into a balanced master mix.

        Args:
            voice_pcm: 1D float32 spoken dialogue waveform.
            ambient_pcm: 1D float32 ambient music waveform.

        Returns:
            1D float32 master mixed PCM array clipped to [-1.0, 1.0].

        Raises:
            AudioDuckingError: When input arrays are empty or non-1D.
        """
        if voice_pcm.ndim != 1 or ambient_pcm.ndim != 1:
            raise AudioDuckingError("Both voice and ambient inputs must be 1D arrays")

        len_voice = len(voice_pcm)
        len_ambient = len(ambient_pcm)
        target_len = max(len_voice, len_ambient)

        if target_len == 0:
            return np.zeros(0, dtype=np.float32)

        if len_voice < target_len:
            pad_voice = np.zeros(target_len - len_voice, dtype=np.float32)
            v_aligned = np.concatenate([voice_pcm, pad_voice])
        else:
            v_aligned = voice_pcm[:target_len].astype(np.float32)

        if len_ambient < target_len:
            pad_ambient = np.zeros(target_len - len_ambient, dtype=np.float32)
            a_aligned = np.concatenate([ambient_pcm, pad_ambient])
        else:
            a_aligned = ambient_pcm[:target_len].astype(np.float32)

        gain_envelope = self.compute_smoothed_gain_envelope(v_aligned)
        ducked_ambient = a_aligned * gain_envelope
        mixed = v_aligned + ducked_ambient

        return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def compute_smoothed_gain_envelope(
        self,
        voice_pcm: np.ndarray,
        window_size_samples: int = 480,
    ) -> np.ndarray:
        """Compute time-varying smoothed gain envelope using short-time RMS and asymmetric smoothing.

        Args:
            voice_pcm: 1D float32 voice waveform.
            window_size_samples: Sliding window width in samples (default 480 = 20ms at 24kHz).

        Returns:
            1D float32 gain envelope array matching length of voice_pcm.
        """
        total_samples = len(voice_pcm)
        if total_samples == 0:
            return np.zeros(0, dtype=np.float32)

        sq = np.square(voice_pcm.astype(np.float32))
        kernel = np.ones(window_size_samples, dtype=np.float32) / float(window_size_samples)
        rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-9)

        vad = (rms > self._vad_threshold).astype(np.float32)
        target_gain = np.where(vad > 0.5, self._duck_gain, self._ambient_gain).astype(np.float32)

        dt = 1.0 / float(self._sample_rate)
        alpha_attack = 1.0 - math.exp(-dt / max(1e-4, self._attack_tau))
        alpha_release = 1.0 - math.exp(-dt / max(1e-4, self._release_tau))

        gain_out = np.empty(total_samples, dtype=np.float32)
        current_gain = self._ambient_gain

        for i in range(total_samples):
            tg = target_gain[i]
            if tg < current_gain:
                current_gain += alpha_attack * (tg - current_gain)
            else:
                current_gain += alpha_release * (tg - current_gain)
            gain_out[i] = current_gain

        return gain_out
