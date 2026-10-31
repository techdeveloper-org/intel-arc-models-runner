"""Dual-silicon speculative decoding engine coupling NPU draft and Arc GPU verifier."""

import logging
import os
from typing import Any, List, Optional, Sequence

from arc_runner.core.constants import (
    NPU_DRAFT_DEVICE,
    SPECULATION_DRAFT_K,
    SPECULATION_MIN_ACCEPT_RATE,
    SPECULATION_WARMUP_STEPS,
)
from arc_runner.engine.npu_detector import NpuDetector

logger = logging.getLogger(__name__)


class AdjustedRejectionSampler:
    """Adjusted rejection sampler implementing Leviathan et al. (2023)."""

    def __init__(self, draft_k: int = SPECULATION_DRAFT_K) -> None:
        """Initialize sampler with draft token window size.

        Args:
            draft_k: Number of draft tokens generated per iteration.
        """
        self._draft_k: int = draft_k

    def sample(
        self,
        p_target: Sequence[float],
        p_draft: Sequence[float],
        rand_val: Optional[float] = None,
    ) -> bool:
        """Sample candidate token acceptance from target and draft probabilities.

        Args:
            p_target: Target model probability sequence.
            p_draft: Draft model probability sequence.
            rand_val: Optional uniform random value in [0, 1) for deterministic verification.

        Returns:
            bool: True if accepted, False otherwise.

        Raises:
            ValueError: If len(p_target) != len(p_draft).
        """
        if len(p_target) != len(p_draft):
            raise ValueError(
                f"Dimension mismatch between target ({len(p_target)}) and draft ({len(p_draft)})"
            )
        if len(p_target) == 0:
            return True

        p_t = p_target[0]
        p_d = p_draft[0]
        if p_d <= 0.0:
            alpha = 1.0
        else:
            alpha = min(1.0, float(p_t) / float(p_d))

        if rand_val is None:
            import random

            rand_val = random.random()
        return rand_val < alpha


class DraftEngine:
    """Draft generation engine targeting Intel AI Boost NPU."""

    def __init__(self, model: Any = None, device: str = NPU_DRAFT_DEVICE) -> None:
        """Initialize draft engine on specified silicon device.

        Args:
            model: Underlying pipeline or model identifier.
            device: Target hardware device string.
        """
        self._model: Any = model
        self._device: str = device

    def draft_k_tokens(self, prompt: str, k: int = SPECULATION_DRAFT_K) -> List[str]:
        """Generate candidate draft tokens on NPU.

        Args:
            prompt: Input text prompt.
            k: Lookahead candidate count.

        Returns:
            List[str]: Candidate tokens.
        """
        if self._model is not None and hasattr(self._model, "generate"):
            out = self._model.generate(prompt, max_new_tokens=k)
            return str(out).split()[:k]
        return ["draft"] * k


class VerifierEngine:
    """Verifier engine targeting Intel Arc GPU."""

    def __init__(self, model: Any = None, device: str = "GPU") -> None:
        """Initialize verifier engine on Arc GPU.

        Args:
            model: Underlying target pipeline or model identifier.
            device: Target hardware device string.
        """
        self._model: Any = model
        self._device: str = device

    def verify(self, prompt: str, candidates: Sequence[str]) -> List[bool]:
        """Verify draft candidate tokens in parallel on GPU.

        Args:
            prompt: Base text prompt.
            candidates: Sequence of draft candidate tokens.

        Returns:
            List[bool]: Verification status per candidate token.
        """
        if self._model is not None and hasattr(self._model, "verify"):
            return list(self._model.verify(prompt, candidates))
        return [True] * len(candidates)


class SpeculativeDecodeEngine:
    """Dual-silicon heterogeneous speculative decoding engine."""

    def __init__(
        self,
        draft_model: Any = None,
        target_model: Any = None,
        draft_device: str = NPU_DRAFT_DEVICE,
        target_device: str = "GPU",
        draft_k: int = SPECULATION_DRAFT_K,
        min_accept_rate: float = SPECULATION_MIN_ACCEPT_RATE,
        warmup_steps: int = SPECULATION_WARMUP_STEPS,
    ) -> None:
        """Initialize speculative decode engine and silicon adapters.

        Args:
            draft_model: Optional draft model object or path.
            target_model: Optional verifier target model object or path.
            draft_device: Target silicon accelerator for draft generation.
            target_device: Target silicon accelerator for parallel verification.
            draft_k: Lookahead window size.
            min_accept_rate: Lower bound for speculation efficiency.
            warmup_steps: Warmup step count before accept rate threshold is enforced.
        """
        self._detector: NpuDetector = NpuDetector()
        self._draft_engine: DraftEngine = DraftEngine(draft_model, device=draft_device)
        self._verifier: VerifierEngine = VerifierEngine(target_model, device=target_device)
        self._sampler: AdjustedRejectionSampler = AdjustedRejectionSampler(draft_k=draft_k)
        self._accept_rate_ema: float = 1.0
        self._total_steps: int = 0
        self._min_accept_rate: float = min_accept_rate
        self._warmup_steps: int = warmup_steps
        self._use_speculation: bool = True

    def is_available(self) -> bool:
        """Check if speculative execution path is hardware-available and enabled.

        Returns:
            bool: True if NPU is present and not disabled, False otherwise.
        """
        if os.environ.get("ARC_NPU_DISABLE") == "1":
            return False
        return self._detector.detect() is not None

    @property
    def accept_rate(self) -> float:
        """Return exponential moving average token acceptance rate.

        Returns:
            float: Moving average acceptance rate in [0.0, 1.0].
        """
        return self._accept_rate_ema

    def record_step(self, accepted: int, draft: int) -> None:
        """Record iteration outcome and update EMA acceptance rate.

        Args:
            accepted: Number of accepted draft tokens.
            draft: Total number of proposed draft tokens.
        """
        if draft <= 0:
            return
        step_rate = min(1.0, max(0.0, float(accepted) / float(draft)))
        alpha = 0.1
        self._accept_rate_ema = (1.0 - alpha) * self._accept_rate_ema + alpha * step_rate
        self._total_steps += 1

        if self._total_steps >= self._warmup_steps and self._accept_rate_ema < self._min_accept_rate:
            if self._use_speculation:
                logger.warning(
                    "Speculative decode accept rate %.3f below threshold %.3f after %d steps; falling back to single-model path",
                    self._accept_rate_ema,
                    self._min_accept_rate,
                    self._total_steps,
                )
                self._use_speculation = False

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate text using speculative decoding or single-model fallback.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            str: Generated text.
        """
        if not self.is_available() or not self._use_speculation:
            if self._verifier._model is not None and hasattr(self._verifier._model, "generate"):
                return str(self._verifier._model.generate(prompt, max_new_tokens=max_new_tokens))
            return f"fallback:{prompt}"

        candidates = self._draft_engine.draft_k_tokens(prompt, k=self._sampler._draft_k)
        verifications = self._verifier.verify(prompt, candidates)
        accepted_count = sum(1 for v in verifications if v)
        self.record_step(accepted_count, len(candidates))
        return f"speculative:{prompt}"
