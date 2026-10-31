"""Context-window budgeting shared by the engines that wrap an OpenVINO GenAI pipeline."""

from typing import Any, Optional

from arc_runner.core.exceptions import ContextWindowExceededError
from arc_runner.core.logging import get_logger
from arc_runner.engine.kv_cache import KVCacheManager

logger = get_logger(__name__)


class ContextBudgetMixin:
    """Prompt token counting and context-window enforcement for pipeline-backed engines.

    Subclasses provide ``_pipeline`` (an object exposing ``get_tokenizer()``) and ``kv_manager``.
    """

    _pipeline: Optional[Any]
    kv_manager: KVCacheManager

    def _count_prompt_tokens(self, prompt: str) -> Optional[int]:
        """Count prompt tokens with the loaded pipeline's own tokenizer.

        Args:
            prompt: Prompt text.

        Returns:
            Optional[int]: Token count, or None if the tokenizer cannot be queried.
        """
        try:
            encoded = self._pipeline.get_tokenizer().encode(prompt)
            return int(encoded.input_ids.shape[-1])
        except Exception as exc:
            logger.warning("Prompt token count unavailable; context window not enforced", extra={"reason": str(exc)})
            return None

    def _enforce_context_window(self, prompt_tokens: int, max_new_tokens: int) -> None:
        """Reject requests whose prompt, or prompt plus requested completion, exceeds the context window.

        Args:
            prompt_tokens: Token count of the prompt.
            max_new_tokens: Requested maximum completion tokens.

        Raises:
            ContextWindowExceededError: If prompt_tokens or prompt_tokens + max_new_tokens is too large.
        """
        remaining = self.kv_manager.check_headroom(prompt_tokens)
        if max_new_tokens > remaining:
            raise ContextWindowExceededError(prompt_tokens + max_new_tokens, self.kv_manager.max_context_length)

    def _check_request_fits_context(self, prompt: str, max_new_tokens: int) -> Optional[int]:
        """Count prompt tokens and enforce the context window before any generation work starts.

        Args:
            prompt: Prompt text.
            max_new_tokens: Requested maximum completion tokens.

        Returns:
            Optional[int]: Prompt token count, or None if it could not be measured.
        """
        prompt_tokens = self._count_prompt_tokens(prompt)
        if prompt_tokens is not None:
            self._enforce_context_window(prompt_tokens, max_new_tokens)
        return prompt_tokens
