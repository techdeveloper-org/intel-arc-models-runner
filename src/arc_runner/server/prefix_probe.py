"""Prefix cache probing with the model's real token ids."""

from typing import Any


def match_prefix_with_real_tokens(batch_pipeline: Any, engine: Any, prompt: str) -> None:
    """Probe the prefix cache with the model's real token ids, or skip when they are unavailable.

    Hash-derived pseudo-token ids are never used: they would pollute cache statistics.

    Args:
        batch_pipeline: Batch pipeline exposing prefix_cache.
        engine: Engine that will serve the request.
        prompt: Prompt text.
    """
    tokenize = getattr(engine, "tokenize", None)
    token_ids = tokenize(prompt) if callable(tokenize) else None
    if token_ids:
        batch_pipeline.prefix_cache.match_prefix(token_ids[:32])
