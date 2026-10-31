"""Context window and KV-cache geometry resolution from the environment and a model's config.json.

The context window enforced by the engines is a software budget. It comes, in priority order, from
the ARC_CONTEXT_WINDOW environment variable, from ``max_position_embeddings`` in the model's own
``config.json``, and finally from DEFAULT_CONTEXT_WINDOW. Model files are untrusted input: the file is
size-capped, must be a JSON object, and every value is type- and range-checked. Nothing here
allocates memory for the window; the KV cache stays lazy.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Union

from arc_runner.core.constants import (
    CONTEXT_WINDOW_ENV,
    DEFAULT_CONTEXT_WINDOW,
    MAX_CONTEXT_WINDOW,
    MAX_MODEL_CONFIG_BYTES,
)
from arc_runner.core.exceptions import ConfigurationError
from arc_runner.core.logging import get_logger

__all__ = [
    "CONTEXT_WINDOW_ENV",
    "MAX_CONTEXT_WINDOW",
    "MAX_MODEL_CONFIG_BYTES",
    "KVGeometry",
    "parse_context_window_env",
    "read_kv_geometry",
    "read_model_config",
    "resolve_context_window",
]

logger = get_logger(__name__)

_ENV_DIGITS = re.compile(r"[0-9]{1,%d}" % len(str(MAX_CONTEXT_WINDOW)))
_MAX_GEOMETRY_VALUE = 2**20


@dataclass(frozen=True)
class KVGeometry:
    """Shape of the key-value cache read from a model config.

    Attributes:
        num_layers: Transformer decoder layers.
        num_kv_heads: Key-value attention heads.
        head_dim: Dimension per attention head.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int


def parse_context_window_env(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Parse ARC_CONTEXT_WINDOW strictly.

    Only ASCII digits are accepted (no sign, whitespace, separators or exponent), at most as many
    digits as the cap has, and the value must be between 1 and MAX_CONTEXT_WINDOW. A leading zero
    is accepted ('08' is 8). An unset or empty variable means not configured.

    Args:
        env: Environment mapping; None reads os.environ.

    Returns:
        Optional[int]: The configured window, or None when unset or empty.

    Raises:
        ConfigurationError: If the value is present but invalid.
    """
    raw = (os.environ if env is None else env).get(CONTEXT_WINDOW_ENV)
    if raw is None or raw == "":
        return None
    value = int(raw) if _ENV_DIGITS.fullmatch(raw) and raw.isascii() else 0
    if not 1 <= value <= MAX_CONTEXT_WINDOW:
        raise ConfigurationError(
            f"{CONTEXT_WINDOW_ENV} must be a positive integer written with ASCII digits only, "
            f"between 1 and {MAX_CONTEXT_WINDOW}; the value was rejected (length {len(raw)})"
        )
    return value


def read_model_config(model_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read ``<model_dir>/config.json`` as untrusted input.

    Args:
        model_dir: Model directory.

    Returns:
        Optional[Dict[str, Any]]: The parsed JSON object, or None (with an INFO or WARNING record
        stating why) when the file is missing, not a regular file, too large, unreadable, not valid
        JSON or not an object.
    """
    path = Path(model_dir) / "config.json"
    try:
        if not path.exists():
            logger.info("Model config.json not found", extra={"model_dir": str(model_dir)})
            return None
        if not path.is_file():
            logger.warning("Model config.json is not a regular file", extra={"model_dir": str(model_dir)})
            return None
        with open(path, "rb") as handle:
            data = handle.read(MAX_MODEL_CONFIG_BYTES + 1)
    except OSError as exc:
        logger.warning("Model config.json unreadable", extra={"model_dir": str(model_dir), "reason": type(exc).__name__})
        return None
    if len(data) > MAX_MODEL_CONFIG_BYTES:
        logger.warning(
            "Model config.json is larger than the size cap and was not parsed",
            extra={"model_dir": str(model_dir), "cap_bytes": MAX_MODEL_CONFIG_BYTES},
        )
        return None
    try:
        document = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        logger.warning("Model config.json is not valid JSON", extra={"model_dir": str(model_dir), "reason": type(exc).__name__})
        return None
    if not isinstance(document, dict):
        logger.warning("Model config.json is not a JSON object", extra={"model_dir": str(model_dir)})
        return None
    return document


def _positive_int(value: Any, upper: int) -> Optional[int]:
    """Return value if it is a real int (not bool) within 1..upper, else None.

    Args:
        value: Parsed JSON value.
        upper: Inclusive upper bound.

    Returns:
        Optional[int]: The value or None.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        return None
    return value


def resolve_context_window(model_dir: Union[str, Path], env: Optional[Mapping[str, str]] = None) -> int:
    """Resolve the context window: environment, then model config, then the default.

    ``max_position_embeddings`` is the only config key read. rope scaling keys are ignored, so a
    model whose usable length depends on scaling reports only what its config states. A model value
    above MAX_CONTEXT_WINDOW is clamped to it with a WARNING. Each outcome is logged with its source.

    Args:
        model_dir: Model directory that may hold config.json.
        env: Environment mapping; None reads os.environ.

    Returns:
        int: Context window in tokens, between 1 and MAX_CONTEXT_WINDOW.

    Raises:
        ConfigurationError: If ARC_CONTEXT_WINDOW is set to an invalid value.
    """
    configured = parse_context_window_env(env)
    if configured is not None:
        logger.info("Context window from environment", extra={"context_window": configured, "source": CONTEXT_WINDOW_ENV})
        return configured
    config = read_model_config(model_dir)
    if config is not None:
        raw = config.get("max_position_embeddings")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > MAX_CONTEXT_WINDOW:
            logger.warning(
                "Model max_position_embeddings above the cap; clamped",
                extra={"model_value": raw, "context_window": MAX_CONTEXT_WINDOW},
            )
            return MAX_CONTEXT_WINDOW
        value = _positive_int(raw, MAX_CONTEXT_WINDOW)
        if value is not None:
            logger.info("Context window from model config", extra={"context_window": value, "source": "max_position_embeddings"})
            return value
        logger.info(
            "Model config has no valid max_position_embeddings (a positive integer is required); using the default",
            extra={"context_window": DEFAULT_CONTEXT_WINDOW},
        )
        return DEFAULT_CONTEXT_WINDOW
    logger.info(
        "Using the default context window because no usable model config.json was read",
        extra={"context_window": DEFAULT_CONTEXT_WINDOW},
    )
    return DEFAULT_CONTEXT_WINDOW


def read_kv_geometry(model_dir: Union[str, Path]) -> Optional[KVGeometry]:
    """Read the KV-cache shape from a model config without inventing missing fields.

    Uses num_hidden_layers, num_key_value_heads (num_attention_heads when absent, the Hugging Face
    convention for multi-head attention) and head_dim (hidden_size / num_attention_heads when
    absent and the division is exact).

    Args:
        model_dir: Model directory that may hold config.json.

    Returns:
        Optional[KVGeometry]: The geometry, or None when any needed field is missing or invalid.
    """
    config = read_model_config(model_dir)
    if config is None:
        return None
    layers = _positive_int(config.get("num_hidden_layers"), _MAX_GEOMETRY_VALUE)
    attention_heads = _positive_int(config.get("num_attention_heads"), _MAX_GEOMETRY_VALUE)
    if "num_key_value_heads" in config:
        kv_heads = _positive_int(config["num_key_value_heads"], _MAX_GEOMETRY_VALUE)
    else:
        kv_heads = attention_heads
    if "head_dim" in config:
        head_dim = _positive_int(config["head_dim"], _MAX_GEOMETRY_VALUE)
    else:
        hidden = _positive_int(config.get("hidden_size"), _MAX_GEOMETRY_VALUE * 64)
        head_dim = hidden // attention_heads if hidden and attention_heads and hidden % attention_heads == 0 else None
    if layers is None or kv_heads is None or head_dim is None:
        logger.info("KV geometry unavailable from model config", extra={"model_dir": str(model_dir)})
        return None
    return KVGeometry(num_layers=layers, num_kv_heads=kv_heads, head_dim=head_dim)
