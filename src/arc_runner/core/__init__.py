"""Core utilities, constants, and domain exceptions for Intel Arc Models Runner."""

from typing import List
from arc_runner.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    STREAM_BUFFER_SIZE,
    PAGE_BLOCK_SIZE,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
from arc_runner.core.exceptions import (
    ArcRunnerError,
    ModelLoadError,
    ModelNotFoundError,
    DeviceNotFoundError,
    ContextWindowExceededError,
    VRAMExhaustedError,
    InferenceTimeoutError,
)
from arc_runner.core.logging import (
    clear_request_context,
    get_logger,
    get_request_id,
    set_request_context,
    setup_logging,
)

__all__: List[str] = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_TOP_K",
    "DEFAULT_REPETITION_PENALTY",
    "STREAM_BUFFER_SIZE",
    "PAGE_BLOCK_SIZE",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ArcRunnerError",
    "ModelLoadError",
    "ModelNotFoundError",
    "DeviceNotFoundError",
    "ContextWindowExceededError",
    "VRAMExhaustedError",
    "InferenceTimeoutError",
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "get_request_id",
]
