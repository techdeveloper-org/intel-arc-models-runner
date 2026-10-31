"""System-wide default constants and execution limits for Intel Arc Models Runner."""

import os
import re
from pathlib import Path

DEFAULT_CONTEXT_WINDOW: int = 4096
CONTEXT_WINDOW_ENV: str = "ARC_CONTEXT_WINDOW"
MAX_CONTEXT_WINDOW: int = 2**21
MAX_MODEL_CONFIG_BYTES: int = 1024 * 1024
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 40
DEFAULT_REPETITION_PENALTY: float = 1.1
STREAM_BUFFER_SIZE: int = 64
MAX_PROMPT_HEADROOM_RATIO: float = 0.85
PAGE_BLOCK_SIZE: int = 16
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 11434
MIN_DECODE_RATE_TOKENS_PER_SEC: float = 8.0
MIN_PREFILL_RATE_TOKENS_PER_SEC: float = 150.0
WATCHDOG_OVERHEAD_SECONDS: float = 3.0
GGUF_MAGIC: bytes = b"GGUF"
GGUF_DEFAULT_ALIGNMENT: int = 32
GGUF_SUPPORTED_VERSIONS: tuple[int, ...] = (2, 3)
DEFAULT_CACHE_DIR: Path = Path.home() / ".arc-runner" / "cache"
DEFAULT_OLLAMA_DIR: Path = Path.home() / ".ollama"
DEFAULT_AUDIO_SAMPLE_RATE: int = 16000
TTS_OUTPUT_SAMPLE_RATE: int = 24000
MAX_AUDIO_DURATION_SECONDS: float = 300.0
DEFAULT_TTS_VOICE: str = "af_heart"
LEVEL_ZERO_SUBMISSION_LATENCY_US: float = 2.1
TIMEOUT_GRACEFUL_SHUTDOWN_S: float = 30.0
NPU_MAX_PROMPT_LEN: int = 1024
NPU_MIN_RESPONSE_LEN: int = 128
CANCEL_JOIN_TIMEOUT_S: float = 30.0
SHUTDOWN_JOIN_S: float = CANCEL_JOIN_TIMEOUT_S
GRACEFUL_SHUTDOWN_S: float = 8.0
GRACEFUL_SHUTDOWN_ENV: str = "ARC_GRACEFUL_SHUTDOWN_S"
_GRACEFUL_SHUTDOWN_PATTERN = re.compile(r"[1-9][0-9]{0,8}", re.ASCII)


def graceful_shutdown_s() -> float:
    """Read the uvicorn graceful shutdown timeout from ARC_GRACEFUL_SHUTDOWN_S.

    The default of 8 seconds stays below Docker's 10 second stop grace period. A set value
    must be 1 to 9 ASCII digits with no leading zero, so it is at least 1.

    Returns:
        float: Timeout in seconds.

    Raises:
        ArcRunnerError: When the variable is set to anything other than 1-9 ASCII digits >= 1.
    """
    value = os.environ.get(GRACEFUL_SHUTDOWN_ENV, "").strip()
    if not value:
        return GRACEFUL_SHUTDOWN_S
    if not _GRACEFUL_SHUTDOWN_PATTERN.fullmatch(value):
        from arc_runner.core.exceptions import ArcRunnerError

        raise ArcRunnerError(f"{GRACEFUL_SHUTDOWN_ENV} must be 1-9 ASCII digits >= 1, got {value!r}")
    return float(value)
