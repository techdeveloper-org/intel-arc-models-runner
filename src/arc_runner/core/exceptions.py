"""Structured domain exception hierarchy for Intel Arc Models Runner.

Provides typed exceptions for model loading, device targeting, memory exhaustion,
and inference timeout conditions in accordance with Rule 19 exception standards.
"""

from typing import List, Optional


class ArcRunnerError(Exception):
    """Root base exception for all runner and engine errors."""

    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", status_code: int = 500) -> None:
        """Initialize the base runner exception.

        Args:
            message: Human-readable error description.
            error_code: Unique machine-readable error token.
            status_code: Corresponding HTTP status code.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class ModelLoadError(ArcRunnerError):
    """Raised when an OpenVINO IR model fails to compile or load on target silicon."""

    def __init__(self, message: str, model_path: str = "", device: str = "") -> None:
        """Initialize the model load failure exception.

        Args:
            message: Failure reason details.
            model_path: File system path to the model directory.
            device: Accelerator device identifier string.
        """
        super().__init__(f"Failed to load model from '{model_path}' on {device}: {message}", "MODEL_LOAD_FAILED", 500)
        self.model_path = model_path
        self.device = device


class ModelNotFoundError(ArcRunnerError):
    """Raised when a specified model directory or file is not found."""

    def __init__(self, model_id: str, search_path: str = "") -> None:
        """Initialize the model not found exception.

        Args:
            model_id: Model name or identifier requested.
            search_path: File system location searched.
        """
        super().__init__(f"Model '{model_id}' was not found at '{search_path}'", "MODEL_NOT_FOUND", 404)
        self.model_id = model_id
        self.search_path = search_path


class DeviceNotFoundError(ArcRunnerError):
    """Raised when a requested accelerator device is not detected."""

    def __init__(self, device: str, available_devices: Optional[List[str]] = None) -> None:
        """Initialize the device not found exception.

        Args:
            device: Requested device string.
            available_devices: List of actually available hardware devices.
        """
        avail_str = ", ".join(available_devices) if available_devices else "None"
        super().__init__(f"Requested device '{device}' is not available. Detected silicon: [{avail_str}]", "DEVICE_NOT_FOUND", 400)
        self.device = device
        self.available_devices = available_devices or []


class ContextWindowExceededError(ArcRunnerError):
    """Raised when prompt and generation tokens exceed model context budget."""

    def __init__(self, prompt_tokens: int, max_context: int) -> None:
        """Initialize the context window exceeded exception.

        Args:
            prompt_tokens: Number of tokens in input prompt.
            max_context: Maximum allowed context window tokens.
        """
        super().__init__(f"Prompt tokens ({prompt_tokens}) exceed maximum context window ({max_context})", "CONTEXT_WINDOW_EXCEEDED", 400)
        self.prompt_tokens = prompt_tokens
        self.max_context = max_context


class VRAMExhaustedError(ArcRunnerError):
    """Raised when GPU USM or discrete VRAM memory allocation fails."""

    def __init__(self, required_mb: float, available_mb: float) -> None:
        """Initialize the memory allocation failure exception.

        Args:
            required_mb: Memory required in megabytes.
            available_mb: Memory currently available in megabytes.
        """
        super().__init__(f"Insufficient memory: requested {required_mb:.1f} MB, available {available_mb:.1f} MB", "VRAM_EXHAUSTED", 507)
        self.required_mb = required_mb
        self.available_mb = available_mb


class InferenceTimeoutError(ArcRunnerError):
    """Raised when inference exceeds maximum watchdog time threshold."""

    def __init__(self, elapsed_seconds: float, timeout_seconds: float) -> None:
        """Initialize the inference watchdog timeout exception.

        Args:
            elapsed_seconds: Elapsed wall-clock time in seconds.
            timeout_seconds: Maximum allowed execution timeout in seconds.
        """
        super().__init__(f"Inference timed out after {elapsed_seconds:.2f}s (threshold: {timeout_seconds:.2f}s)", "INFERENCE_TIMEOUT", 504)
        self.elapsed_seconds = elapsed_seconds
        self.timeout_seconds = timeout_seconds
