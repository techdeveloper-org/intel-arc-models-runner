"""Structured domain exception hierarchy for Intel Arc Models Runner.

Provides typed exceptions for model loading, device targeting, memory exhaustion,
and inference timeout conditions in accordance with Rule 19 exception standards.
"""



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


class NativeLayerError(ArcRunnerError):
    """Raised when the native acceleration layer rejects a request or reports a failure."""

    def __init__(self, message: str, native_status: int = 0, os_error: int = 0) -> None:
        """Initialize the native layer failure exception.

        Args:
            message: Human-readable error description.
            native_status: ArcStatus code reported by the native library, if any.
            os_error: Operating-system error code recorded by the native library, if any.
        """
        super().__init__(message, "NATIVE_LAYER_ERROR", 500)
        self.native_status = native_status
        self.os_error = os_error


class NativeBoundsError(NativeLayerError):
    """Raised when tensor extents, counts, or integers fall outside the supported native bounds."""


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
            search_path: File system location searched; kept for logs and never put in the message.
        """
        super().__init__(f"Model '{model_id}' was not found", "MODEL_NOT_FOUND", 404)
        self.model_id = model_id
        self.search_path = search_path


class DeviceNotFoundError(ArcRunnerError):
    """Raised when a requested accelerator device is not detected."""

    def __init__(self, device: str, available_devices: list[str] | None = None) -> None:
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


class InvalidGGUFError(ArcRunnerError):
    """Raised when a file lacks valid GGUF container signatures or contains unsupported versions."""

    def __init__(self, message: str, file_path: str = "") -> None:
        """Initialize the invalid GGUF container exception."""
        super().__init__(f"Invalid GGUF container '{file_path}': {message}", "INVALID_GGUF_CONTAINER", 400)
        self.file_path = file_path


class GGUFParseError(ArcRunnerError):
    """Raised when decoding GGUF metadata or tensor tables fails due to data corruption or truncation."""

    def __init__(self, message: str, offset: int = 0) -> None:
        """Initialize the GGUF binary decoding exception."""
        super().__init__(f"GGUF decode error at byte offset {offset}: {message}", "GGUF_PARSE_ERROR", 400)
        self.offset = offset


class AOTCompilationError(ArcRunnerError):
    """Raised when ahead-of-time graph compilation fails on target accelerator silicon."""

    def __init__(self, model_id: str, device: str, message: str) -> None:
        """Initialize the AOT compilation failure exception."""
        super().__init__(f"AOT compilation failed for '{model_id}' on {device}: {message}", "AOT_COMPILATION_FAILED", 500)
        self.model_id = model_id
        self.device = device


class AudioEngineError(ArcRunnerError):
    """Base exception for all audio engine operations and runtime failures."""

    def __init__(self, message: str, error_code: str = "AUDIO_ENGINE_ERROR", status_code: int = 500) -> None:
        """Initialize the audio engine exception.

        Args:
            message: Human-readable error description.
            error_code: Unique machine-readable error token.
            status_code: Corresponding HTTP status code.
        """
        super().__init__(message, error_code, status_code)


class AudioTranscriptionError(AudioEngineError):
    """Raised when speech recognition transcription pipeline encounters a failure."""

    def __init__(self, message: str) -> None:
        """Initialize the transcription failure exception.

        Args:
            message: Failure reason details.
        """
        super().__init__(f"Audio transcription failed: {message}", "TRANSCRIPTION_FAILED", 500)


class AudioSynthesisError(AudioEngineError):
    """Raised when text-to-speech synthesis pipeline encounters a failure."""

    def __init__(self, message: str) -> None:
        """Initialize the synthesis failure exception.

        Args:
            message: Failure reason details.
        """
        super().__init__(f"Audio synthesis failed: {message}", "SYNTHESIS_FAILED", 500)


class InvalidAudioFormatError(AudioEngineError):
    """Raised when audio input fails format, channel, or sample rate validation."""

    def __init__(self, message: str) -> None:
        """Initialize the invalid audio format exception.

        Args:
            message: Validation failure details.
        """
        super().__init__(f"Invalid audio format: {message}", "INVALID_AUDIO_FORMAT", 400)


class VoiceNotFoundError(AudioEngineError):
    """Raised when a requested speaker voice embedding cannot be resolved."""

    def __init__(self, voice_name: str, available_voices: list[str] | None = None) -> None:
        """Initialize the voice not found exception.

        Args:
            voice_name: Requested voice identifier.
            available_voices: Optional list of known valid voice slugs.
        """
        avail_str = f" Available: {', '.join(available_voices)}" if available_voices else ""
        super().__init__(f"Voice embedding '{voice_name}' not found.{avail_str}", "VOICE_NOT_FOUND", 404)
        self.voice_name = voice_name
        self.available_voices = available_voices or []


class ConfigurationError(ArcRunnerError):
    """Raised when a configuration or request parameter is invalid or unsafe."""

    def __init__(self, message: str) -> None:
        """Initialize the configuration failure exception.

        Args:
            message: Description of the invalid setting.
        """
        super().__init__(message, "INVALID_CONFIGURATION", 400)
