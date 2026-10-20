"""Abstract base class interface contract for inference engines.

Defines the execution contract, model lifecycle, streaming protocols, and telemetry
collection for all backend accelerator engines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Iterator, Optional, Type, Union

from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk


class BaseEngine(ABC):
    """Abstract interface for local accelerator execution backends."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check whether a model pipeline is loaded and ready for execution.

        Returns:
            bool: True if weights are compiled and allocated in memory.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Union[str, Path], device: str = "GPU", **kwargs: object) -> None:
        """Load and compile model weights on the designated accelerator silicon.

        Args:
            model_path: Path to the directory containing OpenVINO IR files or GGUF weights.
            device: Target hardware device string (e.g., 'GPU', 'NPU', 'CPU').
            **kwargs: Additional engine-specific configuration flags.

        Raises:
            ModelLoadError: If model compilation or device allocation fails.
            ModelNotFoundError: If the path does not exist.
            DeviceNotFoundError: If the requested silicon is unavailable.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Execute synchronous autoregressive generation and return completed text.

        Args:
            prompt: Text prompt input to prefill.
            config: Optional sampling and generation configuration parameters.

        Returns:
            str: Full generated completion text.

        Raises:
            ArcRunnerError: If invoked prior to load_model().
            ContextWindowExceededError: If prompt length exceeds model context capacity.
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None) -> Iterator[TokenChunk]:
        """Execute streaming generation and yield token chunks as they are decoded.

        Args:
            prompt: Text prompt input to prefill.
            config: Optional sampling and generation configuration parameters.

        Yields:
            TokenChunk: Incremental token emissions containing text deltas and finish reasons.

        Raises:
            ArcRunnerError: If invoked prior to load_model().
            ContextWindowExceededError: If prompt length exceeds model context capacity.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release compiled pipeline buffers and free device memory allocations."""
        pass

    @abstractmethod
    def get_telemetry(self) -> EngineTelemetry:
        """Retrieve execution telemetry including throughput and latency metrics.

        Returns:
            EngineTelemetry: Recorded performance telemetry for the most recent run.
        """
        pass

    def __enter__(self) -> "BaseEngine":
        """Enter runtime context manager.

        Returns:
            BaseEngine: The active engine instance.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit runtime context manager and guarantee resource deallocation.

        Args:
            exc_type: Exception class raised if an error occurred.
            exc_val: Exception instance raised.
            exc_tb: Exception traceback.
        """
        self.unload()
