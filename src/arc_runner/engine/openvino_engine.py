"""Native OpenVINO GenAI execution engine wrapping C++ LLMPipeline.

Provides direct integration with Intel Arc GPUs (XMX DPAS matrix hardware),
Intel AI Boost NPUs (stateful MCDM arrays), and host CPUs, with a thread-safe
Producer-Consumer streaming bridge.
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

from arc_runner.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    STREAM_BUFFER_SIZE,
)
from arc_runner.core.exceptions import (
    ArcRunnerError,
    ContextWindowExceededError,
    DeviceNotFoundError,
    ModelLoadError,
    ModelNotFoundError,
)
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.lora_manager import LoRAManager
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk
from arc_runner.utils.dll_loader import initialize_windows_dll_environment


class OpenVINOGenAIEngine(BaseEngine):
    """Execution engine wrapping openvino_genai.LLMPipeline for Intel heterogeneous silicon."""

    VALID_DEVICE_PREFIXES = ("GPU", "NPU", "CPU")

    def __init__(
        self,
        kv_manager: Optional[KVCacheManager] = None,
        lora_manager: Optional[LoRAManager] = None,
    ) -> None:
        """Initialize the OpenVINO GenAI engine instance.

        Args:
            kv_manager: Optional KV-cache manager for context budgeting and headroom tracking.
            lora_manager: Optional LoRA adapter manager for dynamic hot-swapping.
        """
        initialize_windows_dll_environment()
        self.kv_manager = kv_manager or KVCacheManager()
        self.lora_manager = lora_manager or LoRAManager()
        self._pipeline: Any = None
        self._is_loaded: bool = False
        self._model_path: str = ""
        self._device: str = ""
        self._telemetry: EngineTelemetry = EngineTelemetry()

    @property
    def is_loaded(self) -> bool:
        """Check whether a model pipeline is loaded and ready for execution.

        Returns:
            bool: True if LLMPipeline is active in memory.
        """
        return self._is_loaded and self._pipeline is not None

    @property
    def model_path(self) -> str:
        """Return the file system path of the loaded model.

        Returns:
            str: Path string of active model.
        """
        return self._model_path

    @property
    def device(self) -> str:
        """Return the device identifier string currently targeted.

        Returns:
            str: Device string (e.g., 'GPU', 'NPU', 'CPU').
        """
        return self._device

    def load_model(
        self,
        model_path: Union[str, Path],
        device: str = "GPU",
        **kwargs: object,
    ) -> None:
        """Load and compile model weights on the designated accelerator silicon.

        Args:
            model_path: Path to the directory containing OpenVINO IR files.
            device: Target hardware device string (e.g., 'GPU', 'NPU', 'CPU').
            **kwargs: Additional properties passed to openvino_genai.LLMPipeline.

        Raises:
            ModelNotFoundError: If the model path does not exist on disk.
            DeviceNotFoundError: If requested device is not recognized.
            ModelLoadError: If OpenVINO compiler or device driver fails.
        """
        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists() or not resolved_path.is_dir():
            raise ModelNotFoundError(str(model_path), str(resolved_path))

        normalized_device = device.strip().upper()
        if not any(normalized_device.startswith(prefix) for prefix in self.VALID_DEVICE_PREFIXES):
            raise DeviceNotFoundError(device, list(self.VALID_DEVICE_PREFIXES))

        try:
            import openvino_genai as ov_genai
            self._pipeline = ov_genai.LLMPipeline(str(resolved_path), normalized_device, **kwargs)
            self._is_loaded = True
            self._model_path = str(resolved_path)
            self._device = normalized_device
        except DeviceNotFoundError:
            raise
        except ModelNotFoundError:
            raise
        except Exception as exc:
            self._is_loaded = False
            self._pipeline = None
            raise ModelLoadError(str(exc), str(resolved_path), normalized_device) from exc

    def _build_ov_generation_config(self, config: GenerationConfig) -> Any:
        """Convert engine GenerationConfig into an openvino_genai.GenerationConfig instance.

        Args:
            config: Engine sampling configuration.

        Returns:
            openvino_genai.GenerationConfig: Populated native config object.
        """
        try:
            import openvino_genai as ov_genai
            ov_config = ov_genai.GenerationConfig()
            ov_config.max_new_tokens = config.max_new_tokens
            ov_config.temperature = config.temperature
            ov_config.top_p = config.top_p
            ov_config.top_k = config.top_k
            ov_config.repetition_penalty = config.repetition_penalty
            if config.stop_strings:
                ov_config.stop_strings = set(config.stop_strings)
            if config.stop_token_ids:
                ov_config.stop_token_ids = set(config.stop_token_ids)
            if config.adapters:
                ov_adapter_cfg = self.lora_manager.build_adapter_config(config.adapters)
                if ov_adapter_cfg is not None:
                    ov_config.adapters = ov_adapter_cfg
            return ov_config
        except Exception:
            return None

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
        if not self.is_loaded:
            raise ArcRunnerError("Model not loaded. Call load_model() first.", "MODEL_NOT_LOADED", 400)

        if not prompt:
            return ""

        cfg = config or GenerationConfig()
        ov_config = self._build_ov_generation_config(cfg)

        start_time = time.monotonic()
        raw_result = self._pipeline.generate(prompt, ov_config) if ov_config else self._pipeline.generate(prompt)
        elapsed_sec = max(time.monotonic() - start_time, 1e-6)

        text_result = str(raw_result)
        approx_tokens = len(text_result.split())

        self._telemetry = EngineTelemetry(
            model_id=self._model_path,
            device=self._device,
            ttft_ms=elapsed_sec * 500.0,
            tokens_per_second=approx_tokens / elapsed_sec,
            total_tokens=approx_tokens,
            total_latency_ms=elapsed_sec * 1000.0,
        )

        return text_result

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Execute streaming generation and yield token chunks as they are decoded.

        Uses a bounded Producer-Consumer queue to deliver non-blocking token streaming
        with automatic backpressure.

        Args:
            prompt: Text prompt input to prefill.
            config: Optional sampling and generation configuration parameters.

        Yields:
            TokenChunk: Incremental token emissions containing text deltas and finish reasons.

        Raises:
            ArcRunnerError: If invoked prior to load_model().
            ContextWindowExceededError: If prompt length exceeds model context capacity.
        """
        if not self.is_loaded:
            raise ArcRunnerError("Model not loaded. Call load_model() first.", "MODEL_NOT_LOADED", 400)

        cfg = config or GenerationConfig()

        if not prompt:
            yield TokenChunk(text="", is_finished=True, finish_reason="stop", prompt_tokens=0, completion_tokens=0)
            return

        chunk_queue: queue.Queue[Optional[TokenChunk]] = queue.Queue(maxsize=STREAM_BUFFER_SIZE)
        cancel_event = threading.Event()
        stop_sequences = cfg.stop_strings

        start_time = time.monotonic()
        first_token_time: Optional[float] = None
        generated_count = 0

        def streamer_callback(chunk_text: str) -> bool:
            nonlocal first_token_time, generated_count
            if cancel_event.is_set():
                return True

            if first_token_time is None:
                first_token_time = time.monotonic()

            for stop_str in stop_sequences:
                if stop_str in chunk_text:
                    truncated = chunk_text.split(stop_str)[0]
                    cancel_event.set()
                    if truncated:
                        generated_count += 1
                        chunk_queue.put(TokenChunk(text=truncated, is_finished=False, completion_tokens=generated_count))
                    return True

            generated_count += 1
            chunk_queue.put(TokenChunk(text=chunk_text, is_finished=False, completion_tokens=generated_count))
            return False

        ov_config = self._build_ov_generation_config(cfg)

        worker_error: List[Exception] = []

        def worker_target() -> None:
            try:
                if ov_config:
                    self._pipeline.generate(prompt, ov_config, streamer=streamer_callback)
                else:
                    self._pipeline.generate(prompt, streamer=streamer_callback)
            except Exception as exc:
                worker_error.append(exc)
            finally:
                chunk_queue.put(None)

        worker_thread = threading.Thread(target=worker_target, daemon=True)
        worker_thread.start()

        try:
            while True:
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    if not worker_thread.is_alive() and chunk_queue.empty():
                        break
                    continue

                if chunk is None:
                    break

                yield chunk

        finally:
            cancel_event.set()

        if worker_error:
            raise worker_error[0]

        end_time = time.monotonic()
        total_sec = max(end_time - start_time, 1e-6)
        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else total_sec * 500.0
        decode_speed = generated_count / max(end_time - (first_token_time or start_time), 1e-6)

        self._telemetry = EngineTelemetry(
            model_id=self._model_path,
            device=self._device,
            ttft_ms=ttft,
            tokens_per_second=decode_speed,
            total_tokens=generated_count,
            total_latency_ms=total_sec * 1000.0,
        )

        finish_reason = "length" if generated_count >= cfg.max_new_tokens else "stop"
        yield TokenChunk(
            text="",
            is_finished=True,
            finish_reason=finish_reason,
            completion_tokens=generated_count,
        )

    def unload(self) -> None:
        """Release compiled pipeline buffers and free device memory allocations."""
        self._pipeline = None
        self._is_loaded = False
        self._model_path = ""
        self._device = ""

    def get_telemetry(self) -> EngineTelemetry:
        """Retrieve execution telemetry including throughput and latency metrics.

        Returns:
            EngineTelemetry: Recorded performance telemetry for the most recent run.
        """
        return self._telemetry
