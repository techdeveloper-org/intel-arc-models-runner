"""Heterogeneous dual-silicon speculative decoding engine.

Coordinates low-power INT4 draft token generation on the Intel NPU with parallel
matrix verification on the Intel Arc GPU via Level Zero and OpenVINO GenAI C++ kernels.
"""

import collections
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from arc_runner.core.constants import STREAM_BUFFER_SIZE
from arc_runner.core.exceptions import (
    ArcRunnerError,
    ConfigurationError,
    ContextWindowExceededError,
    DeviceNotFoundError,
    DraftModelCompilationError,
    ModelLoadError,
    ModelNotFoundError,
    SpeculativeDecodingError,
    TokenizerMismatchError,
)
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.context_budget import ContextBudgetMixin
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.telemetry import telemetry_from_perf_metrics, telemetry_from_stream, telemetry_unavailable
from arc_runner.engine.models import (
    EngineTelemetry,
    GenerationConfig,
    SpeculativeConfig,
    TokenChunk,
    validate_generation_config,
)

logger = logging.getLogger(__name__)


class HybridSpeculativeEngine(ContextBudgetMixin, BaseEngine):
    """Dual-silicon speculative decoding engine coupling an NPU draft model with an Arc GPU verifier.

    Leverages openvino_genai native draft_model bindings to execute speculative verification
    with zero GPU VRAM consumption for the draft model and dynamic lookahead horizon adaptation.
    """

    def __init__(self, kv_cache_manager: Optional[KVCacheManager] = None) -> None:
        """Initialize the speculative engine instance in an unloaded state.

        Args:
            kv_cache_manager: Optional pre-configured KV-cache memory manager.
        """
        self._pipeline: Optional[Any] = None
        self._is_loaded: bool = False
        self._model_path: str = ""
        self._draft_model_path: str = ""
        self._device: str = "GPU"
        self._draft_device: str = "NPU"
        self._telemetry: EngineTelemetry = EngineTelemetry()
        self._spec_config: Optional[SpeculativeConfig] = None
        self._current_gamma: int = 4
        self._circuit_breaker_tripped: bool = False
        self._circuit_breaker_threshold: float = 0.55
        self._tokens_since_trip: int = 0
        self._acceptance_history: collections.deque = collections.deque(maxlen=15)
        self.kv_manager: KVCacheManager = kv_cache_manager or KVCacheManager()

    @property
    def is_loaded(self) -> bool:
        """Return True if both target and optional draft pipelines are compiled and ready.

        Returns:
            bool: Initialization state flag.
        """
        return self._is_loaded

    @property
    def model_path(self) -> str:
        """Retrieve the filesystem path of the loaded target model.

        Returns:
            str: Resolved target directory path.
        """
        return self._model_path

    @property
    def draft_model_path(self) -> str:
        """Retrieve the filesystem path of the loaded draft model.

        Returns:
            str: Resolved draft directory path, or empty string if single-model.
        """
        return self._draft_model_path

    @property
    def device(self) -> str:
        """Retrieve the active execution target device identifier for the verifier model.

        Returns:
            str: Hardware device string (e.g. 'GPU').
        """
        return self._device

    @property
    def draft_device(self) -> str:
        """Retrieve the active silicon device running the draft model.

        Returns:
            str: Hardware device string (e.g. 'NPU', 'CPU', 'GPU').
        """
        return self._draft_device

    @property
    def current_gamma(self) -> int:
        """Retrieve the current active lookahead token window.

        Returns:
            int: Number of draft tokens evaluated per speculative step.
        """
        return self._current_gamma

    def _validate_path(self, path_str: str, label: str) -> Path:
        """Verify that a specified model directory exists on the filesystem.

        Args:
            path_str: Path string to validate.
            label: Human-readable label for error reporting.

        Returns:
            Path: Resolved filesystem path.

        Raises:
            ModelNotFoundError: If path does not exist.
        """
        path = Path(path_str).resolve()
        if not path.exists():
            raise ModelNotFoundError(f"{label} path not found: {path_str}", str(path))
        return path

    def _sync_draft_tokenizer(self, target_path: Path, draft_path: Path) -> None:
        """Ensure the draft model directory contains tokenizer IR files matching the target.

        Args:
            target_path: Path to the target OpenVINO IR model directory.
            draft_path: Path to the draft model directory.
        """
        import shutil

        tok_files = [
            "openvino_tokenizer.xml",
            "openvino_tokenizer.bin",
            "openvino_detokenizer.xml",
            "openvino_detokenizer.bin",
        ]
        target_has_tok = (target_path / "openvino_tokenizer.xml").exists()
        if not target_has_tok:
            return

        for fname in tok_files:
            src = target_path / fname
            dst = draft_path / fname
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    logger.info("Synchronized tokenizer file %s to draft directory", fname)
                except Exception as exc:
                    logger.warning("Could not synchronize tokenizer file %s: %s", fname, exc)

    def _create_draft_model(
        self,
        draft_path: Path,
        preferred_device: str,
        draft_properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Attempt to instantiate an openvino_genai.draft_model with hardware fallback.

        Args:
            draft_path: Filesystem path to the draft model directory.
            preferred_device: Requested hardware target for the draft model.
            draft_properties: Optional device-specific compilation properties.

        Returns:
            Optional[Any]: Compiled draft model wrapper, or None if fallback exhausted.

        Raises:
            DraftModelCompilationError: If draft compilation fails on all eligible devices.
        """
        import openvino_genai as ov_genai

        candidates = [preferred_device]
        if preferred_device != "CPU":
            candidates.append("CPU")

        props = dict(draft_properties or {})
        last_error: Optional[Exception] = None
        for dev in candidates:
            dev_props = dict(props)
            if dev.startswith("GPU"):
                dev_props.setdefault("PERFORMANCE_HINT", "LATENCY")
                dev_props.setdefault("EXECUTION_MODE_HINT", "PERFORMANCE")
                dev_props.setdefault("INFERENCE_PRECISION_HINT", "f16")
                dev_props.setdefault("GPU_ENABLE_SDPA_OPTIMIZATION", True)
                dev_props.setdefault("GPU_HOST_TASK_PRIORITY", "HIGH")
                dev_props.setdefault("GPU_QUEUE_PRIORITY", "HIGH")
                dev_props.setdefault("GPU_QUEUE_THROTTLE", "LOW")
            else:
                dev_props.pop("GPU_ENABLE_SDPA_OPTIMIZATION", None)
                dev_props.pop("EXECUTION_MODE_HINT", None)
                dev_props.pop("GPU_HOST_TASK_PRIORITY", None)
                dev_props.pop("GPU_QUEUE_PRIORITY", None)
                dev_props.pop("GPU_QUEUE_THROTTLE", None)
            try:
                if dev_props:
                    draft_handle = ov_genai.draft_model(str(draft_path), dev, **dev_props)
                else:
                    draft_handle = ov_genai.draft_model(str(draft_path), dev)
                self._draft_device = dev
                return draft_handle
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Draft model failed on device %s: %s. Attempting fallback.",
                    dev,
                    exc,
                )

        if last_error:
            raise DraftModelCompilationError(
                str(draft_path),
                candidates,
                str(last_error),
            ) from last_error
        return None

    def load_model(
        self,
        model_path: str,
        speculative_config: Optional[SpeculativeConfig] = None,
        device: str = "GPU",
        **kwargs: Any,
    ) -> None:
        """Compile and initialize the target and draft models for dual-silicon inference.

        Args:
            model_path: Filesystem directory containing the target OpenVINO IR model.
            speculative_config: Optional speculative decoding configuration.
            device: Target hardware device for the main verification model.
            **kwargs: Additional device or compilation properties.

        Raises:
            ModelNotFoundError: If target or draft directories do not exist.
            ModelLoadError: If OpenVINO runtime compilation fails.
            TokenizerMismatchError: If tokenizers differ between target and draft models.
        """
        target_path = self._validate_path(model_path, "Target model")
        normalized_target_device = device.upper()

        draft_handle: Optional[Any] = None
        if speculative_config:
            if speculative_config.circuit_breaker_threshold:
                self._circuit_breaker_threshold = speculative_config.circuit_breaker_threshold
            if speculative_config.prompt_lookup:
                self._spec_config = speculative_config
                self._current_gamma = speculative_config.num_assistant_tokens
                self._draft_model_path = ""
                self._draft_device = "PROMPT_LOOKUP"
            elif speculative_config.draft_model_path:
                draft_path = self._validate_path(
                    speculative_config.draft_model_path,
                    "Draft model",
                )
                self._sync_draft_tokenizer(target_path, draft_path)
                requested_device = (
                    "GPU"
                    if getattr(speculative_config, "colocate_on_gpu", False) and normalized_target_device.startswith("GPU")
                    else speculative_config.draft_device.upper()
                )
                draft_handle = self._create_draft_model(
                    draft_path,
                    requested_device,
                    speculative_config.draft_properties,
                )
                self._spec_config = speculative_config
                self._current_gamma = speculative_config.num_assistant_tokens
                self._draft_model_path = str(draft_path)
            else:
                self._spec_config = None
                self._draft_model_path = ""
                self._draft_device = ""
        else:
            self._spec_config = None
            self._draft_model_path = ""
            self._draft_device = ""

        try:
            import openvino_genai as ov_genai

            pipeline_kwargs = dict(kwargs)
            if pipeline_kwargs and normalized_target_device.startswith("GPU"):
                pipeline_kwargs.setdefault("PERFORMANCE_HINT", "LATENCY")
                pipeline_kwargs.setdefault("EXECUTION_MODE_HINT", "PERFORMANCE")
                pipeline_kwargs.setdefault("INFERENCE_PRECISION_HINT", "f16")
                pipeline_kwargs.setdefault("GPU_ENABLE_SDPA_OPTIMIZATION", True)
                pipeline_kwargs.setdefault("GPU_HOST_TASK_PRIORITY", "HIGH")
                pipeline_kwargs.setdefault("GPU_QUEUE_PRIORITY", "HIGH")
                pipeline_kwargs.setdefault("GPU_QUEUE_THROTTLE", "LOW")
                pipeline_kwargs.setdefault("GPU_ENABLE_LOOP_UNROLLING", True)

            if draft_handle is not None:
                pipeline_kwargs["draft_model"] = draft_handle

            self._pipeline = ov_genai.LLMPipeline(
                str(target_path),
                normalized_target_device,
                **pipeline_kwargs,
            )
            self._is_loaded = True
            self._model_path = str(target_path)
            self._device = normalized_target_device
        except Exception as exc:
            self.unload()
            exc_str = str(exc)
            if "Tokenizers for draft and main models are different" in exc_str:
                raise TokenizerMismatchError(
                    str(target_path),
                    self._draft_model_path,
                    exc_str,
                ) from exc
            raise ModelLoadError(
                f"Failed to load target model pipeline: {exc}",
                str(target_path),
                normalized_target_device,
            ) from exc

    def _build_ov_generation_config(self, config: GenerationConfig) -> Any:
        """Translate engine GenerationConfig into an openvino_genai.GenerationConfig.

        Args:
            config: Sampling and generation parameters.

        Returns:
            Any: Populated openvino_genai.GenerationConfig instance.

        Raises:
            ConfigurationError: If a sampling value is out of range or the native binding rejects it.
        """
        validate_generation_config(config)
        import openvino_genai as ov_genai

        try:
            ov_cfg = ov_genai.GenerationConfig()
            ov_cfg.max_new_tokens = config.max_new_tokens
            ov_cfg.temperature = config.temperature
            ov_cfg.top_p = config.top_p
            ov_cfg.top_k = config.top_k
            ov_cfg.repetition_penalty = config.repetition_penalty

            if self._circuit_breaker_tripped:
                ov_cfg.num_assistant_tokens = 0
            elif self._spec_config:
                if self._spec_config.prompt_lookup:
                    ov_cfg.num_assistant_tokens = self._current_gamma
                    ov_cfg.max_ngram_size = self._spec_config.max_ngram_size
                elif self._draft_model_path:
                    ov_cfg.num_assistant_tokens = self._current_gamma

            if config.stop_strings:
                ov_cfg.stop_strings = set(config.stop_strings)
            if config.stop_token_ids:
                ov_cfg.stop_token_ids = set(config.stop_token_ids)
            return ov_cfg
        except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
            logger.warning(
                "Generation config rejected by native binding",
                extra={"error_type": type(exc).__name__, "max_new_tokens": str(config.max_new_tokens), "reason": str(exc)},
            )
            raise ConfigurationError(f"native generation config rejected: {exc}") from exc

    def _adapt_gamma(self, accepted_count: int, draft_count: int) -> None:
        """Update active lookahead horizon gamma based on empirical acceptance rate with circuit breaker.

        Args:
            accepted_count: Number of proposed draft tokens accepted by target.
            draft_count: Total draft tokens proposed in the iteration.
        """
        if draft_count <= 0:
            return

        alpha = accepted_count / float(draft_count)
        self._acceptance_history.append(alpha)
        rolling_alpha = sum(self._acceptance_history) / float(len(self._acceptance_history))

        if rolling_alpha < self._circuit_breaker_threshold and len(self._acceptance_history) >= 5:
            if not self._circuit_breaker_tripped:
                logger.info(
                    "Speculative acceptance rate %.2f dropped below threshold %.2f; tripping circuit breaker to autoregressive fallback.",
                    rolling_alpha,
                    self._circuit_breaker_threshold,
                )
                self._circuit_breaker_tripped = True
                self._tokens_since_trip = 0
            return

        if self._circuit_breaker_tripped:
            self._tokens_since_trip += draft_count
            if self._tokens_since_trip >= 20:
                self._circuit_breaker_tripped = False
                self._current_gamma = 2
                logger.info("Circuit breaker half-open probe: testing gamma=2.")
                return

        if not self._spec_config or not self._spec_config.dynamic_gamma:
            return

        new_gamma = self._current_gamma
        if alpha >= 0.75:
            new_gamma += 1
        elif alpha <= 0.40:
            new_gamma -= 1

        self._current_gamma = max(
            self._spec_config.min_gamma,
            min(self._spec_config.max_gamma, new_gamma),
        )

    def get_speculative_stats(self) -> Dict[str, Any]:
        """Return real-time speculative execution status including gamma and circuit breaker state.

        Returns:
            Dict[str, Any]: Current speculative parameters and acceptance metrics.
        """
        rolling_alpha = (
            sum(self._acceptance_history) / float(len(self._acceptance_history))
            if self._acceptance_history
            else 0.0
        )
        return {
            "current_gamma": self._current_gamma,
            "rolling_acceptance_rate": rolling_alpha,
            "draft_device": self._draft_device,
            "circuit_breaker_tripped": self._circuit_breaker_tripped,
        }

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
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
            raise ArcRunnerError(
                "Model not loaded. Call load_model() first.",
                "MODEL_NOT_LOADED",
                400,
            )

        if not prompt:
            return ""

        cfg = config or GenerationConfig()
        ov_config = self._build_ov_generation_config(cfg)
        prompt_tokens = self._check_request_fits_context(prompt, cfg.max_new_tokens)

        start_time = time.monotonic()
        raw_result = (
            self._pipeline.generate([prompt], ov_config)
            if ov_config
            else self._pipeline.generate([prompt])
        )
        elapsed_sec = max(time.monotonic() - start_time, 1e-6)

        is_spec = bool(self._spec_config and (self._draft_model_path or self._spec_config.prompt_lookup))
        if hasattr(raw_result, "texts") and hasattr(raw_result, "perf_metrics"):
            text_result = raw_result.texts[0] if raw_result.texts else ""
            self._telemetry = telemetry_from_perf_metrics(
                raw_result.perf_metrics,
                self._model_path,
                self._device,
                elapsed_sec,
                is_speculative=is_spec,
                draft_device=self._draft_device if is_spec else "",
            )
        elif isinstance(raw_result, (list, tuple)) and len(raw_result) > 0:
            text_result = str(raw_result[0])
            self._telemetry = telemetry_unavailable(
                self._model_path,
                self._device,
                elapsed_sec,
                is_speculative=is_spec,
                draft_device=self._draft_device if is_spec else "",
            )
        else:
            text_result = str(raw_result)
            self._telemetry = telemetry_unavailable(
                self._model_path,
                self._device,
                elapsed_sec,
                is_speculative=is_spec,
                draft_device=self._draft_device if is_spec else "",
            )

        self._telemetry.prompt_tokens = prompt_tokens or 0
        self._telemetry.circuit_breaker_tripped = self._circuit_breaker_tripped
        return text_result

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[TokenChunk]:
        """Execute non-blocking streaming generation yielding token chunks.

        Utilizes a bounded Producer-Consumer queue with atomic cancellation flags
        to prevent event loop blocking.

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
            raise ArcRunnerError(
                "Model not loaded. Call load_model() first.",
                "MODEL_NOT_LOADED",
                400,
            )

        cfg = config or GenerationConfig()

        if not prompt:
            yield TokenChunk(
                text="",
                is_finished=True,
                finish_reason="stop",
                prompt_tokens=0,
                completion_tokens=0,
            )
            return

        prompt_tokens = self._check_request_fits_context(prompt, cfg.max_new_tokens)

        chunk_queue: queue.Queue[Optional[TokenChunk]] = queue.Queue(
            maxsize=STREAM_BUFFER_SIZE
        )
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
                        chunk_queue.put(
                            TokenChunk(
                                text=truncated,
                                is_finished=False,
                                completion_tokens=generated_count,
                            )
                        )
                    return True

            generated_count += 1
            chunk_queue.put(
                TokenChunk(
                    text=chunk_text,
                    is_finished=False,
                    completion_tokens=generated_count,
                )
            )
            return False

        ov_config = self._build_ov_generation_config(cfg)
        worker_error: List[Exception] = []

        def worker_target() -> None:
            try:
                if ov_config:
                    self._pipeline.generate(
                        prompt,
                        ov_config,
                        streamer=streamer_callback,
                    )
                else:
                    self._pipeline.generate(
                        prompt,
                        streamer=streamer_callback,
                    )
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
        is_spec = bool(self._spec_config and self._draft_model_path)
        self._telemetry = telemetry_from_stream(
            self._model_path,
            self._device,
            start_time,
            end_time,
            first_token_time,
            generated_count,
            is_speculative=is_spec,
            draft_device=self._draft_device if is_spec else "",
        )
        self._telemetry.prompt_tokens = prompt_tokens or 0
        self._telemetry.circuit_breaker_tripped = self._circuit_breaker_tripped

        finish_reason = (
            "length" if generated_count >= cfg.max_new_tokens else "stop"
        )
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
        self._draft_model_path = ""
        self._device = "GPU"
        self._draft_device = "NPU"
        self._spec_config = None

    def get_telemetry(self) -> EngineTelemetry:
        """Retrieve execution telemetry including speculative speedup metrics.

        Returns:
            EngineTelemetry: Performance metrics recorded during the most recent generation.
        """
        return self._telemetry
