"""Native OpenVINO GenAI execution engine wrapping C++ LLMPipeline.

Provides direct integration with Intel Arc GPUs (XMX DPAS matrix hardware),
Intel AI Boost NPUs (stateful MCDM arrays), and host CPUs, with a thread-safe
Producer-Consumer streaming bridge.
"""

import errno
import hashlib
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from arc_runner.core.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    STREAM_BUFFER_SIZE,
)
from arc_runner.core.exceptions import (
    ArcRunnerError,
    ConfigurationError,
    ContextWindowExceededError,
    DeviceNotFoundError,
    ModelLoadError,
    ModelNotFoundError,
)
from arc_runner.core.logging import get_logger
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.context_budget import ContextBudgetMixin
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.lora_manager import LoRAManager
from arc_runner.engine.models import EngineTelemetry, GenerationConfig, TokenChunk, validate_generation_config
from arc_runner.engine.telemetry import telemetry_from_perf_metrics, telemetry_from_stream, telemetry_unavailable
from arc_runner.utils.dll_loader import initialize_windows_dll_environment


logger = get_logger("engine.openvino")

KV_PRECISION_ENV = "ARC_KV_CACHE_PRECISION"
CACHE_DIR_ENV = "ARC_CACHE_DIR"
WARMUP_ENV = "ARC_WARMUP_ON_LOAD"
DEVICE_KEY_MAX_LEN = 64
WARMUP_PROMPT = "Hello."
WARMUP_MAX_NEW_TOKENS = 4
_RESERVED_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)})
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def device_key(device: str, max_len: int = DEVICE_KEY_MAX_LEN) -> str:
    """Turn a device string into a stable, Windows-safe directory name.

    Args:
        device: Device string such as 'GPU.0' or 'AUTO:GPU,CPU'.
        max_len: Maximum key length; longer keys are truncated with a hash suffix.

    Returns:
        str: Non-empty upper-case key made of letters, digits, dot, underscore and hyphen.
    """
    digest = hashlib.sha256(device.encode("utf-8")).hexdigest()[:8].upper()
    key = _INVALID_KEY_CHARS.sub("_", device.upper()).strip("._ ")
    if not key:
        key = f"DEVICE_{digest}"
    if key.split(".", 1)[0] in _RESERVED_DEVICE_NAMES:
        key = "_" + key
    if len(key) > max_len:
        key = f"{key[: max_len - 9]}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:8].upper()}"
    return key


class OpenVINOGenAIEngine(ContextBudgetMixin, BaseEngine):
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
        self._kv_cache_precision: str = ""

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

        device_props = self._build_ov_properties(normalized_device, dict(kwargs))

        cache_dir = device_props.get("CACHE_DIR")
        if cache_dir:
            logger.info("Compile cache in use", extra={"cache_dir": str(cache_dir), "device": normalized_device})
        warmup_on = self._warmup_on_load_enabled()
        try:
            import openvino_genai as ov_genai
            self._pipeline = ov_genai.LLMPipeline(str(resolved_path), normalized_device, **device_props)
            self._is_loaded = True
            self._model_path = str(resolved_path)
            self._device = normalized_device
            if warmup_on:
                logger.warning("warmup_on_load=on", extra={"device": normalized_device})
                self.warmup()
        except DeviceNotFoundError:
            raise
        except ModelNotFoundError:
            raise
        except Exception as exc:
            self.unload()
            detail = f"{exc} (compile cache: {cache_dir})" if cache_dir else str(exc)
            raise ModelLoadError(detail, str(resolved_path), normalized_device) from exc

    @staticmethod
    def _warmup_on_load_enabled() -> bool:
        """Read ARC_WARMUP_ON_LOAD; only 'on', 'off' and empty are valid (case-insensitive).

        Returns:
            bool: True only for 'on'. An invalid value is logged at WARNING and treated as off.
        """
        value = os.environ.get(WARMUP_ENV, "").strip().lower()
        if value in ("", "off"):
            return False
        if value == "on":
            return True
        logger.warning(
            "ARC_WARMUP_ON_LOAD value is not on, off or empty; warm-up stays off",
            extra={"env_var": WARMUP_ENV},
        )
        return False

    def warmup(self) -> None:
        """Run one short fixed generation to compile and touch the pipeline; telemetry is left unchanged.

        Raises:
            ArcRunnerError: If no model is loaded.
        """
        if not self.is_loaded:
            raise ArcRunnerError("Model not loaded. Call load_model() first.", "MODEL_NOT_LOADED", 400)
        saved = self._telemetry
        try:
            ov_config = self._build_ov_generation_config(GenerationConfig(max_new_tokens=WARMUP_MAX_NEW_TOKENS))
            self._invoke_pipeline(WARMUP_PROMPT, ov_config)
        finally:
            self._telemetry = saved

    @staticmethod
    def _resolve_cache_dir(device: str) -> Optional[Path]:
        """Create and return the absolute device-keyed compile cache directory.

        Args:
            device: Normalized device string.

        Returns:
            Optional[Path]: The directory, or None (after a WARNING) if it cannot be created or written.
        """
        raw = os.environ.get(CACHE_DIR_ENV, "").strip()
        root = Path(os.path.abspath(os.path.expanduser(raw))) if raw else Path(os.path.abspath(DEFAULT_CACHE_DIR / "ov_cache"))
        target = root / device_key(device)
        try:
            target.mkdir(parents=True, exist_ok=True)
            if not os.access(target, os.W_OK):
                raise PermissionError(errno.EACCES, "cache directory is not writable")
        except OSError as exc:
            logger.warning(
                "Compile cache disabled: directory unusable",
                extra={"cache_dir": str(target), "device": device, "reason": str(exc), "errno": str(exc.errno)},
            )
            return None
        return target

    def _build_ov_properties(self, device: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Build OpenVINO pipeline properties for a device.

        The u8 KV cache is NOT enabled by default: it is a quality trade-off with no quality gate in
        this project. Opt in with the KV_CACHE_PRECISION keyword or ARC_KV_CACHE_PRECISION=u8.

        Args:
            device: Normalized device string (GPU, NPU, CPU with optional suffix).
            overrides: Caller-supplied properties; they always win over defaults.

        Returns:
            Dict[str, Any]: Properties passed to openvino_genai.LLMPipeline.
        """
        props = dict(overrides)
        if "CACHE_DIR" not in props:
            cache_dir = self._resolve_cache_dir(device)
            if cache_dir is not None:
                props["CACHE_DIR"] = str(cache_dir)
        if device.startswith("GPU"):
            props.setdefault("PERFORMANCE_HINT", "LATENCY")
            props.setdefault("GPU_ENABLE_SDPA_OPTIMIZATION", True)
        requested_precision = os.environ.get(KV_PRECISION_ENV, "").strip()
        if requested_precision:
            props.setdefault("KV_CACHE_PRECISION", requested_precision)
        self._kv_cache_precision = str(props.get("KV_CACHE_PRECISION", ""))
        return props

    def tokenize(self, prompt: str) -> Optional[List[int]]:
        """Return the prompt's token ids from the loaded pipeline's tokenizer.

        Args:
            prompt: Prompt text.

        Returns:
            Optional[List[int]]: Token ids, or None if no model is loaded or the tokenizer cannot be queried.
        """
        if not self.is_loaded:
            return None
        try:
            encoded = self._pipeline.get_tokenizer().encode(prompt)
            return [int(token) for token in encoded.input_ids.data.flatten().tolist()]
        except Exception as exc:
            logger.warning("Tokenization unavailable", extra={"reason": str(exc)})
            return None

    def _invoke_pipeline(self, prompt: str, ov_config: Any) -> Any:
        """Run one synchronous generation using a single call form that returns DecodedResults.

        Args:
            prompt: Prompt text.
            ov_config: openvino_genai.GenerationConfig or None.

        Returns:
            Any: Pipeline result (DecodedResults carrying texts and perf_metrics).
        """
        return self._pipeline.generate([prompt], ov_config) if ov_config else self._pipeline.generate([prompt])

    def _build_ov_generation_config(self, config: GenerationConfig) -> Any:
        """Convert engine GenerationConfig into an openvino_genai.GenerationConfig instance.

        Args:
            config: Engine sampling configuration.

        Returns:
            openvino_genai.GenerationConfig: Populated native config object.

        Raises:
            ConfigurationError: If a sampling value is out of range or the native binding rejects it.
        """
        validate_generation_config(config)
        import openvino_genai as ov_genai
        try:
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
        except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
            logger.warning(
                "Generation config rejected by native binding",
                extra={"error_type": type(exc).__name__, "max_new_tokens": str(config.max_new_tokens), "reason": str(exc)},
            )
            raise ConfigurationError(f"native generation config rejected: {exc}") from exc

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
            self._telemetry = EngineTelemetry(model_id=self._model_path, device=self._device)
            return ""

        cfg = config or GenerationConfig()
        ov_config = self._build_ov_generation_config(cfg)

        prompt_tokens = self._check_request_fits_context(prompt, cfg.max_new_tokens)

        start_time = time.monotonic()
        raw_result = self._invoke_pipeline(prompt, ov_config)
        elapsed_sec = max(time.monotonic() - start_time, 1e-6)

        if hasattr(raw_result, "texts") and hasattr(raw_result, "perf_metrics"):
            text_result = raw_result.texts[0] if raw_result.texts else ""
            self._telemetry = telemetry_from_perf_metrics(
                raw_result.perf_metrics, self._model_path, self._device, elapsed_sec
            )
        else:
            text_result = str(raw_result)
            self._telemetry = telemetry_unavailable(self._model_path, self._device, elapsed_sec)

        self._telemetry.prompt_tokens = prompt_tokens or 0
        self._telemetry.kv_cache_precision = self._kv_cache_precision
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
        emitted: List[str] = []

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
                        emitted.append(truncated)
                        generated_count += 1
                        chunk_queue.put(TokenChunk(text=truncated, is_finished=False, completion_tokens=generated_count))
                    return True

            emitted.append(chunk_text)
            generated_count += 1
            chunk_queue.put(TokenChunk(text=chunk_text, is_finished=False, completion_tokens=generated_count))
            return False

        ov_config = self._build_ov_generation_config(cfg)
        prompt_tokens = self._check_request_fits_context(prompt, cfg.max_new_tokens)

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
        token_count = self._count_prompt_tokens("".join(emitted)) if emitted else None
        self._telemetry = telemetry_from_stream(
            self._model_path, self._device, start_time, end_time, first_token_time, generated_count, token_count=token_count
        )
        self._telemetry.prompt_tokens = prompt_tokens or 0
        self._telemetry.kv_cache_precision = self._kv_cache_precision

        finish_reason = "length" if generated_count >= cfg.max_new_tokens else "stop"
        yield TokenChunk(
            text="",
            is_finished=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens or 0,
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
