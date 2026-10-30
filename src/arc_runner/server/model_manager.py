"""Thread-safe asynchronous model registry and LRU cache manager for Intel Arc Models Runner.

Provides centralized model compilation, lifetime management, heterogeneous device dispatch
(GPU target + NPU draft), and deterministic Level Zero memory reclamation.
"""

from collections import OrderedDict
import os
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from arc_runner.core.exceptions import (
    ArcRunnerError,
    ModelLoadError,
    ModelNotFoundError,
)
from arc_runner.core.logging import get_logger
from arc_runner.core.path_policy import allowed_roots, ensure_path_allowed, resolve_inside
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.embedding_engine import EmbeddingEngine
from arc_runner.engine.lora_manager import LoRAManager
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.model_config import parse_context_window_env, read_kv_geometry, resolve_context_window
from arc_runner.engine.openvino_engine import KV_PRECISION_ENV, OpenVINOGenAIEngine
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine, SpeculativeConfig
from arc_runner.engine.vision_engine import VisionEngine

logger = get_logger("server.model_manager")


class ModelManager:
    """Manages active engine instances and model directories with thread-safe LRU eviction."""

    def __init__(
        self,
        model_dirs: Optional[List[Union[str, Path]]] = None,
        max_loaded_models: int = 2,
    ) -> None:
        """Initialize the model manager with specified model search directories and capacity.

        Args:
            model_dirs: List of directory paths searched for OpenVINO IR models.
            max_loaded_models: Maximum number of compiled engines concurrently kept in memory.

        Raises:
            ConfigurationError: If ARC_CONTEXT_WINDOW is set to an invalid value (checked before any model load).
        """
        parse_context_window_env()
        self._lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._max_loaded_models = max_loaded_models
        self._model_dirs: List[Path] = [Path(d).resolve() for d in (model_dirs or [Path("models")])]
        self._cache: OrderedDict[str, BaseEngine] = OrderedDict()
        self._embedding_cache: OrderedDict[str, EmbeddingEngine] = OrderedDict()
        self._vision_cache: OrderedDict[str, VisionEngine] = OrderedDict()
        self.lora_manager: LoRAManager = LoRAManager()
        self._total_requests_served: int = 0
        self._total_tokens_generated: int = 0

    @staticmethod
    def engine_key(
        model_id: str,
        device: str,
        draft_model_id: Optional[str] = None,
        draft_device: str = "NPU",
        gamma: int = 4,
    ) -> str:
        """Return the canonical cache key for a given engine configuration.

        Args:
            model_id: Primary model identifier or path.
            device: Target accelerator device string.
            draft_model_id: Optional draft model identifier for speculative decoding.
            draft_device: Draft model accelerator device string.
            gamma: Speculative lookahead token count.

        Returns:
            Deterministic colon-separated cache key string.
        """
        return f"{model_id}:{device}:{draft_model_id or 'none'}:{draft_device}:{gamma}"

    @property
    def model_dirs(self) -> List[Path]:
        """Return resolved model search directory paths.

        Returns:
            List of Path objects for model lookup.
        """
        return list(self._model_dirs)

    def add_model_dir(self, directory: Union[str, Path]) -> None:
        """Register an additional model search directory path.

        Args:
            directory: Directory path to append to lookup list.
        """
        resolved = Path(directory).resolve()
        with self._lock:
            if resolved not in self._model_dirs:
                self._model_dirs.append(resolved)
                logger.info(f"Registered model directory: {resolved}")

    def resolve_model_path(self, model_id: str) -> Path:
        """Resolve a model name or filesystem path to an existing directory containing OpenVINO models.

        Args:
            model_id: Model identifier name or direct filesystem directory path.

        Returns:
            Resolved absolute Path pointing to valid model directory.

        Raises:
            ModelNotFoundError: When model directory is not found in search paths.
        """
        direct_path = Path(model_id).resolve()
        if direct_path.is_dir():
            try:
                return ensure_path_allowed(direct_path, allowed_roots(self._model_dirs))
            except ArcRunnerError:
                logger.warning("Refusing a model directory outside the allowed roots", extra={"path": str(direct_path)})

        for base_dir in self._model_dirs:
            for name in (model_id, model_id.split(":")[0]):
                try:
                    candidate = resolve_inside(base_dir, name)
                except ArcRunnerError:
                    continue
                if candidate.is_dir():
                    return candidate

        search_str = ", ".join(str(d) for d in self._model_dirs)
        raise ModelNotFoundError(model_id, search_str)

    @staticmethod
    def _build_kv_manager(model_path: Path) -> KVCacheManager:
        """Build the KV manager for one model with its resolved context window; nothing is preallocated.

        Args:
            model_path: Resolved target model directory.

        Returns:
            KVCacheManager: Manager sized from ARC_CONTEXT_WINDOW, config.json or the default.

        Raises:
            ConfigurationError: If ARC_CONTEXT_WINDOW is invalid.
        """
        return KVCacheManager(
            max_context_length=resolve_context_window(model_path),
            geometry=read_kv_geometry(model_path),
            kv_precision=os.environ.get(KV_PRECISION_ENV, "").strip(),
        )

    def get_or_load_engine(
        self,
        model_id: str,
        device: str = "GPU",
        draft_model_id: Optional[str] = None,
        draft_device: str = "NPU",
        gamma: int = 4,
    ) -> BaseEngine:
        """Retrieve a cached engine or compile and load an engine instance.

        Args:
            model_id: Target model identifier or path.
            device: Target accelerator silicon identifier ('GPU', 'NPU', 'CPU').
            draft_model_id: Optional draft model identifier for hybrid speculative decoding.
            draft_device: Accelerator silicon identifier for speculative draft model.
            gamma: Lookahead draft verification tokens count.

        Returns:
            Ready-to-execute BaseEngine instance.

        Raises:
            ModelNotFoundError: When target or draft model directory does not exist.
            ModelLoadError: When compilation on target hardware fails.
        """
        cache_key = f"{model_id}:{device}:{draft_model_id or 'none'}:{draft_device}:{gamma}"

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                logger.debug(f"Cache hit for engine key: {cache_key}")
                return self._cache[cache_key]

            target_path = self.resolve_model_path(model_id)

            if len(self._cache) >= self._max_loaded_models:
                evicted_key, evicted_engine = self._cache.popitem(last=False)
                logger.info(f"Evicting least recently used engine: {evicted_key}")
                try:
                    evicted_engine.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading evicted engine {evicted_key}: {exc}")

            if draft_model_id:
                draft_path = self.resolve_model_path(draft_model_id)
                logger.info(f"Compiling HybridSpeculativeEngine (Target: {target_path} on {device}, Draft: {draft_path} on {draft_device})")
                spec_engine = HybridSpeculativeEngine(kv_cache_manager=self._build_kv_manager(target_path))
                spec_config = SpeculativeConfig(
                    draft_model_path=str(draft_path),
                    target_device=device,
                    draft_device=draft_device,
                    num_assistant_tokens=gamma,
                )
                spec_engine.load_model(target_path, spec_config)
                self._cache[cache_key] = spec_engine
                return spec_engine
            else:
                logger.info(f"Compiling OpenVINOGenAIEngine (Target: {target_path} on {device})")
                engine = OpenVINOGenAIEngine(lora_manager=self.lora_manager, kv_manager=self._build_kv_manager(target_path))
                engine.load_model(target_path, device=device)
                self._cache[cache_key] = engine
                return engine

    def get_or_load_embedding_engine(
        self,
        model_id: str,
        device: str = "NPU",
    ) -> EmbeddingEngine:
        """Retrieve a cached embedding engine or compile and load an instance.

        Args:
            model_id: Target embedding model identifier or path.
            device: Hardware device target ('NPU', 'GPU', 'CPU').

        Returns:
            Ready-to-execute EmbeddingEngine instance.

        Raises:
            ModelNotFoundError: When embedding model directory does not exist.
            ModelLoadError: When compilation on target hardware fails.
        """
        cache_key = f"{model_id}:{device}"

        with self._lock:
            if cache_key in self._embedding_cache:
                self._embedding_cache.move_to_end(cache_key)
                logger.debug(f"Cache hit for embedding engine key: {cache_key}")
                return self._embedding_cache[cache_key]

            target_path = self.resolve_model_path(model_id)

            if len(self._embedding_cache) >= self._max_loaded_models:
                evicted_key, evicted_emb = self._embedding_cache.popitem(last=False)
                logger.info(f"Evicting least recently used embedding engine: {evicted_key}")
                try:
                    evicted_emb.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading evicted embedding engine {evicted_key}: {exc}")

            logger.info(f"Constructing EmbeddingEngine (Target: {target_path} on {device})")
            engine = EmbeddingEngine(str(target_path), device=device)
            self._embedding_cache[cache_key] = engine
            return engine

    def get_or_load_vision_engine(
        self,
        model_id: str,
        device: str = "GPU",
    ) -> VisionEngine:
        """Retrieve a cached VisionEngine or compile and load an instance.

        Args:
            model_id: Target multimodal vision model identifier or path.
            device: Hardware device target ('GPU', 'CPU', 'NPU').

        Returns:
            Ready-to-execute VisionEngine instance.

        Raises:
            ModelNotFoundError: When vision model directory does not exist.
            ModelLoadError: When compilation on target hardware fails.
        """
        cache_key = f"{model_id}:{device}"

        with self._lock:
            if cache_key in self._vision_cache:
                self._vision_cache.move_to_end(cache_key)
                logger.debug(f"Cache hit for vision engine key: {cache_key}")
                return self._vision_cache[cache_key]

            target_path = self.resolve_model_path(model_id)

            if len(self._vision_cache) >= self._max_loaded_models:
                evicted_key, evicted_vis = self._vision_cache.popitem(last=False)
                logger.info(f"Evicting least recently used vision engine: {evicted_key}")
                try:
                    evicted_vis.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading evicted vision engine {evicted_key}: {exc}")

            logger.info(f"Constructing VisionEngine (Target: {target_path} on {device})")
            engine = VisionEngine(str(target_path), device=device)
            self._vision_cache[cache_key] = engine
            return engine

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Scan registered directories and return list of discoverable models.

        Returns:
            List of dictionaries containing model metadata and filesystem properties.
        """
        results: List[Dict[str, Any]] = []
        discovered_names = set()

        with self._lock:
            for base_dir in self._model_dirs:
                if not base_dir.exists():
                    continue
                for item in base_dir.iterdir():
                    if item.is_dir() and item.name not in discovered_names:
                        xml_files = list(item.glob("*.xml"))
                        if xml_files:
                            total_size = sum(f.stat().st_size for f in item.glob("*") if f.is_file())
                            mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
                            digest_hash = hashlib.sha256(item.name.encode("utf-8")).hexdigest()[:12]
                            discovered_names.add(item.name)
                            results.append({
                                "name": item.name,
                                "model": item.name,
                                "modified_at": mtime,
                                "size": total_size,
                                "digest": digest_hash,
                                "path": str(item),
                                "details": {
                                    "parent_model": "",
                                    "format": "openvino",
                                    "family": "llama" if "llama" in item.name.lower() else "qwen" if "qwen" in item.name.lower() else "custom",
                                    "parameter_size": "3B" if "3b" in item.name.lower() else "1.5B" if "1.5b" in item.name.lower() else "7B" if "7b" in item.name.lower() else "unknown",
                                    "quantization_level": "INT4" if "int4" in item.name.lower() else "INT8" if "int8" in item.name.lower() else "FP16",
                                }
                            })
        return results

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a specific model identifier.

        Args:
            model_id: Target model identifier.

        Returns:
            Metadata dictionary for model details endpoint.
        """
        resolved_path = self.resolve_model_path(model_id)
        name = resolved_path.name
        return {
            "modelfile": f"# Intel Arc OpenVINO Runner Modelfile\nFROM {name}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\n",
            "parameters": "temperature 0.7\ntop_p 0.9\nrepeat_penalty 1.1",
            "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
            "details": {
                "parent_model": "",
                "format": "openvino",
                "family": "llama" if "llama" in name.lower() else "qwen" if "qwen" in name.lower() else "custom",
                "parameter_size": "3B" if "3b" in name.lower() else "1.5B" if "1.5b" in name.lower() else "7B" if "7b" in name.lower() else "unknown",
                "quantization_level": "INT4" if "int4" in name.lower() else "INT8" if "int8" in name.lower() else "FP16",
            }
        }

    def unload_model(self, model_id: str) -> bool:
        """Unload and remove engine instances associated with model_id from all caches.

        Args:
            model_id: Model identifier to evict from LLM, embedding, and vision caches.

        Returns:
            True if any engine instance was evicted, False otherwise.
        """
        evicted = False
        prefix = f"{model_id}:"
        with self._lock:
            for k in [k for k in self._cache if k.startswith(prefix)]:
                engine = self._cache.pop(k)
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading engine {k}: {exc}")
                evicted = True
            for k in [k for k in self._embedding_cache if k.startswith(prefix)]:
                engine = self._embedding_cache.pop(k)
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading embedding engine {k}: {exc}")
                evicted = True
            for k in [k for k in self._vision_cache if k.startswith(prefix)]:
                engine = self._vision_cache.pop(k)
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading vision engine {k}: {exc}")
                evicted = True
        return evicted

    def unload_all(self) -> None:
        """Unload and release all compiled engine instances across all caches."""
        with self._lock:
            for key, engine in list(self._cache.items()):
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning(f"Error during unload_all for {key}: {exc}")
            self._cache.clear()
            for key, engine in list(self._embedding_cache.items()):
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning(f"Error during unload_all (embedding) for {key}: {exc}")
            self._embedding_cache.clear()
            for key, engine in list(self._vision_cache.items()):
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning(f"Error during unload_all (vision) for {key}: {exc}")
            self._vision_cache.clear()

    def record_generation_metrics(self, tokens_generated: int) -> None:
        """Atomically increment server request and token counters.

        Args:
            tokens_generated: Output tokens count produced in completion.
        """
        with self._meta_lock:
            self._total_requests_served += 1
            self._total_tokens_generated += tokens_generated

    def get_telemetry(self) -> Dict[str, Any]:
        """Retrieve aggregated server runtime telemetry.

        Returns:
            Dictionary containing active loaded models count and cumulative stats.
        """
        with self._lock:
            loaded = len(self._cache)
            keys = list(self._cache.keys())
        with self._meta_lock:
            return {
                "loaded_models_count": loaded,
                "loaded_keys": keys,
                "total_requests_served": self._total_requests_served,
                "total_tokens_generated": self._total_tokens_generated,
            }
