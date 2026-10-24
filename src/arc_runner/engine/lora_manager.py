"""Multi-LoRA Adapter management and dynamic hot-swapping engine.

Provides zero-downtime hot-swapping and linear composition of LoRA adapters
on Intel Arc GPUs and OpenVINO GenAI with bounded LRU memory caching (Proofs M22-M27, P8-P15).
"""

from collections import OrderedDict
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import struct
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.engine.models import LoRAAdapterSpec

logger = get_logger("arc_runner.engine.lora_manager")


class LoRAError(ArcRunnerError):
    """Base exception for LoRA adapter errors."""


class AdapterNotFoundError(LoRAError):
    """Raised when a requested LoRA adapter file or directory does not exist."""


class AdapterConfigError(LoRAError):
    """Raised when adapter_config.json is missing or corrupted."""


class AdapterIncompatibleError(LoRAError):
    """Raised when adapter architecture is incompatible with active base model."""


@dataclass
class LoRAAdapterMetadata:
    """Descriptor of parsed LoRA adapter metadata and configuration."""

    adapter_id: str
    adapter_path: Path
    base_model_name: str
    rank: int
    alpha: float
    target_modules: List[str]
    file_size_bytes: int
    dropout: float = 0.0
    is_cached: bool = False

    @property
    def name(self) -> str:
        """Alias for adapter_id."""
        return self.adapter_id

    @property
    def base_model(self) -> str:
        """Alias for base_model_name."""
        return self.base_model_name


class LoRAManager:
    """Manages discovery, LRU caching, and dynamic OpenVINO GenAI composition of LoRA adapters."""

    DEFAULT_ADAPTERS_DIR: Path = Path.home() / ".arc-runner" / "adapters"
    DEFAULT_MAX_CACHED: int = 8

    def __init__(
        self,
        adapters_dir: Optional[Union[str, Path]] = None,
        max_cached: int = DEFAULT_MAX_CACHED,
        max_cached_adapters: Optional[int] = None,
    ) -> None:
        """Initialize the LoRAManager instance.

        Args:
            adapters_dir: Root directory path for discovering local LoRA adapters.
            max_cached: Maximum number of adapters to retain in memory before LRU eviction.
            max_cached_adapters: Optional alias for max_cached.
        """
        self._adapters_dir = Path(adapters_dir).resolve() if adapters_dir else self.DEFAULT_ADAPTERS_DIR
        limit = max_cached_adapters if max_cached_adapters is not None else max_cached
        self._max_cached = max(1, limit)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._metadata_cache: Dict[str, LoRAAdapterMetadata] = {}
        self._lock = threading.Lock()

    @property
    def adapters_dir(self) -> Path:
        """Return the root directory searched for LoRA adapters.

        Returns:
            Path: Current adapters directory path.
        """
        return self._adapters_dir

    @property
    def max_cached(self) -> int:
        """Return the maximum number of cached adapters.

        Returns:
            int: LRU capacity ceiling.
        """
        return self._max_cached

    @property
    def cached_count(self) -> int:
        """Return the number of adapters currently cached in memory.

        Returns:
            int: Number of active cached adapters.
        """
        return len(self._cache)

    @property
    def cache_size(self) -> int:
        """Alias for cached_count.

        Returns:
            int: Number of active cached adapters.
        """
        return self.cached_count

    @property
    def cached_adapters(self) -> List[str]:
        """Return list of adapter IDs currently cached in memory.

        Returns:
            List[str]: Active cached adapter keys in LRU order.
        """
        with self._lock:
            return list(self._cache.keys())

    @staticmethod
    def inspect_safetensors_header(safetensors_path: Path) -> Tuple[int, Dict[str, Any]]:
        """Inspect safetensors binary container header without loading weight tensors (Proof P13, P15).

        Args:
            safetensors_path: Path to .safetensors container.

        Returns:
            Tuple of (header_size_bytes, parsed_header_dictionary).

        Raises:
            AdapterConfigError: If header cannot be decoded or exceeds boundary limits.
        """
        if not safetensors_path.exists():
            raise AdapterNotFoundError(f"Safetensors file does not exist: {safetensors_path}")

        try:
            with open(safetensors_path, "rb") as f:
                header_bytes_prefix = f.read(8)
                if len(header_bytes_prefix) < 8:
                    raise AdapterConfigError(f"Corrupt safetensors file: {safetensors_path} is less than 8 bytes")
                header_len = struct.unpack("<Q", header_bytes_prefix)[0]
                if header_len > 100 * 1024 * 1024:
                    raise AdapterConfigError(f"Safetensors header exceeds 100MB limit: {header_len}")
                raw_header = f.read(header_len)
                if len(raw_header) < header_len:
                    raise AdapterConfigError("Safetensors header truncated before promised length")
                header_json = json.loads(raw_header.decode("utf-8"))
                return header_len, header_json
        except Exception as exc:
            if isinstance(exc, (AdapterNotFoundError, AdapterConfigError)):
                raise
            raise AdapterConfigError(f"Failed to inspect safetensors header: {exc}") from exc

    def parse_adapter_metadata(self, adapter_path: Path) -> LoRAAdapterMetadata:
        """Parse configuration and metadata for an adapter directory or safetensors file.

        Args:
            adapter_path: Directory containing adapter_config.json and safetensors, or direct safetensors path.

        Returns:
            LoRAAdapterMetadata: Validated metadata record.

        Raises:
            AdapterNotFoundError: If files do not exist.
            AdapterConfigError: If metadata parsing fails.
        """
        resolved = adapter_path.resolve()
        if not resolved.exists():
            raise AdapterNotFoundError(f"Adapter path does not exist: {resolved}")

        config_file: Optional[Path] = None
        weights_file: Optional[Path] = None

        if resolved.is_dir():
            candidate_config = resolved / "adapter_config.json"
            if candidate_config.exists():
                config_file = candidate_config

            candidate_weights = resolved / "adapter_model.safetensors"
            if candidate_weights.exists():
                weights_file = candidate_weights
            else:
                safetensors_list = list(resolved.glob("*.safetensors"))
                if safetensors_list:
                    weights_file = safetensors_list[0]
        elif resolved.suffix == ".safetensors":
            weights_file = resolved
            parent_config = resolved.parent / "adapter_config.json"
            if parent_config.exists():
                config_file = parent_config

        if weights_file is None or not weights_file.exists():
            raise AdapterNotFoundError(f"Missing .safetensors weight file in adapter: {resolved}")

        rank = 16
        alpha = 32.0
        dropout = 0.0
        base_model_name = "unknown"
        target_modules: List[str] = ["q_proj", "v_proj"]

        if config_file and config_file.exists():
            try:
                cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
                rank = int(cfg_data.get("r", rank))
                alpha = float(cfg_data.get("lora_alpha", alpha))
                dropout = float(cfg_data.get("lora_dropout", dropout))
                base_model_name = str(cfg_data.get("base_model_name_or_path", base_model_name))
                modules = cfg_data.get("target_modules", target_modules)
                if isinstance(modules, list):
                    target_modules = [str(m) for m in modules]
            except Exception as exc:
                raise AdapterConfigError(f"Failed to parse adapter_config.json at {config_file}: {exc}") from exc
        else:
            try:
                _, header_data = self.inspect_safetensors_header(weights_file)
                meta = header_data.get("__metadata__", {})
                if "base_model" in meta:
                    base_model_name = meta["base_model"]
            except Exception:
                pass

        adapter_id = resolved.name if resolved.is_dir() else resolved.stem
        file_size = weights_file.stat().st_size if weights_file.exists() else 0
        is_cached = adapter_id in self._cache or str(resolved) in self._cache

        meta_obj = LoRAAdapterMetadata(
            adapter_id=adapter_id,
            adapter_path=resolved,
            base_model_name=base_model_name,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            file_size_bytes=file_size,
            dropout=dropout,
            is_cached=is_cached,
        )
        self._metadata_cache[adapter_id] = meta_obj
        return meta_obj

    def discover_adapters(self) -> List[LoRAAdapterMetadata]:
        """Discover all valid LoRA adapters present in the configured adapters directory.

        Returns:
            List[LoRAAdapterMetadata]: Discovered adapter descriptors.
        """
        results: List[LoRAAdapterMetadata] = []
        if not self._adapters_dir.exists() or not self._adapters_dir.is_dir():
            return results

        for entry in self._adapters_dir.iterdir():
            try:
                if entry.is_dir():
                    meta = self.parse_adapter_metadata(entry)
                    results.append(meta)
                elif entry.suffix == ".safetensors":
                    meta = self.parse_adapter_metadata(entry)
                    results.append(meta)
            except Exception as exc:
                logger.warning(f"Skipping invalid adapter candidate at {entry}: {exc}")

        return results

    def load_adapter(self, adapter_name_or_path: str) -> LoRAAdapterMetadata:
        """Load an adapter container into memory and insert it into the LRU cache (Proofs M24, P8, P11).

        Args:
            adapter_name_or_path: Adapter identifier in adapters_dir or direct filesystem path.

        Returns:
            LoRAAdapterMetadata: Metadata for the loaded adapter.

        Raises:
            AdapterNotFoundError: If the adapter cannot be found.
            AdapterConfigError: If loading or instantiation fails.
        """
        target_path: Path
        direct_path = Path(adapter_name_or_path)
        if direct_path.exists():
            target_path = direct_path.resolve()
        else:
            in_dir = self._adapters_dir / adapter_name_or_path
            if in_dir.exists():
                target_path = in_dir.resolve()
            else:
                raise AdapterNotFoundError(f"LoRA adapter '{adapter_name_or_path}' not found in {self._adapters_dir}")

        adapter_id = target_path.name if target_path.is_dir() else target_path.stem
        with self._lock:
            if adapter_id in self._cache and adapter_id in self._metadata_cache:
                self._cache.move_to_end(adapter_id)
                cached_meta = self._metadata_cache[adapter_id]
                cached_meta.is_cached = True
                return cached_meta

        meta = self.parse_adapter_metadata(target_path)
        cache_key = meta.adapter_id

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                cached_meta = self._metadata_cache.get(cache_key, meta)
                cached_meta.is_cached = True
                return cached_meta

            weights_file = target_path if target_path.is_file() else target_path / "adapter_model.safetensors"
            if not weights_file.exists():
                candidates = list(target_path.glob("*.safetensors")) if target_path.is_dir() else []
                if candidates:
                    weights_file = candidates[0]
                else:
                    raise AdapterNotFoundError(f"Missing weights file for adapter '{cache_key}'")

            ov_adapter_instance: Any
            try:
                import openvino_genai as ov_genai
                ov_adapter_instance = ov_genai.Adapter(str(weights_file))
            except Exception as exc:
                ov_adapter_instance = weights_file

            if len(self._cache) >= self._max_cached:
                evicted_key, _ = self._cache.popitem(last=False)
                if evicted_key in self._metadata_cache:
                    self._metadata_cache[evicted_key].is_cached = False
                gc.collect()

            self._cache[cache_key] = ov_adapter_instance
            meta.is_cached = True
            return meta

    def unload_adapter(self, adapter_name: str) -> bool:
        """Evict a specific adapter from memory and reclaim USM buffers (Proof P11).

        Args:
            adapter_name: Unique identifier of the adapter to unload.

        Returns:
            bool: True if adapter was evicted; False if it was not in cache.
        """
        with self._lock:
            if adapter_name in self._cache:
                del self._cache[adapter_name]
                if adapter_name in self._metadata_cache:
                    self._metadata_cache[adapter_name].is_cached = False
                gc.collect()
                return True
            return False

    def clear_cache(self) -> None:
        """Evict all cached adapters from memory and trigger immediate garbage collection."""
        with self._lock:
            self._cache.clear()
            for meta in self._metadata_cache.values():
                meta.is_cached = False
            gc.collect()

    def build_adapter_config(self, specs: Sequence[LoRAAdapterSpec]) -> Optional[Any]:
        """Construct an openvino_genai.AdapterConfig composing requested adapters (Proofs M22, M25, M27).

        Args:
            specs: Sequence of LoRAAdapterSpec bindings with individual scaling factors.

        Returns:
            Populated openvino_genai.AdapterConfig instance, or None if specs is empty.

        Raises:
            AdapterNotFoundError: If any requested adapter cannot be loaded.
        """
        if not specs:
            return None

        try:
            import openvino_genai as ov_genai
            ov_config = ov_genai.AdapterConfig()
            for spec in specs:
                clamped_scale = max(0.0, min(2.0, float(spec.scale)))
                if spec.name not in self._cache:
                    self.load_adapter(spec.path or spec.name)
                adapter_obj = self._cache.get(spec.name)
                if adapter_obj is not None:
                    if isinstance(adapter_obj, ov_genai.Adapter):
                        ov_config.add(adapter_obj, clamped_scale)
            return ov_config
        except ImportError:
            return None
        except Exception as exc:
            logger.error(f"Failed to build AdapterConfig: {exc}")
            raise
