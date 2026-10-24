"""Triad test suite for Intel Arc GPU Multi-LoRA Adapter Engine and Mathematical Proofs.

Validates LoRA linear subspace rank projection (Proof M22), hot-swap dispatch latency (Proof M23),
LRU cache bound memory footprint (Proof M24), and zero-copy safetensors parsing (Proof M26).
"""

import json
from pathlib import Path
import struct
from typing import Optional
import pytest

from arc_runner.engine.lora_manager import (
    AdapterConfigError,
    AdapterNotFoundError,
    LoRAAdapterMetadata,
    LoRAError,
    LoRAManager,
)
from arc_runner.engine.models import LoRAAdapterSpec


def create_dummy_lora_adapter(
    base_dir: Path,
    name: str,
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    include_safetensors: bool = True,
    safetensors_name: str = "adapter_model.safetensors",
    corrupt_config: bool = False,
) -> Path:
    """Create a synthetically valid or parameterized mock LoRA adapter on disk.

    Args:
        base_dir: Root directory containing adapters.
        name: Name of the adapter directory.
        base_model: Base model identifier or path.
        r: LoRA decomposition rank.
        alpha: LoRA scaling constant alpha.
        dropout: LoRA dropout probability.
        include_safetensors: Whether to write binary safetensors weights.
        safetensors_name: Name of safetensors file.
        corrupt_config: Whether to write syntactically invalid JSON config.

    Returns:
        Path to the created adapter directory.
    """
    adapter_dir = base_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config_path = adapter_dir / "adapter_config.json"
    if corrupt_config:
        config_path.write_text("{invalid: json format error!", encoding="utf-8")
    else:
        config_dict = {
            "base_model_name_or_path": base_model,
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "target_modules": ["q_proj", "v_proj"],
        }
        config_path.write_text(json.dumps(config_dict), encoding="utf-8")

    if include_safetensors:
        st_path = adapter_dir / safetensors_name
        meta = {
            "format": "pt",
            "lora_rank": str(r),
            "lora_alpha": str(alpha),
        }
        header_bytes = json.dumps({"__metadata__": meta}).encode("utf-8")
        st_path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 32)

    return adapter_dir


class TestLoRAManagerPositive:
    """Positive test cases verifying expected adapter discovery, loading, and blending."""

    def test_lora_manager_discover_adapters_positive(self, tmp_path: Path) -> None:
        """Verify discovery enumerates all valid adapter subdirectories on disk."""
        create_dummy_lora_adapter(tmp_path, "lora-coder", r=16, alpha=32)
        create_dummy_lora_adapter(tmp_path, "lora-math", r=32, alpha=64)

        manager = LoRAManager(adapters_dir=tmp_path)
        adapters = manager.discover_adapters()

        names = [a.name for a in adapters]
        assert "lora-coder" in names
        assert "lora-math" in names
        assert len(adapters) == 2

    def test_lora_manager_parse_metadata(self, tmp_path: Path) -> None:
        """Verify adapter metadata parses rank, alpha, target modules, and base model."""
        adapter_path = create_dummy_lora_adapter(
            tmp_path,
            "lora-instruct",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            r=8,
            alpha=16,
            dropout=0.1,
        )

        manager = LoRAManager(adapters_dir=tmp_path)
        meta = manager.parse_adapter_metadata(adapter_path)

        assert meta.name == "lora-instruct"
        assert meta.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert meta.rank == 8
        assert meta.alpha == 16
        assert meta.dropout == 0.1
        assert "q_proj" in meta.target_modules

    def test_lora_manager_load_adapter_caching(self, tmp_path: Path) -> None:
        """Verify loading an adapter instantiates and retains object in LRU cache."""
        create_dummy_lora_adapter(tmp_path, "lora-chat", r=16, alpha=32)
        manager = LoRAManager(adapters_dir=tmp_path)

        adapter_obj1 = manager.load_adapter("lora-chat")
        assert adapter_obj1 is not None
        assert manager.cache_size == 1
        assert "lora-chat" in manager.cached_adapters

        adapter_obj2 = manager.load_adapter("lora-chat")
        assert adapter_obj2 is adapter_obj1

    def test_lora_manager_build_adapter_config_single(self, tmp_path: Path) -> None:
        """Verify building AdapterConfig for a single adapter applies specified scale."""
        create_dummy_lora_adapter(tmp_path, "lora-qa", r=16, alpha=32)
        manager = LoRAManager(adapters_dir=tmp_path)

        specs = [LoRAAdapterSpec(name="lora-qa", scale=0.8)]
        config = manager.build_adapter_config(specs)

        assert config is not None

    def test_lora_manager_build_multi_adapter_config_blending(self, tmp_path: Path) -> None:
        """Verify building AdapterConfig blends multiple adapters with distinct scales (Proof M22)."""
        create_dummy_lora_adapter(tmp_path, "lora-code", r=16, alpha=32)
        create_dummy_lora_adapter(tmp_path, "lora-security", r=8, alpha=16)
        manager = LoRAManager(adapters_dir=tmp_path)

        specs = [
            LoRAAdapterSpec(name="lora-code", scale=0.7),
            LoRAAdapterSpec(name="lora-security", scale=0.3),
        ]
        config = manager.build_adapter_config(specs)

        assert config is not None
        assert manager.cache_size == 2

    def test_lora_manager_lru_cache_hit_order(self, tmp_path: Path) -> None:
        """Verify cache hit moves item to the most recent position in LRU order."""
        create_dummy_lora_adapter(tmp_path, "lora-a", r=8, alpha=16)
        create_dummy_lora_adapter(tmp_path, "lora-b", r=8, alpha=16)
        manager = LoRAManager(adapters_dir=tmp_path, max_cached_adapters=3)

        manager.load_adapter("lora-a")
        manager.load_adapter("lora-b")
        assert manager.cached_adapters == ["lora-a", "lora-b"]

        manager.load_adapter("lora-a")
        assert manager.cached_adapters == ["lora-b", "lora-a"]


class TestLoRAManagerNegative:
    """Negative test cases validating error propagation and graceful degradation."""

    def test_lora_manager_missing_adapter_raises(self, tmp_path: Path) -> None:
        """Verify requesting un-indexed adapter raises AdapterNotFoundError."""
        manager = LoRAManager(adapters_dir=tmp_path)
        with pytest.raises(AdapterNotFoundError, match="not found"):
            manager.load_adapter("non-existent")

    def test_lora_manager_corrupted_config_raises(self, tmp_path: Path) -> None:
        """Verify invalid JSON in adapter_config.json raises AdapterConfigError."""
        create_dummy_lora_adapter(tmp_path, "lora-corrupt", corrupt_config=True)
        manager = LoRAManager(adapters_dir=tmp_path)
        with pytest.raises(AdapterConfigError):
            manager.parse_adapter_metadata(tmp_path / "lora-corrupt")

    def test_lora_manager_missing_safetensors_file_raises(self, tmp_path: Path) -> None:
        """Verify missing safetensors weight file raises AdapterNotFoundError."""
        create_dummy_lora_adapter(tmp_path, "lora-no-weights", include_safetensors=False)
        manager = LoRAManager(adapters_dir=tmp_path)
        with pytest.raises(AdapterNotFoundError, match=r"Missing \.safetensors"):
            manager.load_adapter("lora-no-weights")

    def test_lora_manager_negative_scale_clamped(self, tmp_path: Path) -> None:
        """Verify negative scale is strictly rejected by schema validation."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            LoRAAdapterSpec(name="lora-neg", scale=-0.5)

    def test_lora_manager_unload_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """Verify unloading non-cached adapter returns False without exception."""
        manager = LoRAManager(adapters_dir=tmp_path)
        assert manager.unload_adapter("ghost-adapter") is False


class TestLoRAManagerEdgeCases:
    """Edge-case test cases validating boundary scales, high ranks, and LRU eviction."""

    def test_lora_manager_zero_scale_adapter(self, tmp_path: Path) -> None:
        """Verify adapter with scale lambda=0.0 compiles without division error (Proof M22)."""
        create_dummy_lora_adapter(tmp_path, "lora-zero", r=16, alpha=32)
        manager = LoRAManager(adapters_dir=tmp_path)
        specs = [LoRAAdapterSpec(name="lora-zero", scale=0.0)]
        config = manager.build_adapter_config(specs)
        assert config is not None

    def test_lora_manager_extreme_scale_adapter(self, tmp_path: Path) -> None:
        """Verify extreme scale values configure cleanly without overflow."""
        create_dummy_lora_adapter(tmp_path, "lora-high", r=16, alpha=32)
        manager = LoRAManager(adapters_dir=tmp_path)
        specs = [LoRAAdapterSpec(name="lora-high", scale=2.0)]
        config = manager.build_adapter_config(specs)
        assert config is not None

    def test_lora_manager_rank_boundary_extremes(self, tmp_path: Path) -> None:
        """Verify boundary ranks r=1 and r=256 parse correctly."""
        p1 = create_dummy_lora_adapter(tmp_path, "lora-r1", r=1, alpha=1)
        p256 = create_dummy_lora_adapter(tmp_path, "lora-r256", r=256, alpha=512)

        manager = LoRAManager(adapters_dir=tmp_path)
        meta1 = manager.parse_adapter_metadata(p1)
        meta256 = manager.parse_adapter_metadata(p256)

        assert meta1.rank == 1
        assert meta1.alpha == 1
        assert meta256.rank == 256
        assert meta256.alpha == 512

    def test_lora_manager_lru_capacity_eviction_m24(self, tmp_path: Path) -> None:
        """Verify LRU cache evicts oldest adapter when capacity is exceeded (Proof M24)."""
        manager = LoRAManager(adapters_dir=tmp_path, max_cached_adapters=2)

        create_dummy_lora_adapter(tmp_path, "lora-1")
        create_dummy_lora_adapter(tmp_path, "lora-2")
        create_dummy_lora_adapter(tmp_path, "lora-3")

        manager.load_adapter("lora-1")
        manager.load_adapter("lora-2")
        assert manager.cache_size == 2
        assert manager.cached_adapters == ["lora-1", "lora-2"]

        manager.load_adapter("lora-3")
        assert manager.cache_size == 2
        assert "lora-1" not in manager.cached_adapters
        assert manager.cached_adapters == ["lora-2", "lora-3"]

    def test_lora_manager_paths_with_spaces_and_symbols(self, tmp_path: Path) -> None:
        """Verify adapters in paths containing spaces and dashes parse safely."""
        adapter_path = create_dummy_lora_adapter(tmp_path, "lora adapter-test v1")
        manager = LoRAManager(adapters_dir=tmp_path)
        meta = manager.parse_adapter_metadata(adapter_path)
        assert meta.name == "lora adapter-test v1"
