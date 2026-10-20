"""Curated catalog and registry mapping for official Intel-optimized OpenVINO IR models.

Provides alias resolution, architectural metadata, parameter counts, and quantization
specifications for verified models on Intel Arc GPUs and Intel AI Boost NPUs.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CuratedModelInfo(BaseModel):
    """Architectural and repository descriptor for curated OpenVINO models."""

    alias: str = Field(description="Human-friendly lookup alias")
    repo_id: str = Field(description="Canonical Hugging Face repository identifier")
    parameter_size: str = Field(description="Model parameter count estimate")
    quantization: str = Field(description="Quantization precision and grouping format")
    family: str = Field(description="Model architectural family (llama, qwen, phi)")
    description: str = Field(description="Concise model summary description")
    target_device: str = Field(description="Recommended silicon target (GPU, NPU, HYBRID)")
    size_bytes: int = Field(description="Estimated on-disk weight footprint in bytes")
    context_window: int = Field(description="Maximum native context window tokens")
    sha256_checksum: Optional[str] = Field(default=None, description="Published reference checksum")

    @property
    def model_id(self) -> str:
        """Alias property matching model_id contract."""
        return self.alias


class ModelRegistry:
    """Registry maintaining curated Intel-optimized models and alias resolution logic."""

    _MODELS: Dict[str, CuratedModelInfo] = {
        "llama-3.2-3b": CuratedModelInfo(
            alias="llama-3.2-3b",
            repo_id="OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
            parameter_size="3.21B",
            quantization="INT4 (Symmetric, Group 128)",
            family="llama",
            description="Llama 3.2 3B Instruct optimized with INT4 weights for Intel Arc GPUs",
            target_device="GPU",
            size_bytes=1820000000,
            context_window=131072,
        ),
        "llama-3.2-1b": CuratedModelInfo(
            alias="llama-3.2-1b",
            repo_id="OpenVINO/Llama-3.2-1B-Instruct-int4-ov",
            parameter_size="1.23B",
            quantization="INT4 (Symmetric, Group 128)",
            family="llama",
            description="Ultra-compact Llama 3.2 1B Instruct ideal for 5W Intel NPU speculative draft",
            target_device="NPU",
            size_bytes=740000000,
            context_window=131072,
        ),
        "qwen-2.5-3b": CuratedModelInfo(
            alias="qwen-2.5-3b",
            repo_id="OpenVINO/Qwen2.5-3B-Instruct-int4-ov",
            parameter_size="3.09B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="Qwen 2.5 3B Instruct optimized for coding and multilingual tasks on Arc GPUs",
            target_device="GPU",
            size_bytes=1910000000,
            context_window=32768,
        ),
        "qwen-2.5-1.5b": CuratedModelInfo(
            alias="qwen-2.5-1.5b",
            repo_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
            parameter_size="1.54B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="Qwen 2.5 1.5B Instruct lightweight model for Intel NPU cooperative drafting",
            target_device="NPU",
            size_bytes=940000000,
            context_window=32768,
        ),
        "qwen-2.5-7b": CuratedModelInfo(
            alias="qwen-2.5-7b",
            repo_id="OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
            parameter_size="7.61B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="Qwen 2.5 7B Instruct for discrete Intel Arc GPUs (A770, A750, B580)",
            target_device="GPU",
            size_bytes=4420000000,
            context_window=32768,
        ),
        "phi-3.5-mini": CuratedModelInfo(
            alias="phi-3.5-mini",
            repo_id="OpenVINO/Phi-3.5-mini-instruct-int4-ov",
            parameter_size="3.82B",
            quantization="INT4 (Symmetric, Group 128)",
            family="phi",
            description="Microsoft Phi 3.5 mini with 128k context support on Intel Arc silicon",
            target_device="GPU",
            size_bytes=2210000000,
            context_window=131072,
        ),
        "deepseek-r1-1.5b": CuratedModelInfo(
            alias="deepseek-r1-1.5b",
            repo_id="OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
            parameter_size="1.77B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen",
            description="DeepSeek R1 reasoning model distilled onto Qwen 1.5B architecture",
            target_device="HYBRID",
            size_bytes=1120000000,
            context_window=32768,
        ),
        "nomic-embed-text": CuratedModelInfo(
            alias="nomic-embed-text",
            repo_id="OpenVINO/nomic-embed-text-v1.5-fp16-ov",
            parameter_size="137M",
            quantization="FP16 (OpenVINO IR)",
            family="nomic",
            description="Nomic Embed Text v1.5 high-performance dense embedding model for Intel NPU @ 5W",
            target_device="NPU",
            size_bytes=274000000,
            context_window=8192,
        ),
        "bge-m3": CuratedModelInfo(
            alias="bge-m3",
            repo_id="OpenVINO/bge-m3-fp16-ov",
            parameter_size="567M",
            quantization="FP16 (OpenVINO IR)",
            family="bge",
            description="BAAI BGE-M3 multi-lingual, multi-granularity embedding model for RAG systems",
            target_device="NPU",
            size_bytes=1140000000,
            context_window=8192,
        ),
        "all-minilm-l6-v2": CuratedModelInfo(
            alias="all-minilm-l6-v2",
            repo_id="OpenVINO/all-MiniLM-L6-v2-fp16-ov",
            parameter_size="22.7M",
            quantization="FP16 (OpenVINO IR)",
            family="minilm",
            description="Ultra-fast lightweight sentence embedding model (384 dimensions) for local RAG",
            target_device="NPU",
            size_bytes=45000000,
            context_window=512,
        ),
        "llama-3.2-11b-vision": CuratedModelInfo(
            alias="llama-3.2-11b-vision",
            repo_id="OpenVINO/Llama-3.2-11B-Vision-Instruct-int4-ov",
            parameter_size="10.6B",
            quantization="INT4 (Symmetric, Group 128)",
            family="llama-vision",
            description="Llama 3.2 11B Vision Instruct multimodal model accelerated on Arc Xe-LPG GPUs",
            target_device="GPU",
            size_bytes=6200000000,
            context_window=131072,
        ),
        "qwen2-vl-7b": CuratedModelInfo(
            alias="qwen2-vl-7b",
            repo_id="OpenVINO/Qwen2-VL-7B-Instruct-int4-ov",
            parameter_size="7.6B",
            quantization="INT4 (Symmetric, Group 128)",
            family="qwen-vl",
            description="Qwen2-VL 7B Instruct vision-language model for document and visual reasoning",
            target_device="GPU",
            size_bytes=4500000000,
            context_window=32768,
        ),
    }

    @classmethod
    def resolve(cls, alias_or_repo: str) -> CuratedModelInfo:
        """Resolve a model name alias or direct Hugging Face repository path.

        Args:
            alias_or_repo: Short alias name or full organization/model repository string.

        Returns:
            CuratedModelInfo populated with target repository coordinates.
        """
        clean_key = alias_or_repo.strip().lower()
        if clean_key in cls._MODELS:
            return cls._MODELS[clean_key]

        tag_clean = clean_key.split(":")[0]
        if tag_clean in cls._MODELS:
            return cls._MODELS[tag_clean]

        family = "custom"
        if "llama" in clean_key:
            family = "llama"
        elif "qwen" in clean_key:
            family = "qwen"
        elif "phi" in clean_key:
            family = "phi"

        return CuratedModelInfo(
            alias=alias_or_repo,
            repo_id=alias_or_repo,
            parameter_size="Custom",
            quantization="OpenVINO IR",
            family=family,
            description=f"Custom OpenVINO model repository: {alias_or_repo}",
            target_device="GPU",
            size_bytes=2000000000,
            context_window=4096,
        )

    @classmethod
    def get_all(cls) -> Dict[str, CuratedModelInfo]:
        """Retrieve copy of all registered models mapped by slug alias.

        Returns:
            Dictionary mapping alias strings to CuratedModelInfo instances.
        """
        return cls._MODELS.copy()

    @classmethod
    def list_curated(cls) -> List[CuratedModelInfo]:
        """Retrieve ordered list of all registered curated models.

        Returns:
            List of CuratedModelInfo models available for immediate pull.
        """
        return list(cls._MODELS.values())

    @classmethod
    def is_curated(cls, model_id: str) -> bool:
        """Check whether a model identifier matches a known curated alias.

        Args:
            model_id: Identifier string to evaluate.

        Returns:
            True if identifier exists in curated registry, False otherwise.
        """
        clean_key = model_id.strip().lower().split(":")[0]
        return clean_key in cls._MODELS
