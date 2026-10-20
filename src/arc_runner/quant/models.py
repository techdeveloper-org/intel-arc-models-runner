"""Data structures and type definitions for GGUF parsing and quantization.

Provides formal schemas for GGUF headers, metadata dictionaries, tensor descriptor tables,
and Ahead-of-Time compilation telemetry.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional


class GGUFValueType(IntEnum):
    """Enumeration of typed value descriptors supported by the GGUF specification."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """Enumeration of GGML tensor precision and block quantization schemes."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


@dataclass(frozen=True)
class GGUFHeader:
    """Header record parsed from the first 24 bytes of a GGUF container file."""

    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Descriptor table entry defining tensor dimensions, quantization, and memory offset."""

    name: str
    n_dimensions: int
    dimensions: tuple[int, ...]
    tensor_type: GGMLType
    offset: int
    size_bytes: int = 0


@dataclass(frozen=True)
class GGUFSummary:
    """High-level architectural summary derived from parsed GGUF metadata and tensor tables."""

    file_path: str
    file_size_bytes: int
    architecture: str
    context_length: int
    embedding_length: int
    block_count: int
    head_count: int
    tensor_count: int
    total_parameters: int
    quant_distribution: dict[str, float] = field(default_factory=dict)
    primary_quant_type: str = "UNKNOWN"


@dataclass(frozen=True)
class OllamaModelManifest:
    """Parsed representation of a local Ollama Docker/OCI distribution schema v2 manifest."""

    model_name: str
    tag: str
    manifest_path: str
    model_digest: str
    model_blob_path: Optional[str]
    model_size_bytes: int
    system_template: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AOTCompilationResult:
    """Telemetry report produced upon ahead-of-time model graph compilation."""

    model_id: str
    device: str
    compilation_time_ms: float
    warm_load_time_ms: float
    blob_size_bytes: int
    cache_path: str
    cache_hit: bool
    speedup_multiplier: float


@dataclass(frozen=True)
class CacheEntryInfo:
    """Metadata detailing a cached compiled blob entry on disk."""

    model_id: str
    device: str
    cache_path: str
    exists: bool
    size_bytes: int
    modified_timestamp: float
