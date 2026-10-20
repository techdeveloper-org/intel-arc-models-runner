"""Exhaustive 10-test triad suite for binary GGUF container parsing.

Validates GGUF v2/v3 header decoding, recursive typed metadata deserialization,
tensor descriptor extraction, 32-byte memory alignment, and corruption handling.
"""

from pathlib import Path
import struct
import tempfile
from typing import Any
import pytest

from arc_runner.core.exceptions import GGUFParseError, InvalidGGUFError
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.models import GGMLType, GGUFHeader, GGUFSummary, GGUFValueType


def build_synthetic_gguf(
    magic: bytes = b"GGUF",
    version: int = 3,
    tensor_count: int = 2,
    metadata: dict[str, tuple[GGUFValueType, Any]] = None,
    tensors: list[dict[str, Any]] = None,
    truncate_bytes: int = 0,
) -> bytes:
    """Construct binary GGUF container byte sequence for test fixtures.

    Args:
        magic: 4-byte magic signature.
        version: GGUF version integer.
        tensor_count: Number of tensor descriptor entries.
        metadata: Dictionary mapping keys to (GGUFValueType, value) tuples.
        tensors: List of tensor definitions with name, dims, type, and offset.
        truncate_bytes: Number of bytes to slice from end to simulate truncation.

    Returns:
        Packed bytes adhering to GGUF v3 specification.
    """
    if metadata is None:
        metadata = {
            "general.architecture": (GGUFValueType.STRING, "llama"),
            "llama.context_length": (GGUFValueType.UINT32, 4096),
            "llama.embedding_length": (GGUFValueType.UINT32, 2048),
            "llama.block_count": (GGUFValueType.UINT32, 16),
            "llama.attention.head_count": (GGUFValueType.UINT32, 32),
            "general.alignment": (GGUFValueType.UINT32, 32),
        }

    if tensors is None:
        tensors = [
            {"name": "blk.0.attn_q.weight", "dims": (2048, 2048), "type": GGMLType.Q4_0, "offset": 0},
            {"name": "output.weight", "dims": (2048, 32000), "type": GGMLType.F16, "offset": 2097152},
        ]

    buf = bytearray()
    buf.extend(magic)
    buf.extend(struct.pack("<IQQ", version, len(tensors), len(metadata)))

    for k, (v_type, val) in metadata.items():
        k_bytes = k.encode("utf-8")
        buf.extend(struct.pack("<Q", len(k_bytes)))
        buf.extend(k_bytes)
        buf.extend(struct.pack("<I", v_type.value))

        if v_type == GGUFValueType.UINT8:
            buf.extend(struct.pack("<B", val))
        elif v_type == GGUFValueType.INT8:
            buf.extend(struct.pack("<b", val))
        elif v_type == GGUFValueType.UINT16:
            buf.extend(struct.pack("<H", val))
        elif v_type == GGUFValueType.INT16:
            buf.extend(struct.pack("<h", val))
        elif v_type == GGUFValueType.UINT32:
            buf.extend(struct.pack("<I", val))
        elif v_type == GGUFValueType.INT32:
            buf.extend(struct.pack("<i", val))
        elif v_type == GGUFValueType.FLOAT32:
            buf.extend(struct.pack("<f", val))
        elif v_type == GGUFValueType.BOOL:
            buf.extend(struct.pack("<B", 1 if val else 0))
        elif v_type == GGUFValueType.STRING:
            str_bytes = val.encode("utf-8")
            buf.extend(struct.pack("<Q", len(str_bytes)))
            buf.extend(str_bytes)
        elif v_type == GGUFValueType.UINT64:
            buf.extend(struct.pack("<Q", val))
        elif v_type == GGUFValueType.INT64:
            buf.extend(struct.pack("<q", val))
        elif v_type == GGUFValueType.FLOAT64:
            buf.extend(struct.pack("<d", val))
        elif v_type == GGUFValueType.ARRAY:
            elem_type, elem_list = val
            buf.extend(struct.pack("<IQ", elem_type.value, len(elem_list)))
            for elem in elem_list:
                if elem_type == GGUFValueType.STRING:
                    eb = elem.encode("utf-8")
                    buf.extend(struct.pack("<Q", len(eb)))
                    buf.extend(eb)
                elif elem_type == GGUFValueType.UINT32:
                    buf.extend(struct.pack("<I", elem))

    for t in tensors:
        t_name = t["name"].encode("utf-8")
        buf.extend(struct.pack("<Q", len(t_name)))
        buf.extend(t_name)
        dims = t["dims"]
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<IQ", t["type"].value, t["offset"]))

    if truncate_bytes > 0:
        return bytes(buf[:-truncate_bytes])
    return bytes(buf)


def test_parse_valid_gguf_v3_header() -> None:
    """Verify parsing valid GGUF v3 header extracts correct counts and version."""
    payload = build_synthetic_gguf(version=3)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        hdr = parser.header
        assert hdr.magic == b"GGUF"
        assert hdr.version == 3
        assert hdr.tensor_count == 2
        assert hdr.metadata_kv_count == 6
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_parse_metadata_all_types() -> None:
    """Verify recursive typed deserialization across primitive and array types."""
    meta = {
        "m_u8": (GGUFValueType.UINT8, 255),
        "m_i8": (GGUFValueType.INT8, -12),
        "m_u16": (GGUFValueType.UINT16, 65530),
        "m_i16": (GGUFValueType.INT16, -3200),
        "m_u32": (GGUFValueType.UINT32, 1000000),
        "m_i32": (GGUFValueType.INT32, -50000),
        "m_f32": (GGUFValueType.FLOAT32, 3.14159),
        "m_bool": (GGUFValueType.BOOL, True),
        "m_str": (GGUFValueType.STRING, "intel-arc-models-runner"),
        "m_u64": (GGUFValueType.UINT64, 18446744073709551610),
        "m_i64": (GGUFValueType.INT64, -9223372036854775800),
        "m_f64": (GGUFValueType.FLOAT64, 2.718281828459),
        "m_arr": (GGUFValueType.ARRAY, (GGUFValueType.STRING, ["opt1", "opt2", "opt3"])),
    }
    payload = build_synthetic_gguf(metadata=meta, tensors=[])
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        res = parser.metadata
        assert res["m_u8"] == 255
        assert res["m_i8"] == -12
        assert res["m_u16"] == 65530
        assert res["m_bool"] is True
        assert res["m_str"] == "intel-arc-models-runner"
        assert len(res["m_arr"]) == 3
        assert res["m_arr"][1] == "opt2"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_parse_tensor_descriptors_and_alignment() -> None:
    """Verify tensor descriptor shapes, types, and 32-byte alignment offset calculation."""
    payload = build_synthetic_gguf()
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        tensors = parser.tensor_info
        assert len(tensors) == 2
        assert tensors[0].name == "blk.0.attn_q.weight"
        assert tensors[0].dimensions == (2048, 2048)
        assert tensors[0].tensor_type == GGMLType.Q4_0
        assert tensors[1].name == "output.weight"
        assert tensors[1].tensor_type == GGMLType.F16
        assert parser.data_offset % 32 == 0
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_gguf_summary_quant_distribution() -> None:
    """Verify GGUF summary computes parameter totals and quantization distribution percentages."""
    payload = build_synthetic_gguf()
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        summary = parser.get_summary()
        assert summary.architecture == "llama"
        assert summary.context_length == 4096
        assert summary.total_parameters == (2048 * 2048) + (2048 * 32000)
        assert "Q4_0" in summary.quant_distribution
        assert "F16" in summary.quant_distribution
        assert summary.quant_distribution["Q4_0"] == 50.0
        assert summary.quant_distribution["F16"] == 50.0
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_invalid_magic_bytes_raises_error() -> None:
    """Verify InvalidGGUFError raised when magic header does not match 'GGUF'."""
    payload = build_synthetic_gguf(magic=b"NOPE")
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        with pytest.raises(InvalidGGUFError) as exc_info:
            parser.parse_all()
        assert "Expected magic b'GGUF'" in str(exc_info.value)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_unsupported_version_raises_error() -> None:
    """Verify InvalidGGUFError raised for future unsupported GGUF version numbers."""
    payload = build_synthetic_gguf(version=99)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        with pytest.raises(InvalidGGUFError) as exc_info:
            parser.parse_all()
        assert "Unsupported GGUF version 99" in str(exc_info.value)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_truncated_file_raises_eof_error() -> None:
    """Verify GGUFParseError raised when container file is abruptly truncated."""
    payload = build_synthetic_gguf(truncate_bytes=24)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        with pytest.raises(GGUFParseError):
            parser.parse_all()
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_zero_tensor_zero_metadata_gguf() -> None:
    """Verify GGUFParser handles edge case of empty container with 0 metadata and 0 tensors."""
    payload = build_synthetic_gguf(metadata={}, tensors=[])
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        summary = parser.get_summary()
        assert summary.tensor_count == 0
        assert summary.total_parameters == 0
        assert summary.quant_distribution == {}
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_giant_metadata_string_boundary() -> None:
    """Verify handling of large metadata string values up to 64KB."""
    large_string = "A" * 65536
    meta = {
        "large_text": (GGUFValueType.STRING, large_string),
    }
    payload = build_synthetic_gguf(metadata=meta, tensors=[])
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(payload)
        temp_path = f.name

    try:
        parser = GGUFParser(temp_path)
        assert parser.metadata["large_text"] == large_string
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_nonexistent_file_raises_invalid_gguf_error() -> None:
    """Verify InvalidGGUFError raised if target file path does not exist on disk."""
    parser = GGUFParser("C:/nonexistent/invalid_path_model.gguf")
    with pytest.raises(InvalidGGUFError) as exc_info:
        parser.parse_all()
    assert "File not found" in str(exc_info.value)
