"""Slice S-B tests: the GGUF parser must reject hostile length fields without crashing or exhausting memory."""

from pathlib import Path
import struct

import pytest

from arc_runner.core.exceptions import GGUFParseError
from arc_runner.quant.gguf_parser import GGUF_MAX_DIMENSIONS, GGUFParser


def _write(tmp_path: Path, payload: bytes) -> Path:
    """Write a candidate GGUF byte string to a temporary file."""
    path = tmp_path / "hostile.gguf"
    path.write_bytes(payload)
    return path


def _header(tensor_count: int, kv_count: int) -> bytes:
    """Build a GGUF v3 header with the given counts."""
    return b"GGUF" + struct.pack("<IQQ", 3, tensor_count, kv_count)


@pytest.mark.parametrize("declared", [2**31, 2**40, 2**63, 2**64 - 1])
def test_parse_all_metadata_key_length_exceeding_file_raises_parse_error(tmp_path: Path, declared: int) -> None:
    """G1: a huge declared string length yields GGUFParseError, not MemoryError or OverflowError."""
    path = _write(tmp_path, _header(0, 1) + struct.pack("<Q", declared) + b"abc")
    with pytest.raises(GGUFParseError):
        GGUFParser(path).parse_all()


@pytest.mark.parametrize("n_dims", [GGUF_MAX_DIMENSIONS + 1, 2**16, 2**31, 2**32 - 1])
def test_parse_all_tensor_dimension_count_over_limit_raises_parse_error(tmp_path: Path, n_dims: int) -> None:
    """G1: more than four dimensions is rejected before any allocation is attempted."""
    payload = _header(1, 0) + struct.pack("<Q", 1) + b"t" + struct.pack("<I", n_dims)
    with pytest.raises(GGUFParseError):
        GGUFParser(_write(tmp_path, payload)).parse_all()


def test_parse_all_tensor_count_larger_than_file_raises_parse_error(tmp_path: Path) -> None:
    """G1: a tensor count of 2**62 stops at end of file with GGUFParseError."""
    with pytest.raises(GGUFParseError):
        GGUFParser(_write(tmp_path, _header(2**62, 0))).parse_all()


def test_parse_all_string_exactly_filling_the_file_is_accepted(tmp_path: Path) -> None:
    """G1: a declared length equal to the remaining bytes is valid and not over-rejected."""
    payload = _header(0, 1) + struct.pack("<Q", 3) + b"key" + struct.pack("<I", 8) + struct.pack("<Q", 2) + b"hi"
    parser = GGUFParser(_write(tmp_path, payload))
    parser.parse_all()
    assert parser.metadata == {"key": "hi"}
