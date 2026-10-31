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


def _kv_array(elem_type: int, declared_len: int, body: bytes) -> bytes:
    """Build a one-entry metadata block holding an array value."""
    return struct.pack("<Q", 1) + b"k" + struct.pack("<I", 9) + struct.pack("<IQ", elem_type, declared_len) + body


def test_parse_all_array_length_implying_more_bytes_than_remain_raises_before_allocating(tmp_path: Path) -> None:
    """TM-08: an array whose declared length cannot fit the remaining bytes is rejected up front."""
    payload = _header(0, 1) + _kv_array(10, 2**40, b"\x00" * 16)
    with pytest.raises(GGUFParseError, match="remain"):
        GGUFParser(_write(tmp_path, payload)).parse_all()


def test_parse_all_array_elements_over_total_budget_raises_parse_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-08: the sum of array elements across the file is capped, and exceeding the cap is an error, not a truncation."""
    from arc_runner.quant import gguf_parser

    monkeypatch.setattr(gguf_parser, "GGUF_MAX_TOTAL_ARRAY_ELEMENTS", 100)
    payload = _header(0, 1) + _kv_array(0, 101, b"\x01" * 101)
    with pytest.raises(GGUFParseError, match="element"):
        GGUFParser(_write(tmp_path, payload)).parse_all()


def test_parse_all_array_elements_within_total_budget_are_returned_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-08: an array exactly at the cap parses and keeps every element."""
    from arc_runner.quant import gguf_parser

    monkeypatch.setattr(gguf_parser, "GGUF_MAX_TOTAL_ARRAY_ELEMENTS", 100)
    parser = GGUFParser(_write(tmp_path, _header(0, 1) + _kv_array(0, 100, bytes(range(100)))))
    parser.parse_all()
    assert parser.metadata["k"] == list(range(100))


def test_parse_all_nested_empty_arrays_count_against_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-08: an outer array of empty inner arrays cannot dodge the cap by carrying no scalar elements."""
    from arc_runner.quant import gguf_parser

    monkeypatch.setattr(gguf_parser, "GGUF_MAX_TOTAL_ARRAY_ELEMENTS", 50)
    inner = struct.pack("<IQ", 0, 0)
    payload = _header(0, 1) + _kv_array(9, 60, inner * 60)
    with pytest.raises(GGUFParseError, match="element"):
        GGUFParser(_write(tmp_path, payload)).parse_all()


def test_parse_all_budget_resets_between_parses_of_the_same_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-08: the element budget is per parse, so a second parse of a valid file is not starved."""
    from arc_runner.quant import gguf_parser

    monkeypatch.setattr(gguf_parser, "GGUF_MAX_TOTAL_ARRAY_ELEMENTS", 100)
    parser = GGUFParser(_write(tmp_path, _header(0, 1) + _kv_array(0, 80, b"\x01" * 80)))
    parser.parse_all()
    parser.parse_all()


@pytest.mark.parametrize("value_type, body", [(8, struct.pack("<Q", 3) + b"abc"), (6, struct.pack("<f", float("nan"))), (6, struct.pack("<f", float("inf")))])
def test_get_summary_non_integer_context_length_raises_parse_error(tmp_path: Path, value_type: int, body: bytes) -> None:
    """TM-08: a hostile context_length yields GGUFParseError, not ValueError or OverflowError."""
    kv = struct.pack("<Q", len("general.context_length")) + b"general.context_length" + struct.pack("<I", value_type) + body
    payload = _header(0, 1) + kv
    with pytest.raises(GGUFParseError, match="context_length"):
        GGUFParser(_write(tmp_path, payload)).get_summary()


def test_parse_all_counts_larger_than_remaining_bytes_name_the_count(tmp_path: Path) -> None:
    """TM-08: a tensor or kv count that cannot fit the file is rejected before the loop runs."""
    with pytest.raises(GGUFParseError, match="count"):
        GGUFParser(_write(tmp_path, _header(0, 2**40))).parse_all()
    with pytest.raises(GGUFParseError, match="count"):
        GGUFParser(_write(tmp_path, _header(2**40, 0))).parse_all()


def test_index_layers_tensor_name_with_thousands_of_digits_does_not_raise() -> None:
    """TM-08: a tensor named blk.<5000 digits>.w is treated as a non-layer tensor, not a ValueError."""
    import threading
    from types import SimpleNamespace

    from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager

    manager = object.__new__(FlashStreamWeightManager)
    manager._lock = threading.RLock()
    manager._layer_tensor_map = {}
    manager.parser = SimpleNamespace(tensor_info=[SimpleNamespace(name="blk." + "9" * 5000 + ".w"), SimpleNamespace(name="blk.3.w")])
    assert manager.index_layers() == {3: 1}
