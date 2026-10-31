"""Independent hostile-input tests for the GGUF parser.

Specification: gguf_parser docstrings and CLAUDE.md ("treats input as hostile ... raise the specific
ArcRunnerError subclass"; never hang, never allocate absurd memory). Any outcome other than a clean parse
or an ArcRunnerError subclass is a defect.
"""

import random
import struct
import threading
import tracemalloc
from pathlib import Path

import pytest

from arc_runner.core.exceptions import ArcRunnerError, GGUFParseError, InvalidGGUFError
from arc_runner.quant.gguf_parser import GGUFParser

from ind_helpers import ARR, BOOL, F32, F64, I16, I32, I64, I8, STR, U16, U32, U64, U8, build_gguf, pack_string

SEED = 424242
PARSE_TIMEOUT_S = 10.0
MEMORY_CAP_BYTES = 64 * 1024 * 1024


def _valid_file() -> tuple:
    metadata = [
        ("general.architecture", STR, "llama"),
        ("general.alignment", U32, 32),
        ("llama.context_length", U32, 2048),
        ("llama.block_count", U32, 2),
        ("x.u8", U8, 200), ("x.i8", I8, -5), ("x.u16", U16, 60000), ("x.i16", I16, -300),
        ("x.i32", I32, -70000), ("x.f32", F32, 1.5), ("x.bool", BOOL, True),
        ("x.u64", U64, 1 << 60), ("x.i64", I64, -(1 << 40)), ("x.f64", F64, 2.25),
        ("x.arr", ARR, (U32, [1, 2, 3, 4])),
        ("x.strarr", ARR, (STR, ["a", "bb", "ccc"])),
        ("x.nested", ARR, (ARR, [(U8, [1, 2]), (U8, [])])),
    ]
    tensors = [("blk.0.w", (4, 2), 0, 0), ("blk.1.w", (4, 2), 1, 32), ("output.weight", (3,), 0, 64)]
    payload = bytes(range(256))
    return build_gguf(metadata, tensors, payload)


VALID, INFO_END = _valid_file()


def _run_guarded(path: Path):
    """Parse in a worker thread; returns ('ok'|'arc'|'other', detail, peak_bytes)."""
    box = {}

    def work():
        tracemalloc.start()
        try:
            try:
                GGUFParser(path).parse_all()
                box["r"] = ("ok", None)
            except ArcRunnerError as exc:
                box["r"] = ("arc", type(exc).__name__)
            except BaseException as exc:  # noqa: BLE001
                box["r"] = ("other", f"{type(exc).__name__}: {str(exc)[:80]}")
        finally:
            box["peak"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(PARSE_TIMEOUT_S)
    if t.is_alive():
        return ("hang", "parser still running after %.0fs" % PARSE_TIMEOUT_S, 0)
    return (*box["r"], box["peak"])


def _write(tmp_path: Path, data: bytes, name="f.gguf") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_valid_file_parses_and_reports_expected_metadata(tmp_path):
    parser = GGUFParser(_write(tmp_path, VALID))
    parser.parse_all()
    assert parser.header.version == 3
    assert parser.header.tensor_count == 3
    assert parser.metadata["general.architecture"] == "llama"
    assert parser.metadata["x.arr"] == [1, 2, 3, 4]
    assert parser.metadata["x.strarr"] == ["a", "bb", "ccc"]
    assert parser.metadata["x.nested"] == [[1, 2], []]
    assert parser.metadata["x.bool"] is True
    assert parser.metadata["x.i8"] == -5
    assert abs(parser.metadata["x.f32"] - 1.5) < 1e-7
    assert [t.name for t in parser.tensor_info] == ["blk.0.w", "blk.1.w", "output.weight"]
    assert parser.data_offset % 32 == 0
    assert parser.data_offset >= INFO_END
    assert parser.data_offset - INFO_END < 32


def test_every_truncation_before_end_of_tensor_info_raises_arc_error(tmp_path):
    bad = []
    for cut in range(0, INFO_END):
        status, detail, _ = _run_guarded(_write(tmp_path, VALID[:cut]))
        if status != "arc":
            bad.append((cut, status, detail))
    assert bad == []


def test_every_truncation_after_tensor_info_still_parses(tmp_path):
    # payload bytes are never read by the parser, so cutting inside the payload must not matter
    for cut in range(INFO_END, len(VALID) + 1, 7):
        status, detail, _ = _run_guarded(_write(tmp_path, VALID[:cut]))
        assert status == "ok", (cut, status, detail)


def test_bad_magic_and_version_raise_invalid_gguf(tmp_path):
    with pytest.raises(InvalidGGUFError):
        GGUFParser(_write(tmp_path, b"GGUX" + VALID[4:])).parse_all()
    for version in (0, 1, 4, 99, 2 ** 32 - 1):
        data = VALID[:4] + struct.pack("<I", version) + VALID[8:]
        with pytest.raises(InvalidGGUFError):
            GGUFParser(_write(tmp_path, data)).parse_all()
    for version in (2, 3):
        data = VALID[:4] + struct.pack("<I", version) + VALID[8:]
        GGUFParser(_write(tmp_path, data)).parse_all()


def test_missing_file_raises_invalid_gguf(tmp_path):
    with pytest.raises(InvalidGGUFError):
        GGUFParser(tmp_path / "absent.gguf").parse_all()


def test_random_byte_flips_never_hang_or_leak_non_arc_errors(tmp_path):
    rng = random.Random(SEED)
    bad = []
    for i in range(500):
        data = bytearray(VALID[:INFO_END + 8])
        for _ in range(rng.randrange(1, 5)):
            data[rng.randrange(len(data))] = rng.randrange(256)
        status, detail, peak = _run_guarded(_write(tmp_path, bytes(data)))
        if status in ("other", "hang") or peak > MEMORY_CAP_BYTES:
            bad.append((i, status, detail, peak))
    assert bad == [], bad[:5]


def test_random_u64_word_replacement_with_edge_values(tmp_path):
    rng = random.Random(SEED + 1)
    edges = [0, 1, 2, 31, 32, 33, 255, 256, 2 ** 16, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 40, 2 ** 62, 2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1]
    bad = []
    for i in range(400):
        data = bytearray(VALID[:INFO_END + 8])
        pos = rng.randrange(0, len(data) - 8)
        data[pos:pos + 8] = struct.pack("<Q", rng.choice(edges))
        status, detail, peak = _run_guarded(_write(tmp_path, bytes(data)))
        if status in ("other", "hang") or peak > MEMORY_CAP_BYTES:
            bad.append((pos, status, detail, peak))
    assert bad == [], bad[:5]


def test_u32_word_replacement_in_type_fields(tmp_path):
    rng = random.Random(SEED + 2)
    edges = [0, 3, 13, 14, 31, 32, 255, 2 ** 31, 2 ** 32 - 1]
    bad = []
    for i in range(400):
        data = bytearray(VALID[:INFO_END + 8])
        pos = rng.randrange(0, len(data) - 4)
        data[pos:pos + 4] = struct.pack("<I", rng.choice(edges))
        status, detail, peak = _run_guarded(_write(tmp_path, bytes(data)))
        if status in ("other", "hang") or peak > MEMORY_CAP_BYTES:
            bad.append((pos, status, detail, peak))
    assert bad == [], bad[:5]


# ---- targeted hostile constructions -------------------------------------------------------------

def _single_meta(vtype_bytes: bytes, key="k") -> bytes:
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + pack_string(key) + vtype_bytes


@pytest.mark.parametrize("name,blob", [
    ("huge_string_len", _single_meta(struct.pack("<I", STR) + struct.pack("<Q", 2 ** 63))),
    ("max_string_len", _single_meta(struct.pack("<I", STR) + struct.pack("<Q", 2 ** 64 - 1))),
    ("huge_array_len_u8", _single_meta(struct.pack("<IIQ", ARR, U8, 2 ** 63) + b"\x01")),
    ("huge_array_len_str", _single_meta(struct.pack("<IIQ", ARR, STR, 2 ** 64 - 1) + struct.pack("<Q", 0))),
    ("array_len_larger_than_file_zero_size_elems", _single_meta(struct.pack("<IIQ", ARR, ARR, 10 ** 9))),
    ("huge_meta_count", b"GGUF" + struct.pack("<IQQ", 3, 0, 2 ** 64 - 1)),
    ("huge_tensor_count", b"GGUF" + struct.pack("<IQQ", 3, 2 ** 64 - 1, 0)),
    ("tensor_dims_5", b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + pack_string("t") + struct.pack("<I", 5) + b"\x00" * 64),
    ("tensor_dims_huge", b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + pack_string("t") + struct.pack("<I", 2 ** 32 - 1)),
])
def test_targeted_hostile_length_fields_raise_arc_error(tmp_path, name, blob):
    status, detail, peak = _run_guarded(_write(tmp_path, blob, name + ".gguf"))
    assert status == "arc", (name, status, detail)
    assert peak < MEMORY_CAP_BYTES


def test_unknown_metadata_value_type_raises_arc_error(tmp_path):
    for vt in (13, 14, 255, 2 ** 32 - 1):
        status, detail, _ = _run_guarded(_write(tmp_path, _single_meta(struct.pack("<I", vt)), f"vt{vt}.gguf"))
        assert status == "arc", (vt, status, detail)


def test_unknown_array_element_type_raises_arc_error(tmp_path):
    for et in (13, 99, 2 ** 32 - 1):
        blob = _single_meta(struct.pack("<IIQ", ARR, et, 1) + b"\x00" * 8)
        status, detail, _ = _run_guarded(_write(tmp_path, blob, f"et{et}.gguf"))
        assert status == "arc", (et, status, detail)


def test_unknown_ggml_tensor_type_raises_arc_error(tmp_path):
    for tt in (4, 5, 31, 1000, 2 ** 32 - 1):
        blob, _ = build_gguf([], [("t", (1,), tt, 0)], b"\x00" * 32)
        status, detail, _ = _run_guarded(_write(tmp_path, blob, f"tt{tt}.gguf"))
        assert status == "arc", (tt, status, detail)


@pytest.mark.parametrize("alignment_meta", [
    ("general.alignment", U32, 0),
    ("general.alignment", I32, -32),
    ("general.alignment", STR, "32"),
    ("general.alignment", F32, 32.0),
    ("general.alignment", ARR, (U8, [1])),
    ("general.alignment", BOOL, True),
])
def test_hostile_alignment_metadata_raises_arc_error_or_parses(tmp_path, alignment_meta):
    blob, _ = build_gguf([alignment_meta], [], b"")
    status, detail, _ = _run_guarded(_write(tmp_path, blob))
    assert status in ("ok", "arc"), (alignment_meta[:2], status, detail)


def test_alignment_zero_must_not_be_accepted_silently(tmp_path):
    blob, _ = build_gguf([("general.alignment", U32, 0)], [], b"")
    status, detail, _ = _run_guarded(_write(tmp_path, blob))
    assert status == "arc", (status, detail)


@pytest.mark.parametrize("depth", [50, 500, 5000])
def test_deeply_nested_arrays_raise_arc_error_not_recursion_error(tmp_path, depth):
    body = struct.pack("<I", ARR)
    for _ in range(depth):
        body += struct.pack("<IQ", ARR, 1)
    body += struct.pack("<IQ", U8, 0)
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + pack_string("k") + body
    status, detail, _ = _run_guarded(_write(tmp_path, blob))
    assert status in ("ok", "arc"), (depth, status, detail)


def test_empty_and_tiny_files_raise_arc_error(tmp_path):
    for n in range(0, 24):
        status, detail, _ = _run_guarded(_write(tmp_path, VALID[:n], f"tiny{n}.gguf"))
        assert status == "arc", (n, status, detail)


def test_directory_path_raises_arc_error(tmp_path):
    status, detail, _ = _run_guarded(tmp_path)
    assert status in ("arc",), (status, detail)


def test_invalid_utf8_in_strings_does_not_crash(tmp_path):
    raw = b"\xff\xfe\xfd"
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", len(raw)) + raw + struct.pack("<II", U8, 0)[:4] + b"\x07"
    status, detail, _ = _run_guarded(_write(tmp_path, blob))
    assert status in ("ok", "arc"), (status, detail)
