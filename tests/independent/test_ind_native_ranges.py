"""Independent randomized range checks of the native engine against a Python reference.

Specification: SRS.md FR-001 ("every extent is validated against the file size at registration ...
integers validated without truncation or wrapping") and the bridge.py docstrings: an extent
[offset, offset + size) is acceptable iff it lies inside the file; arbitrary-precision Python integer
arithmetic is the reference (no wrapping is possible there).
"""

import random
from pathlib import Path

import numpy as np
import pytest

from arc_runner.core.exceptions import ArcRunnerError, NativeBoundsError, NativeLayerError
from arc_runner.native.bridge import NativeMemoryEngine

U64 = (1 << 64) - 1
SEED = 20241029


def _engine_or_skip(path: Path) -> NativeMemoryEngine:
    eng = NativeMemoryEngine()
    if not eng.is_available:
        pytest.skip("native library not available on this platform (not run)")
    assert eng.open_model(path) is True
    return eng


def _record(name: str, offset: int, size: int) -> dict:
    return {"name": name, "tensor_type": 0, "dimensions": [1], "offset": offset, "size_bytes": size}


def _reference_ok(offset: int, size: int, file_size: int) -> bool:
    return 0 <= offset and 0 <= size and offset + size <= file_size


def _random_extent(rng: random.Random, file_size: int):
    kind = rng.randrange(8)
    if kind == 0:
        return rng.randrange(0, file_size + 1), rng.randrange(1, file_size + 1)
    if kind == 1:
        off = rng.randrange(0, file_size)
        return off, file_size - off  # touches end exactly
    if kind == 2:
        off = rng.randrange(0, file_size)
        return off, file_size - off + 1  # one past the end
    if kind == 3:
        return U64 - rng.randrange(0, 4), rng.randrange(1, 8)  # wraps if summed in u64
    if kind == 4:
        return rng.randrange(0, file_size), U64 - rng.randrange(0, 4)
    if kind == 5:
        return U64, U64
    if kind == 6:
        return 1 << 63, 1 << 63  # sum == 2**64 exactly
    return rng.randrange(0, 2 * file_size + 2), rng.randrange(1, 2 * file_size + 2)


@pytest.fixture(scope="module")
def blob_files(tmp_path_factory):
    rng = random.Random(SEED)
    base = tmp_path_factory.mktemp("native_ranges")
    files = {}
    for size in (1, 7, 64, 4096, 5000):
        p = base / f"f{size}.bin"
        p.write_bytes(bytes(rng.randrange(256) for _ in range(size)))
        files[size] = p
    return files


def test_register_layer_single_extent_random_matches_reference(blob_files):
    rng = random.Random(SEED + 1)
    mismatches = []
    for case in range(400):
        file_size = rng.choice(list(blob_files))
        offset, size = _random_extent(rng, file_size)
        eng = _engine_or_skip(blob_files[file_size])
        try:
            expected = _reference_ok(offset, size, file_size)
            try:
                got = eng.register_layer(0, [_record("w", offset, size)])
                accepted = got is True
            except NativeBoundsError:
                accepted = False
            if accepted != expected:
                mismatches.append((file_size, offset, size, expected, accepted))
        finally:
            eng.close()
    assert mismatches == []


@pytest.mark.parametrize("offset,size,file_size", [
    (0, 1, 1), (0, 2, 1), (1, 1, 1), (1, 0, 1), (U64, 1, 64), (1, U64, 64), (U64, U64, 64),
    (1 << 63, 1 << 63, 64), (U64 - 63, 64, 64), (64 - U64 + U64 - 1, 1, 64), (63, 1, 64), (64, 1, 64),
])
def test_register_layer_uint64_edges_match_reference(blob_files, offset, size, file_size):
    if file_size not in blob_files:
        file_size = 64
    eng = _engine_or_skip(blob_files[file_size])
    try:
        expected = _reference_ok(offset, size, file_size)
        try:
            accepted = eng.register_layer(3, [_record("edge", offset, size)]) is True
        except NativeBoundsError:
            accepted = False
        assert accepted == expected
    finally:
        eng.close()


@pytest.mark.parametrize("bad", [-1, U64 + 1, 1 << 64, 1 << 200, -(1 << 63), 1.5, "3", None, True])
def test_register_layer_non_u64_integers_raise_bounds_error(blob_files, bad):
    eng = _engine_or_skip(blob_files[64])
    try:
        with pytest.raises(NativeBoundsError):
            eng.register_layer(0, [_record("x", bad, 1)])
        with pytest.raises(NativeBoundsError):
            eng.register_layer(0, [_record("x", 0, bad)])
        with pytest.raises(NativeBoundsError):
            eng.register_layer(0, [{"name": "x", "tensor_type": 0, "dimensions": [bad], "offset": 0, "size_bytes": 1}])
    finally:
        eng.close()


def test_register_layer_layer_index_and_counts_boundaries(blob_files):
    eng = _engine_or_skip(blob_files[4096])
    try:
        for bad_idx in (-1, 256, 1 << 40, True, 1.0, "0"):
            with pytest.raises(NativeBoundsError):
                eng.register_layer(bad_idx, [_record("a", 0, 4)])
        assert eng.register_layer(0, [_record(f"t{i}", i * 4, 4) for i in range(64)]) is True
        with pytest.raises(NativeBoundsError):
            eng.register_layer(1, [_record(f"t{i}", i * 4, 4) for i in range(65)])
        assert eng.register_layer(255, [_record("a", 0, 4)]) is True
    finally:
        eng.close()


def test_register_layer_name_and_dimension_limits(blob_files):
    eng = _engine_or_skip(blob_files[64])
    try:
        assert eng.register_layer(0, [_record("n" * 127, 0, 4)]) is True
        with pytest.raises(NativeBoundsError):
            eng.register_layer(1, [_record("n" * 128, 0, 4)])
        with pytest.raises(NativeBoundsError):
            # 127 two-byte characters = 254 bytes: the limit is in bytes, not characters
            eng.register_layer(2, [_record("é" * 127, 0, 4)])
        rec = _record("d", 0, 4)
        rec["dimensions"] = [1, 1, 1, 1]
        assert eng.register_layer(3, [rec]) is True
        rec5 = dict(rec, name="d5", dimensions=[1, 1, 1, 1, 1])
        with pytest.raises(NativeBoundsError):
            eng.register_layer(4, [rec5])
    finally:
        eng.close()


def test_register_layer_multi_tensor_random_each_extent_checked(blob_files):
    rng = random.Random(SEED + 2)
    for case in range(120):
        file_size = rng.choice([64, 4096, 5000])
        n = rng.randrange(1, 6)
        extents = [_random_extent(rng, file_size) for _ in range(n)]
        recs = [_record(f"t{i}", o, s) for i, (o, s) in enumerate(extents)]
        expected = all(_reference_ok(o, s, file_size) for o, s in extents)
        eng = _engine_or_skip(blob_files[file_size])
        try:
            try:
                accepted = eng.register_layer(1, recs) is True
            except NativeBoundsError:
                accepted = False
            assert accepted == expected, (file_size, extents)
        finally:
            eng.close()


def test_get_tensor_buffer_returns_exact_file_bytes_for_random_valid_extents(blob_files):
    rng = random.Random(SEED + 3)
    for case in range(150):
        file_size = rng.choice([64, 4096, 5000])
        data = blob_files[file_size].read_bytes()
        n = rng.randrange(1, 5)
        extents, recs = [], []
        for i in range(n):
            off = rng.randrange(0, file_size)
            size = rng.randrange(1, file_size - off + 1)
            extents.append((off, size))
            recs.append(_record(f"t{i}", off, size))
        eng = _engine_or_skip(blob_files[file_size])
        try:
            assert eng.register_layer(2, recs) is True
            for i, (off, size) in enumerate(extents):
                view = eng.get_tensor_buffer(2, f"t{i}", dtype=np.uint8)
                assert view is not None
                assert view.tobytes() == data[off:off + size]
                assert view.flags.writeable is False
            assert eng.get_tensor_buffer(2, "does-not-exist", dtype=np.uint8) is None
        finally:
            eng.close()


def test_register_layer_after_seal_raises_native_layer_error(blob_files):
    eng = _engine_or_skip(blob_files[64])
    try:
        assert eng.register_layer(0, [_record("a", 0, 8)]) is True
        eng.get_tensor_buffer(0, "a", dtype=np.uint8)  # first lookup seals
        with pytest.raises(NativeLayerError):
            eng.register_layer(1, [_record("b", 0, 8)])
    finally:
        eng.close()


def test_lock_unlock_range_validate_u64_and_never_crash(blob_files):
    rng = random.Random(SEED + 4)
    eng = _engine_or_skip(blob_files[4096])
    try:
        for _ in range(100):
            off, size = _random_extent(rng, 4096)
            result = eng.lock_range(off, size)  # best effort: must return bool, never crash
            assert isinstance(result, bool)
            assert isinstance(eng.unlock_range(off, size), bool)
            if not _reference_ok(off, size, 4096):
                assert result is False, "a lock outside the file must never succeed"
        for bad in (-1, 1 << 64):
            with pytest.raises(NativeBoundsError):
                eng.lock_range(bad, 1)
            with pytest.raises(NativeBoundsError):
                eng.unlock_range(0, bad)
    finally:
        eng.close()


def test_native_errors_are_arc_runner_errors(blob_files):
    eng = _engine_or_skip(blob_files[64])
    try:
        with pytest.raises(ArcRunnerError):
            eng.register_layer(0, [_record("a", 0, 1 << 70)])
    finally:
        eng.close()


def test_open_model_on_missing_or_empty_file_returns_false(tmp_path):
    eng = NativeMemoryEngine()
    if not eng.is_available:
        pytest.skip("native library not available (not run)")
    assert eng.open_model(tmp_path / "missing.bin") is False
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert eng.open_model(empty) is False
    assert eng.is_open is False
