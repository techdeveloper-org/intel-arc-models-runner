"""Slice S-B tests: native bounds validation, ABI/sidecar gating, build script, and C sanitizer harness."""

import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, List

import numpy as np
import pytest

from arc_runner.core.exceptions import NativeBoundsError, NativeLayerError
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager
from arc_runner.native import bridge
from arc_runner.native import build_native
from arc_runner.native.bridge import NativeMemoryEngine, _sidecar_matches, _validate_u64
from tests.test_flash_stream_triad import build_synthetic_flash_gguf

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "src" / "arc_runner" / "native"
FILE_BYTES = 4096
U64_MAX = (1 << 64) - 1

requires_native = pytest.mark.skipif(not bridge._load_native_library(), reason="native library not built or not loadable")


@pytest.fixture
def blob(tmp_path: Path) -> Path:
    """Create a 4 KiB patterned file for mapping tests."""
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(i & 0xFF for i in range(FILE_BYTES)))
    return path


def _tensor(name: str = "w", offset: int = 0, size: int = 16, dims: Any = (4,)) -> dict:
    """Build a tensor metadata record."""
    return {"name": name, "tensor_type": 0, "dimensions": list(dims), "offset": offset, "size_bytes": size}


@pytest.mark.parametrize("value", [-1, U64_MAX + 1, 2**70, True, 1.5, "7", None])
def test_validate_u64_invalid_value_raises_bounds_error(value: Any) -> None:
    """C3: values ctypes would silently wrap or coerce are rejected."""
    with pytest.raises(NativeBoundsError):
        _validate_u64(value, "field")


@pytest.mark.parametrize("value", [0, 1, 2**63, U64_MAX])
def test_validate_u64_valid_value_returns_value(value: int) -> None:
    """C3: the full unsigned 64-bit range is accepted unchanged."""
    assert _validate_u64(value, "field") == value


@requires_native
@pytest.mark.parametrize(
    "offset,size",
    [(U64_MAX - 3, 8), (0, U64_MAX), (FILE_BYTES - 6, 7), (FILE_BYTES + 1, 0)],
)
def test_register_layer_out_of_file_extent_raises_bounds_error(blob: Path, offset: int, size: int) -> None:
    """C1/C2: wrapped or out-of-file extents are rejected before any pointer is derived."""
    engine = NativeMemoryEngine(blob)
    try:
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [_tensor(offset=offset, size=size)])
    finally:
        engine.close()


@requires_native
def test_register_layer_exact_end_of_file_is_accepted(blob: Path) -> None:
    """C1: an extent ending exactly at the last byte is valid."""
    engine = NativeMemoryEngine(blob)
    try:
        assert engine.register_layer(0, [_tensor(offset=FILE_BYTES - 8, size=8)]) is True
    finally:
        engine.close()


@requires_native
@pytest.mark.parametrize("layer_idx", [-1, 256, 10**6])
def test_register_layer_index_out_of_range_raises_bounds_error(blob: Path, layer_idx: int) -> None:
    """C2: layer indices outside [0, 256) are rejected."""
    engine = NativeMemoryEngine(blob)
    try:
        with pytest.raises(NativeBoundsError):
            engine.register_layer(layer_idx, [_tensor()])
    finally:
        engine.close()


@requires_native
def test_register_layer_more_than_64_tensors_raises_instead_of_truncating(blob: Path) -> None:
    """C2/C3: 65 tensors raise; they are not silently cut to 64."""
    engine = NativeMemoryEngine(blob)
    try:
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [_tensor(name=f"t{i}") for i in range(65)])
        assert engine.register_layer(0, [_tensor(name=f"t{i}") for i in range(64)]) is True
    finally:
        engine.close()


@requires_native
def test_register_layer_name_too_long_raises_instead_of_truncating(blob: Path) -> None:
    """C3: a 128-byte name raises; a 127-byte name round-trips exactly."""
    engine = NativeMemoryEngine(blob)
    try:
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [_tensor(name="a" * 128)])
        long_name = "b" * 127
        assert engine.register_layer(1, [_tensor(name=long_name, offset=8, size=8)]) is True
        assert engine.get_tensor_buffer(1, long_name, dtype=np.uint8) is not None
    finally:
        engine.close()


@requires_native
def test_register_layer_more_than_four_dims_raises(blob: Path) -> None:
    """C3: 5 dimensions raise; they are not silently cut to 4."""
    engine = NativeMemoryEngine(blob)
    try:
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [_tensor(dims=(1, 2, 3, 4, 5))])
    finally:
        engine.close()


@requires_native
@pytest.mark.parametrize("field", ["offset", "size_bytes"])
@pytest.mark.parametrize("value", [-1, U64_MAX + 1])
def test_register_layer_unrepresentable_integer_raises(blob: Path, field: str, value: int) -> None:
    """C3: negative or oversized integers raise instead of wrapping in ctypes."""
    engine = NativeMemoryEngine(blob)
    record = _tensor()
    record[field] = value
    try:
        with pytest.raises(NativeBoundsError):
            engine.register_layer(0, [record])
    finally:
        engine.close()


@requires_native
def test_get_tensor_buffer_returns_exact_mapped_bytes(blob: Path) -> None:
    """E3: a tensor read returns exactly the file bytes at its offset."""
    engine = NativeMemoryEngine(blob)
    try:
        engine.register_layer(2, [_tensor(offset=4090, size=6)])
        view = engine.get_tensor_buffer(2, "w", dtype=np.uint8)
        assert view is not None
        assert view.tolist() == [(4090 + i) & 0xFF for i in range(6)]
    finally:
        engine.close()


@requires_native
def test_get_tensor_buffer_size_not_multiple_of_itemsize_returns_none(blob: Path) -> None:
    """C4: a payload that does not divide into dtype items is refused, not truncated."""
    engine = NativeMemoryEngine(blob)
    try:
        engine.register_layer(0, [_tensor(offset=0, size=6)])
        assert engine.get_tensor_buffer(0, "w", dtype=np.float32) is None
    finally:
        engine.close()


@requires_native
def test_register_layer_after_first_read_raises_sealed_error(blob: Path) -> None:
    """The first read seals the table; later registration raises NativeLayerError, not a bounds error."""
    engine = NativeMemoryEngine(blob)
    try:
        engine.register_layer(0, [_tensor(offset=0, size=8)])
        assert engine.get_tensor_buffer(0, "w", dtype=np.uint8) is not None
        with pytest.raises(NativeLayerError) as excinfo:
            engine.register_layer(1, [_tensor(offset=8, size=8)])
        assert not isinstance(excinfo.value, NativeBoundsError)
    finally:
        engine.close()


@requires_native
def test_lock_range_invalid_arguments_are_rejected(blob: Path) -> None:
    """C7: negative values raise; empty and out-of-file ranges return False; no raw OS codes leak."""
    engine = NativeMemoryEngine(blob)
    try:
        with pytest.raises(NativeBoundsError):
            engine.lock_range(-1, 8)
        assert engine.lock_range(0, 0) is False
        assert engine.lock_range(U64_MAX - 3, 8) is False
        assert engine.lock_range(FILE_BYTES, 1) is False
        assert isinstance(engine.lock_range(0, FILE_BYTES), bool)
        engine.unlock_range(0, FILE_BYTES)
    finally:
        engine.close()


@requires_native
def test_close_is_idempotent_and_blocks_further_use(blob: Path) -> None:
    """N7: double close is safe and a closed engine refuses operations."""
    engine = NativeMemoryEngine(blob)
    engine.close()
    engine.close()
    assert engine.is_open is False
    assert engine.register_layer(0, [_tensor()]) is False
    assert engine.get_tensor_buffer(0, "w") is None


def test_native_disabled_by_environment_returns_no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """N5: ARC_NATIVE_DISABLE forces the Python fallback."""
    monkeypatch.setenv("ARC_NATIVE_DISABLE", "1")
    assert bridge._load_native_library() is None
    assert NativeMemoryEngine().is_available is False


@requires_native
def test_abi_version_mismatch_returns_no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """N5: a library built with another ABI version is refused."""
    monkeypatch.setattr(bridge, "_EXPECTED_ABI_VERSION", 99)
    assert bridge._load_native_library() is None


@requires_native
def test_missing_abi_symbol_returns_no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1/N5: a library without the ABI symbol (for example the pre-fix DLL) is refused."""

    class _OldLibrary:
        def __getattr__(self, name: str) -> Any:
            raise AttributeError(name)

    monkeypatch.setattr(bridge.ctypes, "CDLL", lambda _path: _OldLibrary())
    assert bridge._load_native_library() is None


def test_sidecar_matches_detects_hash_mismatch_and_malformed_content(tmp_path: Path) -> None:
    """The sidecar check passes on an equal hash and fails when the hash or JSON is wrong."""
    lib = tmp_path / "libarc_native_sys.so"
    lib.write_bytes(b"binary")
    sidecar = tmp_path / "libarc_native_sys.so.build.json"
    sidecar.write_text(json.dumps({"sha256": hashlib.sha256(b"binary").hexdigest()}), encoding="utf-8")
    assert _sidecar_matches(lib) is True
    sidecar.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    assert _sidecar_matches(lib) is False
    sidecar.write_text("{not json", encoding="utf-8")
    assert _sidecar_matches(lib) is False


@pytest.mark.parametrize("content", ["[1, 2]", "null", '"text"', '{"sha256": 5}', '{"sha256": null}', "{}"])
def test_sidecar_matches_non_object_or_non_string_hash_returns_false_without_raising(tmp_path: Path, content: str) -> None:
    """TM-07: a sidecar that parses as JSON but has the wrong shape is a failed check, not an AttributeError."""
    lib = tmp_path / "libarc_native_sys.so"
    lib.write_bytes(b"binary")
    (tmp_path / "libarc_native_sys.so.build.json").write_text(content, encoding="utf-8")
    assert _sidecar_matches(lib) is False


def test_sidecar_matches_missing_sidecar_refuses_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-07: deleting the sidecar must not turn the integrity check off."""
    monkeypatch.delenv("ARC_NATIVE_ALLOW_MISSING_SIDECAR", raising=False)
    lib = tmp_path / "libarc_native_sys.so"
    lib.write_bytes(b"binary")
    assert _sidecar_matches(lib) is False


def test_sidecar_matches_missing_sidecar_allowed_only_with_explicit_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-07: ARC_NATIVE_ALLOW_MISSING_SIDECAR=1 (and only 1) permits a library with no sidecar."""
    lib = tmp_path / "libarc_native_sys.so"
    lib.write_bytes(b"binary")
    monkeypatch.setenv("ARC_NATIVE_ALLOW_MISSING_SIDECAR", "1")
    assert _sidecar_matches(lib) is True
    for other in ("0", "", "true", "yes"):
        monkeypatch.setenv("ARC_NATIVE_ALLOW_MISSING_SIDECAR", other)
        assert _sidecar_matches(lib) is False


def test_sidecar_opt_out_does_not_excuse_a_present_but_wrong_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-07: the opt-out covers a missing sidecar only; a mismatching one still refuses."""
    monkeypatch.setenv("ARC_NATIVE_ALLOW_MISSING_SIDECAR", "1")
    lib = tmp_path / "libarc_native_sys.so"
    lib.write_bytes(b"binary")
    (tmp_path / "libarc_native_sys.so.build.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    assert _sidecar_matches(lib) is False


def test_load_native_library_missing_sidecar_falls_back_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TM-07: with the sidecar absent the loader returns None (Python read path) without calling CDLL."""
    name = "libarc_native_sys.dll" if sys.platform == "win32" else "libarc_native_sys.so"
    (tmp_path / name).write_bytes(b"binary")
    monkeypatch.delenv("ARC_NATIVE_ALLOW_MISSING_SIDECAR", raising=False)
    monkeypatch.delenv("ARC_NATIVE_DISABLE", raising=False)
    monkeypatch.setattr(bridge, "__file__", str(tmp_path / "bridge.py"))

    def forbidden(_path: str) -> Any:
        raise AssertionError("CDLL must not be reached")

    monkeypatch.setattr(bridge.ctypes, "CDLL", forbidden)
    assert bridge._load_native_library() is None


@requires_native
def test_native_and_python_paths_return_identical_tensor_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N5: with the native layer disabled, layer weights are byte-identical to the native path."""
    model = build_synthetic_flash_gguf(tmp_path / "model.gguf", num_layers=2, hidden_dim=32)

    def read_layer(disable: bool) -> dict:
        if disable:
            monkeypatch.setenv("ARC_NATIVE_DISABLE", "1")
        else:
            monkeypatch.delenv("ARC_NATIVE_DISABLE", raising=False)
        with FlashStreamWeightManager(model, ring_buffer_slots=2) as manager:
            assert manager._native_engine.is_available is (not disable)
            return {key: np.array(value, copy=True) for key, value in manager.acquire_layer_weights(0).items()}

    native_weights = read_layer(disable=False)
    python_weights = read_layer(disable=True)
    assert native_weights.keys() == python_weights.keys()
    for key in native_weights:
        assert native_weights[key].tobytes() == python_weights[key].tobytes()


def test_find_compiler_unknown_name_raises_runtime_error() -> None:
    """build_native reports a missing compiler clearly."""
    with pytest.raises(RuntimeError):
        build_native.find_compiler("definitely-not-a-compiler-xyz")


def test_library_name_matches_target_platform() -> None:
    """Library file names follow the target platform."""
    assert build_native.library_name(True) == "libarc_native_sys.dll"
    assert build_native.library_name(False) == "libarc_native_sys.so"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
@pytest.mark.skipif(sys.platform == "win32", reason="builds the POSIX .so (needs -ldl); the Windows DLL is built by build_native.py itself")
def test_compile_library_produces_loadable_abi2_library_with_matching_sidecar(tmp_path: Path) -> None:
    """The build script emits a -Werror-clean library whose sidecar hash matches and ABI is 3."""
    output = tmp_path / "libarc_native_sys.so"
    build_native.compile_library(output, target_windows=False)
    sidecar = build_native.write_build_info(output, build_native.find_compiler(), target_windows=False)
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    assert info["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert info["abi_version"] == 3
    assert "-mavx2" not in json.dumps(info["flags"])
    library = ctypes.CDLL(str(output))
    library.arc_native_abi_version.restype = ctypes.c_uint32
    assert library.arc_native_abi_version() == 3


@pytest.mark.skipif(shutil.which("x86_64-w64-mingw32-gcc") is None, reason="mingw-w64 cross compiler not available")
def test_windows_branch_compiles_clean_with_werror(tmp_path: Path) -> None:
    """The Win32 code path compiles under -Wall -Wextra -Wpedantic -Werror via mingw-w64."""
    result = subprocess.run(
        [
            "x86_64-w64-mingw32-gcc", "-std=c17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
            "-c", str(NATIVE_DIR / "arc_flash_stream_mmap.c"), "-o", str(tmp_path / "mmap_win.o"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_c_sanitizer_harness_passes_under_asan_and_ubsan(tmp_path: Path) -> None:
    """The C harness (1e6 fuzz iterations plus boundary tables) passes with ASan and UBSan."""
    harness = tmp_path / "mmap_harness"
    compile_cmd: List[str] = [
        "gcc", "-std=c17", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
        str(REPO_ROOT / "tests" / "native_c" / "mmap_harness.c"),
        str(NATIVE_DIR / "arc_flash_stream_mmap.c"),
        "-o", str(harness),
    ]
    built = subprocess.run(compile_cmd, capture_output=True, text=True)
    if built.returncode != 0 and any(token in built.stderr for token in ("sanitize", "-lasan", "-lubsan")):
        pytest.skip("sanitizers unavailable on this toolchain")
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(harness), str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "failures=0" in ran.stdout
