"""Tests for pybind11 native binding and ABI version 4 (WP-06 CC-1..CC-5).

Verifies ABI version consistency, ArcMmapContext RAII lifecycle,
TensorPtrMap operations, environment disabling, and ABI mismatch detection.
"""

from pathlib import Path
import re
import pytest

from arc_runner.core.exceptions import NativeBoundsError, NativeLayerError


def test_abi_version_consistency_cc1() -> None:
    """CC-1: ARC_NATIVE_ABI_VERSION == 4 in header, bridge.py, and build_native.py."""
    repo_root = Path(__file__).resolve().parents[1]
    header = (repo_root / "src/arc_runner/native/arc_native_pybind.h").read_text(encoding="utf-8")
    match_header = re.search(r"#define\s+ARC_NATIVE_ABI_VERSION\s+(\d+)", header)
    assert match_header is not None
    assert int(match_header.group(1)) == 4

    from arc_runner.native.bridge import ARC_NATIVE_ABI_VERSION

    assert ARC_NATIVE_ABI_VERSION == 4

    build_script = (repo_root / "src/arc_runner/native/build_native.py").read_text(encoding="utf-8")
    match_build = re.search(r"ABI_VERSION\s*=\s*(\d+)", build_script)
    assert match_build is not None
    assert int(match_build.group(1)) == 4


def test_arc_mmap_context_raii_context_manager_cc2(tmp_path: Path) -> None:
    """CC-2: ArcMmapContext is usable as a Python context manager."""
    from arc_runner.native.bridge import ArcMmapContext

    test_file = tmp_path / "model.bin"
    payload = b"X" * 4096
    test_file.write_bytes(payload)

    with ArcMmapContext(str(test_file), len(payload)) as ctx:
        assert ctx.size() == len(payload)
        assert ctx.ptr() is not None
        assert ctx.ptr() != 0

    assert ctx.is_closed()


def test_tensor_ptr_map_operations_cc3() -> None:
    """CC-3: TensorPtrMap supports insert, get, and len() operations."""
    from arc_runner.native.bridge import TensorPtrMap

    tmap = TensorPtrMap()
    assert len(tmap) == 0

    tmap.insert("layer.0.weight", 0x1000, 2048)
    tmap.insert("layer.0.bias", 0x2000, 512)

    assert len(tmap) == 2
    entry = tmap.get("layer.0.weight")
    assert entry is not None
    assert entry.ptr == 0x1000
    assert entry.size == 2048

    assert tmap.get("nonexistent") is None


def test_native_disabled_by_env_raises_layer_error_cc4(monkeypatch: pytest.MonkeyPatch) -> None:
    """CC-4: ARC_NATIVE_DISABLE=1 raises NativeLayerError with 'native layer disabled'."""
    monkeypatch.setenv("ARC_NATIVE_DISABLE", "1")
    from arc_runner.native.bridge import load_native_library

    with pytest.raises(NativeLayerError) as exc_info:
        load_native_library()
    assert "native layer disabled" in str(exc_info.value).lower()


def test_stale_abi_version_raises_bounds_error_cc5() -> None:
    """CC-5: Stale library ABI version triggers NativeBoundsError at load time."""
    from arc_runner.native.bridge import verify_abi_version

    with pytest.raises(NativeBoundsError) as exc_info:
        verify_abi_version(3)
    assert "ABI version mismatch" in str(exc_info.value)
