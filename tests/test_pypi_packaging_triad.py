"""Triad test suite for PyPI standalone binary wheel packaging and distribution.

Enforces Triad testing patterns (Positive, Negative, Edge-Case) verifying
archive structure, PEP 561 compliance, dual CLI entrypoints, and cryptographic
hash integrity for arc-runner distribution packages.
"""

import base64
import hashlib
from pathlib import Path
import tempfile
from typing import Dict, List, Set
import zipfile
import pytest

from scripts.build_wheel import WheelBuilder


class TestPyPIPackagingPositiveTriad:
    """Positive test cases verifying nominal distribution package invariants."""

    def test_wheel_builder_initialization(self) -> None:
        """Verify that WheelBuilder initializes with valid root and dist paths."""
        builder = WheelBuilder()
        assert builder.root_dir.is_dir()
        assert builder.dist_dir.name == "dist"

    def test_wheel_metadata_conformance(self) -> None:
        """Inspect the METADATA descriptor inside the compiled wheel archive.

        Asserts that package name, version, license, and required dependencies
        match the project specification.
        """
        builder = WheelBuilder()
        wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))
        if not wheel_candidates:
            builder.compile_distribution_packages()
            wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))

        wheel_path = wheel_candidates[0]
        with zipfile.ZipFile(wheel_path, "r") as archive:
            metadata_name = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")][0]
            metadata_text = archive.read(metadata_name).decode("utf-8")

            assert "Name: arc-runner" in metadata_text
            assert "Version: 0.1.0" in metadata_text
            assert "License: Apache-2.0" in metadata_text
            assert "Requires-Python: >=3.10" in metadata_text
            assert "Requires-Dist: openvino" in metadata_text
            assert "Requires-Dist: fastapi" in metadata_text

    def test_wheel_entry_points_dual_aliasing(self) -> None:
        """Verify that both arc-runner and intel-arc-models-runner CLI aliases are defined."""
        builder = WheelBuilder()
        wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))
        assert len(wheel_candidates) > 0, "No built wheel found in dist/"

        wheel_path = wheel_candidates[0]
        with zipfile.ZipFile(wheel_path, "r") as archive:
            entrypoint_file = [n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt")][0]
            content = archive.read(entrypoint_file).decode("utf-8")

            commands: Dict[str, str] = {}
            for line in content.splitlines():
                if "=" in line and not line.strip().startswith("["):
                    key, val = line.split("=", 1)
                    commands[key.strip()] = val.strip()

            assert "arc-runner" in commands
            assert "intel-arc-models-runner" in commands
            assert commands["arc-runner"] == "arc_runner.cli.main:main"
            assert commands["intel-arc-models-runner"] == "arc_runner.cli.main:main"

    def test_wheel_static_assets_completeness(self) -> None:
        """Verify that all non-Python HTML, CSS, JS and PEP 561 markers are bundled."""
        builder = WheelBuilder()
        wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))
        wheel_path = wheel_candidates[0]

        with zipfile.ZipFile(wheel_path, "r") as archive:
            names = set(archive.namelist())

            assert any("arc_runner/py.typed" in n for n in names)
            assert any("templates/index.html" in n for n in names)
            assert any("static/css/style.css" in n for n in names)
            assert any("static/js/app.js" in n for n in names)

    def test_wheel_record_hash_integrity(self) -> None:
        """Verify that every bundled file strictly matches its SHA256 RECORD hash."""
        builder = WheelBuilder()
        wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))
        wheel_path = wheel_candidates[0]

        with zipfile.ZipFile(wheel_path, "r") as archive:
            record_name = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")][0]
            record_lines = archive.read(record_name).decode("utf-8").splitlines()

            for line in record_lines:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[1].startswith("sha256="):
                    file_name = parts[0]
                    expected_b64 = parts[1].split("=")[1]
                    file_bytes = archive.read(file_name)
                    calc_b64 = base64.urlsafe_b64encode(hashlib.sha256(file_bytes).digest()).decode("utf-8").rstrip("=")
                    calc_hex = hashlib.sha256(file_bytes).hexdigest()
                    assert expected_b64 in (calc_b64, calc_hex)


class TestPyPIPackagingNegativeTriad:
    """Negative test cases verifying failure detection and exception guards."""

    def test_wheel_audit_fails_on_missing_file(self) -> None:
        """Verify that inspect_and_verify_wheel raises FileNotFoundError for non-existent file."""
        builder = WheelBuilder()
        with pytest.raises(FileNotFoundError):
            builder.inspect_and_verify_wheel(Path("non_existent_wheel_archive.whl"))

    def test_wheel_audit_detects_missing_py_typed(self) -> None:
        """Verify that a wheel missing py.typed triggers ValueError."""
        builder = WheelBuilder()
        with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(tmp_path, "w") as arc:
                arc.writestr("arc_runner/__init__.py", "# empty")
                arc.writestr("arc_runner-0.1.0.dist-info/entry_points.txt", "[console_scripts]\narc-runner=foo:main\nintel-arc-models-runner=foo:main\n")
                arc.writestr("arc_runner/server/templates/index.html", "<html></html>")

            with pytest.raises(ValueError, match="py.typed marker missing"):
                builder.inspect_and_verify_wheel(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_wheel_audit_detects_hash_tampering(self) -> None:
        """Verify that a tampered file payload triggers a checksum mismatch ValueError."""
        builder = WheelBuilder()
        with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with zipfile.ZipFile(tmp_path, "w") as arc:
                arc.writestr("arc_runner/py.typed", "")
                arc.writestr("arc_runner/server/templates/index.html", "<html></html>")
                arc.writestr("arc_runner-0.1.0.dist-info/entry_points.txt", "[console_scripts]\narc-runner=foo:main\nintel-arc-models-runner=foo:main\n")
                arc.writestr("arc_runner/sample.py", "original_content")
                arc.writestr("arc_runner-0.1.0.dist-info/RECORD", "arc_runner/sample.py,sha256=invalidhash999,16\n")

            with pytest.raises(ValueError, match="Checksum mismatch"):
                builder.inspect_and_verify_wheel(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestPyPIPackagingEdgeCaseTriad:
    """Edge-case test cases verifying boundary conditions, locale safety, and isolation."""

    def test_wheel_archive_compression_ratio(self) -> None:
        """Verify Proof M115: compression ratio satisfies R_comp >= 1.25."""
        builder = WheelBuilder()
        wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))
        wheel_path = wheel_candidates[0]

        with zipfile.ZipFile(wheel_path, "r") as archive:
            total_uncompressed = sum(info.file_size for info in archive.infolist())
            total_compressed = sum(info.compress_size for info in archive.infolist())

            compression_ratio = total_uncompressed / max(total_compressed, 1)
            assert compression_ratio >= 1.25

    def test_wheel_templates_utf8_decodable(self) -> None:
        """Verify that all bundled static templates decode cleanly as strict UTF-8."""
        builder = WheelBuilder()
        wheel_candidates = list(builder.dist_dir.glob("arc_runner-*.whl"))
        wheel_path = wheel_candidates[0]

        with zipfile.ZipFile(wheel_path, "r") as archive:
            for name in archive.namelist():
                if name.endswith((".html", ".css", ".js", ".json", ".txt")):
                    content = archive.read(name)
                    decoded = content.decode("utf-8")
                    assert len(decoded) > 0

    def test_wheel_sdist_mirroring(self) -> None:
        """Verify that source distribution (.tar.gz) mirrors critical project files."""
        builder = WheelBuilder()
        sdists = list(builder.dist_dir.glob("arc_runner-*.tar.gz"))
        assert len(sdists) > 0, "No sdist package generated in dist/"
        sdist_path = sdists[0]
        assert sdist_path.stat().st_size > 50000
