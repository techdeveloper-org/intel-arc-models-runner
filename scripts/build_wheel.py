"""Automated wheel building, inspection, and verification script for arc-runner.

Compiles standalone binary wheel and source distribution packages, audits
the archive for critical asset inclusions, and verifies cryptographic integrity.
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile


class WheelBuilder:
    """Builds and verifies arc-runner standalone distribution packages."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """Initialize the WheelBuilder instance.

        Args:
            root_dir: Project root directory path. Defaults to parent of scripts/.
        """
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent
        self.dist_dir = self.root_dir / "dist"
        self.build_dir = self.root_dir / "build"
        self.src_dir = self.root_dir / "src"

    def clean_build_artifacts(self) -> None:
        """Remove previous build outputs, egg-info directories, and stale wheels."""
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        for egg_dir in self.src_dir.glob("*.egg-info"):
            shutil.rmtree(egg_dir)

        self.dist_dir.mkdir(parents=True, exist_ok=True)

    def compile_distribution_packages(self) -> Tuple[Path, Path]:
        """Compile binary wheel and source distribution archives via build module.

        Returns:
            Tuple[Path, Path]: Paths to the created wheel (.whl) and sdist (.tar.gz).

        Raises:
            RuntimeError: If the build process returns a non-zero exit code.
        """
        cmd = [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(self.dist_dir), str(self.root_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Package compilation failed (code {result.returncode}):\n{result.stderr}")

        wheels = list(self.dist_dir.glob("*.whl"))
        sdists = list(self.dist_dir.glob("*.tar.gz"))

        if not wheels or not sdists:
            raise RuntimeError("Build finished without generating required .whl or .tar.gz artifacts.")

        return wheels[0], sdists[0]

    def inspect_and_verify_wheel(self, wheel_path: Path) -> Dict[str, Any]:
        """Inspect the contents of a compiled wheel archive for compliance.

        Verifies presence of entrypoints, typing marker, static UI assets, and
        validates file checksums against the internal RECORD manifest.

        Args:
            wheel_path: Path to the .whl archive to inspect.

        Returns:
            Dict[str, Any]: Detailed inspection report.

        Raises:
            ValueError: If required assets or metadata entries are missing.
        """
        if not wheel_path.exists():
            raise FileNotFoundError(f"Wheel file not found: {wheel_path}")

        report: Dict[str, Any] = {
            "wheel_name": wheel_path.name,
            "size_bytes": wheel_path.stat().st_size,
            "entries_count": 0,
            "has_py_typed": False,
            "has_template_html": False,
            "has_entrypoints": False,
            "entrypoint_commands": [],
            "hash_verification_passed": True,
        }

        with zipfile.ZipFile(wheel_path, "r") as archive:
            namelist = archive.namelist()
            report["entries_count"] = len(namelist)

            for name in namelist:
                if name.endswith("arc_runner/py.typed"):
                    report["has_py_typed"] = True
                elif "templates/index.html" in name:
                    report["has_template_html"] = True
                elif name.endswith(".dist-info/entry_points.txt"):
                    report["has_entrypoints"] = True
                    content = archive.read(name).decode("utf-8")
                    for line in content.splitlines():
                        if "=" in line and not line.strip().startswith("["):
                            cmd_name = line.split("=")[0].strip()
                            report["entrypoint_commands"].append(cmd_name)

            if not report["has_py_typed"]:
                raise ValueError("Wheel verification failed: PEP 561 py.typed marker missing from archive.")
            if not report["has_template_html"]:
                raise ValueError("Wheel verification failed: Studio Web UI template templates/index.html missing from archive.")
            if not report["has_entrypoints"]:
                raise ValueError("Wheel verification failed: entry_points.txt missing from distribution.")

            expected_commands = {"arc-runner", "intel-arc-models-runner"}
            found_commands = set(report["entrypoint_commands"])
            if not expected_commands.issubset(found_commands):
                raise ValueError(
                    f"Wheel verification failed: Expected entry points {expected_commands}, found {found_commands}"
                )

            record_name = [n for n in namelist if n.endswith(".dist-info/RECORD")]
            if record_name:
                record_content = archive.read(record_name[0]).decode("utf-8")
                for line in record_content.splitlines():
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[1].startswith("sha256="):
                        entry_file = parts[0]
                        expected_hash = parts[1].split("=")[1]
                        file_data = archive.read(entry_file)
                        computed_hash = hashlib.sha256(file_data).hexdigest()
                        import base64
                        b64_hash = base64.urlsafe_b64encode(hashlib.sha256(file_data).digest()).decode("utf-8").rstrip("=")
                        if expected_hash not in (computed_hash, b64_hash):
                            report["hash_verification_passed"] = False
                            raise ValueError(f"Checksum mismatch for file {entry_file} inside wheel archive.")

        return report

    def run_complete_build_pipeline(self) -> int:
        """Execute the full build, inspection, and verification pipeline.

        Returns:
            int: 0 on success, non-zero on failure.
        """
        print("[1/3] Cleaning previous build artifacts...")
        self.clean_build_artifacts()

        print("[2/3] Compiling standalone wheel and sdist packages...")
        wheel_path, sdist_path = self.compile_distribution_packages()
        print(f"      Wheel created: {wheel_path.name} ({wheel_path.stat().st_size:,} bytes)")
        print(f"      Sdist created: {sdist_path.name} ({sdist_path.stat().st_size:,} bytes)")

        print("[3/3] Auditing wheel contents and package integrity...")
        report = self.inspect_and_verify_wheel(wheel_path)
        print(f"      Entries audited: {report['entries_count']}")
        print(f"      CLI Entrypoints: {report['entrypoint_commands']}")
        print(f"      PEP 561 py.typed: {'PASSED' if report['has_py_typed'] else 'FAILED'}")
        print(f"      Studio Web UI:    {'PASSED' if report['has_template_html'] else 'FAILED'}")
        print(f"      RECORD Hashes:    {'PASSED' if report['hash_verification_passed'] else 'FAILED'}")

        print("\n[SUCCESS] Standalone PyPI distribution package built and verified successfully!")
        return 0


def main() -> int:
    """CLI script entrypoint for WheelBuilder.

    Returns:
        int: Exit status code.
    """
    builder = WheelBuilder()
    try:
        return builder.run_complete_build_pipeline()
    except Exception as exc:
        print(f"\n[ERROR] Build pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
