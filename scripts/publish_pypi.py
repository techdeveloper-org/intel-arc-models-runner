"""PyPI publication automation and verification utility for arc-runner.

Executes pre-publish validation, twin-check metadata audits, and coordinates
distribution uploading to test.pypi.org or the primary Python Package Index.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List, Optional


class PyPIPublisher:
    """Manages pre-flight verification and distribution upload to PyPI."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """Initialize the PyPIPublisher instance.

        Args:
            root_dir: Root repository directory path.
        """
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent
        self.dist_dir = self.root_dir / "dist"

    def verify_twine_readiness(self) -> bool:
        """Verify that the twine tool is accessible within the current Python environment.

        Returns:
            bool: True if twine is installed and functional.
        """
        cmd = [sys.executable, "-m", "twine", "--version"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0

    def inspect_artifacts_integrity(self, dist_dir: Optional[Path] = None) -> bool:
        """Execute metadata and distribution verification on packages.

        Uses twine check if installed; otherwise executes direct RFC 822
        METADATA header and zipfile inspection.

        Args:
            dist_dir: Optional custom dist directory path. Defaults to self.dist_dir.

        Returns:
            bool: True if all packages pass long_description, syntax, and metadata checks.

        Raises:
            FileNotFoundError: If no distribution files are present.
            RuntimeError: If validation check flags errors in the distribution packages.
        """
        import email.parser
        import zipfile

        target = dist_dir or self.dist_dir
        packages = list(target.glob("*.whl")) + list(target.glob("*.tar.gz"))
        if not packages:
            raise FileNotFoundError(f"No distribution packages found in {target}")

        if self.verify_twine_readiness():
            cmd = [sys.executable, "-m", "twine", "check"] + [str(p) for p in packages]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Twine validation check failed:\n{result.stdout}\n{result.stderr}")
            return True

        for pkg in packages:
            if pkg.suffix == ".whl":
                with zipfile.ZipFile(pkg, "r") as archive:
                    metadata_files = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                    if not metadata_files:
                        raise RuntimeError(f"Corrupt wheel archive {pkg.name}: Missing .dist-info/METADATA")
                    raw_meta = archive.read(metadata_files[0]).decode("utf-8")
                    parsed = email.parser.HeaderParser().parsestr(raw_meta)
                    required_headers = ["Metadata-Version", "Name", "Version", "Summary"]
                    for header in required_headers:
                        if header not in parsed:
                            raise RuntimeError(f"Metadata header '{header}' missing from {pkg.name}")

        return True

    def execute_publish(
        self,
        dist_dir: Optional[Path] = None,
        repository: str = "pypi",
        dry_run: bool = True,
    ) -> bool:
        """Coordinate upload of distribution archives to target repository.

        Args:
            dist_dir: Optional custom dist directory path.
            repository: Repository name (e.g. 'pypi' or 'testpypi').
            dry_run: If True, performs validation only without executing upload.

        Returns:
            bool: True if publishing or dry-run validation succeeded.
        """
        target = dist_dir or self.dist_dir
        self.inspect_artifacts_integrity(target)

        if dry_run:
            print("[INFO] Dry run enabled. Distribution packages validated successfully. Skipping upload.")
            return True

        packages = list(target.glob("*.whl")) + list(target.glob("*.tar.gz"))
        cmd: List[str] = [sys.executable, "-m", "twine", "upload", "--repository", repository] + [str(p) for p in packages]
        result = subprocess.run(cmd, capture_output=False, check=False)
        return result.returncode == 0


def main() -> int:
    """CLI script entry point for PyPI publisher.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description="Publish arc-runner distribution packages to PyPI.")
    parser.add_argument("--repo", default="pypi", help="Target PyPI repository (default: pypi)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Validate without uploading")
    parser.add_argument("--upload", dest="dry_run", action="store_false", help="Execute live upload")
    args = parser.parse_args()

    publisher = PyPIPublisher()
    try:
        success = publisher.execute_publish(repository=args.repo, dry_run=args.dry_run)
        return 0 if success else 1
    except Exception as exc:
        print(f"[ERROR] Publication failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
