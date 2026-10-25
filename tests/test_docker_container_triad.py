"""Triad test suite for Docker containerization and Ubuntu 24.04 runtime deployment.

Enforces Triad testing patterns (Positive, Negative, Edge-Case) verifying
Dockerfile multi-stage structure, Rule 29-container-deployment non-root security,
/dev/dri hardware device mapping, entrypoint fallback semantics, and Linux
Level Zero dynamic library resolution.
"""

from pathlib import Path
import subprocess
import sys
from typing import Dict, List
import unittest.mock as mock
import pytest
import yaml

from arc_runner.devices.sysman import LevelZeroSysman
from arc_runner.utils.dll_loader import (
    discover_linux_level_zero_paths,
    initialize_driver_environment,
    initialize_linux_driver_environment,
)


class TestDockerContainerPositiveTriad:
    """Positive test cases verifying nominal Docker container configuration invariants."""

    def test_dockerfile_syntax_and_multi_stage_structure(self) -> None:
        """Verify Dockerfile contains valid multi-stage builder and runtime targets."""
        dockerfile_path = Path("Dockerfile")
        assert dockerfile_path.is_file(), "Dockerfile not found in project root"

        content = dockerfile_path.read_text(encoding="utf-8")
        assert "FROM ubuntu:24.04 AS builder" in content
        assert "FROM ubuntu:24.04 AS runtime" in content
        assert "COPY --from=builder /opt/venv /opt/venv" in content
        assert "intel-level-zero-gpu" in content
        assert "intel-opencl-icd" in content

    def test_dockerfile_non_root_security_enforcement(self) -> None:
        """Verify strict adherence to Rule 29-container-deployment non-root user rules."""
        content = Path("Dockerfile").read_text(encoding="utf-8")

        assert "USER appuser" in content
        assert "groupadd --system --gid 1000 appgroup" in content
        assert "useradd --system --uid 1000" in content
        assert "usermod -aG video,render appuser" in content

    def test_dockerfile_healthcheck_contract(self) -> None:
        """Verify HEALTHCHECK definition complies with platform standards."""
        content = Path("Dockerfile").read_text(encoding="utf-8")

        assert "HEALTHCHECK" in content
        assert "http://localhost:8000/v1/health" in content
        assert "--interval=30s" in content
        assert "--timeout=5s" in content
        assert "--retries=3" in content

    def test_docker_compose_device_passthrough_config(self) -> None:
        """Parse docker-compose.yml and verify /dev/dri pass-through and caching volumes."""
        compose_path = Path("docker-compose.yml")
        assert compose_path.is_file()

        with open(compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "services" in data
        assert "arc-runner" in data["services"]
        service = data["services"]["arc-runner"]

        assert "/dev/dri:/dev/dri" in service.get("devices", [])
        assert "8000:8000" in service.get("ports", [])
        assert "volumes" in data
        assert "arc-models-cache" in data["volumes"]
        assert "arc-aot-cache" in data["volumes"]

    def test_docker_entrypoint_executes_cli(self) -> None:
        """Verify entrypoint script structure executes CLI commands without subshell wrap."""
        entrypoint_path = Path("docker-entrypoint.sh")
        assert entrypoint_path.is_file()

        content = entrypoint_path.read_text(encoding="utf-8")
        assert "#!/bin/sh" in content
        assert "set -e" in content
        assert 'exec arc-runner "$@"' in content
        assert 'exec "$@"' in content


class TestDockerContainerNegativeTriad:
    """Negative test cases verifying edge error handling and permission guards."""

    def test_docker_entrypoint_missing_dri_graceful_notice(self) -> None:
        """Verify that when /dev/dri is missing, entrypoint emits informative notice."""
        content = Path("docker-entrypoint.sh").read_text(encoding="utf-8")
        assert 'if [ -d "/dev/dri" ];' in content
        assert "/dev/dri not mapped into container" in content
        assert "Heterogeneous GPU offload unavailable; using CPU" in content

    def test_docker_entrypoint_permission_denied_warning(self) -> None:
        """Verify that entrypoint detects unwritable renderD128 and prints guidance."""
        content = Path("docker-entrypoint.sh").read_text(encoding="utf-8")
        assert 'if [ -r "/dev/dri/renderD128" ] && [ -w "/dev/dri/renderD128" ];' in content
        assert "Verify host video/render GIDs" in content

    def test_linux_level_zero_loader_handles_absent_libraries(self) -> None:
        """Verify that LevelZeroSysman handles missing Level Zero libraries without crash."""
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("ctypes.CDLL", side_effect=OSError("Cannot load library")):
                sysman = LevelZeroSysman()
                assert not sysman.is_available
                total, free = sysman.query_gpu_memory(0)
                assert total is None
                assert free is None


class TestDockerContainerEdgeCaseTriad:
    """Edge-case test cases verifying Linux driver paths, signal propagation, and hygiene."""

    def test_linux_level_zero_dynamic_loader_fallback(self) -> None:
        """Verify that LevelZeroSysman searches candidate .so names on Linux platforms."""
        attempted_loads: List[str] = []

        def mock_cdll(name: str) -> mock.MagicMock:
            attempted_loads.append(name)
            if name == "libze_loader.so.1":
                return mock.MagicMock()
            raise OSError(f"File {name} not found")

        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("ctypes.CDLL", side_effect=mock_cdll):
                sysman = LevelZeroSysman()
                assert sysman.is_available
                assert "libze_loader.so.1" in attempted_loads

    def test_discover_linux_level_zero_paths_filtering(self) -> None:
        """Verify discover_linux_level_zero_paths correctly filters existing shared libraries."""
        with mock.patch("os.path.isdir", side_effect=lambda d: d == "/usr/lib/x86_64-linux-gnu"):
            with mock.patch("os.path.isfile", side_effect=lambda f: "libze_loader.so.1" in f):
                paths = discover_linux_level_zero_paths()
                assert len(paths) == 1
                assert paths[0] == "/usr/lib/x86_64-linux-gnu/libze_loader.so.1"

    def test_dockerignore_excludes_all_sensitive_and_intermediate_artifacts(self) -> None:
        """Verify that .dockerignore excludes heavy or sensitive repository assets."""
        dockerignore_path = Path(".dockerignore")
        assert dockerignore_path.is_file()

        lines = [line.strip() for line in dockerignore_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        critical_exclusions = {".git", ".pytest_cache", ".coverage", "build", "dist", "models", "tests", "*.log"}

        for exclusion in critical_exclusions:
            assert exclusion in lines, f"Missing critical exclusion '{exclusion}' in .dockerignore"
