"""Unit and integration test suite for OS-aware ModelPathProvider.

Validates platform-specific path resolution, environment variable overrides,
and filesystem directory provisioning adhering to system standards.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from arc_runner.core.model_paths import ENV_MODELS_DIR, ModelPathProvider


class TestModelPathProvider:
    """Test suite covering OS-aware path resolution and provisioning."""

    def test_default_models_dir_platform_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify standard path resolution when no environment override is present."""
        monkeypatch.delenv(ENV_MODELS_DIR, raising=False)
        resolved = ModelPathProvider.get_default_models_dir()
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()
        assert "arc-runner" in str(resolved)
        assert str(resolved).endswith("models")

    def test_env_override_takes_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that ARC_RUNNER_MODELS_DIR overrides default platform paths."""
        custom_dir = tmp_path / "custom_models_storage"
        monkeypatch.setenv(ENV_MODELS_DIR, str(custom_dir))
        resolved = ModelPathProvider.get_default_models_dir()
        assert resolved == custom_dir.resolve()

    def test_ensure_models_dir_creates_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that ensure_models_dir provisions non-existent storage folders."""
        target_dir = tmp_path / "provisioned_models"
        assert not target_dir.exists()
        created = ModelPathProvider.ensure_models_dir(target_dir)
        assert created.exists()
        assert created.is_dir()
        assert created == target_dir.resolve()

    def test_ensure_models_dir_idempotent(self, tmp_path: Path) -> None:
        """Verify that repeated calls to ensure_models_dir do not raise errors."""
        target_dir = tmp_path / "idempotent_models"
        first = ModelPathProvider.ensure_models_dir(target_dir)
        second = ModelPathProvider.ensure_models_dir(target_dir)
        assert first == second
        assert first.exists()
