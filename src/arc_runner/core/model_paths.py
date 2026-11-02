"""OS-aware unified model storage resolution for Intel Arc Models Runner.

Provides cross-platform resolution and creation of the unified models directory
following standard operating system application data specifications (Windows LOCALAPPDATA,
Linux XDG Base Directory Specification, and macOS Application Support), with environment
variable override support.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional, Union

ENV_MODELS_DIR: str = "ARC_RUNNER_MODELS_DIR"


class ModelPathProvider:
    """Resolves and provisions OS-compliant model storage directories."""

    @classmethod
    def get_default_models_dir(cls) -> Path:
        """Resolve the standard platform-specific base directory for storing models.

        Inspects the ARC_RUNNER_MODELS_DIR environment variable first. If unset,
        resolves the platform-specific application data directory:
        - Windows: %LOCALAPPDATA%\\arc-runner\\models
        - macOS: ~/Library/Application Support/arc-runner/models
        - Linux / Other: $XDG_DATA_HOME/arc-runner/models or ~/.local/share/arc-runner/models

        Returns:
            Path: Resolved absolute path to the designated models directory.
        """
        env_override = os.environ.get(ENV_MODELS_DIR)
        if env_override:
            return Path(env_override).expanduser().resolve()

        if sys.platform == "win32":
            local_app_data = os.environ.get("LOCALAPPDATA")
            base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
            return (base / "arc-runner" / "models").resolve()

        if sys.platform == "darwin":
            return (Path.home() / "Library" / "Application Support" / "arc-runner" / "models").resolve()

        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
        return (base / "arc-runner" / "models").resolve()

    @classmethod
    def ensure_models_dir(cls, target_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create the model storage directory if it does not exist with secure permissions.

        Args:
            target_dir: Optional explicit directory path. If omitted, uses the default
                OS-specific path returned by get_default_models_dir().

        Returns:
            Path: Resolved absolute path of the created directory.

        Raises:
            OSError: If directory creation fails due to filesystem permissions or invalid paths.
        """
        resolved_path = Path(target_dir).expanduser().resolve() if target_dir else cls.get_default_models_dir()
        resolved_path.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            try:
                resolved_path.chmod(0o700)
            except OSError:
                pass
        return resolved_path
