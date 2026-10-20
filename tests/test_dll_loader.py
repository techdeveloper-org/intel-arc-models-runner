"""Unit tests for Windows DLL pre-flight loader.

Validates driver discovery, runtime path extraction, and safe os.add_dll_directory
registration in accordance with Rule 41 testing standards.
"""

import os
import platform
import unittest.mock as mock
from typing import List

import pytest

from arc_runner.utils.dll_loader import (
    discover_intel_driver_store_paths,
    discover_openvino_runtime_paths,
    initialize_windows_dll_environment,
)


def test_discover_openvino_runtime_paths_returns_existing_directories() -> None:
    """Verify that OpenVINO runtime discovery identifies existing directories on disk."""
    paths: List[str] = discover_openvino_runtime_paths()
    assert isinstance(paths, list)
    for path in paths:
        assert os.path.isdir(path)


def test_discover_intel_driver_store_paths_on_windows() -> None:
    """Verify that DriverStore scanner returns a list of strings without raising errors."""
    paths: List[str] = discover_intel_driver_store_paths()
    assert isinstance(paths, list)
    if platform.system() == "Windows":
        for path in paths:
            assert isinstance(path, str)


def test_discover_intel_driver_store_paths_on_non_windows() -> None:
    """Verify that DriverStore scanner returns empty list on non-Windows platforms."""
    with mock.patch("platform.system", return_value="Linux"):
        paths: List[str] = discover_intel_driver_store_paths()
        assert paths == []


def test_initialize_windows_dll_environment_execution() -> None:
    """Verify that initialize_windows_dll_environment runs without unhandled exceptions."""
    registered: List[str] = initialize_windows_dll_environment()
    assert isinstance(registered, list)


def test_initialize_windows_dll_environment_on_linux() -> None:
    """Verify that environment initialization safely no-ops on Linux."""
    with mock.patch("platform.system", return_value="Linux"):
        registered: List[str] = initialize_windows_dll_environment()
        assert registered == []


def test_initialize_windows_dll_environment_handles_os_error() -> None:
    """Verify that os.add_dll_directory exceptions are caught and suppressed."""
    if not hasattr(os, "add_dll_directory"):
        pytest.skip("os.add_dll_directory not supported on this platform")

    with mock.patch("os.add_dll_directory", side_effect=OSError("Invalid path")):
        registered: List[str] = initialize_windows_dll_environment()
        assert registered == []
