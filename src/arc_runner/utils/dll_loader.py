"""Windows dynamic link library pre-flight resolution module.

Ensures that Intel Graphics drivers, Intel AI Boost NPU drivers, Level Zero
loaders, and OpenVINO runtime dependencies are discoverable prior to loading
native C++ extensions or SYCL engines on Microsoft Windows systems.
"""

import glob
import logging
import os
import platform
import sys
from typing import List

logger = logging.getLogger("arc_runner.utils.dll_loader")


def discover_intel_driver_store_paths() -> List[str]:
    """Scan the Windows DriverStore directory for Intel Graphics and NPU drivers.

    Returns:
        List[str]: Absolute directory paths containing candidate Intel driver libraries.
    """
    if platform.system() != "Windows":
        return []

    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    file_repository = os.path.join(system_root, "System32", "DriverStore", "FileRepository")
    if not os.path.isdir(file_repository):
        return []

    patterns = [
        os.path.join(file_repository, "iigd_dch.inf_*"),
        os.path.join(file_repository, "igdlh64.inf_*"),
        os.path.join(file_repository, "intel_npu.inf_*"),
        os.path.join(file_repository, "*intel*graphics*"),
        os.path.join(file_repository, "*intel*npu*"),
    ]

    discovered: List[str] = []
    for pattern in patterns:
        for match in glob.glob(pattern):
            if os.path.isdir(match) and match not in discovered:
                discovered.append(match)

    return discovered


def discover_openvino_runtime_paths() -> List[str]:
    """Identify directories containing OpenVINO runtime DLLs in the Python environment.

    Returns:
        List[str]: Absolute directory paths containing OpenVINO shared libraries.
    """
    paths: List[str] = []

    try:
        import openvino
        ov_dir = os.path.dirname(openvino.__file__)
        candidate_dirs = [
            ov_dir,
            os.path.join(ov_dir, "libs"),
            os.path.join(ov_dir, "..", "openvino_genai"),
            os.path.join(ov_dir, "..", "openvino_tokenizers"),
        ]
        for candidate in candidate_dirs:
            abs_path = os.path.abspath(candidate)
            if os.path.isdir(abs_path) and abs_path not in paths:
                paths.append(abs_path)
    except ImportError:
        pass

    return paths


def initialize_windows_dll_environment() -> List[str]:
    """Register all Intel driver and OpenVINO runtime directories on Windows.

    Calls os.add_dll_directory() for each valid candidate directory to prevent
    STATUS_DLL_NOT_FOUND (0xC0000135) and missing dependency errors when
    instantiating Level Zero or OpenVINO execution pipelines.

    Returns:
        List[str]: All directory paths successfully registered with the OS loader.
    """
    if platform.system() != "Windows":
        return []

    if not hasattr(os, "add_dll_directory"):
        return []

    candidate_paths: List[str] = []

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    if os.path.isdir(system32):
        candidate_paths.append(system32)

    candidate_paths.extend(discover_intel_driver_store_paths())
    candidate_paths.extend(discover_openvino_runtime_paths())

    path_env = os.environ.get("PATH", "")
    for segment in path_env.split(os.pathsep):
        if segment and os.path.isdir(segment) and segment not in candidate_paths:
            if "intel" in segment.lower() or "openvino" in segment.lower() or "oneapi" in segment.lower():
                candidate_paths.append(segment)

    registered: List[str] = []
    for directory in candidate_paths:
        try:
            os.add_dll_directory(directory)
            registered.append(directory)
        except (OSError, ValueError):
            pass

    return registered
