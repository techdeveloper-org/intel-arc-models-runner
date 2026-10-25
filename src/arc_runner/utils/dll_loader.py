"""Cross-platform dynamic link library and shared object pre-flight resolution module.

Ensures that Intel Graphics drivers, Intel AI Boost NPU drivers, Level Zero
loaders, and OpenVINO runtime dependencies are discoverable prior to loading
native C++ extensions or SYCL engines across Windows and Linux systems.
"""

import ctypes
import glob
import logging
import os
import platform
import sys
from typing import List, Optional

logger = logging.getLogger("arc_runner.utils.dll_loader")

LINUX_CANDIDATE_PATHS: List[str] = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/local/lib",
    "/opt/intel/oneapi/compiler/latest/lib",
    "/opt/intel/oneapi/redist/lib",
    "/opt/intel/oneapi/lib64",
]

LEVEL_ZERO_SO_NAMES: List[str] = [
    "libze_loader.so.1",
    "libze_loader.so",
    "libze_intel_gpu.so.1",
]


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


def discover_linux_level_zero_paths() -> List[str]:
    """Scan candidate Linux directories for Intel Level Zero and oneAPI shared libraries.

    Returns:
        List[str]: Absolute file paths to discovered Level Zero shared objects.
    """
    discovered: List[str] = []
    for base_dir in LINUX_CANDIDATE_PATHS:
        if not os.path.isdir(base_dir):
            continue
        for so_name in LEVEL_ZERO_SO_NAMES:
            so_path = os.path.join(base_dir, so_name).replace("\\", "/")
            if os.path.isfile(so_path) and so_path not in discovered:
                discovered.append(so_path)
    return discovered


def initialize_linux_driver_environment() -> List[str]:
    """Pre-load Intel Level Zero runtime shared libraries on Linux platforms.

    Loads discovered shared libraries into the process global symbol table using
    RTLD_GLOBAL to ensure OpenVINO and hardware execution backends can locate
    the Level Zero symbols without runtime link failure.

    Returns:
        List[str]: File paths of all shared libraries loaded into memory.
    """
    if platform.system() == "Windows":
        return []

    loaded: List[str] = []
    candidate_sos = discover_linux_level_zero_paths()
    for so_path in candidate_sos:
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(so_path)
            logger.info("Successfully loaded Level Zero runtime library: %s", so_path)
        except (OSError, Exception) as exc:
            logger.debug("Could not load candidate library %s: %s", so_path, exc)

    return loaded


def initialize_driver_environment() -> List[str]:
    """Initialize driver dependencies across Windows and Linux platforms.

    Returns:
        List[str]: Registered or loaded driver library paths.
    """
    if platform.system() == "Windows":
        return initialize_windows_dll_environment()
    return initialize_linux_driver_environment()
