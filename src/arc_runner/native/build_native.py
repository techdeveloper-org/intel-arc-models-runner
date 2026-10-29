"""Build script that compiles the native acceleration library and records its provenance.

Compiles arc_flash_stream_mmap.c and arc_level_zero_bridge.c into libarc_native_sys,
writes the result atomically, and emits a sidecar JSON file with the sha256, compiler
version, flags, and ABI version so the Python bridge can verify the binary it loads.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

ABI_VERSION = 2
BUILD_INFO_SUFFIX = ".build.json"
MMAP_SOURCE = "arc_flash_stream_mmap.c"
LEVEL_ZERO_SOURCE = "arc_level_zero_bridge.c"
COMMON_FLAGS = ["-O2", "-std=c17", "-Wall", "-Wextra", "-Werror"]
STRICT_FLAGS = ["-Wpedantic"]
LINK_FLAGS = ["-shared", "-Wl,--no-undefined"]


def find_compiler(override: Optional[str] = None) -> str:
    """Locate the C compiler to use.

    Args:
        override: Compiler executable name or path; defaults to the CC environment variable, then gcc.

    Returns:
        str: Absolute path of the compiler executable.

    Raises:
        RuntimeError: If no compiler can be found on PATH.
    """
    candidate = override or os.environ.get("CC") or "gcc"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise RuntimeError(f"C compiler {candidate!r} not found on PATH; install gcc (MinGW-w64 on Windows)")
    return resolved


def library_name(target_windows: Optional[bool] = None) -> str:
    """Return the shared library file name for the target platform.

    Args:
        target_windows: True for a Windows target, False for POSIX; defaults to the host platform.

    Returns:
        str: libarc_native_sys.dll or libarc_native_sys.so.
    """
    is_windows = sys.platform == "win32" if target_windows is None else target_windows
    return "libarc_native_sys.dll" if is_windows else "libarc_native_sys.so"


def _object_flags(is_windows: bool) -> Dict[str, List[str]]:
    """Return per-source compile flags for the target platform.

    Args:
        is_windows: True when compiling for Windows.

    Returns:
        Dict[str, List[str]]: Flags keyed by source file name.
    """
    position_independent = [] if is_windows else ["-fPIC"]
    return {
        MMAP_SOURCE: COMMON_FLAGS + STRICT_FLAGS + position_independent,
        LEVEL_ZERO_SOURCE: COMMON_FLAGS + position_independent,
    }


def _run(cmd: List[str]) -> None:
    """Run a compiler command and raise with its output on failure.

    Args:
        cmd: Command and arguments.

    Raises:
        RuntimeError: If the command exits non-zero.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Native build step failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def compile_library(output: Path, compiler: Optional[str] = None, target_windows: Optional[bool] = None) -> None:
    """Compile and link the native sources, replacing the output atomically.

    Args:
        output: Destination path of the shared library.
        compiler: Optional compiler executable (for example x86_64-w64-mingw32-gcc).
        target_windows: Override of the target platform; defaults to the host.

    Raises:
        RuntimeError: If compilation or linking fails.
    """
    native_dir = Path(__file__).resolve().parent
    cc = find_compiler(compiler)
    is_windows = sys.platform == "win32" if target_windows is None else target_windows
    flags = _object_flags(is_windows)

    with tempfile.TemporaryDirectory() as tmp:
        objects: List[str] = []
        for source, source_flags in flags.items():
            obj = str(Path(tmp) / (Path(source).stem + ".o"))
            _run([cc, *source_flags, "-c", str(native_dir / source), "-o", obj])
            objects.append(obj)

        staged = Path(tmp) / output.name
        link_libs = [] if is_windows else ["-ldl"]
        _run([cc, *LINK_FLAGS, *objects, *link_libs, "-o", str(staged)])
        pending = output.with_name(output.name + ".tmp")
        shutil.copyfile(staged, pending)
        os.replace(pending, output)


def _git_sha(repo_dir: Path) -> str:
    """Return the current git commit SHA, or 'unknown' outside a repository.

    Args:
        repo_dir: Directory inside the repository.

    Returns:
        str: Full commit SHA or 'unknown'.
    """
    result = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_build_info(library: Path, compiler: str, target_windows: Optional[bool] = None) -> Path:
    """Write the sidecar JSON describing a built library.

    Args:
        library: Path to the built shared library.
        compiler: Compiler executable used.
        target_windows: Target platform override used for the build.

    Returns:
        Path: Path of the written sidecar file.
    """
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    is_windows = sys.platform == "win32" if target_windows is None else target_windows
    info = {
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "abi_version": ABI_VERSION,
        "compiler": version[0] if version else compiler,
        "flags": _object_flags(is_windows),
        "link_flags": LINK_FLAGS,
        "git_sha": _git_sha(library.parent),
        "host": platform.platform(),
    }
    sidecar = library.with_name(library.name + BUILD_INFO_SUFFIX)
    sidecar.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def main(argv: Optional[List[str]] = None) -> int:
    """Build the library for the host platform and write its sidecar.

    Args:
        argv: Optional arguments; supports --cc <compiler> and --windows to cross-compile.

    Returns:
        int: Process exit code, 0 on success.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    compiler = args[args.index("--cc") + 1] if "--cc" in args else None
    target_windows = True if "--windows" in args else None

    output = Path(__file__).resolve().parent / library_name(target_windows)
    try:
        compile_library(output, compiler, target_windows)
        sidecar = write_build_info(output, find_compiler(compiler), target_windows)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Built {output} ({output.stat().st_size} bytes); sidecar {sidecar.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
