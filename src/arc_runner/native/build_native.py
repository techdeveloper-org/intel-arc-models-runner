"""Build script to compile native C/C++ acceleration library using GCC.

Compiles arc_flash_stream_mmap.c and arc_level_zero_bridge.c into
libarc_native_sys.dll for high-performance memory streaming.
"""

from pathlib import Path
import subprocess
import sys


def compile_native_library() -> Path:
    """Compile C source files into shared library.

    Returns:
        Path: Path to the generated shared library.

    Raises:
        RuntimeError: If compiler exits with non-zero status.
    """
    native_dir = Path(__file__).resolve().parent
    src_files = [
        str(native_dir / "arc_flash_stream_mmap.c"),
        str(native_dir / "arc_level_zero_bridge.c"),
    ]

    out_file = native_dir / ("libarc_native_sys.dll" if sys.platform == "win32" else "libarc_native_sys.so")

    cmd = [
        "gcc",
        "-O3",
        "-mavx2",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        *src_files,
        "-o",
        str(out_file),
    ]

    print(f"Compiling native acceleration library: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Native compilation failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    print(f"Successfully built {out_file} ({out_file.stat().st_size} bytes)")
    return out_file


if __name__ == "__main__":
    compile_native_library()
