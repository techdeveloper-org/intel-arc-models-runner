"""Probe the hardware facts the plan cannot learn from the cloud and print them as JSON.

Run once on the target machine:  python scripts/probe_hardware.py > probe.json
Each fact has a status of "ok" or "unavailable" and the raw value; nothing is guessed. Stdlib only,
with optional imports (numpy, openvino, ctypes access to Level Zero).
"""

import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def fact(read: Callable[[], Any]) -> Dict[str, Any]:
    """Run a probe and wrap its result as an ok/unavailable record.

    Args:
        read: Callable returning a JSON-serializable value.

    Returns:
        Dict[str, Any]: {"status": "ok", "value": ...} or {"status": "unavailable", "reason": ...}.
    """
    try:
        return {"status": "ok", "value": read()}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def memory_bandwidth_gbs() -> Dict[str, float]:
    """Measure copy and triad bandwidth with NumPy on a buffer larger than the caches.

    Returns:
        Dict[str, float]: Sustained GB/s for copy and triad (best of five).
    """
    import numpy as np

    n = 64 * 1024 * 1024 // 8
    a, b, c = np.ones(n), np.full(n, 2.0), np.zeros(n)
    best = {"copy": 0.0, "triad": 0.0}
    for _ in range(5):
        start = time.perf_counter()
        np.copyto(c, a)
        best["copy"] = max(best["copy"], 2 * a.nbytes / (time.perf_counter() - start) / 1e9)
        start = time.perf_counter()
        np.multiply(b, 3.0, out=c)
        np.add(c, a, out=c)
        best["triad"] = max(best["triad"], 3 * a.nbytes / (time.perf_counter() - start) / 1e9)
    return {k: round(v, 2) for k, v in best.items()}


def level_zero_devices() -> Any:
    """Enumerate Level Zero drivers and devices with their type, id and name.

    Returns:
        Any: List of {driver, type, device_id, name} records.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from arc_runner.devices.sysman import LevelZeroSysman, _ZeDeviceProps

    lib_names = ["ze_loader.dll"] if platform.system() == "Windows" else ["libze_loader.so.1", "libze_loader.so"]
    lib = None
    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError("Level Zero loader not found")
    lib.zeInit.argtypes = [ctypes.c_uint32]
    lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.zeDeviceGet.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.zeDeviceGetProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZeDeviceProps)]
    if lib.zeInit(0) != 0:
        raise RuntimeError("zeInit failed")
    count = ctypes.c_uint32(0)
    lib.zeDriverGet(ctypes.byref(count), None)
    drivers = (ctypes.c_void_p * count.value)()
    lib.zeDriverGet(ctypes.byref(count), drivers)
    records = []
    for index, driver in enumerate(drivers):
        dev_count = ctypes.c_uint32(0)
        lib.zeDeviceGet(driver, ctypes.byref(dev_count), None)
        devices = (ctypes.c_void_p * dev_count.value)()
        lib.zeDeviceGet(driver, ctypes.byref(dev_count), devices)
        for device in devices:
            props = _ZeDeviceProps()
            props.stype = 0x3
            lib.zeDeviceGetProperties(device, ctypes.byref(props))
            records.append({"driver": index, "type": props.type, "device_id": hex(props.deviceId), "name": props.name.decode(errors="replace")})
    _ = LevelZeroSysman
    return records


def cpu_description() -> str:
    """Return a CPU description string.

    Returns:
        str: Model name where the platform exposes it, else platform.processor().
    """
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Windows":
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20).stdout
        return " ".join(out.split()[1:])
    return platform.processor()


def memory_modules() -> Any:
    """List physical memory modules (capacity, configured speed, type) on Windows.

    Returns:
        Any: Raw wmic output lines.
    """
    if platform.system() != "Windows":
        raise RuntimeError("only implemented on Windows")
    out = subprocess.run(
        ["wmic", "memorychip", "get", "Capacity,ConfiguredClockSpeed,Speed,SMBIOSMemoryType"],
        capture_output=True, text=True, timeout=20,
    ).stdout
    return [line.split() for line in out.splitlines() if line.strip()]


def openvino_devices() -> Any:
    """List the devices OpenVINO can see.

    Returns:
        Any: Device names with their FULL_DEVICE_NAME property.
    """
    import openvino as ov

    core = ov.Core()
    return {device: str(core.get_property(device, "FULL_DEVICE_NAME")) for device in core.available_devices}


def perf_metrics_api() -> Any:
    """Report the installed openvino_genai version and the PerfMetrics accessors present.

    Returns:
        Any: Version and accessor names.
    """
    import openvino_genai as genai

    names = [n for n in dir(genai.PerfMetrics) if n.startswith("get_")]
    return {"version": genai.__version__, "accessors": names}


def main() -> int:
    """Run every probe and print one JSON document.

    Returns:
        int: Always 0; unavailable facts are reported, not fatal.
    """
    report = {
        "schema": "probe_hardware.v1",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": fact(cpu_description),
        "logical_cpus": os.cpu_count(),
        "memory_modules": fact(memory_modules),
        "memory_bandwidth_gbs": fact(memory_bandwidth_gbs),
        "level_zero_devices": fact(level_zero_devices),
        "openvino_devices": fact(openvino_devices),
        "openvino_genai": fact(perf_metrics_api),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
