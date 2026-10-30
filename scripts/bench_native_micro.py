"""Micro-benchmarks of the native mmap layer versus the Python fallback (plan benchmarks B8/B9).

Measures: tensor lookup latency, sequential read bandwidth of a mapped tensor versus file.read,
and ring-slot acquire latency under 1, 4 and 8 threads. Prints JSON; commits nothing. Numbers
describe the machine they ran on (record it); the cloud VM is not the target hardware.

Usage: python scripts/bench_native_micro.py [--smoke] [--out FILE]
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from arc_runner.benchmark.results import capture_environment  # noqa: E402
from arc_runner.benchmark.stats import iqr, median, student_t_ci95  # noqa: E402
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager  # noqa: E402
from arc_runner.native.bridge import NativeMemoryEngine  # noqa: E402


def summarize(values: List[float], unit: str) -> Dict[str, Any]:
    """Summarize repeated measurements.

    Args:
        values: Measurements.
        unit: Unit label.

    Returns:
        Dict[str, Any]: n, median, IQR and the 95% Student-t half-width.
    """
    _, half = student_t_ci95(values)
    return {"n": len(values), "median": round(median(values), 4), "iqr": round(iqr(values), 4), "ci95_half_width": round(half, 4), "unit": unit}


def repeat(fn: Callable[[], float], runs: int, warmup: int = 0) -> List[float]:
    """Call a measurement function repeatedly, discarding warmup invocations.

    Args:
        fn: Function returning one measurement.
        runs: Number of timed repetitions to return.
        warmup: Number of initial calls to discard before timing starts.

    Returns:
        List[float]: Timed measurements (warmup calls excluded).
    """
    for _ in range(warmup):
        fn()
    return [fn() for _ in range(runs)]


def lookup_latency_us(engine: NativeMemoryEngine, calls: int) -> float:
    """Mean microseconds per get_tensor_buffer call on a registered tensor.

    Args:
        engine: Engine with layer 0 tensor "w" registered.
        calls: Calls per measurement.
    """
    start = time.perf_counter()
    for _ in range(calls):
        engine.get_tensor_buffer(0, "w", dtype=np.uint8)
    return (time.perf_counter() - start) / calls * 1e6


def read_bandwidth_gbs(read: Callable[[], int]) -> float:
    """Run a read function and return GB/s.

    Args:
        read: Function returning the number of bytes it touched.
    """
    start = time.perf_counter()
    nbytes = read()
    return nbytes / (time.perf_counter() - start) / 1e9


def touch_mapped_tensor(engine: NativeMemoryEngine) -> int:
    """Read every byte of the mapped tensor through NumPy and return the byte count.

    Args:
        engine: Engine with layer 0 tensor "w" registered.

    Returns:
        int: Number of bytes read.
    """
    view = engine.get_tensor_buffer(0, "w", dtype=np.uint8)
    int(view.sum(dtype=np.uint64))
    return int(view.nbytes)


def acquire_latency_us(manager: FlashStreamWeightManager, threads: int, per_thread: int) -> float:
    """Mean microseconds per acquire of a resident layer with several threads.

    Args:
        manager: Manager with layer 0 already resident.
        threads: Number of concurrent threads.
        per_thread: Acquires per thread.
    """
    manager.acquire_layer_weights(0)
    barrier = threading.Barrier(threads)
    durations: List[float] = []

    def worker() -> None:
        barrier.wait()
        start = time.perf_counter()
        for _ in range(per_thread):
            manager.acquire_layer_weights(0)
        durations.append((time.perf_counter() - start) / per_thread * 1e6)

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return statistics.fmean(durations)


def main(argv: List[str]) -> int:
    """Run the micro-benchmarks and print a JSON report.

    Args:
        argv: Command line arguments without the program name.

    Returns:
        int: 0 on success.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, only to prove the harness runs")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations discarded before timing (default 2)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic blob content (default 0)")
    args = parser.parse_args(argv)
    runs, calls, size_mb, per_thread = (3, 200, 8, 200) if args.smoke else (15, 20000, 256, 20000)

    report: Dict[str, Any] = {"schema": "native_micro.v1", "smoke": args.smoke, "environment": capture_environment(REPO_ROOT)}
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp) / "blob.bin"
        blob.write_bytes(np.random.default_rng(args.seed).integers(0, 255, size_mb * 1024 * 1024, dtype=np.uint8).tobytes())
        engine = NativeMemoryEngine(blob)
        if not engine.is_open:
            report["native"] = {"status": "unavailable", "reason": "native library not loadable"}
        else:
            engine.register_layer(0, [{"name": "w", "tensor_type": 0, "dimensions": [size_mb * 1024 * 1024], "offset": 0, "size_bytes": size_mb * 1024 * 1024}])
            report["native"] = {
                "lookup_latency_us": summarize(repeat(lambda: lookup_latency_us(engine, calls), runs, warmup=args.warmup), "us/call"),
                "mapped_read_warm_gbs": summarize(repeat(lambda: read_bandwidth_gbs(lambda: touch_mapped_tensor(engine)), runs, warmup=args.warmup), "GB/s"),
            }
            engine.close()

        def file_read() -> int:
            with open(blob, "rb") as handle:
                return len(handle.read())

        report["python_file_read_gbs"] = summarize(repeat(lambda: read_bandwidth_gbs(file_read), runs, warmup=args.warmup), "GB/s")

        sys.path.insert(0, str(REPO_ROOT))
        from tests.test_flash_stream_triad import build_synthetic_flash_gguf

        model = build_synthetic_flash_gguf(Path(tmp) / "ring.gguf", num_layers=4, hidden_dim=32)
        report["ring_acquire_latency_us"] = {}
        for threads in (1, 4, 8):
            with FlashStreamWeightManager(model, ring_buffer_slots=3) as manager:
                report["ring_acquire_latency_us"][str(threads)] = summarize(repeat(lambda: acquire_latency_us(manager, threads, per_thread), runs, warmup=args.warmup), "us/acquire")

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
