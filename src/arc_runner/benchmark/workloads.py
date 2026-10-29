"""Shared measurement harness for non-LLM workloads (embeddings, VLM, audio, image, video).

Every value comes from a timed run of the real workload; package power and energy come from the
Sysman energy counter and are None when no counter exists. Nothing is estimated.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from arc_runner.benchmark.power import PowerMeter, PowerReading
from arc_runner.benchmark.results import build_result, capture_environment, write_result
from arc_runner.benchmark.stats import BenchSample


@dataclass(frozen=True)
class TimedRun:
    """One executed run of a workload.

    Attributes:
        iteration: Zero-based index across warmups and timed runs.
        is_warmup: True for discarded warmup runs.
        elapsed_s: Wall-clock seconds of the run.
        power: Package energy and power over the run (values None if unmeasurable).
    """

    iteration: int
    is_warmup: bool
    elapsed_s: float
    power: PowerReading


def measure_workload(
    fn: Callable[[], Any],
    warmups: int,
    runs: int,
    sysman: Any = None,
    device_index: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> List[TimedRun]:
    """Run a workload for warmups then timed runs, measuring latency and package power for each.

    Args:
        fn: The workload; called with no arguments. Exceptions propagate and stop the measurement.
        warmups: Discarded warmup runs executed first (zero or more).
        runs: Timed runs (at least one).
        sysman: LevelZeroSysman (or compatible) for package energy, or None for no power measurement.
        device_index: GPU index whose package domain is read.
        clock: Monotonic clock returning seconds.

    Returns:
        List[TimedRun]: One record per executed run, warmups first.

    Raises:
        ValueError: If warmups is negative or runs is below one.
    """
    if warmups < 0 or runs < 1:
        raise ValueError(f"need warmups >= 0 and runs >= 1, got warmups={warmups}, runs={runs}")
    records: List[TimedRun] = []
    for iteration in range(warmups + runs):
        meter = PowerMeter(sysman, device_index)
        meter.start()
        started = clock()
        fn()
        elapsed = clock() - started
        records.append(TimedRun(iteration, iteration < warmups, elapsed, meter.stop()))
    return records


def workload_samples(
    model_id: str,
    backend: str,
    workload: str,
    runs: Iterable[TimedRun],
    extra: Optional[Callable[[TimedRun], Dict[str, Optional[float]]]] = None,
    fallback_used: bool = False,
) -> List[BenchSample]:
    """Convert timed runs into schema-v1 samples.

    Each sample reports latency_s, package_power_w and package_energy_j (the latter two None when no
    counter existed) plus any metrics derived by the caller from the measured latency and known
    workload constants (for example rtf = latency_s / audio_duration_s).

    Args:
        model_id: Model and workload-tier identifier.
        backend: Backend name; its device/quantization label goes in the protocol labels.
        workload: One of the schema workload names other than llm.
        runs: Timed runs from measure_workload.
        extra: Optional function returning additional metrics for a run.
        fallback_used: True if the run executed on a fallback device or path.

    Returns:
        List[BenchSample]: Measured samples in run order.
    """
    samples: List[BenchSample] = []
    for run in runs:
        metrics: Dict[str, Optional[float]] = {
            "latency_s": run.elapsed_s,
            "package_power_w": run.power.avg_power_w,
            "package_energy_j": run.power.energy_j,
        }
        if extra is not None:
            metrics.update(extra(run))
        samples.append(BenchSample(
            model_id, backend, run.iteration, None, None, 0, 0, "measured",
            is_warmup=run.is_warmup, fallback_used=fallback_used, workload=workload, metrics=metrics,
        ))
    return samples


def export_workload_result(
    path: Path,
    run_id: str,
    samples: List[BenchSample],
    labels: Dict[str, str],
    warmups: int,
    runs: int,
    repo_dir: Path,
) -> None:
    """Validate and write a schema-v1 result file for non-LLM samples.

    Args:
        path: Destination file.
        run_id: Unique run identifier.
        samples: Samples from workload_samples.
        labels: Backend -> device and quantization label printed next to every row.
        warmups: Warmup runs per series.
        runs: Timed runs per series.
        repo_dir: Repository directory, for the git SHA in the environment block.

    Raises:
        ValueError: If the document violates the schema.
    """
    protocol = {"warmup_runs": warmups, "timed_runs": runs, "labels": labels}
    write_result(path, build_result(run_id, samples, protocol, capture_environment(repo_dir)))
