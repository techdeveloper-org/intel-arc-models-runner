"""Shared measurement harness for non-LLM workloads (embeddings, VLM, audio, image, video).

Every value comes from a timed run of the real workload; package power and energy come from the
Sysman energy counter and are None when no counter exists. Nothing is estimated.
"""

from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from arc_runner.benchmark.power import PowerMeter, PowerReading
from arc_runner.benchmark.results import build_result, capture_environment, write_result
from arc_runner.benchmark.stats import MIN_WARMUPS, WARMUP_CV, BenchSample, warmup_converged


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


def _timed_call(
    fn: Callable[[], Any], iteration: int, is_warmup: bool, sysman: Any, device_index: int, clock: Callable[[], float],
) -> TimedRun:
    """Execute one run of a workload under a power meter and a clock.

    Args:
        fn: The workload.
        iteration: Index recorded on the run.
        is_warmup: Whether the run is a discarded warmup.
        sysman: Sysman for package energy, or None.
        device_index: GPU index whose package domain is read.
        clock: Monotonic clock returning seconds.
    """
    meter = PowerMeter(sysman, device_index)
    meter.start()
    started = clock()
    fn()
    elapsed = clock() - started
    return TimedRun(iteration, is_warmup, elapsed, meter.stop())


def _tail_converged(values: List[Optional[float]]) -> bool:
    """Tell whether the last warmup values exist and agree within the warmup coefficient of variation.

    Args:
        values: Warmup metric values in order; None where the metric was not measured.
    """
    tail = values[-MIN_WARMUPS:]
    return len(tail) == MIN_WARMUPS and all(v is not None for v in tail) and warmup_converged([v for v in tail if v is not None], MIN_WARMUPS, WARMUP_CV)


def interleaved_order(names: Sequence[str], iteration: int, seed: int) -> List[str]:
    """Return a seeded random order of the systems for one iteration; the same inputs give the same order.

    Args:
        names: System names.
        iteration: Iteration index (warmups and timed runs share one counter).
        seed: Run seed.

    Returns:
        List[str]: The names in the order they run this iteration.
    """
    order = list(names)
    random.Random(f"{seed}:{iteration}").shuffle(order)
    return order


def measure_workload(
    fn: Callable[[], Any],
    warmups: int,
    runs: int,
    sysman: Any = None,
    device_index: int = 0,
    clock: Callable[[], float] = time.perf_counter,
    max_warmups: Optional[int] = None,
    warmup_metric: Optional[Callable[[TimedRun], Optional[float]]] = None,
) -> List[TimedRun]:
    """Run a workload for warmups then timed runs, measuring latency and package power for each.

    With max_warmups set, warmups continue past `warmups` until the last three values of the warmup metric
    have a coefficient of variation of at most 5% or max_warmups warmups have run. Whether a series
    converged is not returned; it is recomputed from the warmup rows (harness_common.series_quality).

    Args:
        fn: The workload; called with no arguments. Exceptions propagate and stop the measurement.
        warmups: Warmup runs executed first (zero or more); the minimum when max_warmups is set.
        runs: Timed runs (at least one).
        sysman: LevelZeroSysman (or compatible) for package energy, or None for no power measurement.
        device_index: GPU index whose package domain is read.
        clock: Monotonic clock returning seconds.
        max_warmups: Upper bound for convergence-driven warmups, or None for exactly `warmups`.
        warmup_metric: Maps a warmup run to the value whose convergence is checked (default elapsed_s).

    Returns:
        List[TimedRun]: One record per executed run, warmups first.

    Raises:
        ValueError: If warmups is negative or runs is below one.
    """
    if warmups < 0 or runs < 1:
        raise ValueError(f"need warmups >= 0 and runs >= 1, got warmups={warmups}, runs={runs}")
    metric = warmup_metric or (lambda run: run.elapsed_s)
    limit = warmups if max_warmups is None else max(warmups, max_warmups)
    records: List[TimedRun] = []
    values: List[Optional[float]] = []
    while len(records) < limit:
        if len(records) >= warmups and _tail_converged(values):
            break
        record = _timed_call(fn, len(records), True, sysman, device_index, clock)
        records.append(record)
        values.append(metric(record))
    for _ in range(runs):
        records.append(_timed_call(fn, len(records), False, sysman, device_index, clock))
    return records


def measure_interleaved(
    fns: Dict[str, Callable[[], Any]],
    warmups: int,
    runs: int,
    seed: int = 0,
    sysman: Any = None,
    device_index: int = 0,
    clock: Callable[[], float] = time.perf_counter,
    max_warmups: Optional[int] = None,
    warmup_metrics: Optional[Dict[str, Callable[[TimedRun], Optional[float]]]] = None,
) -> Dict[str, List[TimedRun]]:
    """Measure several systems with their runs interleaved per iteration in a seeded random order.

    Every iteration runs each system once, in interleaved_order(names, iteration, seed). Warmup rounds
    continue past `warmups` (up to max_warmups) until every system's warmup metric has converged. All
    systems must be loaded at the same time.

    Args:
        fns: System name -> workload callable.
        warmups: Minimum warmup iterations.
        runs: Timed iterations (at least one).
        seed: Order seed.
        sysman: Sysman for package energy, or None.
        device_index: GPU index whose package domain is read.
        clock: Monotonic clock returning seconds.
        max_warmups: Upper bound for convergence-driven warmup iterations, or None for exactly `warmups`.
        warmup_metrics: System name -> warmup metric function (default elapsed_s).

    Returns:
        Dict[str, List[TimedRun]]: Per system, warmups first, in iteration order.

    Raises:
        ValueError: If fns is empty, warmups is negative or runs is below one.
    """
    if not fns or warmups < 0 or runs < 1:
        raise ValueError(f"need systems, warmups >= 0 and runs >= 1, got {len(fns)}, {warmups}, {runs}")
    metrics = {name: (warmup_metrics or {}).get(name) or (lambda run: run.elapsed_s) for name in fns}
    limit = warmups if max_warmups is None else max(warmups, max_warmups)
    records: Dict[str, List[TimedRun]] = {name: [] for name in fns}
    values: Dict[str, List[Optional[float]]] = {name: [] for name in fns}
    iteration = 0
    while iteration < limit:
        if iteration >= warmups and all(_tail_converged(v) for v in values.values()):
            break
        for name in interleaved_order(list(fns), iteration, seed):
            record = _timed_call(fns[name], iteration, True, sysman, device_index, clock)
            records[name].append(record)
            values[name].append(metrics[name](record))
        iteration += 1
    for _ in range(runs):
        for name in interleaved_order(list(fns), iteration, seed):
            records[name].append(_timed_call(fns[name], iteration, False, sysman, device_index, clock))
        iteration += 1
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
    failures: Optional[List[Dict[str, str]]] = None,
    protocol_extra: Optional[Dict[str, Any]] = None,
    environment_extra: Optional[Dict[str, Any]] = None,
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
        failures: Optional records of tiers that failed on the device (kept in the file, never faked).
        protocol_extra: Optional extra protocol keys (harness settings, series quality).
        environment_extra: Optional keys merged over the captured environment (on_ac_power, preconditions).

    Raises:
        ValueError: If the document violates the schema.
    """
    protocol = {"warmup_runs": warmups, "timed_runs": runs, "labels": labels, **(protocol_extra or {})}
    environment = {**capture_environment(repo_dir), **(environment_extra or {})}
    write_result(path, build_result(run_id, samples, protocol, environment, failures))
