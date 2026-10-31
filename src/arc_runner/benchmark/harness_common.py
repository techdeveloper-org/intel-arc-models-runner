"""Helpers shared by the live benchmark scripts: device choice, optional power, paths, model download.

Also the METHOD.md protocol pieces the live scripts share: AC-power, idle-CPU and memory pre-checks
(each records its result, or None with the reason), the harness settings (converged warmup, interleaved
order, cooldown), and the per-series quality summary (warmup convergence, non-normal flag).
"""

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from arc_runner.benchmark.stats import (
    MAX_WARMUPS, MIN_WARMUPS, WARMUP_CV, BenchSample, coefficient_of_variation, non_normal_flag, warmup_converged,
)
from arc_runner.core.exceptions import ConfigurationError
from arc_runner.devices.sysman import LevelZeroSysman

IDLE_CPU_MAX_PCT = 10.0
IDLE_WINDOW_S = 30.0
COOLDOWN_S = 20.0
MEMORY_HEADROOM = 1.2


def optional_sysman() -> Optional[Any]:
    """Return a Sysman instance for power measurement, or None if Level Zero or a GPU is missing.

    Returns:
        Optional[Any]: LevelZeroSysman with at least one GPU device, else None.
    """
    sysman = LevelZeroSysman()
    return sysman if sysman.is_available and sysman._devices else None


def target_device(topology: Any, preferred: str) -> Tuple[str, bool]:
    """Pick the device to benchmark and say whether it is a fallback from the preferred one.

    Args:
        topology: Detected device topology (has_intel_arc, has_intel_npu).
        preferred: Wanted device: GPU, NPU or CPU.

    Returns:
        Tuple[str, bool]: (device, fallback_used). Rows produced with fallback_used=True are refused by
        the claim gates.
    """
    preferred = preferred.upper()
    available = {"GPU": bool(topology.has_intel_arc), "NPU": bool(topology.has_intel_npu), "CPU": True}
    if available.get(preferred, False):
        return preferred, False
    return "CPU", True


def default_result_path(repo_root: Path, workload: str, model_alias: str, now: Optional[datetime] = None) -> Path:
    """Build the schema-v1 result file path for a run.

    Args:
        repo_root: Repository root.
        workload: Workload name.
        model_alias: Model alias.
        now: Timestamp override (tests).

    Returns:
        Path: docs/benchmarks/results/<workload>_<alias>_<UTC timestamp>.v1.json under repo_root.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "docs" / "benchmarks" / "results" / f"{workload}_{model_alias}_{stamp}.v1.json"


def download_snapshot(repo_id: str, dest_dir: Path, marker_files: Sequence[str]) -> float:
    """Download a Hugging Face snapshot unless one of the marker files already exists.

    Args:
        repo_id: Hugging Face repository id.
        dest_dir: Destination directory.
        marker_files: File names whose presence means the model is already staged.

    Returns:
        float: Download seconds, 0.0 if nothing was downloaded.
    """
    if dest_dir.exists() and any((dest_dir / name).exists() for name in marker_files):
        return 0.0
    dest_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    snapshot_download(repo_id=repo_id, local_dir=str(dest_dir))
    return time.perf_counter() - started


GPU_CONTEXT_LOST_MARKER = "CL_OUT_OF_RESOURCES"
RETAINED_ENGINES: List[Any] = []


def retain_lost_engine(engine: Any) -> None:
    """Keep an engine whose GPU context was lost alive for the rest of the process.

    Dropping the last reference would run the OpenVINO destructors, which issue OpenCL calls that may
    hang after CL_OUT_OF_RESOURCES; the process leaves through os._exit instead.

    Args:
        engine: Engine that hit CL_OUT_OF_RESOURCES.
    """
    RETAINED_ENGINES.append(engine)


def select_tiers(spec: Dict[str, Any], tiers: Optional[str]) -> Dict[str, Any]:
    """Return a copy of a model spec restricted to the requested tiers, in spec order.

    Args:
        spec: Model spec with a "workloads" mapping.
        tiers: Comma-separated tier names, or None for all.

    Returns:
        Dict[str, Any]: Spec whose workloads contain only the selected tiers.

    Raises:
        ValueError: If a requested tier does not exist.
    """
    if not tiers:
        return {**spec, "workloads": dict(spec["workloads"])}
    wanted = [name.strip() for name in tiers.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in spec["workloads"]]
    if unknown:
        raise ValueError(f"unknown tier(s) {unknown}; available: {list(spec['workloads'])}")
    return {**spec, "workloads": {name: load for name, load in spec["workloads"].items() if name in wanted}}


def gpu_context_lost(failures: List[Dict[str, str]]) -> bool:
    """Tell whether any recorded failure was a GPU out-of-resources error.

    After CL_OUT_OF_RESOURCES the Intel driver may hang on any further OpenCL call, so callers must not
    unload the engine and should leave the process without normal interpreter shutdown.

    Args:
        failures: Failure records from collect_tiers.

    Returns:
        bool: True if the GPU context must be treated as lost.
    """
    return any(GPU_CONTEXT_LOST_MARKER in item.get("error", "") for item in failures)


def collect_tiers(
    workloads: Dict[str, Any],
    alias: str,
    backend: str,
    make_samples: Callable[[Any, str], List[Any]],
    failures: List[Dict[str, str]],
) -> Tuple[List[Any], bool]:
    """Measure each tier, recording failures instead of dropping or faking them.

    A GPU out-of-resources error stops the remaining tiers of this backend (the driver may hang); any
    other error is recorded and the next tier still runs.

    Args:
        workloads: Tier name -> workload entry.
        alias: Model alias.
        backend: Backend name.
        make_samples: Function (workload, model_id) -> samples; raises on failure.
        failures: List that receives {model_id, backend, error} records.

    Returns:
        Tuple[List[Any], bool]: Samples from the tiers that ran, and whether the GPU context was lost.
    """
    samples: List[Any] = []
    for tier, workload in workloads.items():
        model_id = f"{alias}/{tier}"
        try:
            samples += make_samples(workload, model_id)
        except Exception as exc:
            lost = GPU_CONTEXT_LOST_MARKER in str(exc)
            text = str(exc)
            if lost and GPU_CONTEXT_LOST_MARKER not in text[:500]:
                text = f"{GPU_CONTEXT_LOST_MARKER}: {text}"
            failures.append({"model_id": model_id, "backend": backend, "error": text[:500]})
            if lost:
                return samples, True
    return samples, False


def _read_ac_power() -> Tuple[Optional[bool], str]:
    """Read whether the machine runs on AC power.

    Returns:
        Tuple[Optional[bool], str]: (True/False, "") when the OS reports it; (None, reason) otherwise.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PowerStatus(ctypes.Structure):
                _fields_ = [("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE), ("BatteryLifePercent", wintypes.BYTE),
                            ("SystemStatusFlag", wintypes.BYTE), ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

            status = PowerStatus()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
                return None, "GetSystemPowerStatus failed"
            if status.ACLineStatus in (0, 1):
                return bool(status.ACLineStatus), ""
            return None, "GetSystemPowerStatus reports the AC line status as unknown"
        supplies = Path("/sys/class/power_supply")
        mains = [d for d in sorted(supplies.glob("*")) if (d / "type").is_file() and (d / "type").read_text().strip() == "Mains" and (d / "online").is_file()]
        if not mains:
            return None, "no mains power supply entry under /sys/class/power_supply"
        return any((d / "online").read_text().strip() == "1" for d in mains), ""
    except (OSError, ValueError, AttributeError) as exc:
        return None, f"AC power not readable: {exc}"


def _read_cpu_times() -> Optional[Tuple[float, float]]:
    """Read cumulative (busy, total) CPU time of the whole machine.

    Returns:
        Optional[Tuple[float, float]]: Counters in arbitrary equal units, or None if unreadable.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):  # type: ignore[attr-defined]
                return None
            to_int = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime  # noqa: E731
            total = to_int(kernel) + to_int(user)
            return float(total - to_int(idle)), float(total)
        fields = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        idle_all = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
        total = sum(fields[:8])
        return total - idle_all, total
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _read_memory() -> Optional[Tuple[int, int]]:
    """Read (total, available) physical memory in bytes.

    Returns:
        Optional[Tuple[int, int]]: The pair, or None if unreadable.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return None
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024
        return info["MemTotal"], info["MemAvailable"]
    except (OSError, ValueError, KeyError, IndexError, AttributeError):
        return None


def total_memory_bytes(reader: Callable[[], Optional[Tuple[int, int]]] = _read_memory) -> Optional[int]:
    """Return measured physical memory in bytes, or None if the OS could not report it.

    Args:
        reader: Memory reader returning (total, available) bytes or None (tests).
    """
    pair = reader()
    return pair[0] if pair else None


def directory_size_bytes(path: Path) -> Optional[int]:
    """Return the summed size of all files under a directory, or None if it does not exist.

    Args:
        path: Model directory.
    """
    if not path.is_dir():
        return None
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@dataclass(frozen=True)
class Probes:
    """OS readers and the sleep function used by check_preconditions; replaced in tests."""

    ac_power: Callable[[], Tuple[Optional[bool], str]] = _read_ac_power
    cpu_times: Callable[[], Optional[Tuple[float, float]]] = _read_cpu_times
    memory: Callable[[], Optional[Tuple[int, int]]] = _read_memory
    sleep: Callable[[float], None] = time.sleep


@dataclass(frozen=True)
class PreconditionReport:
    """Result of the pre-run checks. A value that could not be read is None with its reason.

    Attributes:
        on_ac_power: True/False, or None if unreadable or skipped.
        ac_reason: Why on_ac_power is None, else empty.
        cpu_busy_pct: Machine CPU busy percent over the idle window, or None.
        cpu_window_s: Length of the window the percentage covers.
        cpu_reason: Why cpu_busy_pct is None, else empty.
        memory_available_bytes: Available physical memory at check time, or None.
        memory_required_bytes: Model bytes times MEMORY_HEADROOM, or None when no size was given.
        memory_ok: True/False, or None when it could not be decided.
        memory_reason: Why memory_ok is None, else empty.
        violations: Known violations; an unknown value is never a violation.
    """

    on_ac_power: Optional[bool]
    ac_reason: str
    cpu_busy_pct: Optional[float]
    cpu_window_s: float
    cpu_reason: str
    memory_available_bytes: Optional[int]
    memory_required_bytes: Optional[int]
    memory_ok: Optional[bool]
    memory_reason: str
    violations: Tuple[str, ...] = ()

    @property
    def noisy(self) -> bool:
        """True when any known precondition is violated; rows are then flagged noisy."""
        return bool(self.violations)

    def as_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-ready mapping for the result file."""
        return {
            "on_ac_power": self.on_ac_power, "ac_reason": self.ac_reason,
            "cpu_busy_pct": self.cpu_busy_pct, "cpu_window_s": self.cpu_window_s, "cpu_reason": self.cpu_reason,
            "idle_cpu_max_pct": IDLE_CPU_MAX_PCT,
            "memory_available_bytes": self.memory_available_bytes, "memory_required_bytes": self.memory_required_bytes,
            "memory_ok": self.memory_ok, "memory_reason": self.memory_reason,
            "violations": list(self.violations),
        }


def check_preconditions(
    model_bytes: Optional[int] = None, idle_window_s: float = IDLE_WINDOW_S, skip: bool = False, probes: Probes = Probes(),
) -> PreconditionReport:
    """Check AC power, idle CPU and free memory before a run and record each result.

    AC power must be on. The machine CPU must be busy at most IDLE_CPU_MAX_PCT percent over idle_window_s
    seconds (measured as the delta of the OS cumulative CPU counters). Available physical memory must be at
    least model_bytes * MEMORY_HEADROOM (a fixed convention for weights plus runtime overhead, not a
    measured requirement). A check that cannot be read is recorded as None with the reason and is not
    treated as a violation; a skipped check is recorded the same way.

    Args:
        model_bytes: Bytes the run keeps resident (model files on disk times resident engines), or None.
        idle_window_s: Length of the idle-CPU observation window.
        skip: Record every check as skipped without sleeping or reading anything.
        probes: OS readers and sleep function.

    Returns:
        PreconditionReport: Results and violations.
    """
    if skip:
        return PreconditionReport(None, "skipped by flag", None, 0.0, "skipped by flag", None, None, None, "skipped by flag")
    violations: List[str] = []
    ac, ac_reason = probes.ac_power()
    if ac is False:
        violations.append("machine is on battery power")
    busy_pct: Optional[float] = None
    cpu_reason = ""
    first = probes.cpu_times()
    if first is None:
        cpu_reason = "CPU counters not readable"
    else:
        probes.sleep(idle_window_s)
        second = probes.cpu_times()
        if second is None or second[1] <= first[1]:
            cpu_reason = "CPU counters not readable or did not advance"
        else:
            busy_pct = 100.0 * (second[0] - first[0]) / (second[1] - first[1])
            if busy_pct > IDLE_CPU_MAX_PCT:
                violations.append(f"CPU busy {busy_pct:.1f}% over {idle_window_s:.0f} s exceeds {IDLE_CPU_MAX_PCT:.0f}%")
    memory = probes.memory()
    available = memory[1] if memory else None
    required = int(model_bytes * MEMORY_HEADROOM) if model_bytes is not None else None
    memory_ok: Optional[bool] = None
    memory_reason = ""
    if available is None:
        memory_reason = "available memory not readable"
    elif required is None:
        memory_reason = "model size not given"
    else:
        memory_ok = available >= required
        if not memory_ok:
            violations.append(f"available memory {available} bytes is below required {required} bytes")
    return PreconditionReport(ac, ac_reason, busy_pct, idle_window_s, cpu_reason, available, required, memory_ok, memory_reason, tuple(violations))


@dataclass(frozen=True)
class HarnessConfig:
    """Harness settings of one live run, recorded in the result protocol.

    Attributes:
        max_warmups: Upper bound for convergence-driven warmups, or None for a fixed warmup count.
        interleave: Run systems interleaved per iteration in a seeded random order (all engines resident).
        seed: Order seed.
        cooldown_s: Pause when switching between backends or tiers.
        idle_window_s: Idle-CPU observation window.
        skip_preconditions: Skip the AC/idle/memory checks (recorded as skipped).
    """

    max_warmups: Optional[int] = MAX_WARMUPS
    interleave: bool = False
    seed: int = 0
    cooldown_s: float = COOLDOWN_S
    idle_window_s: float = IDLE_WINDOW_S
    skip_preconditions: bool = False
    sleep: Callable[[float], None] = field(default=time.sleep, compare=False)

    def cooldown(self) -> None:
        """Pause for cooldown_s seconds (no-op when zero)."""
        if self.cooldown_s > 0:
            self.sleep(self.cooldown_s)

    def as_dict(self) -> Dict[str, Any]:
        """Return the settings as a JSON-ready mapping for the result protocol."""
        return {
            "max_warmups": self.max_warmups, "interleaved": self.interleave, "interleave_seed": self.seed,
            "cooldown_s": self.cooldown_s, "idle_window_s": self.idle_window_s, "warmup_cv_threshold": WARMUP_CV,
        }


def add_harness_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared protocol options to a live script's parser.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument("--fixed-warmups", action="store_true", help="run exactly --warmups warmups instead of until converged (max 5)")
    parser.add_argument("--max-warmups", type=int, default=MAX_WARMUPS, help="upper bound for convergence-driven warmups")
    parser.add_argument("--interleave", action="store_true", help="llm/vlm only: interleave backends per iteration in a seeded random order (keeps all engines loaded)")
    parser.add_argument("--seed", type=int, default=0, help="interleave order seed")
    parser.add_argument("--cooldown-s", type=float, default=COOLDOWN_S, help="pause between backend/tier switches; 0 disables")
    parser.add_argument("--idle-window-s", type=float, default=IDLE_WINDOW_S, help="idle-CPU observation window")
    parser.add_argument("--skip-preconditions", action="store_true", help="skip the AC-power, idle-CPU and memory checks (recorded as skipped)")


def add_keep_args(parser: argparse.ArgumentParser) -> None:
    """Add the model-retention options: keep is the default, --delete-after opts into deletion.

    Args:
        parser: Parser to extend.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--keep", action="store_true", help="keep the model files on disk (default; accepted for compatibility)")
    group.add_argument("--delete-after", action="store_true", help="delete the model directory after the run")


def reclaim_model_dir(delete_after: bool, model_dir: Path, models_root: Path) -> bool:
    """Delete a model directory only when requested and only when it lies inside the models root.

    Args:
        delete_after: True when the user passed --delete-after.
        model_dir: Directory of the model alias.
        models_root: Directory that holds all model aliases.

    Returns:
        bool: True if a directory was removed; False if deletion was not requested or it did not exist.

    Raises:
        ConfigurationError: If model_dir is the models root or is not inside it.
    """
    if not delete_after:
        return False
    root = Path(models_root).resolve()
    target = Path(model_dir).resolve()
    if target == root or root not in target.parents:
        raise ConfigurationError(f"refusing to delete '{target}': not a directory inside the models root '{root}'")
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


def config_from_args(args: argparse.Namespace) -> HarnessConfig:
    """Build the harness settings from parsed shared options.

    Args:
        args: Namespace produced by a parser extended with add_harness_args.
    """
    return HarnessConfig(
        max_warmups=None if args.fixed_warmups else max(args.max_warmups, args.warmups), interleave=args.interleave,
        seed=args.seed, cooldown_s=args.cooldown_s, idle_window_s=args.idle_window_s, skip_preconditions=args.skip_preconditions,
    )


def mark_noisy(samples: Sequence[BenchSample], report: PreconditionReport) -> List[BenchSample]:
    """Flag every sample noisy when a precondition was violated; otherwise return the samples unchanged.

    Args:
        samples: Samples of the run.
        report: Pre-run check result.
    """
    return [replace(sample, noisy=True) for sample in samples] if report.noisy else list(samples)


def _series_metrics(workload: str) -> Tuple[str, ...]:
    """Return the metric keys summarized per series for a workload.

    Args:
        workload: Workload name.
    """
    return ("ttft_ms", "decode_tps") if workload == "llm" else ("latency_s",)


def _value(sample: BenchSample, key: str) -> Optional[float]:
    """Read one metric of a sample (field for ttft_ms/decode_tps, else metrics key).

    Args:
        sample: Sample to read.
        key: Metric key.
    """
    return getattr(sample, key) if key in ("ttft_ms", "decode_tps") else (sample.metrics or {}).get(key)


def series_quality(samples: Sequence[BenchSample]) -> List[Dict[str, Any]]:
    """Summarize each (workload, backend, model_id) series: warmups run, convergence, CV, non-normal flag.

    Warmup convergence is recomputed from the warmup rows: the last three values of the warmup metric
    (ttft_ms for llm, latency_s otherwise) must have a coefficient of variation of at most 5%; None when
    fewer than three warmup values exist. non_normal uses stats.non_normal_flag on the timed values.
    Nothing here changes a claim gate; it is recorded next to the data.

    Args:
        samples: Samples of one run.

    Returns:
        List[Dict[str, Any]]: One mapping per series, in first-seen order.
    """
    order: List[Tuple[str, str, str]] = []
    groups: Dict[Tuple[str, str, str], List[BenchSample]] = {}
    for sample in samples:
        key = (sample.workload, sample.backend, sample.model_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(sample)
    summary: List[Dict[str, Any]] = []
    for key in order:
        workload, backend, model_id = key
        rows = groups[key]
        keys = _series_metrics(workload)
        warm = [_value(s, keys[0]) for s in rows if s.is_warmup]
        warm_values = [v for v in warm if v is not None]
        converged: Optional[bool] = None
        if len(warm) >= MIN_WARMUPS:
            converged = len(warm_values) == len(warm) and warmup_converged(warm_values, MIN_WARMUPS, WARMUP_CV)
        timed = {k: [v for v in (_value(s, k) for s in rows if not s.is_warmup) if v is not None] for k in keys}
        summary.append({
            "workload": workload, "backend": backend, "model_id": model_id,
            "warmups_run": len(warm), "warmup_metric": keys[0], "warmup_converged": converged,
            "timed_n": {k: len(v) for k, v in timed.items()},
            "cv": {k: (coefficient_of_variation(v) if len(v) >= 2 else None) for k, v in timed.items()},
            "non_normal": {k: non_normal_flag(v) for k, v in timed.items()},
        })
    return summary


def begin_run(config: HarnessConfig, model_bytes: Optional[int], probes: Probes = Probes()) -> PreconditionReport:
    """Run the pre-run checks for a live script and print what they found.

    Args:
        config: Harness settings.
        model_bytes: Bytes the run keeps resident, or None.
        probes: OS readers (tests).

    Returns:
        PreconditionReport: Results; violations make every row of the run noisy via finish_run.
    """
    report = check_preconditions(model_bytes, config.idle_window_s, config.skip_preconditions, probes)
    for violation in report.violations:
        print(f"PRECONDITION VIOLATED (rows will be flagged noisy): {violation}")
    for label, value, reason in (("AC power", report.on_ac_power, report.ac_reason), ("idle CPU", report.cpu_busy_pct, report.cpu_reason), ("memory", report.memory_ok, report.memory_reason)):
        if value is None:
            print(f"precondition not verified: {label}: {reason}")
    return report


def finish_run(
    samples: Sequence[BenchSample], config: HarnessConfig, report: PreconditionReport,
) -> Tuple[List[BenchSample], Dict[str, Any], Dict[str, Any]]:
    """Apply the pre-run result to the samples and build the extra result-file blocks.

    Args:
        samples: Samples of the run.
        config: Harness settings.
        report: Result of begin_run.

    Returns:
        Tuple[List[BenchSample], Dict[str, Any], Dict[str, Any]]: Samples (noisy when a precondition was
        violated), extra protocol keys (harness, series_quality) and extra environment keys
        (on_ac_power, preconditions).
    """
    marked = mark_noisy(samples, report)
    protocol_extra = {"harness": config.as_dict(), "series_quality": series_quality(marked)}
    environment_extra = {"on_ac_power": report.on_ac_power, "preconditions": report.as_dict()}
    return marked, protocol_extra, environment_extra
