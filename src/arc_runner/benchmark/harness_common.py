"""Helpers shared by the live benchmark scripts: device choice, optional power, paths, model download."""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from arc_runner.devices.sysman import LevelZeroSysman


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
