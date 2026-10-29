"""Helpers shared by the live benchmark scripts: device choice, optional power, paths, model download."""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Optional, Sequence, Tuple

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
