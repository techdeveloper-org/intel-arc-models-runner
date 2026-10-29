"""Schema-v1 benchmark result files: environment capture, writing, and validation.

Result files are the only allowed source of numbers in the benchmark documentation. Values are never
edited by hand; every row carries a source of measured, derived, or projected.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

from arc_runner.benchmark.stats import BenchSample

SCHEMA_VERSION = 1
ALLOWED_SOURCES = ("measured", "derived", "projected")
_SAMPLE_REQUIRED = (
    "model_id", "backend", "iteration", "ttft_ms", "decode_tps", "prompt_tokens",
    "completion_tokens", "source", "is_warmup", "fallback_used",
)
WORKLOADS = ("llm", "embeddings", "vlm", "audio_stt", "audio_tts", "image", "video")
_PROTOCOL_REQUIRED = ("warmup_runs", "timed_runs", "labels")
_PROTOCOL_REQUIRED_LLM = ("temperature", "max_new_tokens")
_METRIC_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_ENVIRONMENT_REQUIRED = ("os", "python", "git_sha", "hardware")


def _git_sha(repo_dir: Path) -> str:
    """Return the current git commit SHA or 'unknown'.

    Args:
        repo_dir: Directory inside the repository.
    """
    try:
        result = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _package_version(name: str) -> Optional[str]:
    """Return an installed package version, or None if it is not installed.

    Args:
        name: Distribution name.
    """
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def capture_environment(repo_dir: Path, ollama_version: Optional[str] = None, on_ac_power: Optional[bool] = None) -> Dict[str, Any]:
    """Record the facts a reader needs to interpret a benchmark run.

    Args:
        repo_dir: Repository directory for the git SHA.
        ollama_version: Version string of the Ollama server, if one was used.
        on_ac_power: True/False if known, None if unknown.

    Returns:
        Dict[str, Any]: The environment block of a result file.
    """
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "git_sha": _git_sha(repo_dir),
        "hardware": f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical CPUs)",
        "openvino_genai": _package_version("openvino-genai"),
        "ollama_version": ollama_version,
        "driver_versions": {},
        "on_ac_power": on_ac_power,
    }


def build_result(run_id: str, samples: Iterable[BenchSample], protocol: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a schema-v1 result document.

    Args:
        run_id: Unique run identifier.
        samples: Measurements with provenance.
        protocol: Protocol block (warmup_runs, timed_runs, temperature, max_new_tokens, labels).
        environment: Environment block from capture_environment.

    Returns:
        Dict[str, Any]: Document ready for write_result.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "protocol": dict(protocol),
        "samples": [asdict(sample) for sample in samples],
    }


def validate_result(document: Any) -> List[str]:
    """Check a result document against the schema-v1 rules.

    Args:
        document: Parsed JSON value.

    Returns:
        List[str]: Human-readable problems; empty if the document is valid.
    """
    problems: List[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    for key in ("schema_version", "run_id", "created_utc", "environment", "protocol", "samples"):
        if key not in document:
            problems.append(f"missing key: {key}")
    if problems:
        return problems
    if document["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    problems += [f"environment missing {k}" for k in _ENVIRONMENT_REQUIRED if k not in document["environment"]]
    problems += [f"protocol missing {k}" for k in _PROTOCOL_REQUIRED if k not in document["protocol"]]
    if not isinstance(document["samples"], list):
        return problems + ["samples must be a list"]
    if any(isinstance(row, dict) and row.get("workload", "llm") == "llm" for row in document["samples"]):
        problems += [f"protocol missing {k}" for k in _PROTOCOL_REQUIRED_LLM if k not in document["protocol"]]
    for index, sample in enumerate(document["samples"]):
        problems += _validate_sample(index, sample)
    return problems


def _validate_sample(index: int, sample: Any) -> List[str]:
    """Check one sample row.

    Args:
        index: Position of the row, for messages.
        sample: Parsed row.

    Returns:
        List[str]: Problems found in this row.
    """
    if not isinstance(sample, dict):
        return [f"samples[{index}] must be an object"]
    problems = [f"samples[{index}] missing {k}" for k in _SAMPLE_REQUIRED if k not in sample]
    if problems:
        return problems
    if sample["source"] not in ALLOWED_SOURCES:
        problems.append(f"samples[{index}].source must be one of {ALLOWED_SOURCES}, got {sample['source']!r}")
    for key in ("ttft_ms", "decode_tps"):
        value = sample[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
            problems.append(f"samples[{index}].{key} must be null or a non-negative number")
    problems += _validate_workload_and_metrics(index, sample)
    for key in ("is_warmup", "fallback_used"):
        if not isinstance(sample[key], bool):
            problems.append(f"samples[{index}].{key} must be a boolean")
    for key in ("iteration", "prompt_tokens", "completion_tokens"):
        if isinstance(sample[key], bool) or not isinstance(sample[key], int) or sample[key] < 0:
            problems.append(f"samples[{index}].{key} must be a non-negative integer")
    return problems


def _validate_workload_and_metrics(index: int, sample: Dict[str, Any]) -> List[str]:
    """Check the workload name and the metrics mapping of one sample row.

    Args:
        index: Position of the row, for messages.
        sample: Parsed row.

    Returns:
        List[str]: Problems found; non-LLM workloads must carry at least one metric.
    """
    workload = sample.get("workload", "llm")
    if workload not in WORKLOADS:
        return [f"samples[{index}].workload must be one of {WORKLOADS}, got {workload!r}"]
    metrics = sample.get("metrics")
    if metrics is None:
        return [f"samples[{index}] needs metrics for workload {workload}"] if workload != "llm" else []
    if not isinstance(metrics, dict) or (workload != "llm" and not metrics):
        return [f"samples[{index}].metrics must be a non-empty object"]
    problems: List[str] = []
    for key, value in metrics.items():
        if not isinstance(key, str) or not _METRIC_KEY.match(key):
            problems.append(f"samples[{index}].metrics key {key!r} must be lowercase letters, digits and underscores")
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            problems.append(f"samples[{index}].metrics.{key} must be null or a finite non-negative number")
    return problems


def write_result(path: Path, document: Dict[str, Any]) -> None:
    """Validate and write a result document.

    Args:
        path: Destination file.
        document: Result document.

    Raises:
        ValueError: If the document violates the schema.
    """
    problems = validate_result(document)
    if problems:
        raise ValueError("invalid benchmark result: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_samples(document: Dict[str, Any]) -> List[BenchSample]:
    """Convert the samples of a valid document back into BenchSample objects.

    Args:
        document: Valid result document.

    Returns:
        List[BenchSample]: Samples in file order.
    """
    return [BenchSample(**{k: v for k, v in row.items() if k in BenchSample.__dataclass_fields__}) for row in document["samples"]]
