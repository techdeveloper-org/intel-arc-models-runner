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


def build_result(
    run_id: str,
    samples: Iterable[BenchSample],
    protocol: Dict[str, Any],
    environment: Dict[str, Any],
    failures: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Assemble a schema-v1 result document.

    Args:
        run_id: Unique run identifier.
        samples: Measurements with provenance.
        protocol: Protocol block (warmup_runs, timed_runs, temperature, max_new_tokens, labels).
        environment: Environment block from capture_environment.
        failures: Optional records {model_id, backend, error} of tiers that could not be measured.

    Returns:
        Dict[str, Any]: Document ready for write_result.
    """
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "protocol": dict(protocol),
        "samples": [asdict(sample) for sample in samples],
    }
    if failures:
        document["failures"] = [dict(item) for item in failures]
    return document


def _is_int(value: Any, minimum: int) -> bool:
    """Tell whether a value is a real integer (not bool, not float) at or above a minimum.

    Args:
        value: Candidate.
        minimum: Smallest allowed value.

    Returns:
        bool: True for an int >= minimum.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_number(value: Any) -> bool:
    """Tell whether a value is a finite, non-negative real number (not bool).

    Args:
        value: Candidate.

    Returns:
        bool: True for a finite number >= 0.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _validate_environment_and_protocol(document: Dict[str, Any]) -> List[str]:
    """Check the environment and protocol blocks against the published schema.

    Args:
        document: Document whose environment and protocol keys exist.

    Returns:
        List[str]: Problems found; never raises on malformed input.
    """
    problems: List[str] = []
    environment, protocol = document["environment"], document["protocol"]
    if not isinstance(environment, dict):
        problems.append("environment must be an object")
    else:
        for key in _ENVIRONMENT_REQUIRED:
            if key not in environment:
                problems.append(f"environment missing {key}")
            elif not isinstance(environment[key], str):
                problems.append(f"environment.{key} must be a string")
        for key in ("openvino_genai", "ollama_version"):
            if environment.get(key) is not None and not isinstance(environment[key], str):
                problems.append(f"environment.{key} must be a string or null")
        if "driver_versions" in environment and not isinstance(environment["driver_versions"], dict):
            problems.append("environment.driver_versions must be an object")
        if environment.get("on_ac_power") is not None and not isinstance(environment["on_ac_power"], bool):
            problems.append("environment.on_ac_power must be a boolean or null")
    if not isinstance(protocol, dict):
        problems.append("protocol must be an object")
        return problems
    problems += [f"protocol missing {k}" for k in _PROTOCOL_REQUIRED if k not in protocol]
    if "warmup_runs" in protocol and not _is_int(protocol["warmup_runs"], 0):
        problems.append("protocol.warmup_runs must be an integer >= 0")
    if "timed_runs" in protocol and not _is_int(protocol["timed_runs"], 1):
        problems.append("protocol.timed_runs must be an integer >= 1")
    if "labels" in protocol and not isinstance(protocol["labels"], dict):
        problems.append("protocol.labels must be an object")
    if "max_new_tokens" in protocol and not _is_int(protocol["max_new_tokens"], 1):
        problems.append("protocol.max_new_tokens must be an integer >= 1")
    if "temperature" in protocol and (isinstance(protocol["temperature"], bool) or not isinstance(protocol["temperature"], (int, float)) or not math.isfinite(protocol["temperature"])):
        problems.append("protocol.temperature must be a finite number")
    return problems


def validate_result(document: Any) -> List[str]:
    """Check a result document against the schema-v1 rules.

    Args:
        document: Parsed JSON value.

    Returns:
        List[str]: Human-readable problems; empty if the document is valid. Never raises on malformed input.
    """
    problems: List[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    for key in ("schema_version", "run_id", "created_utc", "environment", "protocol", "samples"):
        if key not in document:
            problems.append(f"missing key: {key}")
    if problems:
        return problems
    if document["schema_version"] is not SCHEMA_VERSION and not (type(document["schema_version"]) is int and document["schema_version"] == SCHEMA_VERSION):
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(document["run_id"], str) or not document["run_id"]:
        problems.append("run_id must be a non-empty string")
    if not isinstance(document["created_utc"], str):
        problems.append("created_utc must be a string")
    problems += _validate_environment_and_protocol(document)
    failures = document.get("failures")
    if failures is not None:
        if not isinstance(failures, list) or not all(
            isinstance(item, dict) and all(isinstance(item.get(key), str) for key in ("model_id", "backend", "error")) for item in failures
        ):
            problems.append("failures must be a list of {model_id, backend, error} strings")
    if not isinstance(document["samples"], list):
        return problems + ["samples must be a list"]
    protocol = document["protocol"]
    if isinstance(protocol, dict) and any(isinstance(row, dict) and row.get("workload", "llm") == "llm" for row in document["samples"]):
        problems += [f"protocol missing {k}" for k in _PROTOCOL_REQUIRED_LLM if k not in protocol]
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
    if not isinstance(sample["source"], str) or sample["source"] not in ALLOWED_SOURCES:
        problems.append(f"samples[{index}].source must be one of {ALLOWED_SOURCES}, got {sample['source']!r}")
    for key in ("ttft_ms", "decode_tps"):
        value = sample[key]
        if value is not None and not _is_number(value):
            problems.append(f"samples[{index}].{key} must be null or a finite non-negative number")
    for key in ("model_id", "backend"):
        if not isinstance(sample[key], str):
            problems.append(f"samples[{index}].{key} must be a string")
    if "noisy" in sample and not isinstance(sample["noisy"], bool):
        problems.append(f"samples[{index}].noisy must be a boolean")
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
