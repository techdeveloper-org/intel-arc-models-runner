"""Tiers that fail on the device (for example CL_OUT_OF_RESOURCES on an iGPU) are recorded, never faked."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List

import pytest

from arc_runner.benchmark.harness_common import GPU_CONTEXT_LOST_MARKER, collect_tiers, select_tiers
from arc_runner.benchmark.results import validate_result
from arc_runner.benchmark.workloads import export_workload_result

REPO_ROOT = Path(__file__).resolve().parents[1]
OOM = RuntimeError("Exception from src\\plugins\\intel_gpu\\src\\runtime\\ocl\\ocl_common.hpp:62: [GPU] CL_OUT_OF_RESOURCES exception.")


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scratch" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_tiers_filters_and_keeps_order() -> None:
    """--tiers low,medium keeps only those tiers in spec order; None keeps all; unknown names raise."""
    spec = {"workloads": {"low": 1, "medium": 2, "hard": 3}}
    assert list(select_tiers(spec, None)["workloads"]) == ["low", "medium", "hard"]
    assert list(select_tiers(spec, "medium,low")["workloads"]) == ["low", "medium"]
    with pytest.raises(ValueError):
        select_tiers(spec, "low,huge")


def test_collect_tiers_records_failure_and_stops_after_gpu_context_loss() -> None:
    """After CL_OUT_OF_RESOURCES the remaining tiers are not attempted (the driver may hang) and the loss is reported."""
    attempted: List[str] = []

    def make(workload: Any, model_id: str) -> list:
        attempted.append(model_id)
        if workload == "hard":
            raise OOM
        return [model_id]

    failures: List[dict] = []
    samples, lost = collect_tiers({"low": "low", "hard": "hard", "extra": "extra"}, "m", "arc-runner", make, failures)
    assert samples == ["m/low"] and lost is True
    assert attempted == ["m/low", "m/hard"]
    assert failures == [{"model_id": "m/hard", "backend": "arc-runner", "error": failures[0]["error"]}]
    assert GPU_CONTEXT_LOST_MARKER in failures[0]["error"]


def test_collect_tiers_continues_after_an_ordinary_failure() -> None:
    """A non-fatal error is recorded and the next tier still runs."""
    def make(workload: Any, model_id: str) -> list:
        if workload == "bad":
            raise ValueError("bad request")
        return [model_id]

    failures: List[dict] = []
    samples, lost = collect_tiers({"bad": "bad", "ok": "ok"}, "m", "b", make, failures)
    assert samples == ["m/ok"] and lost is False and failures[0]["model_id"] == "m/bad"


@pytest.mark.parametrize("script,spec_name", [("run_live_image_benchmarks", "IMAGE_MODEL_WORKLOAD_SPECS"), ("run_live_video_benchmarks", "VIDEO_MODEL_WORKLOAD_SPECS")])
def test_scripts_keep_earlier_tiers_and_skip_unload_when_gpu_context_is_lost(script: str, spec_name: str) -> None:
    """Low-tier samples survive an OOM on a later tier; unload is not called on the lost engine; CPU baseline still runs."""
    module = _script(script)
    specs = getattr(module, spec_name)
    alias = next(iter(specs))
    tiers = list(specs[alias]["workloads"])
    unloads: List[str] = []

    class _Engine:
        def __init__(self, device: str) -> None:
            self.device = device

        def generate(self, request: Any) -> Any:
            if self.device == "GPU" and request.tier == tiers[1]:
                raise OOM
            breakdown = SimpleNamespace(step_latency_ms=None, iteration_rate_it_s=None, frames_per_second=1.0, real_time_factor=1.0)
            return SimpleNamespace(breakdown=breakdown, width=64, height=64, num_frames=8, duration_sec=1.0, total_latency_s=1.0, total_latency_ms=1000.0)

        def unload(self) -> None:
            unloads.append(self.device)

    def builder(workload: Any) -> Any:
        return SimpleNamespace(tier=next(t for t, w in specs[alias]["workloads"].items() if w is workload), width=64, height=64, num_frames=8, fps=8)

    failures: List[dict] = []
    samples = module.collect_samples(alias, specs[alias], _Engine, builder, "GPU", False, True, 0, 1, None, failures=failures)
    gpu = [s for s in samples if s.backend == "arc-runner"]
    cpu = [s for s in samples if s.backend == "cpu-baseline"]
    assert {s.model_id for s in gpu} == {f"{alias}/{tiers[0]}"}
    assert len(cpu) == len(tiers)
    assert unloads == ["CPU"]
    assert failures and GPU_CONTEXT_LOST_MARKER in failures[0]["error"]
    assert module.gpu_context_lost(failures) is True


def test_failures_round_trip_through_the_result_file_and_validate(tmp_path: Path) -> None:
    """The result document carries a failures list; malformed failures are rejected; old files without it stay valid."""
    failures = [{"model_id": "m/hard", "backend": "arc-runner", "error": "CL_OUT_OF_RESOURCES"}]
    target = tmp_path / "r.json"
    export_workload_result(target, "run", [], {"arc-runner": "x"}, 0, 1, REPO_ROOT, failures=failures)
    import json

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["failures"] == failures and validate_result(document) == []
    del document["failures"]
    assert validate_result(document) == []
    document["failures"] = [{"model_id": "m"}]
    assert any("failures" in p for p in validate_result(document))
    document["failures"] = "nope"
    assert any("failures" in p for p in validate_result(document))
