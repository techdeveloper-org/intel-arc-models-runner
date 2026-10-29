"""Shared harness for non-LLM benchmark workloads: timing, power, samples, schema-v1 export."""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pytest

from arc_runner.benchmark.results import load_samples, validate_result
from arc_runner.benchmark.workloads import export_workload_result, measure_workload, workload_samples

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Clock:
    """Deterministic clock advancing by scripted amounts per call pair."""

    def __init__(self, durations: List[float]) -> None:
        self._events: Iterator[float] = iter(self._expand(durations))

    @staticmethod
    def _expand(durations: List[float]) -> List[float]:
        now, events = 0.0, []
        for d in durations:
            events += [now, now + d]
            now += d + 1.0
        return events

    def __call__(self) -> float:
        return next(self._events)


class _Sysman:
    """Sysman stand-in: constant 10 W, one snapshot pair per run."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self._t = 0

    def read_package_energy(self, device_index: int = 0) -> Optional[Tuple[int, int]]:
        if not self.available:
            return None
        self._t += 1
        return self._t * 5_000_000, self._t * 500_000

    def get_unavailable_reason(self, metric: str) -> Optional[str]:
        return "no counter"


def test_measure_workload_runs_warmups_then_timed_and_flags_them() -> None:
    """Warmup runs come first, are flagged, and every run executes the workload once."""
    calls: List[int] = []
    runs = measure_workload(lambda: calls.append(1), warmups=2, runs=3, clock=_Clock([1.0] * 5))
    assert len(calls) == 5
    assert [r.is_warmup for r in runs] == [True, True, False, False, False]
    assert [r.iteration for r in runs] == [0, 1, 2, 3, 4]
    assert all(r.elapsed_s == pytest.approx(1.0) for r in runs)


def test_measure_workload_exception_propagates_and_stops() -> None:
    """A failing workload is an error, not a skipped sample."""
    calls: List[int] = []

    def fn() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("device lost")

    with pytest.raises(RuntimeError):
        measure_workload(fn, warmups=0, runs=5, clock=_Clock([1.0] * 5))
    assert len(calls) == 2


def test_measure_workload_attaches_measured_power_per_run() -> None:
    """Each run carries energy and average power from its own counter interval."""
    runs = measure_workload(lambda: None, warmups=0, runs=2, sysman=_Sysman(), clock=_Clock([1.0, 1.0]))
    assert all(r.power.avg_power_w == pytest.approx(10.0) for r in runs)
    assert all(r.power.energy_j == pytest.approx(5.0) for r in runs)


def test_measure_workload_without_power_source_keeps_none_and_reason() -> None:
    """No counter: power is None with a reason, latency still measured."""
    runs = measure_workload(lambda: None, warmups=0, runs=1, sysman=_Sysman(available=False), clock=_Clock([2.0]))
    assert runs[0].power.avg_power_w is None and runs[0].power.reason
    assert runs[0].elapsed_s == pytest.approx(2.0)


def test_measure_workload_rejects_non_positive_run_counts() -> None:
    """Zero timed runs is a usage error."""
    with pytest.raises(ValueError):
        measure_workload(lambda: None, warmups=0, runs=0)
    with pytest.raises(ValueError):
        measure_workload(lambda: None, warmups=-1, runs=1)


def test_workload_samples_include_latency_power_and_extra_metrics() -> None:
    """Samples carry latency, package power/energy (None when absent) and caller-derived metrics."""
    runs = measure_workload(lambda: None, warmups=1, runs=2, sysman=_Sysman(), clock=_Clock([2.0, 4.0, 3.0]))
    samples = workload_samples("whisper", "arc", "audio_stt", runs, extra=lambda run: {"rtf": run.elapsed_s / 10.0})
    assert [s.is_warmup for s in samples] == [True, False, False]
    assert samples[1].metrics["latency_s"] == pytest.approx(4.0)
    assert samples[1].metrics["rtf"] == pytest.approx(0.4)
    assert samples[1].metrics["package_power_w"] == pytest.approx(10.0)
    assert samples[1].workload == "audio_stt" and samples[1].source == "measured"
    assert samples[1].ttft_ms is None and samples[1].decode_tps is None


def test_workload_samples_power_none_when_unavailable() -> None:
    """Missing power is None in metrics, not zero."""
    runs = measure_workload(lambda: None, warmups=0, runs=1, sysman=None, clock=_Clock([1.0]))
    sample = workload_samples("m", "arc", "image", runs)[0]
    assert sample.metrics["package_power_w"] is None and sample.metrics["package_energy_j"] is None


def test_workload_samples_fallback_flag_is_recorded() -> None:
    """A run on a fallback device is flagged so claim gates refuse it."""
    runs = measure_workload(lambda: None, warmups=0, runs=1, clock=_Clock([1.0]))
    assert workload_samples("m", "arc", "video", runs, fallback_used=True)[0].fallback_used is True


def test_export_workload_result_writes_valid_schema_v1_file(tmp_path: Path) -> None:
    """The exported file validates and loads back with its workload and metrics."""
    runs = measure_workload(lambda: None, warmups=1, runs=2, clock=_Clock([1.0, 2.0, 3.0]))
    samples = workload_samples("clip", "arc", "embeddings", runs)
    target = tmp_path / "emb.json"
    export_workload_result(target, "emb-run", samples, labels={"arc": "iGPU FP16"}, warmups=1, runs=2, repo_dir=REPO_ROOT)
    document = json.loads(target.read_text(encoding="utf-8"))
    assert validate_result(document) == []
    assert document["protocol"]["warmup_runs"] == 1 and document["protocol"]["timed_runs"] == 2
    assert "temperature" not in document["protocol"]
    assert load_samples(document)[2].metrics["latency_s"] == pytest.approx(3.0)
