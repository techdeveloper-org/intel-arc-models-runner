"""METHOD.md protocol pieces in the harness: converged warmup, interleaving, pre-checks, series quality."""

import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from arc_runner.benchmark.harness_common import (
    HarnessConfig, PreconditionReport, Probes, add_harness_args, begin_run, check_preconditions, config_from_args,
    directory_size_bytes, finish_run, mark_noisy, series_quality, total_memory_bytes,
)
from arc_runner.benchmark.stats import BenchSample, non_normal_flag
from arc_runner.benchmark.workloads import interleaved_order, measure_interleaved, measure_workload


class FakeClock:
    """Clock whose consecutive reads differ by the next scripted duration (start read, then end read)."""

    def __init__(self, durations: List[float]) -> None:
        self.durations = list(durations)
        self.now = 0.0
        self.reads = 0

    def __call__(self) -> float:
        if self.reads % 2 == 1:
            self.now += self.durations.pop(0) if self.durations else self.now
        self.reads += 1
        return self.now


def _probes(ac: Tuple[Optional[bool], str] = (True, ""), cpu: Any = ((0.0, 100.0), (5.0, 200.0)), memory: Any = (16_000, 8_000)) -> Probes:
    """Build probes returning scripted readings; cpu is a pair of consecutive counter readings."""
    readings = iter(cpu) if cpu is not None else None
    slept: List[float] = []
    probes = Probes(
        ac_power=lambda: ac,
        cpu_times=(lambda: next(readings)) if readings is not None else (lambda: None),
        memory=lambda: memory,
        sleep=slept.append,
    )
    object.__setattr__(probes, "slept", slept)
    return probes


def test_measure_workload_warmup_stops_at_minimum_when_series_is_already_converged() -> None:
    """Three equal warmups converge at once, so no extra warmup runs."""
    calls: List[int] = []
    clock = FakeClock([1.0] * 20)
    records = measure_workload(lambda: calls.append(1), 3, 4, clock=clock, max_warmups=5)
    assert sum(r.is_warmup for r in records) == 3
    assert len(calls) == 7


def test_measure_workload_warmup_extends_until_last_three_converge() -> None:
    """A slow-then-steady series needs warmups beyond the minimum, ending as soon as the last three agree."""
    durations = [9.0, 5.0, 3.0, 1.0, 1.0, 1.0] + [1.0] * 4
    records = measure_workload(lambda: None, 3, 4, clock=FakeClock(durations), max_warmups=5)
    warmups = [r for r in records if r.is_warmup]
    assert [round(r.elapsed_s, 1) for r in warmups] == [9.0, 5.0, 3.0, 1.0, 1.0]
    assert len(records) == 9


def test_measure_workload_warmup_stops_at_maximum_when_never_converging() -> None:
    """A series that keeps changing stops at max_warmups and the runs still happen."""
    durations = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0] + [1.0] * 3
    records = measure_workload(lambda: None, 3, 3, clock=FakeClock(durations), max_warmups=5)
    assert sum(r.is_warmup for r in records) == 5


def test_measure_workload_without_max_warmups_runs_exactly_the_requested_warmups() -> None:
    """The default keeps the fixed warmup count (previous behaviour)."""
    records = measure_workload(lambda: None, 3, 2, clock=FakeClock([9.0, 1.0, 5.0, 1.0, 1.0]))
    assert sum(r.is_warmup for r in records) == 3


def test_measure_workload_warmup_metric_none_values_do_not_count_as_converged() -> None:
    """A warmup metric that is unavailable cannot declare convergence, so the maximum is used."""
    records = measure_workload(lambda: None, 3, 1, clock=FakeClock([1.0] * 10), max_warmups=5, warmup_metric=lambda run: None)
    assert sum(r.is_warmup for r in records) == 5


def test_interleaved_order_is_deterministic_for_a_seed_and_covers_all_systems() -> None:
    """Same seed and iteration give the same permutation; different iterations are not all identical."""
    names = ["gpu", "cpu", "npu"]
    assert interleaved_order(names, 4, 7) == interleaved_order(names, 4, 7)
    assert all(sorted(interleaved_order(names, i, 7)) == sorted(names) for i in range(10))
    assert len({tuple(interleaved_order(names, i, 7)) for i in range(30)}) > 1


def test_measure_interleaved_runs_every_system_once_per_iteration_in_the_seeded_order() -> None:
    """Each iteration contains each system once, in the order interleaved_order gives."""
    trace: List[str] = []
    fns = {"a": lambda: trace.append("a"), "b": lambda: trace.append("b")}
    records = measure_interleaved(fns, 2, 3, seed=3, clock=FakeClock([1.0] * 40))
    assert len(records["a"]) == len(records["b"]) == 5
    expected: List[str] = []
    for iteration in range(5):
        expected += interleaved_order(["a", "b"], iteration, 3)
    assert trace == expected
    assert [r.is_warmup for r in records["a"]] == [True, True, False, False, False]


def test_measure_interleaved_extends_warmups_until_all_systems_converge() -> None:
    """Warmup rounds continue while any system has not converged, up to the maximum."""
    seq = {"a": iter([5.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0]), "b": iter([1.0] * 7)}
    current = {"name": ""}

    def make(name: str) -> Any:
        def call() -> None:
            current["name"] = name
        return call

    def clock_factory() -> Any:
        state = {"n": 0, "t": 0.0}

        def clock() -> float:
            if state["n"] % 2 == 1:
                state["t"] += next(seq[current["name"]])
            state["n"] += 1
            return state["t"]
        return clock

    records = measure_interleaved({"a": make("a"), "b": make("b")}, 3, 1, seed=0, clock=clock_factory(), max_warmups=5)
    assert sum(r.is_warmup for r in records["a"]) == 5
    assert sum(r.is_warmup for r in records["b"]) == 5


def test_measure_interleaved_rejects_empty_systems() -> None:
    """No systems is an error, not an empty result."""
    with pytest.raises(ValueError):
        measure_interleaved({}, 1, 1)


def test_non_normal_flag_false_for_tight_series_true_for_heavy_outlier_series_none_when_undefined() -> None:
    """Student-t and bootstrap intervals agree on tight data and disagree when one value dominates."""
    assert non_normal_flag([100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.0]) is False
    assert non_normal_flag([1.0] * 9 + [100.0]) is True
    assert non_normal_flag([1.0, 2.0]) is None
    assert non_normal_flag([0.0, 0.0, 0.0]) is None


def test_check_preconditions_records_every_result_and_no_violation_when_all_ok() -> None:
    """AC on, CPU 5% busy, enough memory: nothing violated and every value is recorded."""
    probes = _probes()
    report = check_preconditions(model_bytes=1_000, idle_window_s=30.0, probes=probes)
    assert report.on_ac_power is True and report.cpu_busy_pct == pytest.approx(5.0)
    assert report.memory_ok is True and report.violations == () and not report.noisy
    assert probes.slept == [30.0]  # type: ignore[attr-defined]
    assert report.as_dict()["cpu_window_s"] == 30.0


def test_check_preconditions_flags_battery_busy_cpu_and_low_memory() -> None:
    """Each known violation is listed and makes the run noisy."""
    probes = _probes(ac=(False, ""), cpu=((0.0, 100.0), (50.0, 200.0)), memory=(16_000, 500))
    report = check_preconditions(model_bytes=1_000, probes=probes)
    assert report.noisy and len(report.violations) == 3
    assert report.memory_ok is False and report.cpu_busy_pct == pytest.approx(50.0)


def test_check_preconditions_unreadable_values_are_none_with_reason_and_not_violations() -> None:
    """Unknown is recorded as None plus a reason; it never invents a pass or a violation."""
    probes = _probes(ac=(None, "no mains"), cpu=None, memory=None)
    report = check_preconditions(model_bytes=1_000, probes=probes)
    assert report.on_ac_power is None and report.ac_reason == "no mains"
    assert report.cpu_busy_pct is None and report.cpu_reason
    assert report.memory_ok is None and report.memory_reason
    assert report.violations == () and not report.noisy


def test_check_preconditions_cpu_counter_that_does_not_advance_gives_none() -> None:
    """A counter with no progress cannot give a percentage."""
    report = check_preconditions(probes=_probes(cpu=((1.0, 100.0), (1.0, 100.0))))
    assert report.cpu_busy_pct is None and "advance" in report.cpu_reason


def test_check_preconditions_without_model_size_leaves_memory_undecided() -> None:
    """No model size means no memory verdict."""
    report = check_preconditions(model_bytes=None, probes=_probes())
    assert report.memory_ok is None and report.memory_reason == "model size not given"


def test_check_preconditions_skip_records_skipped_without_sleeping() -> None:
    """Skipping never reads or sleeps and says so for each check."""
    probes = _probes()
    report = check_preconditions(1_000, skip=True, probes=probes)
    assert report.ac_reason == report.cpu_reason == report.memory_reason == "skipped by flag"
    assert probes.slept == []  # type: ignore[attr-defined]


def test_mark_noisy_flags_all_samples_only_for_a_violating_report() -> None:
    """Noisy rows are what the claim gates refuse."""
    rows = [BenchSample("m", "b", 0, 1.0, 2.0, 1, 1, "measured")]
    ok = PreconditionReport(True, "", 1.0, 30.0, "", 10, 5, True, "")
    bad = PreconditionReport(False, "", 1.0, 30.0, "", 10, 5, True, "", ("battery",))
    assert mark_noisy(rows, ok)[0].noisy is False
    assert mark_noisy(rows, bad)[0].noisy is True


def _series(backend: str, warm: List[float], timed: List[float]) -> List[BenchSample]:
    """LLM rows with the given TTFT warmups and timed values."""
    rows = [BenchSample("m/low", backend, i, v, 10.0, 1, 1, "measured", is_warmup=True) for i, v in enumerate(warm)]
    rows += [BenchSample("m/low", backend, len(warm) + i, v, 10.0, 1, 1, "measured") for i, v in enumerate(timed)]
    return rows


def test_series_quality_reports_convergence_from_warmup_rows_and_none_when_too_few() -> None:
    """Convergence is recomputed from the last three warmup TTFTs; fewer than three gives None."""
    summary = series_quality(_series("a", [900, 100, 100, 101], [100.0] * 4) + _series("b", [100.0, 100.0], [100.0] * 4) + _series("c", [500, 300, 100], [100.0] * 4))
    by_backend = {row["backend"]: row for row in summary}
    assert by_backend["a"]["warmup_converged"] is True and by_backend["a"]["warmups_run"] == 4
    assert by_backend["b"]["warmup_converged"] is None
    assert by_backend["c"]["warmup_converged"] is False
    assert by_backend["a"]["timed_n"]["ttft_ms"] == 4


def test_series_quality_uses_latency_for_non_llm_workloads_and_flags_non_normal() -> None:
    """VLM series are summarized on latency_s; a dominated series is flagged non-normal."""
    rows = [BenchSample("v/low", "arc", i, None, None, 0, 0, "measured", is_warmup=i < 3, workload="vlm", metrics={"latency_s": 1.0 if i < 12 else 50.0}) for i in range(13)]
    quality = series_quality(rows)[0]
    assert quality["warmup_metric"] == "latency_s" and quality["warmup_converged"] is True
    assert quality["non_normal"]["latency_s"] is True


def test_finish_run_marks_noisy_and_builds_protocol_and_environment_blocks() -> None:
    """The result blocks carry harness settings, series quality, AC state and the precondition report."""
    bad = PreconditionReport(False, "", 1.0, 30.0, "", 10, 5, True, "", ("battery",))
    samples, protocol, environment = finish_run(_series("a", [1, 1, 1], [1.0] * 3), HarnessConfig(cooldown_s=0.0), bad)
    assert all(s.noisy for s in samples)
    assert protocol["harness"]["max_warmups"] == 5 and protocol["harness"]["interleaved"] is False
    assert protocol["series_quality"][0]["backend"] == "a"
    assert environment["on_ac_power"] is False and environment["preconditions"]["violations"] == ["battery"]


def test_begin_run_prints_violations_and_unverified_checks(capsys: pytest.CaptureFixture) -> None:
    """Violations and unverifiable checks are visible on the console."""
    begin_run(HarnessConfig(idle_window_s=1.0), 1_000, _probes(ac=(False, ""), memory=None))
    out = capsys.readouterr().out
    assert "PRECONDITION VIOLATED" in out and "precondition not verified: memory" in out


def test_harness_args_defaults_follow_method_and_flags_override() -> None:
    """Defaults are converged warmup up to 5, 20 s cooldown, no interleave; flags change them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    add_harness_args(parser)
    default = config_from_args(parser.parse_args([]))
    assert (default.max_warmups, default.interleave, default.cooldown_s, default.idle_window_s) == (5, False, 20.0, 30.0)
    custom = config_from_args(parser.parse_args(["--fixed-warmups", "--interleave", "--cooldown-s", "0", "--seed", "9"]))
    assert custom.max_warmups is None and custom.interleave and custom.cooldown_s == 0.0 and custom.seed == 9


def test_harness_cooldown_sleeps_only_when_positive() -> None:
    """A zero cooldown never sleeps."""
    slept: List[float] = []
    HarnessConfig(cooldown_s=0.0, sleep=slept.append).cooldown()
    HarnessConfig(cooldown_s=20.0, sleep=slept.append).cooldown()
    assert slept == [20.0]


def test_total_memory_and_directory_size_report_absent_as_none(tmp_path: Path) -> None:
    """Unreadable memory and a missing directory are None, not zero."""
    assert total_memory_bytes(lambda: None) is None
    assert total_memory_bytes(lambda: (123, 5)) == 123
    assert directory_size_bytes(tmp_path / "missing") is None
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert directory_size_bytes(tmp_path) == 15


def test_result_document_with_harness_blocks_validates_and_malformed_blocks_are_rejected() -> None:
    """The optional harness, series_quality and preconditions blocks are accepted and type-checked."""
    from arc_runner.benchmark.results import build_result, capture_environment, validate_result

    samples, protocol, environment = finish_run(
        _series("a", [1, 1, 1], [1.0] * 3), HarnessConfig(cooldown_s=0.0), PreconditionReport(True, "", 1.0, 30.0, "", 10, 5, True, ""),
    )
    document = build_result("t", samples, {"warmup_runs": 3, "timed_runs": 3, "temperature": 0.0, "max_new_tokens": 8, "labels": {"a": "x"}, **protocol}, {**capture_environment(Path(".")), **environment})
    assert validate_result(document) == []
    document["protocol"]["harness"] = []
    document["protocol"]["series_quality"] = {}
    document["environment"]["preconditions"] = 3
    problems = " ".join(validate_result(document))
    assert "protocol.harness" in problems and "protocol.series_quality" in problems and "environment.preconditions" in problems
