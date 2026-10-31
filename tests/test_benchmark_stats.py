"""Slice S-E tests: statistics helpers and claim gates."""

import math
import statistics
from typing import List

import pytest

from arc_runner.benchmark.runner import BenchmarkRunner
from arc_runner.benchmark.stats import (
    BenchSample,
    bootstrap_ci,
    bootstrap_ratio_ci,
    coefficient_of_variation,
    evaluate_noninferiority_gate,
    evaluate_speedup_gate,
    iqr,
    is_within_roofline,
    mean,
    median,
    student_t_ci95,
    student_t_critical_95,
    warmup_converged,
)


def _samples(values: List[float], backend: str = "arc", source: str = "measured", **overrides) -> List[BenchSample]:
    """Build timed decode_tps samples from values."""
    return [
        BenchSample("m", backend, i, 100.0, v, 10, 32, source, **overrides)  # type: ignore[arg-type]
        for i, v in enumerate(values)
    ]


FAST = [100.0 + (i % 3) for i in range(12)]
SLOW = [50.0 + (i % 3) for i in range(12)]


@pytest.mark.parametrize("df,expected", [(1, 12.706), (10, 2.228), (11, 2.201), (15, 2.131), (19, 2.093), (30, 2.042), (40, 2.021), (120, 1.980), (500, 1.960)])
def test_student_t_critical_95_matches_published_table(df: int, expected: float) -> None:
    """The previous table skipped df 11-19 and returned a value that was too small."""
    assert student_t_critical_95(df) == pytest.approx(expected, abs=0.001)


def test_student_t_critical_95_interpolates_between_tail_points_monotonically() -> None:
    """Values between 40, 60 and 120 lie between their neighbours and never increase with df."""
    assert 2.000 < student_t_critical_95(50) < 2.021
    assert 1.980 < student_t_critical_95(90) < 2.000
    previous = student_t_critical_95(1)
    for df in range(2, 300):
        current = student_t_critical_95(df)
        assert current <= previous
        previous = current


def test_student_t_critical_95_rejects_zero_degrees_of_freedom() -> None:
    """df below 1 is invalid."""
    with pytest.raises(ValueError):
        student_t_critical_95(0)


def test_student_t_ci95_uses_exact_multiplier_for_twelve_samples() -> None:
    """12 samples (df 11) use 2.201, not the 2.086 the old round-up returned."""
    values = [float(i) for i in range(12)]
    mu, half = student_t_ci95(values)
    expected = 2.201 * statistics.stdev(values) / math.sqrt(12)
    assert mu == pytest.approx(statistics.fmean(values))
    assert half == pytest.approx(expected, rel=1e-3)


def test_runner_confidence_interval_delegates_to_shared_implementation() -> None:
    """Both runners use one implementation."""
    values = [10.0, 11.0, 9.5, 10.5, 10.2, 9.9]
    assert BenchmarkRunner()._compute_confidence_interval_95(values) == student_t_ci95(values)


def test_student_t_ci95_edge_cases() -> None:
    """Empty and single-sample inputs return zero half-width."""
    assert student_t_ci95([]) == (0.0, 0.0)
    assert student_t_ci95([5.0]) == (5.0, 0.0)


def test_median_iqr_cv_basic_values() -> None:
    """Descriptive statistics on a known set."""
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median([]) == 0.0
    assert iqr([1.0, 2.0, 3.0, 4.0, 5.0]) == 2.0
    assert iqr([1.0]) == 0.0
    assert coefficient_of_variation([10.0, 10.0, 10.0]) == 0.0
    assert coefficient_of_variation([0.0, 0.0]) == 0.0
    assert mean([]) == 0.0


def test_bootstrap_ci_deterministic_with_seed() -> None:
    """The same seed gives the same interval; it brackets the mean."""
    values = [9.0, 10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1]
    first = bootstrap_ci(values, seed=7)
    assert first == bootstrap_ci(values, seed=7)
    assert first[0] < statistics.fmean(values) < first[1]
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ratio_ci_brackets_true_ratio_and_validates_input() -> None:
    """The ratio interval contains the observed ratio and excludes 1."""
    lo, hi = bootstrap_ratio_ci(FAST, SLOW, seed=1)
    assert lo < mean(FAST) / mean(SLOW) < hi and lo > 1.5
    with pytest.raises(ValueError):
        bootstrap_ratio_ci([], SLOW)
    with pytest.raises(ValueError):
        bootstrap_ratio_ci(FAST, [0.0, 0.0])


def test_warmup_converged_requires_enough_stable_values() -> None:
    """Convergence needs a full window with low variation."""
    assert warmup_converged([100.0, 50.0, 30.0, 30.2, 29.9]) is True
    assert warmup_converged([100.0, 50.0, 30.0]) is False
    assert warmup_converged([30.0, 30.0]) is False
    assert warmup_converged([1.0, 2.0, 3.0], window=1) is False


def test_is_within_roofline_flags_rates_above_the_bandwidth_ceiling() -> None:
    """B / (W + KV) is the ceiling: 74.5 GB/s over 1 GB per token is 74.5 tok/s."""
    assert is_within_roofline(70.0, 1e9, 0.0, 74.5) is True
    assert is_within_roofline(80.0, 1e9, 0.0, 74.5) is False
    assert is_within_roofline(80.0, 0.0, 0.0, 74.5) is True


def test_gate_passes_for_clearly_faster_measured_system() -> None:
    """A 2x faster, low-variance, measured system passes."""
    result = evaluate_speedup_gate(_samples(FAST), _samples(SLOW, "ref"))
    assert result.passed and result.ratio_lo > 1.0 and result.n_arc == 12


def test_gate_rejects_projected_rows() -> None:
    """Projected rows can never support a claim."""
    result = evaluate_speedup_gate(_samples(FAST, source="projected"), _samples(SLOW, "ref"))
    assert not result.passed and "not source=measured" in result.reason


def test_gate_rejects_derived_fallback_noisy_and_too_few_rows() -> None:
    """Derived, fallback, noisy rows and short runs are all refused."""
    ref = _samples(SLOW, "ref")
    assert not evaluate_speedup_gate(_samples(FAST, source="derived"), ref).passed
    assert "fallback" in evaluate_speedup_gate(_samples(FAST, fallback_used=True), ref).reason
    assert "noisy" in evaluate_speedup_gate(_samples(FAST, noisy=True), ref).reason
    assert "need 10" in evaluate_speedup_gate(_samples(FAST[:5]), ref).reason


def test_gate_ignores_warmup_rows_when_counting_and_judging() -> None:
    """Warmup rows are excluded, so noisy warmups do not fail the gate but also do not count as timed values."""
    warmups = _samples([1.0, 500.0], is_warmup=True)
    result = evaluate_speedup_gate(warmups + _samples(FAST), _samples(SLOW, "ref"))
    assert result.passed and result.n_arc == 12


def test_gate_rejects_high_variance_and_equal_systems() -> None:
    """High CV and overlapping systems do not pass."""
    noisy = [50.0, 150.0] * 6
    assert "variation" in evaluate_speedup_gate(_samples(noisy), _samples(SLOW, "ref")).reason
    equal = evaluate_speedup_gate(_samples(SLOW), _samples(SLOW, "ref"))
    assert not equal.passed and equal.ratio_lo < 1.0


def test_gate_marks_results_above_the_roofline_as_suspect() -> None:
    """A median above the bandwidth ceiling is SUSPECT and fails."""
    result = evaluate_speedup_gate(_samples(FAST), _samples(SLOW, "ref"), roofline_tps=60.0)
    assert not result.passed and result.is_suspect


def test_gate_lower_is_better_for_ttft() -> None:
    """For ttft_ms the ratio is ref/arc, so a smaller arc TTFT passes."""
    arc = [BenchSample("m", "arc", i, 100.0 + (i % 3), None, 1, 1, "measured") for i in range(12)]
    ref = [BenchSample("m", "ref", i, 300.0 + (i % 3), None, 1, 1, "measured") for i in range(12)]
    assert evaluate_speedup_gate(arc, ref, metric="ttft_ms").passed
    assert not evaluate_speedup_gate(ref, arc, metric="ttft_ms").passed


def test_noninferiority_gate_accepts_same_speed_and_rejects_a_ten_percent_slowdown() -> None:
    """Within the 3% margin passes; a 10% regression fails."""
    old = _samples(SLOW, "old")
    assert evaluate_noninferiority_gate(_samples(SLOW, "new"), old).passed
    slower = _samples([v * 0.90 for v in SLOW], "new")
    assert not evaluate_noninferiority_gate(slower, old).passed


def test_noninferiority_gate_uses_old_over_new_for_ttft() -> None:
    """A faster new TTFT passes; a much slower one fails."""
    def rows(base: float, label: str) -> List[BenchSample]:
        return [BenchSample("m", label, i, base + (i % 3), None, 1, 1, "measured") for i in range(12)]

    assert evaluate_noninferiority_gate(rows(90.0, "new"), rows(100.0, "old"), metric="ttft_ms").passed
    assert not evaluate_noninferiority_gate(rows(130.0, "new"), rows(100.0, "old"), metric="ttft_ms").passed
