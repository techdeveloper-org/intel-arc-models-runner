"""Independent statistics and claim-gate tests for arc_runner.benchmark.stats.

Specification: docs/benchmarks/METHOD.md "Claim gates", SRS.md FR-005 and the stats.py docstrings.
Reference values are derived in this file by formula (sample mean, n-1 variance, Student-t quantile by
numerical inversion of the regularized incomplete beta function); the module under test is never used to
produce an expected value.
"""

import math
import random

import numpy as np
import pytest

from arc_runner.benchmark import stats
from arc_runner.benchmark.stats import BenchSample

SEED = 7771


# ---------------------------------------------------------------- reference statistics -------------

def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def _t_crit_975(df):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < 0.975:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _ref_mean(xs):
    return math.fsum(xs) / len(xs)


def _ref_std(xs):
    mu = _ref_mean(xs)
    return math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


TEXTBOOK_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980}


def test_reference_t_quantile_matches_textbook_table():
    for df, expected in TEXTBOOK_T975.items():
        assert abs(_t_crit_975(df) - expected) < 6e-4, df


# ---------------------------------------------------------------- Student t ------------------------

@pytest.mark.parametrize("df,expected", sorted(TEXTBOOK_T975.items()))
def test_student_t_critical_95_matches_textbook_values(df, expected):
    assert abs(stats.student_t_critical_95(df) - expected) < 6e-4


def test_student_t_critical_95_within_table_range_matches_numeric_inverse():
    errors = [(df, stats.student_t_critical_95(df), _t_crit_975(df)) for df in range(1, 121)
              if abs(stats.student_t_critical_95(df) - _t_crit_975(df)) > 2.5e-3]
    assert errors == []


def test_student_t_critical_95_is_monotone_non_increasing_in_df():
    values = [stats.student_t_critical_95(df) for df in range(1, 400)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_student_t_critical_95_beyond_120_is_the_documented_normal_value_with_bounded_gap():
    # Docstring: "the normal value beyond 120". True t(0.975, 121) = 1.9798, so the step is anti-conservative
    # by ~1% of the multiplier; asserted here as documented behaviour with a 1.5% bound (see report).
    assert stats.student_t_critical_95(121) == 1.960
    assert stats.student_t_critical_95(10 ** 6) == 1.960
    worst = max(_t_crit_975(df) - stats.student_t_critical_95(df) for df in (121, 150, 200, 500))
    assert 0 < worst < 0.015 * 1.98


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_student_t_critical_95_rejects_df_below_one(bad):
    with pytest.raises(ValueError):
        stats.student_t_critical_95(bad)


def test_student_t_ci95_matches_hand_computation_for_random_samples():
    rng = random.Random(SEED)
    for _ in range(300):
        n = rng.randrange(2, 31)
        xs = [rng.uniform(-50, 500) for _ in range(n)]
        mu, half = stats.student_t_ci95(xs)
        assert mu == pytest.approx(_ref_mean(xs), rel=1e-12, abs=1e-9)
        expected_half = _t_crit_975(n - 1) * _ref_std(xs) / math.sqrt(n)
        assert half == pytest.approx(expected_half, rel=2e-3, abs=1e-9)


def test_student_t_ci95_known_worked_example():
    # data 2,4,4,4,5,5,7,9: mean 5, sum sq dev = 32, s^2 = 32/7, df 7, t = 2.365
    mu, half = stats.student_t_ci95([2, 4, 4, 4, 5, 5, 7, 9])
    assert mu == 5.0
    assert half == pytest.approx(2.365 * math.sqrt(32 / 7) / math.sqrt(8), rel=1e-3)


def test_student_t_ci95_degenerate_inputs():
    assert stats.student_t_ci95([]) == (0.0, 0.0)
    assert stats.student_t_ci95([3.5]) == (3.5, 0.0)
    mu, half = stats.student_t_ci95([4.0, 4.0, 4.0])
    assert (mu, half) == (4.0, 0.0)


def test_student_t_ci95_is_translation_invariant_and_scale_equivariant():
    rng = random.Random(SEED + 1)
    xs = [rng.uniform(1, 9) for _ in range(12)]
    mu, half = stats.student_t_ci95(xs)
    mu2, half2 = stats.student_t_ci95([x + 1000 for x in xs])
    assert half2 == pytest.approx(half, rel=1e-9)
    assert mu2 == pytest.approx(mu + 1000, rel=1e-12)
    mu3, half3 = stats.student_t_ci95([3 * x for x in xs])
    assert half3 == pytest.approx(3 * half, rel=1e-9)


def test_student_t_ci95_empirical_coverage_close_to_95_percent():
    rng = random.Random(SEED + 2)
    hits, trials = 0, 3000
    for _ in range(trials):
        xs = [rng.gauss(10.0, 2.0) for _ in range(8)]
        mu, half = stats.student_t_ci95(xs)
        hits += (mu - half) <= 10.0 <= (mu + half)
    assert 0.935 <= hits / trials <= 0.965


# ---------------------------------------------------------------- descriptive helpers ---------------

def test_mean_median_iqr_cv_match_reference_for_random_samples():
    rng = random.Random(SEED + 3)
    for _ in range(200):
        n = rng.randrange(2, 40)
        xs = [rng.uniform(0.1, 100) for _ in range(n)]
        assert stats.mean(xs) == pytest.approx(_ref_mean(xs), rel=1e-12)
        assert stats.median(xs) == pytest.approx(float(np.median(xs)), rel=1e-12)
        q = np.percentile(xs, [25, 75])  # linear interpolation, type 7
        assert stats.iqr(xs) == pytest.approx(float(q[1] - q[0]), rel=1e-9, abs=1e-12)
        assert stats.coefficient_of_variation(xs) == pytest.approx(_ref_std(xs) / abs(_ref_mean(xs)), rel=1e-9)


def test_descriptive_helpers_empty_and_single_sample_behaviour():
    assert stats.mean([]) == 0.0 and stats.median([]) == 0.0
    assert stats.iqr([5.0]) == 0.0 and stats.coefficient_of_variation([5.0]) == 0.0
    assert stats.coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0


def test_warmup_converged_follows_trailing_window_cv():
    assert stats.warmup_converged([100, 50, 10.0, 10.1, 9.9]) is True
    assert stats.warmup_converged([10.0, 10.1, 9.9, 50]) is False
    assert stats.warmup_converged([10.0, 10.0]) is False  # fewer than window=3
    assert stats.warmup_converged([1, 1, 1], window=1) is False  # window must be at least 2
    rng = random.Random(SEED + 4)
    for _ in range(100):
        xs = [rng.uniform(1, 100) for _ in range(rng.randrange(3, 10))]
        tail = xs[-3:]
        expected = _ref_std(tail) / _ref_mean(tail) <= 0.05
        assert stats.warmup_converged(xs) is expected


def test_is_within_roofline_matches_formula():
    rng = random.Random(SEED + 5)
    for _ in range(200):
        w, kv, bw = rng.uniform(1e8, 1e10), rng.uniform(0, 1e9), rng.uniform(5, 200)
        ceiling = bw * 1e9 / (w + kv)
        assert stats.is_within_roofline(ceiling * 0.99, w, kv, bw) is True
        assert stats.is_within_roofline(ceiling * 1.01, w, kv, bw) is False
    assert stats.is_within_roofline(1e12, 0, 0, 100) is True  # no traffic: no ceiling can be stated
    assert stats.is_within_roofline(1e12, 1e9, 0, 0) is True  # unknown bandwidth: no ceiling


# ---------------------------------------------------------------- bootstrap -------------------------

def test_bootstrap_ci_properties():
    xs = [3.0, 5.0, 8.0, 13.0, 21.0, 34.0]
    lo, hi = stats.bootstrap_ci(xs, seed=1)
    assert min(xs) <= lo <= hi <= max(xs)
    assert (lo, hi) == stats.bootstrap_ci(xs, seed=1)  # deterministic per seed
    assert stats.bootstrap_ci([7.0], seed=3) == (7.0, 7.0)
    assert stats.bootstrap_ci([2.0, 2.0, 2.0], seed=3) == (2.0, 2.0)
    lo2, hi2 = stats.bootstrap_ci([x + 100 for x in xs], seed=1)
    assert lo2 == pytest.approx(lo + 100) and hi2 == pytest.approx(hi + 100)
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])
    wide = stats.bootstrap_ci(xs, alpha=0.01, seed=1)
    narrow = stats.bootstrap_ci(xs, alpha=0.5, seed=1)
    assert wide[0] <= narrow[0] and narrow[1] <= wide[1]


def test_bootstrap_ci_is_centred_near_sample_mean_for_large_resamples():
    rng = random.Random(SEED + 6)
    xs = [rng.gauss(50, 5) for _ in range(40)]
    lo, hi = stats.bootstrap_ci(xs, n_resamples=3000, seed=2)
    se = _ref_std(xs) / math.sqrt(len(xs))
    # percentile interval of the mean approx mean +- 1.96 se
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.12)
    assert lo < _ref_mean(xs) < hi


def test_bootstrap_ratio_ci_constant_samples_collapse_to_the_exact_ratio():
    lo, hi = stats.bootstrap_ratio_ci([20.0] * 12, [10.0] * 15, seed=4)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_bootstrap_ratio_ci_scale_properties():
    rng = random.Random(SEED + 7)
    num = [rng.uniform(15, 25) for _ in range(15)]
    den = [rng.uniform(8, 12) for _ in range(15)]
    lo, hi = stats.bootstrap_ratio_ci(num, den, seed=9)
    lo_k, hi_k = stats.bootstrap_ratio_ci([3 * x for x in num], den, seed=9)
    assert lo_k == pytest.approx(3 * lo, rel=1e-9) and hi_k == pytest.approx(3 * hi, rel=1e-9)
    lo_d, hi_d = stats.bootstrap_ratio_ci(num, [2 * x for x in den], seed=9)
    assert lo_d == pytest.approx(lo / 2, rel=1e-9) and hi_d == pytest.approx(hi / 2, rel=1e-9)
    assert lo <= hi
    assert lo < _ref_mean(num) / _ref_mean(den) < hi


def test_bootstrap_ratio_ci_validation_errors():
    with pytest.raises(ValueError):
        stats.bootstrap_ratio_ci([], [1.0])
    with pytest.raises(ValueError):
        stats.bootstrap_ratio_ci([1.0], [])
    with pytest.raises(ValueError):
        stats.bootstrap_ratio_ci([1.0], [0.0, 0.0])


def test_bootstrap_ratio_ci_empirical_coverage_of_true_ratio():
    rng = random.Random(SEED + 8)
    hits, trials = 0, 60
    for t in range(trials):
        num = [rng.gauss(20, 2) for _ in range(30)]
        den = [rng.gauss(10, 1) for _ in range(30)]
        lo, hi = stats.bootstrap_ratio_ci(num, den, n_resamples=300, seed=t)
        hits += lo <= 2.0 <= hi
    assert hits / trials >= 0.85


# ---------------------------------------------------------------- claim gates -----------------------

def _samples(values, source="measured", fallback=False, noisy=False, warmup=False, metric="decode_tps", start=0):
    out = []
    for i, v in enumerate(values):
        kw = {"ttft_ms": None, "decode_tps": None}
        kw[metric] = v
        out.append(BenchSample("m", "b", start + i, kw["ttft_ms"], kw["decode_tps"], 8, 16, source,
                               is_warmup=warmup, fallback_used=fallback, noisy=noisy))
    return out


def _tight(center, n=12, spread=0.01, seed=0):
    rng = random.Random(seed)
    return [center * (1 + rng.uniform(-spread, spread)) for _ in range(n)]


def test_speedup_gate_passes_for_exactly_known_ratio_of_two():
    r = stats.evaluate_speedup_gate(_samples([20.0] * 10), _samples([10.0] * 10))
    assert r.passed is True
    assert r.ratio_lo == pytest.approx(2.0) and r.ratio_hi == pytest.approx(2.0)
    assert (r.n_arc, r.n_ref) == (10, 10)


def test_speedup_gate_fails_when_ratio_is_exactly_one():
    r = stats.evaluate_speedup_gate(_samples([10.0] * 10), _samples([10.0] * 10))
    assert r.passed is False  # lower bound must be strictly above 1.0


def test_speedup_gate_fails_when_arc_is_slower():
    r = stats.evaluate_speedup_gate(_samples(_tight(8, seed=1)), _samples(_tight(10, seed=2)))
    assert r.passed is False


def test_speedup_gate_passes_clear_win_with_noise_and_fails_overlapping_means():
    assert stats.evaluate_speedup_gate(_samples(_tight(15, seed=3)), _samples(_tight(10, seed=4))).passed is True
    assert stats.evaluate_speedup_gate(_samples(_tight(10.05, seed=5, spread=0.04)), _samples(_tight(10, seed=6, spread=0.04))).passed is False


def test_speedup_gate_minimum_sample_boundary_is_ten():
    assert stats.evaluate_speedup_gate(_samples([20.0] * 9), _samples([10.0] * 10)).passed is False
    assert stats.evaluate_speedup_gate(_samples([20.0] * 10), _samples([10.0] * 9)).passed is False
    assert stats.evaluate_speedup_gate(_samples([20.0] * 10), _samples([10.0] * 10)).passed is True


def test_speedup_gate_warmup_rows_are_ignored_when_counting_and_validating():
    arc = _samples([20.0] * 10) + _samples([1.0, 2.0], source="projected", warmup=True, start=10)
    ref = _samples([10.0] * 10) + _samples([99.0], noisy=True, fallback=True, warmup=True, start=10)
    r = stats.evaluate_speedup_gate(arc, ref)
    assert r.passed is True and (r.n_arc, r.n_ref) == (10, 10)
    # warm-up rows only: nothing timed, cannot pass
    assert stats.evaluate_speedup_gate(_samples([20.0] * 12, warmup=True), _samples([10.0] * 12)).passed is False


@pytest.mark.parametrize("source", ["derived", "projected"])
def test_speedup_gate_single_non_measured_timed_row_vetoes_either_side(source):
    good_arc, good_ref = _samples([20.0] * 12), _samples([10.0] * 12)
    for idx in (0, 5, 11):
        bad = list(good_arc)
        bad[idx] = _samples([20.0], source=source, start=idx)[0]
        assert stats.evaluate_speedup_gate(bad, good_ref).passed is False
        bad_ref = list(good_ref)
        bad_ref[idx] = _samples([10.0], source=source, start=idx)[0]
        assert stats.evaluate_speedup_gate(good_arc, bad_ref).passed is False


@pytest.mark.parametrize("flag", ["fallback", "noisy"])
def test_speedup_gate_single_fallback_or_noisy_row_vetoes(flag):
    arc = _samples([20.0] * 12)
    ref = _samples([10.0] * 12)
    marked_arc = _samples([20.0], start=3, **{flag: True})[0]
    marked_ref = _samples([10.0], start=3, **{flag: True})[0]
    assert stats.evaluate_speedup_gate(arc[:3] + [marked_arc] + arc[4:], ref).passed is False
    assert stats.evaluate_speedup_gate(arc, ref[:3] + [marked_ref] + ref[4:]).passed is False


def test_speedup_gate_cv_threshold_boundary():
    # build 12 values with sample CV exactly controllable: two-point distribution around mean 20
    def with_cv(cv):
        n = 12
        d = cv * 20.0 / math.sqrt(n / (n - 1))
        return [20.0 + d] * (n // 2) + [20.0 - d] * (n // 2)

    ref = _samples([10.0] * 12)
    assert stats.coefficient_of_variation(with_cv(0.09)) == pytest.approx(0.09)
    assert stats.evaluate_speedup_gate(_samples(with_cv(0.09)), ref).passed is True
    assert stats.evaluate_speedup_gate(_samples(with_cv(0.11)), ref).passed is False
    assert stats.evaluate_speedup_gate(ref, _samples(with_cv(0.11))).passed is False


def test_speedup_gate_roofline_marks_suspect_and_never_passes():
    arc, ref = _samples([200.0] * 12), _samples([10.0] * 12)
    r = stats.evaluate_speedup_gate(arc, ref, roofline_tps=100.0)
    assert r.passed is False and r.is_suspect is True
    ok = stats.evaluate_speedup_gate(arc, ref, roofline_tps=250.0)
    assert ok.passed is True and ok.is_suspect is False


def test_speedup_gate_ttft_direction_is_lower_is_better():
    arc = _samples([50.0] * 10, metric="ttft_ms")
    ref = _samples([100.0] * 10, metric="ttft_ms")
    r = stats.evaluate_speedup_gate(arc, ref, metric="ttft_ms")
    assert r.passed is True and r.ratio_lo == pytest.approx(2.0)
    assert stats.evaluate_speedup_gate(ref, arc, metric="ttft_ms").passed is False


def test_speedup_gate_is_deterministic_for_a_seed():
    a, b = _samples(_tight(13, seed=8)), _samples(_tight(10, seed=9))
    assert stats.evaluate_speedup_gate(a, b, seed=5) == stats.evaluate_speedup_gate(a, b, seed=5)


def test_speedup_gate_zero_throughput_reference_returns_a_verdict_not_an_exception():
    # a reference whose every timed decode_tps is 0.0 is measured, has CV 0 and n>=10; the gate must say
    # "not passed" (or pass: any ratio above 1 of x/0 is unbounded) but must not raise
    ref = _samples([0.0] * 12)
    arc = _samples([20.0] * 12)
    result = stats.evaluate_speedup_gate(arc, ref)
    assert result.passed is False


def test_speedup_gate_zero_ttft_arc_returns_a_verdict_not_an_exception():
    arc = _samples([0.0] * 12, metric="ttft_ms")
    ref = _samples([100.0] * 12, metric="ttft_ms")
    result = stats.evaluate_speedup_gate(arc, ref, metric="ttft_ms")
    assert result.passed is False


def test_noninferiority_gate_margin_semantics_with_exact_ratios():
    old = _samples([10.0] * 12)
    assert stats.evaluate_noninferiority_gate(_samples([10.0] * 12), old).passed is True
    assert stats.evaluate_noninferiority_gate(_samples([9.8] * 12), old).passed is True   # -2% within 3%
    assert stats.evaluate_noninferiority_gate(_samples([9.69] * 12), old).passed is False  # -3.1%
    assert stats.evaluate_noninferiority_gate(_samples([9.71] * 12), old).passed is True
    assert stats.evaluate_noninferiority_gate(_samples([9.0] * 12), old, margin=0.15).passed is True
    assert stats.evaluate_noninferiority_gate(_samples([8.4] * 12), old, margin=0.15).passed is False
    assert stats.evaluate_noninferiority_gate(_samples([10.0] * 12), old, margin=0.0).passed is False  # strict


def test_noninferiority_gate_ttft_uses_old_over_new():
    old = _samples([100.0] * 12, metric="ttft_ms")
    assert stats.evaluate_noninferiority_gate(_samples([102.0] * 12, metric="ttft_ms"), old, metric="ttft_ms").passed is True
    assert stats.evaluate_noninferiority_gate(_samples([104.0] * 12, metric="ttft_ms"), old, metric="ttft_ms").passed is False


def test_noninferiority_gate_applies_same_eligibility_rules():
    old = _samples([10.0] * 12)
    assert stats.evaluate_noninferiority_gate(_samples([10.0] * 9), old).passed is False
    assert stats.evaluate_noninferiority_gate(_samples([10.0] * 12, source="derived"), old).passed is False
    assert stats.evaluate_noninferiority_gate(_samples([10.0] * 12, fallback=True), old).passed is False
    assert stats.evaluate_noninferiority_gate(_samples([10.0] * 12), _samples([10.0] * 12, noisy=True)).passed is False
