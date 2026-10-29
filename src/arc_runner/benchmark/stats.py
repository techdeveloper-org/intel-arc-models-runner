"""Statistics and claim gates for benchmark results.

One implementation of confidence intervals, bootstrap ratios, warmup convergence, roofline checks
and the gates that decide whether a speed claim may be published. Rows that were projected, derived,
noisy, or produced through a fallback can never support a claim.
"""

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Literal, Optional, Sequence, Tuple

SampleSource = Literal["measured", "derived", "projected"]

MIN_TIMED_ITERATIONS = 10
MAX_CV_FOR_CLAIMS = 0.10
DEFAULT_NONINFERIORITY_MARGIN = 0.03

_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}
_T_95_TAIL = ((40, 2.021), (60, 2.000), (120, 1.980))
_T_95_INFINITY = 1.960


@dataclass(frozen=True)
class BenchSample:
    """One timed (or warmup) measurement with its provenance.

    LLM samples use ttft_ms, decode_tps and the token counts. Other workloads (embeddings, VLM, audio,
    image, video) leave those None/0 and report named measurements in metrics, whose keys carry their unit
    as a suffix (latency_s, rtf, throughput_fps, energy_j); a metric that could not be measured is None.
    """

    model_id: str
    backend: str
    iteration: int
    ttft_ms: Optional[float]
    decode_tps: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    source: SampleSource
    is_warmup: bool = False
    fallback_used: bool = False
    noisy: bool = False
    workload: str = "llm"
    metrics: Optional[Dict[str, Optional[float]]] = None


@dataclass(frozen=True)
class ClaimGateResult:
    """Outcome of a claim gate."""

    passed: bool
    reason: str
    ratio_lo: float = 0.0
    ratio_hi: float = 0.0
    n_arc: int = 0
    n_ref: int = 0
    is_suspect: bool = False


def student_t_critical_95(df: int) -> float:
    """Return the two-sided 95% Student's t critical value.

    Exact table values for df 1-30; table values at 40, 60 and 120 with interpolation in 1/df for the
    gaps; the normal value beyond 120. (The previous runner table skipped df 11-19 and rounded up to the
    next key, which returned too small a multiplier.)

    Args:
        df: Degrees of freedom, at least 1.

    Returns:
        float: Critical value.

    Raises:
        ValueError: If df is smaller than 1.
    """
    if df < 1:
        raise ValueError(f"degrees of freedom must be >= 1, got {df}")
    if df in _T_95:
        return _T_95[df]
    points = [(30, _T_95[30]), *_T_95_TAIL]
    for (low_df, low_t), (high_df, high_t) in zip(points, points[1:]):
        if low_df < df <= high_df:
            weight = (1.0 / low_df - 1.0 / df) / (1.0 / low_df - 1.0 / high_df)
            return low_t + weight * (high_t - low_t)
    return _T_95_INFINITY


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty sequence.

    Args:
        values: Samples.
    """
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    """Return the median, or 0.0 for an empty sequence.

    Args:
        values: Samples.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted sequence.

    Args:
        ordered: Sorted samples, non-empty.
        q: Quantile in [0, 1].
    """
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def iqr(values: Sequence[float]) -> float:
    """Return the interquartile range, or 0.0 for fewer than two samples.

    Args:
        values: Samples.
    """
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    return _quantile(ordered, 0.75) - _quantile(ordered, 0.25)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return sample standard deviation divided by the mean (0.0 if fewer than 2 samples or mean is 0).

    Args:
        values: Samples.
    """
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    if mu == 0.0:
        return 0.0
    variance = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / abs(mu)


def student_t_ci95(values: Sequence[float]) -> Tuple[float, float]:
    """Return the sample mean and the 95% Student's t half-width.

    Args:
        values: Samples.

    Returns:
        Tuple[float, float]: (mean, half_width); half_width is 0.0 for fewer than two samples.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = mean(values)
    if n < 2:
        return mu, 0.0
    std = math.sqrt(sum((x - mu) ** 2 for x in values) / (n - 1))
    return mu, student_t_critical_95(n - 1) * std / math.sqrt(n)


def bootstrap_ci(
    values: Sequence[float], alpha: float = 0.05, n_resamples: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean, deterministic for a given seed.

    Args:
        values: Samples, non-empty.
        alpha: Total tail probability (0.05 gives a 95% interval).
        n_resamples: Number of bootstrap resamples.
        seed: Seed of the private random generator.

    Returns:
        Tuple[float, float]: (lower, upper).

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_resamples))
    return _quantile(means, alpha / 2.0), _quantile(means, 1.0 - alpha / 2.0)


def bootstrap_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap interval of mean(numerator) / mean(denominator) with independent resampling.

    Args:
        numerator: Samples of the numerator system.
        denominator: Samples of the denominator system; must have a positive mean.
        alpha: Total tail probability.
        n_resamples: Number of resamples.
        seed: Seed of the private random generator.

    Returns:
        Tuple[float, float]: (lower, upper) of the ratio.

    Raises:
        ValueError: If either sequence is empty or the denominator mean is not positive.
    """
    if not numerator or not denominator or mean(denominator) <= 0:
        raise ValueError("ratio needs non-empty samples and a positive denominator mean")
    rng = random.Random(seed)
    n, m = len(numerator), len(denominator)
    ratios: List[float] = []
    for _ in range(n_resamples):
        top = sum(rng.choice(numerator) for _ in range(n)) / n
        bottom = sum(rng.choice(denominator) for _ in range(m)) / m
        if bottom > 0:
            ratios.append(top / bottom)
    ratios.sort()
    return _quantile(ratios, alpha / 2.0), _quantile(ratios, 1.0 - alpha / 2.0)


def warmup_converged(values: Sequence[float], window: int = 3, eps: float = 0.05) -> bool:
    """Return True when the last `window` values vary by no more than eps (coefficient of variation).

    Args:
        values: Warmup measurements in order.
        window: Number of trailing values to inspect, at least 2.
        eps: Maximum coefficient of variation.
    """
    if window < 2 or len(values) < window:
        return False
    return coefficient_of_variation(values[-window:]) <= eps


def is_within_roofline(tps: float, weight_bytes: float, kv_bytes: float, bandwidth_gbs: float) -> bool:
    """Check that a decode rate does not exceed the memory-bandwidth ceiling B / (W + KV).

    Args:
        tps: Measured decode tokens per second.
        weight_bytes: Bytes of weights read per token.
        kv_bytes: Bytes of KV cache read per token.
        bandwidth_gbs: Sustained memory bandwidth in GB/s.

    Returns:
        bool: True if tps is at or below the ceiling.
    """
    traffic = weight_bytes + kv_bytes
    if traffic <= 0 or bandwidth_gbs <= 0:
        return True
    return tps <= (bandwidth_gbs * 1e9) / traffic


def _timed_values(samples: Sequence[BenchSample], metric: str) -> List[float]:
    """Collect non-warmup, non-null metric values from samples.

    Args:
        samples: Samples to read.
        metric: Attribute name (decode_tps or ttft_ms).
    """
    return [getattr(s, metric) for s in samples if not s.is_warmup and getattr(s, metric) is not None]


def _ineligible_reason(samples: Sequence[BenchSample], label: str, metric: str, min_n: int, max_cv: float) -> str:
    """Return why a sample set cannot support a claim, or an empty string.

    Args:
        samples: Samples of one system.
        label: Name used in the reason text.
        metric: Metric attribute.
        min_n: Minimum number of timed values.
        max_cv: Maximum coefficient of variation.
    """
    timed = [s for s in samples if not s.is_warmup]
    if any(s.source != "measured" for s in timed):
        return f"{label} contains rows that are not source=measured"
    if any(s.fallback_used for s in timed):
        return f"{label} contains rows that used a fallback"
    if any(s.noisy for s in timed):
        return f"{label} contains rows flagged noisy"
    values = _timed_values(samples, metric)
    if len(values) < min_n:
        return f"{label} has {len(values)} timed values, need {min_n}"
    if coefficient_of_variation(values) > max_cv:
        return f"{label} coefficient of variation exceeds {max_cv:.0%}"
    return ""


def evaluate_speedup_gate(
    arc: Sequence[BenchSample],
    ref: Sequence[BenchSample],
    metric: str = "decode_tps",
    min_n: int = MIN_TIMED_ITERATIONS,
    max_cv: float = MAX_CV_FOR_CLAIMS,
    roofline_tps: Optional[float] = None,
    seed: int = 0,
) -> ClaimGateResult:
    """Decide whether arc may be claimed faster than ref.

    Passes only if every timed row of both systems is measured, fallback-free and not noisy, both have at
    least min_n timed values with a coefficient of variation within max_cv, and the lower bound of the
    95% bootstrap interval of the speed ratio is above 1.0. For ttft_ms (lower is better) the ratio is
    ref/arc. A median above the roofline ceiling is SUSPECT and never passes.

    Args:
        arc: Samples of the system being promoted.
        ref: Samples of the reference system.
        metric: decode_tps or ttft_ms.
        min_n: Minimum timed values per system.
        max_cv: Maximum coefficient of variation.
        roofline_tps: Optional decode-rate ceiling for the SUSPECT check.
        seed: Bootstrap seed.

    Returns:
        ClaimGateResult: Verdict and the ratio interval.
    """
    n_arc, n_ref = len(_timed_values(arc, metric)), len(_timed_values(ref, metric))
    for samples, label in ((arc, "arc"), (ref, "ref")):
        reason = _ineligible_reason(samples, label, metric, min_n, max_cv)
        if reason:
            return ClaimGateResult(False, reason, n_arc=n_arc, n_ref=n_ref)

    arc_values, ref_values = _timed_values(arc, metric), _timed_values(ref, metric)
    if metric == "decode_tps" and roofline_tps is not None and median(arc_values) > roofline_tps:
        return ClaimGateResult(False, "median exceeds the roofline ceiling (SUSPECT)", n_arc=n_arc, n_ref=n_ref, is_suspect=True)

    try:
        lo, hi = bootstrap_ratio_ci(arc_values, ref_values, seed=seed) if metric == "decode_tps" else bootstrap_ratio_ci(ref_values, arc_values, seed=seed)
    except ValueError as exc:
        return ClaimGateResult(False, f"ratio undefined: {exc}", n_arc=n_arc, n_ref=n_ref)
    passed = lo > 1.0
    reason = "ratio lower bound above 1.0" if passed else f"ratio lower bound {lo:.3f} does not clear 1.0"
    return ClaimGateResult(passed, reason, lo, hi, n_arc, n_ref)


def evaluate_noninferiority_gate(
    new: Sequence[BenchSample],
    old: Sequence[BenchSample],
    metric: str = "decode_tps",
    margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    min_n: int = MIN_TIMED_ITERATIONS,
    max_cv: float = MAX_CV_FOR_CLAIMS,
    seed: int = 0,
) -> ClaimGateResult:
    """Decide whether a new version is no slower than the old one by more than a margin.

    For decode_tps the new/old ratio lower bound must exceed 1 - margin; for ttft_ms (lower is better)
    the old/new ratio lower bound must exceed 1 - margin.

    Args:
        new: Samples of the new version.
        old: Samples of the old version.
        metric: decode_tps or ttft_ms.
        margin: Allowed relative slowdown, for example 0.03.
        min_n: Minimum timed values per version.
        max_cv: Maximum coefficient of variation.
        seed: Bootstrap seed.

    Returns:
        ClaimGateResult: Verdict and the ratio interval.
    """
    n_new, n_old = len(_timed_values(new, metric)), len(_timed_values(old, metric))
    for samples, label in ((new, "new"), (old, "old")):
        reason = _ineligible_reason(samples, label, metric, min_n, max_cv)
        if reason:
            return ClaimGateResult(False, reason, n_arc=n_new, n_ref=n_old)

    new_values, old_values = _timed_values(new, metric), _timed_values(old, metric)
    try:
        lo, hi = bootstrap_ratio_ci(new_values, old_values, seed=seed) if metric == "decode_tps" else bootstrap_ratio_ci(old_values, new_values, seed=seed)
    except ValueError as exc:
        return ClaimGateResult(False, f"ratio undefined: {exc}", n_arc=n_new, n_ref=n_old)
    passed = lo > 1.0 - margin
    reason = f"lower bound {lo:.3f} above {1.0 - margin:.3f}" if passed else f"lower bound {lo:.3f} is below {1.0 - margin:.3f}"
    return ClaimGateResult(passed, reason, lo, hi, n_new, n_old)
