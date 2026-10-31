"""Red-first tests for WP-02 item 02b: bench_arms.py paired gate runner.

Covers:
  - bootstrap_paired_ratio_ci in stats.py (joint round resampling)
  - evaluate_paired_gate in stats.py (adopt / reject / inconclusive / harmful)
  - decide_adoption in stats.py (AD-1..AD-6 pure function)
  - bench_arms --evaluate mode (reads schema-v1 files, prints verdicts)
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.stats import (
    DEFAULT_NONINFERIORITY_MARGIN,
    PairedGateResult,
    bootstrap_paired_ratio_ci,
    decide_adoption,
    evaluate_paired_gate,
)


def _make_result_doc(
    samples: List[Dict[str, Any]],
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a minimal valid schema-v1 document for evaluate-mode tests."""
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "created_utc": "2026-10-30T00:00:00+00:00",
        "environment": {
            "os": "Windows-10",
            "python": "3.10.0",
            "git_sha": "abc1234",
            "hardware": "test-cpu (4 logical CPUs)",
        },
        "protocol": protocol,
        "samples": samples,
    }


def _make_sample(
    backend: str,
    iteration: int,
    decode_tps: float,
    ttft_ms: float = 10.0,
    is_warmup: bool = False,
) -> Dict[str, Any]:
    """Build one schema-v1 sample row."""
    return {
        "model_id": "test-model/medium",
        "backend": backend,
        "iteration": iteration,
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "prompt_tokens": 20,
        "completion_tokens": 64,
        "source": "measured",
        "is_warmup": is_warmup,
        "fallback_used": False,
    }


def _protocol(warmup: int = 0, timed: int = 18) -> Dict[str, Any]:
    """Minimal protocol block for a paired-arms run."""
    return {
        "warmup_runs": warmup,
        "timed_runs": timed,
        "temperature": 0.0,
        "max_new_tokens": 64,
        "labels": {"control": "control arm", "fast_arm": "fast arm"},
    }


# ---------------------------------------------------------------------------
# bootstrap_paired_ratio_ci
# ---------------------------------------------------------------------------


class TestBootstrapPairedRatioCI:
    """Tests for the new paired bootstrap interval in stats.py."""

    def test_identical_arms_interval_contains_one(self):
        """When arm and control have the same values the ratio CI must contain 1."""
        values = [10.0, 10.5, 9.8, 10.2, 11.0] * 4
        lo, hi = bootstrap_paired_ratio_ci(values, values, alpha=0.05, seed=42)
        assert lo <= 1.0 <= hi

    def test_faster_arm_lower_bound_above_one(self):
        """An arm that is 30% faster should have lower bound above 1.0."""
        control = [10.0] * 20
        arm = [13.0] * 20
        lo, hi = bootstrap_paired_ratio_ci(arm, control, alpha=0.05, seed=0)
        assert lo > 1.0

    def test_slower_arm_upper_bound_below_one(self):
        """An arm that is 30% slower should have upper bound below 1.0."""
        control = [13.0] * 20
        arm = [10.0] * 20
        lo, hi = bootstrap_paired_ratio_ci(arm, control, alpha=0.05, seed=0)
        assert hi < 1.0

    def test_mismatched_lengths_raises(self):
        """Different sequence lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same number of rounds"):
            bootstrap_paired_ratio_ci([1.0, 2.0], [1.0], alpha=0.05)

    def test_empty_raises(self):
        """Empty sequences must raise ValueError."""
        with pytest.raises(ValueError):
            bootstrap_paired_ratio_ci([], [], alpha=0.05)

    def test_zero_control_mean_raises(self):
        """A zero-mean control must raise ValueError."""
        with pytest.raises(ValueError):
            bootstrap_paired_ratio_ci([1.0, 2.0], [0.0, 0.0], alpha=0.05)

    def test_bonferroni_correction_widens_interval(self):
        """alpha = 0.05 / 6 (K=6) should give a wider interval than alpha = 0.05."""
        values = [10.0 + i * 0.05 for i in range(18)]
        lo1, hi1 = bootstrap_paired_ratio_ci(values, [10.0] * 18, alpha=0.05, seed=0)
        lo2, hi2 = bootstrap_paired_ratio_ci(values, [10.0] * 18, alpha=0.05 / 6, seed=0)
        assert lo2 < lo1 or hi2 > hi1

    def test_paired_cancels_common_drift(self):
        """Common additive drift should not inflate the ratio interval when pairing is correct."""
        base = 10.0
        drift = [i * 0.5 for i in range(18)]
        arm = [base + 3.0 + d for d in drift]
        ctrl = [base + d for d in drift]
        lo, hi = bootstrap_paired_ratio_ci(arm, ctrl, alpha=0.05, seed=0)
        assert lo > 1.0, "paired gate should detect 30% speedup despite common drift"


# ---------------------------------------------------------------------------
# evaluate_paired_gate
# ---------------------------------------------------------------------------


class TestEvaluatePairedGate:
    """Tests for evaluate_paired_gate in stats.py."""

    def test_adopt_when_arm_clearly_faster(self):
        """Arm 40% faster should yield verdict='adopt'."""
        ctrl = [10.0] * 18
        arm = [14.0] * 18
        result = evaluate_paired_gate(arm, ctrl, metric="decode_tps", k=1, seed=0)
        assert result.verdict == "adopt"
        assert isinstance(result, PairedGateResult)

    def test_reject_when_arm_at_same_speed(self):
        """Arm at identical speed should yield verdict='reject' (lower bound <= 1+margin)."""
        ctrl = [10.0] * 18
        arm = [10.0] * 18
        result = evaluate_paired_gate(arm, ctrl, metric="decode_tps", k=1, seed=0)
        assert result.verdict == "reject"

    def test_harmful_flag_when_arm_much_slower(self):
        """Arm 10% slower should set harmful=True."""
        ctrl = [10.0] * 18
        arm = [8.0] * 18
        result = evaluate_paired_gate(arm, ctrl, metric="decode_tps", k=1, seed=0)
        assert result.harmful is True

    def test_ttft_convention_faster_arm_adopt(self):
        """For ttft_ms the arm with lower latency should adopt (control/arm ratio above 1+margin)."""
        ctrl = [100.0] * 18
        arm = [65.0] * 18
        result = evaluate_paired_gate(arm, ctrl, metric="ttft_ms", k=1, seed=0)
        assert result.verdict == "adopt"

    def test_point_ratio_decode_tps(self):
        """Point ratio for decode_tps should be arm_mean / control_mean."""
        ctrl = [10.0] * 18
        arm = [12.0] * 18
        result = evaluate_paired_gate(arm, ctrl, metric="decode_tps", k=1, seed=0)
        assert abs(result.point_ratio - 1.2) < 0.001

    def test_k_bonferroni_correction_applied(self):
        """With K=6 the gate should be less likely to adopt a marginal arm."""
        ctrl = [10.0] * 18
        arm = [10.5] * 18
        r_k1 = evaluate_paired_gate(arm, ctrl, metric="decode_tps", k=1, seed=0)
        r_k6 = evaluate_paired_gate(arm, ctrl, metric="decode_tps", k=6, seed=0)
        assert r_k6.ratio_lo <= r_k1.ratio_lo

    def test_result_fields_present(self):
        """PairedGateResult must have all expected fields."""
        ctrl = [10.0] * 18
        arm = [11.0] * 18
        result = evaluate_paired_gate(arm, ctrl, arm_name="test_arm", metric="decode_tps", k=1)
        assert result.arm == "test_arm"
        assert result.metric == "decode_tps"
        assert isinstance(result.n_rounds, int)
        assert isinstance(result.half_width, float)
        assert isinstance(result.reason, str)


# ---------------------------------------------------------------------------
# decide_adoption (AD-1..AD-6 pure function)
# ---------------------------------------------------------------------------


def _gate(arm: str, verdict: str, metric: str = "decode_tps", point_ratio: float = 1.1, harmful: bool = False) -> PairedGateResult:
    """Build a PairedGateResult for decide_adoption tests."""
    lo = 1.05 if verdict == "adopt" else (0.85 if verdict == "reject" else 0.98)
    hi = 1.25 if verdict == "adopt" else (0.97 if verdict == "reject" else 1.1)
    if harmful:
        hi = 0.9 - DEFAULT_NONINFERIORITY_MARGIN - 0.01
        lo = 0.8
    return PairedGateResult(
        verdict=verdict,
        harmful=harmful,
        point_ratio=point_ratio,
        ratio_lo=lo,
        ratio_hi=hi,
        n_rounds=18,
        half_width=(hi - lo) / 2,
        reason="test",
        arm=arm,
        metric=metric,
    )


class TestDecideAdoption:
    """Tests for decide_adoption in stats.py (AD-1..AD-6)."""

    def test_ad1_harmful_arm_not_adopted(self):
        """AD-1: an arm with any harmful cell must not be adopted."""
        gates = [_gate("bad_arm", "adopt", harmful=True)]
        outcomes = decide_adoption(gates, device="GPU")
        assert outcomes["bad_arm"] == "harmful"

    def test_ad1_takes_priority_over_adopt_verdict(self):
        """AD-1: harmful overrides adopt verdict."""
        gates = [_gate("arm", "adopt", harmful=True, point_ratio=1.5)]
        outcomes = decide_adoption(gates, device="GPU")
        assert outcomes["arm"] == "harmful"

    def test_all_reject_gives_no_adopt_cell(self):
        """An arm with all-reject and no harmful cells should get no_adopt_cell."""
        gates = [_gate("arm", "reject")]
        outcomes = decide_adoption(gates, device="GPU")
        assert outcomes["arm"] == "no_adopt_cell"

    def test_inconclusive_gives_rescreen(self):
        """An arm with inconclusive and no adopt cells should get rescreen (AD-2)."""
        gates = [_gate("arm", "inconclusive")]
        outcomes = decide_adoption(gates, device="GPU")
        assert outcomes["arm"] == "rescreen"

    def test_adopt_arm_on_gpu_is_adopted(self):
        """A clean adopt arm on GPU should be adopted."""
        gates = [_gate("fast_arm", "adopt")]
        outcomes = decide_adoption(gates, device="GPU")
        assert outcomes["fast_arm"] == "adopted"

    def test_ad4_one_per_key_lower_ratio_loses(self):
        """AD-4: two arms sharing a key; only the higher-ratio one is adopted."""
        gates = [
            _gate("dq0", "adopt", point_ratio=1.15),
            _gate("dq64", "adopt", point_ratio=1.10),
        ]
        arm_keys = {"dq0": ["DYNAMIC_QUANTIZATION_GROUP_SIZE"], "dq64": ["DYNAMIC_QUANTIZATION_GROUP_SIZE"]}
        outcomes = decide_adoption(gates, arm_keys=arm_keys, device="GPU")
        assert outcomes["dq0"] == "adopted"
        assert outcomes["dq64"] == "lost_key"

    def test_ad5_one_per_device_cpu(self):
        """AD-5: only the highest-ratio arm is adopted on CPU; others get one_per_device."""
        gates = [
            _gate("pcore", "adopt", point_ratio=1.12),
            _gate("ht_off", "adopt", point_ratio=1.08),
        ]
        outcomes = decide_adoption(gates, device="CPU")
        assert outcomes["pcore"] == "adopted"
        assert outcomes["ht_off"] == "one_per_device"

    def test_ad5_one_per_device_npu(self):
        """AD-5: same one-per-device rule applies on NPU."""
        gates = [
            _gate("prefill_static", "adopt", point_ratio=1.20),
            _gate("gen_bestperf", "adopt", point_ratio=1.15),
        ]
        outcomes = decide_adoption(gates, device="NPU")
        assert outcomes["prefill_static"] == "adopted"
        assert outcomes["gen_bestperf"] == "one_per_device"

    def test_ad6_outcome_codes_exhaustive(self):
        """AD-6: every arm in a mixed batch ends with exactly one outcome code."""
        gates = [
            _gate("harmful_arm", "adopt", harmful=True),
            _gate("reject_arm", "reject"),
            _gate("incon_arm", "inconclusive"),
            _gate("adopt_arm", "adopt", point_ratio=1.1),
        ]
        outcomes = decide_adoption(gates, device="GPU")
        valid_codes = {"adopted", "harmful", "no_adopt_cell", "rescreen", "lost_key", "bracket_failed", "one_per_device"}
        for arm, code in outcomes.items():
            assert code in valid_codes, f"{arm} got unexpected code {code!r}"

    def test_control_arm_excluded_from_outcomes(self):
        """The control arm must not appear in outcomes."""
        gates = [
            _gate("control", "adopt"),
            _gate("my_arm", "adopt"),
        ]
        outcomes = decide_adoption(gates, device="GPU")
        assert "control" not in outcomes
        assert "my_arm" in outcomes

    def test_aa_arms_excluded_from_outcomes(self):
        """aa_* arms are A/A calibration and must not appear in adoption outcomes."""
        gates = [
            _gate("aa_control", "adopt"),
            _gate("my_arm", "adopt"),
        ]
        outcomes = decide_adoption(gates, device="GPU")
        assert "aa_control" not in outcomes
        assert "my_arm" in outcomes


# ---------------------------------------------------------------------------
# bench_arms --evaluate mode (end-to-end without hardware)
# ---------------------------------------------------------------------------


class TestBenchArmsEvaluateMode:
    """End-to-end tests for 'python scripts/bench_arms.py --evaluate FILE' without hardware."""

    def _write_result(self, path: Path, arms_data: Dict[str, List[float]]) -> None:
        """Write a synthetic schema-v1 result file with the given arms and decode_tps values."""
        warmup_rounds = 2
        timed_rounds = 18
        samples = []
        for arm, values in arms_data.items():
            for i, v in enumerate(values[:warmup_rounds]):
                samples.append(_make_sample(arm, i, decode_tps=v, is_warmup=True))
            for i, v in enumerate(values[warmup_rounds:warmup_rounds + timed_rounds]):
                samples.append(_make_sample(arm, warmup_rounds + i, decode_tps=v))
        doc = _make_result_doc(samples, _protocol(warmup=warmup_rounds, timed=timed_rounds))
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    def test_evaluate_single_file_exits_zero_when_arm_adopts(self):
        """--evaluate with a clearly faster arm should exit 0."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m10_a.v1.json"
            ctrl_values = [10.0] * 20
            arm_values = [14.0] * 20
            self._write_result(f, {"control": ctrl_values, "fast_arm": arm_values})
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "bench_arms.py"),
                 "--evaluate", str(f)],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr

    def test_evaluate_prints_arm_name_and_verdict(self):
        """--evaluate output must include the arm name and a verdict keyword."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m10_a.v1.json"
            self._write_result(f, {
                "control": [10.0] * 20,
                "fast_arm": [14.0] * 20,
            })
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "bench_arms.py"),
                 "--evaluate", str(f)],
                capture_output=True, text=True,
            )
            out = result.stdout + result.stderr
            assert "fast_arm" in out
            assert any(kw in out.lower() for kw in ("adopt", "reject", "inconclusive"))

    def test_evaluate_multiple_files_aggregates_k(self):
        """--evaluate over two files should aggregate K from both files."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            f1 = Path(tmp) / "m10_a.v1.json"
            f2 = Path(tmp) / "m10_b.v1.json"
            self._write_result(f1, {"control": [10.0] * 20, "arm_a": [14.0] * 20})
            self._write_result(f2, {"control": [10.0] * 20, "arm_b": [13.0] * 20})
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "bench_arms.py"),
                 "--evaluate", str(f1), "--evaluate", str(f2)],
                capture_output=True, text=True,
            )
            out = result.stdout + result.stderr
            assert "arm_a" in out
            assert "arm_b" in out

    def test_evaluate_exits_nonzero_when_no_result_files(self):
        """--evaluate with a non-existent file should exit non-zero."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "bench_arms.py"),
             "--evaluate", "/nonexistent/path.v1.json"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_evaluate_missing_control_arm_exits_nonzero(self):
        """--evaluate on a file with no control backend should fail clearly."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "bad.v1.json"
            self._write_result(f, {"not_control": [10.0] * 20, "arm": [12.0] * 20})
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "bench_arms.py"),
                 "--evaluate", str(f)],
                capture_output=True, text=True,
            )
            assert result.returncode != 0

    def test_evaluate_aa_arm_prints_aa_check(self):
        """--evaluate output must include an A/A interval check when aa_control is present."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m10_d.v1.json"
            doc = _make_result_doc(
                [_make_sample("control", i, decode_tps=10.0) for i in range(20)]
                + [_make_sample("aa_control", i, decode_tps=10.0) for i in range(20)],
                _protocol(),
            )
            f.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "bench_arms.py"),
                 "--evaluate", str(f)],
                capture_output=True, text=True,
            )
            out = result.stdout + result.stderr
            assert "aa_control" in out
