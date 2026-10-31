"""Tests for bench_native_micro.py WP-02 02c: --warmup and --seed flags.

Red-first: TestRepeatWarmup tests fail before the implementation changes because
repeat() does not yet accept a warmup kwarg (TypeError). The argparse tests fail
because --warmup and --seed are not registered (SystemExit(2)). The seed test
fails because default_rng(0) is hardcoded instead of using args.seed.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load():
    """Load or reload bench_native_micro, picking up source changes."""
    if "bench_native_micro" in sys.modules:
        return importlib.reload(sys.modules["bench_native_micro"])
    return importlib.import_module("bench_native_micro")


# ---------------------------------------------------------------------------
# TestRepeatWarmup — direct unit tests for the repeat() helper
# RED-FIRST: all three fail before fix with TypeError (unexpected kwarg 'warmup')
# ---------------------------------------------------------------------------

class TestRepeatWarmup:
    """repeat() must accept a warmup kwarg and discard those calls from results."""

    def test_warmup_discards_initial_calls_returns_timed_results(self):
        """repeat(fn, runs=2, warmup=3) must invoke fn 5 times, return last 2."""
        bnm = _load()
        seq: List[float] = []

        def probe() -> float:
            seq.append(float(len(seq) + 1))
            return seq[-1]

        result = bnm.repeat(probe, runs=2, warmup=3)

        assert len(seq) == 5, (
            f"Expected 5 total calls (3 warmup + 2 timed), got {len(seq)}"
        )
        assert len(result) == 2, f"Expected 2 timed results (runs=2), got {len(result)}"
        assert result == [4.0, 5.0], (
            f"Expected timed results [4.0, 5.0] (warmup=3 discarded), got {result}"
        )

    def test_zero_warmup_equals_original_behaviour(self):
        """repeat(fn, runs=3, warmup=0) must return all 3 calls in order."""
        bnm = _load()
        seq: List[float] = []

        def probe() -> float:
            seq.append(float(len(seq) + 1))
            return seq[-1]

        result = bnm.repeat(probe, runs=3, warmup=0)

        assert result == [1.0, 2.0, 3.0]
        assert len(seq) == 3

    def test_warmup_parameter_present_in_signature(self):
        """repeat() must have 'warmup' as an explicit parameter."""
        bnm = _load()
        import inspect
        sig = inspect.signature(bnm.repeat)
        assert "warmup" in sig.parameters, (
            f"repeat() must declare 'warmup' parameter. Current signature: {sig}"
        )


# ---------------------------------------------------------------------------
# TestArgparseFlagsRegistered — main()'s parser must register --warmup/--seed
# RED-FIRST: fails before fix because argparse raises SystemExit(2)
# ---------------------------------------------------------------------------

class _StopAfterParse(Exception):
    """Raised by the interceptor to stop main() immediately after parse_args."""

    def __init__(self, namespace: argparse.Namespace) -> None:
        self.namespace = namespace


def _intercept_parse_args(real_parse):
    """Wrap ArgumentParser.parse_args to stop main() after parsing succeeds."""

    def wrapper(self, args=None, namespace=None):
        ns = real_parse(self, args, namespace)
        raise _StopAfterParse(ns)

    return wrapper


class TestArgparseFlagsRegistered:
    """main()'s argparse must register --warmup and --seed."""

    def test_warmup_flag_registered_in_main_parser(self):
        """--warmup 5 --smoke must parse successfully inside main() (no SystemExit(2))."""
        bnm = _load()
        real_parse = argparse.ArgumentParser.parse_args

        with pytest.raises(_StopAfterParse) as exc_info, \
             pytest.MonkeyPatch().context() as mp:
            mp.setattr(argparse.ArgumentParser, "parse_args",
                       _intercept_parse_args(real_parse))
            bnm.main(["--warmup", "5", "--smoke"])

        ns = exc_info.value.namespace
        assert hasattr(ns, "warmup"), "--warmup must be registered in main()'s parser"
        assert ns.warmup == 5, f"Expected warmup=5, got {ns.warmup}"

    def test_seed_flag_registered_in_main_parser(self):
        """--seed 42 --smoke must parse successfully inside main() (no SystemExit(2))."""
        bnm = _load()
        real_parse = argparse.ArgumentParser.parse_args

        with pytest.raises(_StopAfterParse) as exc_info, \
             pytest.MonkeyPatch().context() as mp:
            mp.setattr(argparse.ArgumentParser, "parse_args",
                       _intercept_parse_args(real_parse))
            bnm.main(["--seed", "42", "--smoke"])

        ns = exc_info.value.namespace
        assert hasattr(ns, "seed"), "--seed must be registered in main()'s parser"
        assert ns.seed == 42, f"Expected seed=42, got {ns.seed}"

    def test_warmup_default_is_non_negative_integer(self):
        """--warmup default value must be a non-negative int."""
        bnm = _load()
        real_parse = argparse.ArgumentParser.parse_args

        with pytest.raises(_StopAfterParse) as exc_info, \
             pytest.MonkeyPatch().context() as mp:
            mp.setattr(argparse.ArgumentParser, "parse_args",
                       _intercept_parse_args(real_parse))
            bnm.main(["--smoke"])

        ns = exc_info.value.namespace
        assert hasattr(ns, "warmup"), "--warmup must have a default"
        assert isinstance(ns.warmup, int) and ns.warmup >= 0, (
            f"warmup default must be non-negative int, got {ns.warmup!r}"
        )

    def test_seed_default_is_zero(self):
        """--seed default value must be 0 (matches bench_regression.py convention)."""
        bnm = _load()
        real_parse = argparse.ArgumentParser.parse_args

        with pytest.raises(_StopAfterParse) as exc_info, \
             pytest.MonkeyPatch().context() as mp:
            mp.setattr(argparse.ArgumentParser, "parse_args",
                       _intercept_parse_args(real_parse))
            bnm.main(["--smoke"])

        ns = exc_info.value.namespace
        assert hasattr(ns, "seed"), "--seed must have a default"
        assert ns.seed == 0, f"seed default must be 0, got {ns.seed!r}"


# ---------------------------------------------------------------------------
# TestSeedForwardedToRng — args.seed must reach np.random.default_rng
# RED-FIRST: before fix, default_rng(0) is hardcoded; seed=7 is never seen
# ---------------------------------------------------------------------------

class TestSeedForwardedToRng:
    """main() must call np.random.default_rng(args.seed), not np.random.default_rng(0)."""

    def test_seed_arg_reaches_default_rng(self):
        """main(['--seed', '7', '--smoke']) must call np.random.default_rng with 7."""
        bnm = _load()
        import numpy as np
        from unittest.mock import patch, MagicMock

        seeds_seen: List[int] = []
        real_rng = np.random.default_rng

        def spy_rng(seed=None):
            seeds_seen.append(seed)
            return real_rng(seed)

        # Patch Path.write_bytes so blob write does not fail in a non-existent tmp dir;
        # the rng is still called because Python evaluates the argument first.
        with patch.object(bnm.np.random, "default_rng", side_effect=spy_rng), \
             patch.object(bnm, "capture_environment", return_value={}), \
             patch.object(bnm, "NativeMemoryEngine") as eng_cls, \
             patch.object(bnm, "FlashStreamWeightManager") as fsm_cls, \
             patch.object(bnm, "summarize", return_value={}), \
             patch.object(bnm, "repeat", return_value=[0.1, 0.1, 0.1]), \
             patch("tempfile.TemporaryDirectory") as mock_td, \
             patch.object(Path, "write_bytes", MagicMock()):
            mock_td.return_value.__enter__ = MagicMock(return_value="/fake/tmp")
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            eng_cls.return_value.is_open = False
            fsm_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            fsm_cls.return_value.__exit__ = MagicMock(return_value=False)

            try:
                bnm.main(["--seed", "7", "--smoke"])
            except SystemExit as exc:
                if exc.code == 2:
                    pytest.fail(
                        "--seed not registered; got SystemExit(2) from argparse before "
                        "np.random.default_rng could be checked"
                    )
            except Exception:
                pass  # Failures after blob creation are irrelevant to this assertion

        assert 7 in seeds_seen, (
            f"main() must call np.random.default_rng(7) when --seed 7 is passed. "
            f"Actual calls: {seeds_seen}. "
            f"Before fix: default_rng(0) is hardcoded; after fix: default_rng(args.seed) is used."
        )
