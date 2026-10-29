"""Independent tests for kv_quality maths and harness_common helpers.

Specification: kv_quality.py module docstring ("A prompt passes if the outputs agree for at least
min_prefix_tokens tokens (or entirely, when the reference output is shorter); the gate passes if at least
required_pass_fraction of the prompts pass") and the harness_common docstrings. The gate threshold is
evaluated here with exact rational arithmetic.
"""

import json
import random
from fractions import Fraction

import pytest

from arc_runner.benchmark.harness_common import (
    GPU_CONTEXT_LOST_MARKER, collect_tiers, default_result_path, gpu_context_lost, select_tiers, target_device,
)
from arc_runner.benchmark.kv_quality import common_prefix_length, compare_kv_precision, write_report

SEED = 31337


def _ref_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def test_common_prefix_length_matches_reference_on_random_sequences():
    rng = random.Random(SEED)
    for _ in range(2000):
        a = [rng.randrange(4) for _ in range(rng.randrange(0, 12))]
        b = list(a[:rng.randrange(0, len(a) + 1)]) + [rng.randrange(4) for _ in range(rng.randrange(0, 6))]
        got = common_prefix_length(a, b)
        assert got == _ref_prefix(a, b)
        assert got == common_prefix_length(b, a)
        assert 0 <= got <= min(len(a), len(b))
        assert common_prefix_length(a, a) == len(a)


def test_common_prefix_length_edge_cases():
    assert common_prefix_length([], []) == 0
    assert common_prefix_length([1], []) == 0
    assert common_prefix_length((1, 2, 3), [1, 2, 3, 4]) == 3
    assert common_prefix_length([9], [8]) == 0


class _Engine:
    def __init__(self, table, log):
        self.table, self.log = table, log

    def generate(self, prompt, config):
        self.log.append((prompt, config))
        return prompt  # text is the prompt key; tokenize looks it up

    def tokenize(self, text):
        return list(self.table[text])


def _factory_for(ref_by_prompt, cand_by_prompt, log=None, precisions=None):
    log = log if log is not None else []

    def factory(precision):
        if precisions is not None:
            precisions.append(precision)
        return _Engine(ref_by_prompt if precision == "" else cand_by_prompt, log)

    return factory


def _expected_report(refs, cands, min_prefix, fraction):
    passes, exact = 0, 0
    per = []
    for r, c in zip(refs, cands):
        prefix = _ref_prefix(r, c)
        is_exact = r == c
        needed = min(min_prefix, len(r))
        ok = is_exact or prefix >= needed
        per.append((len(r), len(c), prefix, is_exact, ok))
        passes += ok
        exact += is_exact
    n = len(refs)
    gate = Fraction(passes, n) >= Fraction(str(fraction))
    return per, passes, exact, gate


def test_compare_kv_precision_random_cases_match_exact_reference_for_usual_fractions():
    rng = random.Random(SEED + 1)
    for case in range(300):
        n = rng.randrange(1, 11)
        min_prefix = rng.choice([1, 2, 4, 8, 32])
        fraction = rng.choice([0.5, 0.75, 0.8, 0.9, 0.95, 1.0])
        prompts = [f"p{case}-{i}" for i in range(n)]
        refs, cands = [], []
        for _ in prompts:
            r = [rng.randrange(3) for _ in range(rng.randrange(0, 40))]
            kind = rng.randrange(4)
            if kind == 0:
                c = list(r)
            elif kind == 1 and r:
                cut = rng.randrange(len(r))
                c = r[:cut] + [rng.randrange(3, 6)] + r[cut:]
            elif kind == 2:
                c = r[:rng.randrange(len(r) + 1)]
            else:
                c = [rng.randrange(3) for _ in range(rng.randrange(0, 40))]
            refs.append(r)
            cands.append(c)
        report = compare_kv_precision(
            _factory_for(dict(zip(prompts, refs)), dict(zip(prompts, cands))), prompts, 16,
            min_prefix_tokens=min_prefix, required_pass_fraction=fraction)
        per, passes, exact, gate = _expected_report(refs, cands, min_prefix, fraction)
        got = [(p.reference_tokens, p.candidate_tokens, p.prefix_tokens, p.exact_match, p.passes) for p in report.prompts]
        assert got == per
        assert report.pass_count == passes and report.exact_matches == exact
        assert report.passed is gate, (n, passes, fraction)
        assert [p.prompt for p in report.prompts] == prompts


@pytest.mark.parametrize("fraction", [0.07, 0.14, 0.28, 0.55, 0.57, 0.58])
def test_gate_threshold_is_exact_when_pass_fraction_equals_required_fraction(fraction):
    # choose n = 100 prompts and exactly fraction*100 passing: pass_count/n == required fraction, "at least" holds
    n = 100
    k = int(Fraction(str(fraction)) * n)
    assert Fraction(k, n) == Fraction(str(fraction))
    prompts = [f"q{i}" for i in range(n)]
    refs = {p: [1, 2, 3, 4] for p in prompts}
    cands = {p: ([1, 2, 3, 4] if i < k else [9, 9, 9, 9]) for i, p in enumerate(prompts)}
    report = compare_kv_precision(_factory_for(refs, cands), prompts, 8, min_prefix_tokens=4, required_pass_fraction=fraction)
    assert report.pass_count == k
    assert report.passed is True, f"{k}/{n} passing meets required_pass_fraction={fraction}"


def test_gate_fails_one_below_threshold():
    prompts = [f"q{i}" for i in range(10)]
    refs = {p: [1, 2, 3] for p in prompts}
    cands = {p: ([1, 2, 3] if i < 8 else [7]) for i, p in enumerate(prompts)}
    report = compare_kv_precision(_factory_for(refs, cands), prompts, 8, min_prefix_tokens=3, required_pass_fraction=0.9)
    assert report.pass_count == 8 and report.passed is False
    cands2 = {p: ([1, 2, 3] if i < 9 else [7]) for i, p in enumerate(prompts)}
    assert compare_kv_precision(_factory_for(refs, cands2), prompts, 8, min_prefix_tokens=3, required_pass_fraction=0.9).passed is True


def test_prompt_with_short_reference_passes_only_on_full_agreement_of_that_reference():
    refs = {"a": [1, 2], "b": [1, 2]}
    cands = {"a": [1, 2, 5, 6], "b": [1, 3]}
    report = compare_kv_precision(_factory_for(refs, cands), ["a", "b"], 8, min_prefix_tokens=32, required_pass_fraction=1.0)
    assert [p.passes for p in report.prompts] == [True, False]
    assert [p.exact_match for p in report.prompts] == [False, False]
    assert report.passed is False


def test_candidate_shorter_than_required_prefix_fails_even_if_it_is_a_prefix_of_reference():
    refs = {"a": list(range(40))}
    cands = {"a": list(range(10))}
    report = compare_kv_precision(_factory_for(refs, cands), ["a"], 8, min_prefix_tokens=32, required_pass_fraction=1.0)
    assert report.prompts[0].prefix_tokens == 10 and report.prompts[0].passes is False


def test_engine_factory_called_with_default_then_candidate_and_greedy_config():
    log, precisions = [], []
    refs = {"a": [1]}
    compare_kv_precision(_factory_for(refs, refs, log, precisions), ["a"], 17, candidate_precision="f16")
    assert precisions == ["", "f16"]
    assert len(log) == 2
    for _, config in log:
        assert config.temperature == 0.0 and config.max_new_tokens == 17


def test_empty_prompt_list_and_untokenizable_output_raise():
    with pytest.raises(ValueError):
        compare_kv_precision(lambda p: None, [], 8)

    class _NoTok:
        def generate(self, prompt, config):
            return "x"

        def tokenize(self, text):
            return None

    with pytest.raises(RuntimeError):
        compare_kv_precision(lambda p: _NoTok(), ["a"], 8)


def test_write_report_round_trips_fields(tmp_path):
    refs = {"a": [1, 2, 3], "b": [4]}
    cands = {"a": [1, 2, 3], "b": [5]}
    report = compare_kv_precision(_factory_for(refs, cands), ["a", "b"], 8, min_prefix_tokens=2, required_pass_fraction=0.5)
    out = tmp_path / "sub" / "kv.json"
    write_report(out, report, "model-x", "GPU")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["model"] == "model-x" and doc["device"] == "GPU"
    assert doc["passed"] is report.passed and doc["pass_count"] == report.pass_count
    assert doc["exact_matches"] == 1
    assert [p["prefix_tokens"] for p in doc["prompts"]] == [3, 0]
    assert "KL" in doc["note"] or "perplexity" in doc["note"]


# ------------------------------------------------------------------ harness_common --------------------

class _Topo:
    def __init__(self, arc, npu):
        self.has_intel_arc, self.has_intel_npu = arc, npu


@pytest.mark.parametrize("arc,npu,preferred,expected", [
    (True, True, "GPU", ("GPU", False)), (True, True, "gpu", ("GPU", False)), (False, True, "GPU", ("CPU", True)),
    (True, False, "NPU", ("CPU", True)), (False, True, "NPU", ("NPU", False)), (False, False, "CPU", ("CPU", False)),
    (False, False, "cpu", ("CPU", False)), (True, True, "TPU", ("CPU", True)), (True, True, "", ("CPU", True)),
])
def test_target_device_choice_and_fallback_flag(arc, npu, preferred, expected):
    assert target_device(_Topo(arc, npu), preferred) == expected


def test_select_tiers_keeps_spec_order_and_validates_names():
    spec = {"name": "m", "workloads": {"a": 1, "b": 2, "c": 3}}
    assert list(select_tiers(spec, "c,a")["workloads"]) == ["a", "c"]  # spec order, not request order
    assert select_tiers(spec, " b , ")["workloads"] == {"b": 2}
    assert select_tiers(spec, "a")["name"] == "m"
    assert select_tiers(spec, None)["workloads"] == spec["workloads"]
    assert select_tiers(spec, "")["workloads"] == spec["workloads"]
    with pytest.raises(ValueError) as err:
        select_tiers(spec, "a,zzz")
    assert "zzz" in str(err.value)
    assert spec["workloads"] == {"a": 1, "b": 2, "c": 3}, "the input spec must not be modified"


def test_select_tiers_returns_a_copy_as_documented():
    spec = {"workloads": {"a": 1}}
    out = select_tiers(spec, "a")
    out["workloads"]["z"] = 9
    assert "z" not in spec["workloads"]
    out2 = select_tiers(spec, None)
    out2["extra"] = 1
    assert "extra" not in spec, "docstring: 'Return a copy of a model spec'"


def test_collect_tiers_records_failure_and_continues():
    calls = []

    def make(workload, model_id):
        calls.append(model_id)
        if workload == "bad":
            raise RuntimeError("x" * 900)
        return [model_id]

    failures = []
    samples, lost = collect_tiers({"t1": "ok", "t2": "bad", "t3": "ok"}, "alias", "gpu", make, failures)
    assert samples == ["alias/t1", "alias/t3"] and lost is False
    assert calls == ["alias/t1", "alias/t2", "alias/t3"]
    assert len(failures) == 1
    assert failures[0]["model_id"] == "alias/t2" and failures[0]["backend"] == "gpu"
    assert failures[0]["error"] == "x" * 500


def test_collect_tiers_stops_on_gpu_out_of_resources_and_keeps_earlier_samples():
    def make(workload, model_id):
        if workload == "oor":
            raise RuntimeError(f"clEnqueue failed: {GPU_CONTEXT_LOST_MARKER}")
        return [model_id]

    failures = []
    samples, lost = collect_tiers({"a": 1, "b": "oor", "c": 3}, "m", "gpu", make, failures)
    assert samples == ["m/a"] and lost is True
    assert [f["model_id"] for f in failures] == ["m/b"]
    assert gpu_context_lost(failures) is True


def test_collect_tiers_marker_beyond_500_chars_is_still_detected():
    def make(workload, model_id):
        raise RuntimeError("y" * 700 + GPU_CONTEXT_LOST_MARKER)

    failures = []
    _, lost = collect_tiers({"a": 1, "b": 2}, "m", "gpu", make, failures)
    assert lost is True and len(failures) == 1
    # fixed after R06: the marker is prefixed when the 500-character cut would remove it
    assert gpu_context_lost(failures) is True


def test_gpu_context_lost_matches_substring_only():
    assert gpu_context_lost([]) is False
    assert gpu_context_lost([{"error": "timeout"}]) is False
    assert gpu_context_lost([{"error": "timeout"}, {"error": "a CL_OUT_OF_RESOURCES b"}]) is True
    assert gpu_context_lost([{"model_id": "m"}]) is False


def test_default_result_path_format():
    from datetime import datetime, timezone
    from pathlib import Path

    when = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    got = default_result_path(Path("/repo"), "llm", "qwen", when)
    assert got == Path("/repo/docs/benchmarks/results/llm_qwen_20260304T050607Z.v1.json")
