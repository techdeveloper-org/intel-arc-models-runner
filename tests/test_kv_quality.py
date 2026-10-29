"""u8 KV-cache quality gate: greedy generations at two precisions must agree for a minimum prefix."""

import json
import zlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from arc_runner.benchmark.kv_quality import common_prefix_length, compare_kv_precision, write_report


class _Engine:
    """Greedy engine: output is a fixed token list rendered as text; tokenize splits on spaces."""

    def __init__(self, outputs: Dict[str, str]) -> None:
        self.outputs = outputs
        self.calls: List[Any] = []

    def generate(self, prompt: str, config: Any = None) -> str:
        self.calls.append(config)
        return self.outputs[prompt]

    def tokenize(self, text: str) -> List[int]:
        return [zlib.crc32(piece.encode()) for piece in text.split()]


def _factory(by_precision: Dict[str, Dict[str, str]]) -> Any:
    def make(precision: str) -> _Engine:
        return _Engine(by_precision[precision])

    return make


def test_common_prefix_length_counts_matching_leading_elements() -> None:
    """Prefix length stops at the first difference and handles empty and unequal lengths."""
    assert common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([5], [6]) == 0


def test_identical_outputs_pass_the_gate_with_full_prefix() -> None:
    """Greedy outputs identical at both precisions: exact match, gate passes."""
    text = " ".join(f"w{i}" for i in range(40))
    report = compare_kv_precision(_factory({"": {"p": text}, "u8": {"p": text}}), ["p"], max_new_tokens=40, min_prefix_tokens=32)
    assert report.passed and report.exact_matches == 1
    assert report.prompts[0].prefix_tokens == 40 and report.prompts[0].exact_match


def test_early_divergence_fails_the_gate() -> None:
    """Outputs that diverge before min_prefix_tokens fail the prompt and the gate."""
    base = " ".join(f"w{i}" for i in range(40))
    drift = " ".join([f"w{i}" for i in range(5)] + ["X"] + [f"w{i}" for i in range(6, 40)])
    report = compare_kv_precision(_factory({"": {"p": base}, "u8": {"p": drift}}), ["p"], max_new_tokens=40, min_prefix_tokens=32)
    assert not report.passed
    assert report.prompts[0].prefix_tokens == 5 and not report.prompts[0].passes


def test_short_reference_output_must_match_entirely() -> None:
    """If the reference output is shorter than min_prefix_tokens the whole output must match."""
    report = compare_kv_precision(_factory({"": {"p": "a b c"}, "u8": {"p": "a b d"}}), ["p"], max_new_tokens=8, min_prefix_tokens=32)
    assert not report.passed
    ok = compare_kv_precision(_factory({"": {"p": "a b c"}, "u8": {"p": "a b c"}}), ["p"], max_new_tokens=8, min_prefix_tokens=32)
    assert ok.passed


def test_gate_requires_the_stated_fraction_of_prompts_to_pass() -> None:
    """With required_pass_fraction 0.5, one of two failing prompts still passes; one of three does not."""
    good = " ".join(f"w{i}" for i in range(40))
    bad = "z " + good
    outputs_ref = {"a": good, "b": good, "c": good}
    outputs_u8 = {"a": good, "b": bad, "c": bad}
    report = compare_kv_precision(_factory({"": outputs_ref, "u8": outputs_u8}), ["a", "b", "c"], 40, min_prefix_tokens=32, required_pass_fraction=0.5)
    assert report.pass_count == 1 and not report.passed
    report2 = compare_kv_precision(_factory({"": outputs_ref, "u8": {"a": good, "b": good, "c": bad}}), ["a", "b", "c"], 40, min_prefix_tokens=32, required_pass_fraction=0.5)
    assert report2.passed


def test_generation_is_greedy_and_uses_the_same_budget_for_both_precisions() -> None:
    """Both engines get temperature 0 and the same max_new_tokens."""
    engines: List[_Engine] = []

    def make(precision: str) -> _Engine:
        engine = _Engine({"p": "a b c"})
        engines.append(engine)
        return engine

    compare_kv_precision(make, ["p"], max_new_tokens=17)
    assert len(engines) == 2
    assert all(e.calls[0].temperature == 0.0 and e.calls[0].max_new_tokens == 17 for e in engines)


def test_missing_tokenizer_is_an_error_not_a_pass() -> None:
    """If tokenize returns None the comparison cannot be made and raises."""

    class _NoTok(_Engine):
        def tokenize(self, text: str) -> Any:
            return None

    with pytest.raises(RuntimeError):
        compare_kv_precision(lambda precision: _NoTok({"p": "a"}), ["p"], 4)


def test_empty_prompt_list_is_rejected() -> None:
    """A gate over zero prompts would pass vacuously."""
    with pytest.raises(ValueError):
        compare_kv_precision(_factory({"": {}, "u8": {}}), [], 4)


def test_write_report_records_thresholds_and_verdict(tmp_path: Path) -> None:
    """The report file holds the parameters, per-prompt prefixes and the verdict; it makes no default change."""
    text = " ".join(f"w{i}" for i in range(40))
    report = compare_kv_precision(_factory({"": {"p": text}, "u8": {"p": text}}), ["p"], 40)
    target = tmp_path / "kv.json"
    write_report(target, report, model="m", device="GPU")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["passed"] is True and data["min_prefix_tokens"] == 32 and data["required_pass_fraction"] == 0.9
    assert data["model"] == "m" and data["prompts"][0]["prefix_tokens"] == 40
    assert "default" in data["note"]


def _script() -> Any:
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bench_kv_quality", root / "scripts" / "bench_kv_quality.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_factory_passes_kv_precision_only_for_the_candidate(tmp_path: Path) -> None:
    """The default engine is loaded without the property; the candidate with KV_CACHE_PRECISION=u8."""
    loads: List[Dict[str, Any]] = []

    class _Loader:
        def load_model(self, path: Any, device: str = "GPU", **kwargs: Any) -> None:
            loads.append({"device": device, **kwargs})

    factory = _script().build_factory(_Loader, tmp_path, "GPU")
    factory("")
    factory("u8")
    assert loads == [{"device": "GPU"}, {"device": "GPU", "KV_CACHE_PRECISION": "u8"}]


def test_script_default_prompt_set_is_fixed_and_non_trivial() -> None:
    """The registered prompt set has at least eight distinct prompts."""
    prompts = _script().DEFAULT_PROMPTS
    assert len(prompts) >= 8 and len(set(prompts)) == len(prompts)


def test_script_missing_model_dir_exits_2_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """No model directory: exit code 2, no report."""
    out = tmp_path / "r.json"
    assert _script().main(["--model-dir", str(tmp_path / "nope"), "--out", str(out)]) == 2
    assert not out.exists()
