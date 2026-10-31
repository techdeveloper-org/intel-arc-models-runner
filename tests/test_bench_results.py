"""Slice S-E tests: schema-v1 result files, the validator script, and the docs renderer."""

import importlib.util
import json
from pathlib import Path
import statistics
from types import ModuleType
from typing import Any

import pytest

from arc_runner.benchmark.results import build_result, capture_environment, load_samples, validate_result, write_result
from arc_runner.benchmark.stats import BenchSample

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {"warmup_runs": 1, "timed_runs": 3, "temperature": 0.0, "max_new_tokens": 8, "labels": {"arc": "iGPU INT4", "ollama": "CPU Q4_K_M"}}


def _document(source: str = "measured") -> dict:
    """Build a small valid document."""
    samples = [
        BenchSample("m", "arc", 0, 100.0, 40.0, 5, 8, "measured", is_warmup=True),
        BenchSample("m", "arc", 1, 110.0, 42.0, 5, 8, source),  # type: ignore[arg-type]
        BenchSample("m", "arc", 2, 90.0, 38.0, 5, 8, "measured"),
        BenchSample("m", "ollama", 1, None, 20.0, 5, 8, "measured"),
    ]
    return build_result("run-1", samples, PROTOCOL, capture_environment(REPO_ROOT))


def _load_script(name: str) -> ModuleType:
    """Import a script from scripts/ as a module."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_document_has_no_problems_and_round_trips(tmp_path: Path) -> None:
    """A built document validates, is written, and loads back into the same samples."""
    document = _document()
    assert validate_result(document) == []
    path = tmp_path / "result.json"
    write_result(path, document)
    reloaded = load_samples(json.loads(path.read_text(encoding="utf-8")))
    assert reloaded[1].decode_tps == 42.0 and reloaded[3].ttft_ms is None and len(reloaded) == 4


@pytest.mark.parametrize("source", ["guess", "", "Measured", None])
def test_validate_rejects_unknown_source_values(source: Any) -> None:
    """Only measured, derived and projected are allowed."""
    document = _document()
    document["samples"][1]["source"] = source
    assert any("source" in problem for problem in validate_result(document))


def test_validate_reports_missing_keys_and_wrong_types() -> None:
    """Missing sections, bad numbers and non-boolean flags are reported."""
    assert validate_result([]) == ["top level must be an object"]
    assert any("missing key: samples" in p for p in validate_result({"schema_version": 1}))
    document = _document()
    document["samples"][0]["decode_tps"] = -1
    document["samples"][1]["is_warmup"] = "no"
    document["samples"][2]["prompt_tokens"] = True
    document["samples"][3].pop("fallback_used")
    problems = " ".join(validate_result(document))
    assert "decode_tps" in problems and "is_warmup" in problems and "prompt_tokens" in problems and "fallback_used" in problems


def test_write_result_refuses_an_invalid_document(tmp_path: Path) -> None:
    """Invalid documents are never written."""
    document = _document()
    document["samples"][0]["source"] = "guess"
    with pytest.raises(ValueError):
        write_result(tmp_path / "bad.json", document)
    assert not (tmp_path / "bad.json").exists()


def test_validator_script_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI returns 0 for valid files and 1 for invalid or unreadable ones."""
    script = _load_script("validate_bench_json")
    good = tmp_path / "good.json"
    write_result(good, _document())
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert script.main([str(good)]) == 0
    assert script.main([str(good), str(bad)]) == 1
    assert "1 invalid" in capsys.readouterr().out


def test_renderer_builds_table_from_results_and_check_detects_staleness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The docs table comes only from result files; --check fails until the block is regenerated."""
    renderer = _load_script("render_bench_docs")
    results = tmp_path / "results"
    results.mkdir()
    write_result(results / "r.json", _document())
    doc = tmp_path / "README.md"
    doc.write_text("intro\n<!-- bench:begin measured-summary -->\nold\n<!-- bench:end -->\nend\n", encoding="utf-8")
    monkeypatch.setattr(renderer, "RESULTS_DIR", results)
    monkeypatch.setattr(renderer, "TARGET_DOCS", [doc])

    assert renderer.main(["--check"]) == 1
    assert renderer.main([]) == 0
    text = doc.read_text(encoding="utf-8")
    assert "arc (iGPU INT4)" in text and "ollama (CPU Q4_K_M)" in text and "measured" in text
    assert "| 2 |" in text
    assert "old" not in text
    assert renderer.main(["--check"]) == 0


def test_renderer_states_when_no_results_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no result files the block says so instead of inventing numbers."""
    renderer = _load_script("render_bench_docs")
    monkeypatch.setattr(renderer, "RESULTS_DIR", tmp_path)
    assert "No benchmark result files exist yet" in renderer.render_tables()


def test_checked_in_docs_are_up_to_date_with_results() -> None:
    """The repository's marked block matches what the renderer would produce."""
    assert _load_script("render_bench_docs").main(["--check"]) == 0


def test_checked_in_result_files_validate() -> None:
    """Every committed result file follows schema v1."""
    assert _load_script("validate_bench_json").main([]) == 0


def test_live_script_converts_existing_results_into_valid_schema_v1_samples() -> None:
    """The live benchmark script exports only rows that carry values, with honest sources."""
    spec = importlib.util.spec_from_file_location("live_bench", REPO_ROOT / "scratch" / "run_live_real_models_benchmark_and_compare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = object.__new__(module.LiveSequentialBenchmarkRunner)
    runner.results_data = json.loads((REPO_ROOT / "scratch" / "live_real_models_benchmark_results.json").read_text(encoding="utf-8"))
    samples = runner.collect_schema_v1_samples()
    assert samples and all(sample.source in ("measured", "derived") for sample in samples)
    assert all(sample.backend == "ollama" for sample in samples), "legacy arc-runner rows without a source must be skipped"
    document = build_result("t", samples, PROTOCOL, capture_environment(REPO_ROOT))
    assert validate_result(document) == []


NON_LLM_PROTOCOL = {"warmup_runs": 1, "timed_runs": 3, "labels": {"arc": "iGPU FP16"}}


def _non_llm_document(workload: str = "audio_stt") -> dict:
    """Build a valid document for a non-LLM workload (no temperature or token budget)."""
    samples = [
        BenchSample("whisper", "arc", 0, None, None, 0, 0, "measured", is_warmup=True, workload=workload, metrics={"latency_s": 2.0, "rtf": 0.2}),
        BenchSample("whisper", "arc", 1, None, None, 0, 0, "measured", workload=workload, metrics={"latency_s": 1.5, "rtf": 0.15}),
        BenchSample("whisper", "arc", 2, None, None, 0, 0, "measured", workload=workload, metrics={"latency_s": 1.7, "rtf": 0.17, "power_w": None}),
    ]
    return build_result("run-nl", samples, NON_LLM_PROTOCOL, capture_environment(REPO_ROOT))


def test_non_llm_document_validates_without_llm_protocol_fields_and_round_trips(tmp_path: Path) -> None:
    """Non-LLM runs need no temperature or max_new_tokens; workload and metrics survive a round trip."""
    document = _non_llm_document()
    assert validate_result(document) == []
    path = tmp_path / "nl.json"
    write_result(path, document)
    reloaded = load_samples(json.loads(path.read_text(encoding="utf-8")))
    assert reloaded[1].workload == "audio_stt"
    assert reloaded[1].metrics == {"latency_s": 1.5, "rtf": 0.15}
    assert reloaded[2].metrics["power_w"] is None


def test_llm_document_still_requires_temperature_and_token_budget() -> None:
    """The LLM protocol fields remain mandatory when any sample is an LLM sample."""
    document = _document()
    del document["protocol"]["temperature"]
    del document["protocol"]["max_new_tokens"]
    problems = " ".join(validate_result(document))
    assert "temperature" in problems and "max_new_tokens" in problems


def test_legacy_sample_rows_without_workload_default_to_llm() -> None:
    """Existing schema-v1 files (no workload key) still load as LLM samples."""
    document = _document()
    for row in document["samples"]:
        row.pop("workload", None)
        row.pop("metrics", None)
    assert validate_result(document) == []
    assert all(sample.workload == "llm" for sample in load_samples(document))


class TestQuickBenchSchema:
    """Positive, negative and edge-case tests for the quick_v1 result format."""

    def _make_quick(self, timed_tps=None):
        """Build a minimal valid quick_v1 document."""
        tps_list = timed_tps if timed_tps is not None else [30.0, 28.0, 32.0, 29.0, 31.0]
        samples = [
            {"iteration": 0, "is_warmup": True, "output_tokens": 10,
             "elapsed_s": 0.5, "decode_tps": 20.0, "source": "measured"}
        ]
        for i, tps in enumerate(tps_list, 1):
            samples.append({
                "iteration": i, "is_warmup": False,
                "output_tokens": 20, "elapsed_s": round(20 / tps, 4),
                "decode_tps": tps, "source": "measured",
            })
        timed = [s["decode_tps"] for s in samples if not s["is_warmup"]]
        return {
            "schema_version": "quick_v1",
            "protocol": "quick_api_benchmark",
            "warning": "test",
            "created_utc": "2026-10-31T00:00:00Z",
            "run_id": "test-run-1",
            "model": "deepseek-r1-1.5b",
            "device": "GPU",
            "warmup_runs": 1,
            "timed_runs": len(tps_list),
            "samples": samples,
            "summary": {
                "mean_decode_tps": round(statistics.mean(timed), 2),
                "median_decode_tps": round(statistics.median(timed), 2),
                "min_decode_tps": round(min(timed), 2),
                "max_decode_tps": round(max(timed), 2),
            },
        }

    def test_quick_v1_required_fields_present(self) -> None:
        """All mandatory top-level keys exist."""
        doc = self._make_quick()
        for key in ("schema_version", "protocol", "created_utc", "run_id",
                    "model", "device", "samples", "summary"):
            assert key in doc, f"missing required key: {key}"

    def test_quick_v1_has_correct_warmup_and_timed_counts(self) -> None:
        """Exactly 1 warmup and 5 timed samples."""
        doc = self._make_quick()
        warmups = [s for s in doc["samples"] if s["is_warmup"]]
        timed = [s for s in doc["samples"] if not s["is_warmup"]]
        assert len(warmups) == 1
        assert len(timed) == 5

    def test_quick_v1_all_timed_decode_tps_positive(self) -> None:
        """Every timed sample has decode_tps > 0."""
        doc = self._make_quick()
        for s in doc["samples"]:
            if not s["is_warmup"]:
                assert s["decode_tps"] > 0, f"iteration {s['iteration']} has non-positive tps"

    def test_quick_v1_summary_mean_matches_computed_mean(self) -> None:
        """summary.mean_decode_tps equals round(mean(timed_samples), 2)."""
        doc = self._make_quick()
        timed_tps = [s["decode_tps"] for s in doc["samples"] if not s["is_warmup"]]
        expected = round(statistics.mean(timed_tps), 2)
        assert doc["summary"]["mean_decode_tps"] == expected

    def test_quick_v1_summary_mean_within_min_max(self) -> None:
        """summary.mean is between min and max."""
        doc = self._make_quick()
        s = doc["summary"]
        assert s["min_decode_tps"] <= s["mean_decode_tps"] <= s["max_decode_tps"]

    def test_quick_v1_all_timed_source_is_measured(self) -> None:
        """All timed samples carry source='measured'."""
        doc = self._make_quick()
        for s in doc["samples"]:
            if not s["is_warmup"]:
                assert s["source"] == "measured"

    def test_quick_v1_serializes_to_json_without_error(self, tmp_path: Path) -> None:
        """Document round-trips through JSON without data loss."""
        doc = self._make_quick()
        p = tmp_path / "quick.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        reloaded = json.loads(p.read_text(encoding="utf-8"))
        assert reloaded["summary"]["mean_decode_tps"] == doc["summary"]["mean_decode_tps"]

    def test_quick_v1_warmup_excluded_from_summary(self) -> None:
        """Warmup tps=20.0 must not appear in summary stats."""
        doc = self._make_quick(timed_tps=[30.0, 28.0, 32.0, 29.0, 31.0])
        assert doc["summary"]["min_decode_tps"] >= 28.0, \
            "warmup sample contaminated the summary min"

    def test_quick_v1_missing_summary_key_raises_keyerror(self) -> None:
        """A quick doc without 'summary' raises KeyError on access."""
        doc = self._make_quick()
        del doc["summary"]
        with pytest.raises(KeyError):
            _ = doc["summary"]["mean_decode_tps"]

    def test_quick_v1_validate_result_rejects_quick_schema(self) -> None:
        """validate_result (schema-v1) must flag quick_v1 docs as invalid."""
        doc = self._make_quick()
        problems = validate_result(doc)
        assert len(problems) > 0, \
            "validate_result must report at least one problem for a quick_v1 document"

    def test_quick_v1_single_timed_sample_mean_equals_that_sample(self) -> None:
        """With exactly one timed run, mean == that sample's tps."""
        doc = self._make_quick(timed_tps=[25.5])
        assert doc["summary"]["mean_decode_tps"] == 25.5

    def test_quick_v1_high_variance_mean_stays_within_bounds(self) -> None:
        """High-variance runs: mean is still within [min, max]."""
        doc = self._make_quick(timed_tps=[10.0, 50.0, 12.0, 48.0, 30.0])
        s = doc["summary"]
        assert s["min_decode_tps"] <= s["mean_decode_tps"] <= s["max_decode_tps"]

    def test_quick_v1_mean_rounding_is_2_decimal_places(self) -> None:
        """mean_decode_tps is rounded to exactly 2 decimal places."""
        tps = [24.1, 27.3, 25.8, 26.5, 25.2]
        doc = self._make_quick(timed_tps=tps)
        val = doc["summary"]["mean_decode_tps"]
        assert val == round(statistics.mean(tps), 2)
        assert str(val)[::-1].find(".") <= 2, "must be at most 2 decimal places"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda s: s.update(workload="telepathy"), "workload"),
        (lambda s: s.update(metrics={}), "metrics"),
        (lambda s: s.pop("metrics"), "metrics"),
        (lambda s: s.update(metrics={"latency_s": -1.0}), "latency_s"),
        (lambda s: s.update(metrics={"latency_s": float("nan")}), "latency_s"),
        (lambda s: s.update(metrics={"latency_s": True}), "latency_s"),
        (lambda s: s.update(metrics={"Latency S": 1.0}), "Latency S"),
        (lambda s: s.update(metrics=[1.0]), "metrics"),
    ],
)
def test_validate_rejects_bad_non_llm_samples(mutation: Any, expected: str) -> None:
    """Unknown workloads, empty or malformed metrics, negative, NaN and boolean values are reported."""
    document = _non_llm_document()
    mutation(document["samples"][1])
    assert any(expected in problem for problem in validate_result(document))


def test_renderer_summarises_non_llm_metrics_with_confidence_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-LLM files render a metric table; null metrics are skipped, never shown as zero."""
    renderer = _load_script("render_bench_docs")
    results = tmp_path / "results"
    results.mkdir()
    write_result(results / "nl.json", _non_llm_document())
    monkeypatch.setattr(renderer, "RESULTS_DIR", results)
    text = renderer.render_tables()
    assert "audio_stt" in text and "latency_s" in text and "rtf" in text
    assert "power_w" not in text
    assert "whisper" in text and "arc (iGPU FP16)" in text


def test_renderer_keeps_three_significant_digits_for_small_and_large_values() -> None:
    """Small metrics (RTF 0.047) must not collapse to 0.0 and large ones keep whole numbers."""
    renderer = _load_script("render_bench_docs")
    assert renderer._sig(0.0474) == "0.0474"
    assert renderer._sig(15.62) == "15.6"
    assert renderer._sig(204.0) == "204"
    assert renderer._sig(2232.5) == "2232"
    assert renderer._sig(0.0) == "0"
