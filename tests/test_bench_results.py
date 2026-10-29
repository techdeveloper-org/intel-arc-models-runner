"""Slice S-E tests: schema-v1 result files, the validator script, and the docs renderer."""

import importlib.util
import json
from pathlib import Path
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
