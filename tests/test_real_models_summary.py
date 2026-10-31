"""The real-models live script prints only measured values or states them absent."""

import importlib.util
import io
from pathlib import Path
from typing import Any

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    """Import the live real-models script as a module."""
    spec = importlib.util.spec_from_file_location("live_real", REPO_ROOT / "scratch" / "run_live_real_models_benchmark_and_compare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(results: dict) -> str:
    """Render the summary table of a runner holding the given results into text."""
    module = _load_module()
    runner = object.__new__(module.LiveSequentialBenchmarkRunner)
    buffer = io.StringIO()
    runner.console = Console(file=buffer, width=200, force_terminal=False)
    runner.results_data = results
    runner.render_summary_table()
    return buffer.getvalue()


def test_render_summary_table_regular_model_row_states_ram_not_measured_without_hardcoded_usm_text() -> None:
    """No literal footprint is printed for a model whose memory was not measured."""
    results = {
        "evaluated_models": {"m": {"tiers": {"low": {
            "arc_runner": {"tokens_per_second": 10.0, "ttft_ms": 5.0, "source": "ov_perf_metrics"},
            "ollama": {"status": "PASS", "tokens_per_second": 4.0, "ttft_ms": 9.0},
        }}}},
        "flash_stream_heavy_models": {},
    }
    text = _render(results)
    assert "1.3 GB" not in text
    assert "not measured" in text


def test_render_summary_table_header_states_ram_not_measured_when_total_is_absent() -> None:
    """A missing total-RAM measurement is printed as absent, never as a constant."""
    module = _load_module()
    runner = object.__new__(module.LiveSequentialBenchmarkRunner)
    runner.results_data = {"testbed": {"ram_total_gb": None}}
    assert runner._ram_text() == "not measured"
    runner.results_data = {"testbed": {"ram_total_gb": 12.5}}
    assert runner._ram_text() == "12.50 GB"


def test_header_ram_is_not_a_hardcoded_literal() -> None:
    """The panel header never contains the old fixed 15.43 GB string."""
    source = (REPO_ROOT / "scratch" / "run_live_real_models_benchmark_and_compare.py").read_text(encoding="utf-8")
    assert "15.43" not in source and "1.3 GB" not in source
