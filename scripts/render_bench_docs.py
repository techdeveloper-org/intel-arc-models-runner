"""Render benchmark tables in the docs from schema-v1 result files.

Tables live between marker comments:
    <!-- bench:begin measured-summary -->  ...  <!-- bench:end -->
Numbers come only from docs/benchmarks/results/*.json, never from hand edits.

Usage: python scripts/render_bench_docs.py [--check]
  --check   exit 1 if a marked block is out of date instead of rewriting it
"""

import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.results import load_samples  # noqa: E402
from arc_runner.benchmark.stats import BenchSample, median, student_t_ci95  # noqa: E402

RESULTS_DIR = REPO_ROOT / "docs" / "benchmarks" / "results"
TARGET_DOCS = [REPO_ROOT / "docs" / "benchmarks" / "README.md"]
BLOCK = re.compile(r"(<!-- bench:begin measured-summary -->\n)(.*?)(<!-- bench:end -->)", re.S)


def _sig(value: float) -> str:
    """Format a number with three significant digits (whole numbers from 1000 up).

    Args:
        value: Number to format.

    Returns:
        str: Text such as 0.0474, 15.6, 204 or 2230.
    """
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def _format(values: List[float]) -> str:
    """Format a metric as median with a 95% Student-t half-width.

    Args:
        values: Timed values.
    """
    if not values:
        return "n/a"
    _, half = student_t_ci95(values)
    return f"{_sig(median(values))} +/- {_sig(half)}"


def render_tables() -> str:
    """Build the markdown summary from every result file.

    Returns:
        str: LLM table and, where non-LLM results exist, a metric table; or a sentence stating that
        no measured results exist yet.
    """
    llm_rows: List[Tuple[str, str, str, int, str, str, str]] = []
    metric_rows: List[Tuple[str, str, str, str, int, str, str, str]] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        grouped: Dict[Tuple[str, str, str], List[BenchSample]] = {}
        for sample in load_samples(document):
            if not sample.is_warmup:
                grouped.setdefault((sample.workload, sample.model_id, sample.backend), []).append(sample)
        labels = document["protocol"]["labels"]
        for (workload, model, backend), samples in sorted(grouped.items()):
            sources = ",".join(sorted({s.source for s in samples}))
            label = f"{backend} ({labels.get(backend, 'unlabelled')})"
            if workload == "llm":
                llm_rows.append((
                    model, label, path.name, len(samples),
                    _format([s.ttft_ms for s in samples if s.ttft_ms is not None]),
                    _format([s.decode_tps for s in samples if s.decode_tps is not None]), sources,
                ))
                continue
            names = sorted({key for s in samples for key in (s.metrics or {})})
            for name in names:
                values = [s.metrics[name] for s in samples if s.metrics and s.metrics.get(name) is not None]
                if values:
                    metric_rows.append((workload, model, label, path.name, len(values), name, _format(values), sources))
    if not llm_rows and not metric_rows:
        return "No benchmark result files exist yet. Run the benchmark on real hardware; this table is generated from `docs/benchmarks/results/*.json`.\n"
    lines: List[str] = []
    if llm_rows:
        lines += ["| Model | Backend (labels) | Result file | n | TTFT ms (median +/- 95% CI) | Decode tok/s (median +/- 95% CI) | Source |", "|:--|:--|:--|--:|--:|--:|:--|"]
        lines += [f"| {m} | {b} | {f} | {n} | {t} | {d} | {src} |" for m, b, f, n, t, d, src in llm_rows]
    if metric_rows:
        lines += [""] if lines else []
        lines += ["| Workload | Model | Backend (labels) | Result file | n | Metric | Value (median +/- 95% CI) | Source |", "|:--|:--|:--|:--|--:|:--|--:|:--|"]
        lines += [f"| {w} | {m} | {b} | {f} | {n} | {name} | {v} | {src} |" for w, m, b, f, n, name, v, src in metric_rows]
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    """Rewrite or check the marked blocks.

    Args:
        argv: Command line arguments without the program name.

    Returns:
        int: 0 when up to date (or rewritten), 1 when --check finds a stale block.
    """
    check = "--check" in argv
    table = render_tables()
    stale = 0
    for doc in TARGET_DOCS:
        text = doc.read_text(encoding="utf-8")
        if not BLOCK.search(text):
            continue
        updated = BLOCK.sub(lambda m: m.group(1) + table + m.group(3), text)
        if updated != text:
            stale += 1
            if not check:
                doc.write_text(updated, encoding="utf-8")
    if check and stale:
        print(f"{stale} document(s) out of date; run python scripts/render_bench_docs.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
