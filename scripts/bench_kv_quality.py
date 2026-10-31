"""Run the u8 KV-cache quality gate on a real model (needs the model and, usually, Arc silicon).

Loads the model twice (runtime-default KV precision and KV_CACHE_PRECISION=u8), generates greedily for a
fixed prompt set and writes docs/benchmarks/quality/kv_u8_<model>.json. The exit code is 0 when the gate
passes, 1 when it fails. A pass is evidence for a human decision about the default; it never changes it.

Usage: python scripts/bench_kv_quality.py --model-dir <dir> [--device GPU] [--max-new-tokens 64]
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.kv_quality import compare_kv_precision, write_report  # noqa: E402

DEFAULT_PROMPTS: List[str] = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Summarise the causes of the French Revolution in five sentences.",
    "What is the difference between TCP and UDP?",
    "Translate 'Where is the nearest railway station?' into French, German and Spanish.",
    "List the planets of the solar system in order from the Sun.",
    "Describe how garbage collection works in a generational collector.",
    "Give three tips for writing readable SQL queries.",
]


def build_factory(engine_cls: Callable[[], Any], model_dir: Path, device: str) -> Callable[[str], Any]:
    """Return a factory that loads the model with a given KV precision.

    Args:
        engine_cls: Engine class (OpenVINOGenAIEngine or a test double).
        model_dir: Model directory.
        device: Device name.

    Returns:
        Callable[[str], Any]: Function taking a KV precision ("" for the runtime default).
    """

    def make(precision: str) -> Any:
        engine = engine_cls()
        if precision:
            engine.load_model(model_dir, device=device, KV_CACHE_PRECISION=precision)
        else:
            engine.load_model(model_dir, device=device)
        return engine

    return make


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments without the program name, or None for sys.argv.

    Returns:
        int: 0 if the gate passes, 1 if it fails, 2 if the model directory is missing, 3 if the runtime refused the
        reduced KV precision (recorded in the report, nothing was compared).
    """
    parser = argparse.ArgumentParser(description="u8 KV-cache quality gate.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.model_dir.is_dir():
        print(f"model directory {args.model_dir} not found", file=sys.stderr)
        return 2

    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    report = compare_kv_precision(build_factory(OpenVINOGenAIEngine, args.model_dir, args.device), DEFAULT_PROMPTS, args.max_new_tokens)
    out = args.out or REPO_ROOT / "docs" / "benchmarks" / "quality" / f"kv_u8_{args.model_dir.name}.json"
    write_report(out, report, args.model_dir.name, args.device)
    if report.candidate_error is not None:
        print(f"CANDIDATE NOT RUN: the runtime refused the reduced KV precision ({report.candidate_error[:200]}); wrote {out}")
        return 3
    print(f"{'PASS' if report.passed else 'FAIL'}: {report.pass_count}/{len(report.prompts)} prompts agree for >= {report.min_prefix_tokens} tokens; wrote {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
