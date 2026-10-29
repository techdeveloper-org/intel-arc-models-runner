"""Validate benchmark result files against schema v1 (stdlib plus the arc_runner package).

Usage: python scripts/validate_bench_json.py [files...]   (default: docs/benchmarks/results/*.json)
Exit code 0 when every file is valid; an empty results directory is valid.
"""

import json
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arc_runner.benchmark.results import validate_result  # noqa: E402


def main(argv: List[str]) -> int:
    """Validate the given files or every file in docs/benchmarks/results.

    Args:
        argv: Command line arguments without the program name.

    Returns:
        int: 0 if all files are valid, 1 otherwise.
    """
    paths = [Path(a) for a in argv] or sorted((REPO_ROOT / "docs" / "benchmarks" / "results").glob("*.json"))
    failures = 0
    for path in paths:
        try:
            problems = validate_result(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            problems = [f"cannot read: {exc}"]
        for problem in problems:
            print(f"{path}: {problem}")
        failures += bool(problems)
    print(f"checked {len(paths)} file(s), {failures} invalid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
