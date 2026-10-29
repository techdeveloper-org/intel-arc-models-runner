"""Inventory the benchmark-style numbers in documents that are marked UNVERIFIED.

Usage: python scripts/list_unverified_numbers.py [--json]
For every Markdown file under docs/ and the README that carries a 'Provenance status: UNVERIFIED' (or
UNVERIFIED illustration) banner, counts table rows containing a number followed by a unit typical of a
benchmark (tok/s, ms, s, W, GB, x, %, fps) so they can be replaced by results from
docs/benchmarks/results/*.json. Exit code 0 always; this is a worklist, not a gate.
"""

import json
from pathlib import Path
import re
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
BANNER = re.compile(r"Provenance status: UNVERIFIED", re.I)
ROW = re.compile(r"^\|.*\d+(\.\d+)?\s*(tok/s|t/s|ms|s\b|W\b|GB|MB|x\b|×|%|fps|it/s|MP/s).*\|\s*$", re.M)


def inventory(root: Path = REPO_ROOT) -> List[Dict[str, object]]:
    """Scan documents for UNVERIFIED banners and count benchmark-looking table rows.

    Args:
        root: Repository root.

    Returns:
        List[Dict[str, object]]: One record per marked file with its path and numeric-row count.
    """
    records = []
    for path in sorted(list((root / "docs").rglob("*.md")) + [root / "README.md"]):
        text = path.read_text(encoding="utf-8", errors="replace")
        if BANNER.search(text):
            records.append({"file": str(path.relative_to(root)), "numeric_table_rows": len(ROW.findall(text))})
    return records


def main(argv: List[str]) -> int:
    """Print the inventory.

    Args:
        argv: Command line arguments without the program name.

    Returns:
        int: Always 0.
    """
    records = inventory()
    if "--json" in argv:
        print(json.dumps(records, indent=2))
        return 0
    for record in records:
        print(f"{record['numeric_table_rows']:5d} rows  {record['file']}")
    print(f"{len(records)} document(s) carry an UNVERIFIED banner; replace their numbers with results from docs/benchmarks/results/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
