"""Render every Mermaid block in the repository's diagram documents with mermaid-cli to prove they parse.

Usage: python scripts/check_mermaid.py [--require] [files...]
  --require   fail (exit 1) instead of skipping (exit 0) when mmdc is not installed.
Needs `mmdc` (npm i @mermaid-js/mermaid-cli) and a Chromium; set PUPPETEER_EXECUTABLE_PATH to use an
existing browser. The Chromium sandbox is disabled through a temporary puppeteer config.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)
DEFAULT_FILES = ["uml/*.md", "docs/architecture.md", "docs/diagrams/README.md", "docs/research/*.md"]


def collect_blocks(paths: List[Path]) -> List[Tuple[str, int, str]]:
    """Extract Mermaid blocks.

    Args:
        paths: Markdown files to scan.

    Returns:
        List[Tuple[str, int, str]]: (file, block index, source) for every block.
    """
    blocks = []
    for path in paths:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        blocks += [(str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path), i, m.group(1)) for i, m in enumerate(BLOCK.finditer(text))]
    return blocks


def main(argv: List[str]) -> int:
    """Render all blocks and report failures.

    Args:
        argv: Command line arguments without the program name.

    Returns:
        int: 0 if all blocks render (or mmdc is absent without --require), 1 otherwise.
    """
    require = "--require" in argv
    names = [a for a in argv if not a.startswith("--")]
    mmdc = shutil.which("mmdc") or shutil.which("mmdc", path=str(REPO_ROOT / "node_modules" / ".bin"))
    if mmdc is None:
        print("mmdc not installed; Mermaid blocks not rendered")
        return 1 if require else 0
    paths = [Path(n) for n in names] if names else sorted(p for pattern in DEFAULT_FILES for p in REPO_ROOT.glob(pattern))
    config = {"args": ["--no-sandbox", "--disable-gpu"]}
    if os.environ.get("PUPPETEER_EXECUTABLE_PATH"):
        config["executablePath"] = os.environ["PUPPETEER_EXECUTABLE_PATH"]
    failures = 0
    blocks = collect_blocks(paths)
    with tempfile.TemporaryDirectory() as tmp:
        pp = Path(tmp) / "pp.json"
        pp.write_text(json.dumps(config), encoding="utf-8")
        for file, index, source in blocks:
            src = Path(tmp) / "block.mmd"
            src.write_text(source, encoding="utf-8")
            result = subprocess.run([mmdc, "-p", str(pp), "-i", str(src), "-o", str(Path(tmp) / "out.svg"), "-q"], capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                failures += 1
                detail = next((line for line in result.stderr.splitlines() if "rror" in line), result.stderr[:200])
                print(f"FAIL {file} block {index}: {detail.strip()}")
    print(f"rendered {len(blocks)} block(s), {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
