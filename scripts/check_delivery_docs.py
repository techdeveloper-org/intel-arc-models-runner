"""Validate the documents under docs/delivery (stdlib only).

Rules: every markdown file has front matter with doc_id, title, stage, version (semver), status, created and
updated; doc_ids are unique; each document has a revision-history row for its version; INDEX.md lists every
document with its current version and path and nothing else; DECISION_LOG entries are unique and increasing;
files referenced from TRACEABILITY.md exist; a document that changed against HEAD must have a version that is
not lower than, and not equal to, the committed one.

Usage: python scripts/check_delivery_docs.py
"""

from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = REPO_ROOT / "docs" / "delivery"
REQUIRED = ("doc_id", "title", "stage", "version", "status", "created", "updated")
STATUSES = {"draft", "reviewed", "approved", "superseded"}
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def front_matter(text: str) -> Dict[str, str]:
    """Parse the simple key: value front matter of a document.

    Args:
        text: Document text.

    Returns:
        Dict[str, str]: Keys and values, empty if there is no front matter.
    """
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    pairs = (line.split(":", 1) for line in text[4:end].splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in pairs}


def semver_tuple(version: str) -> Tuple[int, ...]:
    """Convert x.y.z to a comparable tuple.

    Args:
        version: Version string.
    """
    return tuple(int(part) for part in version.split("."))


def committed_text(path: Path) -> str:
    """Return the text of a file at HEAD, or an empty string if it is not committed.

    Args:
        path: File path inside the repository.
    """
    relative = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relative}"], capture_output=True, encoding="utf-8")
    return result.stdout if result.returncode == 0 else ""


def check_document(path: Path, seen: Dict[str, Path]) -> List[str]:
    """Validate one document.

    Args:
        path: Markdown file.
        seen: doc_id to path map, updated in place.

    Returns:
        List[str]: Problems found.
    """
    name = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    meta = front_matter(text)
    problems = [f"{name}: front matter missing {key}" for key in REQUIRED if key not in meta]
    if problems:
        return problems
    if not SEMVER.match(meta["version"]):
        problems.append(f"{name}: version {meta['version']} is not x.y.z")
        return problems
    if meta["status"] not in STATUSES:
        problems.append(f"{name}: status {meta['status']} not in {sorted(STATUSES)}")
    if meta["doc_id"] in seen:
        problems.append(f"{name}: doc_id {meta['doc_id']} already used by {seen[meta['doc_id']].relative_to(REPO_ROOT).as_posix()}")
    seen[meta["doc_id"]] = path
    if not re.search(rf"^\|\s*{re.escape(meta['version'])}\s*\|", text, re.M):
        problems.append(f"{name}: no revision-history row for version {meta['version']}")
    old = committed_text(path)
    if old and old != text:
        old_version = front_matter(old).get("version", "")
        if SEMVER.match(old_version):
            if semver_tuple(meta["version"]) < semver_tuple(old_version):
                problems.append(f"{name}: version went backwards ({old_version} -> {meta['version']})")
            elif meta["version"] == old_version:
                problems.append(f"{name}: changed without a version bump (still {old_version})")
    return problems


def check_index(seen: Dict[str, Path]) -> List[str]:
    """Check INDEX.md against the documents found.

    Args:
        seen: doc_id to path map.

    Returns:
        List[str]: Problems found.
    """
    index = DELIVERY / "INDEX.md"
    if not index.exists():
        return ["docs/delivery/INDEX.md is missing"]
    rows = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0].startswith("DOC-"):
            rows[cells[0]] = cells
    problems: List[str] = []
    for doc_id, path in seen.items():
        meta = front_matter(path.read_text(encoding="utf-8"))
        row = rows.get(doc_id)
        relative = path.relative_to(DELIVERY).as_posix()
        if row is None:
            problems.append(f"INDEX.md lacks {doc_id} ({relative})")
        elif row[2] != meta["version"] or row[5] != relative:
            problems.append(f"INDEX.md row for {doc_id} is stale (version {row[2]} path {row[5]}; document has {meta['version']} {relative})")
    problems += [f"INDEX.md lists unknown {doc_id}" for doc_id in rows if doc_id not in seen]
    return problems


def check_decision_log() -> List[str]:
    """Check DL numbering in DECISION_LOG.md.

    Returns:
        List[str]: Problems found.
    """
    log = DELIVERY / "DECISION_LOG.md"
    numbers = [int(m.group(1)) for m in re.finditer(r"^\| DL-(\d+) \|", log.read_text(encoding="utf-8"), re.M)] if log.exists() else []
    problems = []
    if numbers != sorted(set(numbers)):
        problems.append("DECISION_LOG.md: DL numbers are not unique and increasing")
    if numbers and numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        problems.append("DECISION_LOG.md: DL numbers have gaps")
    return problems


def check_traceability() -> List[str]:
    """Check that files named in TRACEABILITY.md exist.

    Returns:
        List[str]: Problems found.
    """
    trace = DELIVERY / "TRACEABILITY.md"
    if not trace.exists():
        return []
    problems = []
    for reference in sorted(set(re.findall(r"`((?:tests|scripts|docs|src)/[A-Za-z0-9_./-]+\.[a-z]+)", trace.read_text(encoding="utf-8")))):
        if not (REPO_ROOT / reference).exists():
            problems.append(f"TRACEABILITY.md references missing file {reference}")
    return problems


def check() -> List[str]:
    """Run every delivery check.

    Returns:
        List[str]: All problems found.
    """
    if not DELIVERY.exists():
        return ["docs/delivery does not exist"]
    seen: Dict[str, Path] = {}
    problems: List[str] = []
    for path in sorted(DELIVERY.rglob("*.md")):
        problems += check_document(path, seen)
    return problems + check_index(seen) + check_decision_log() + check_traceability()


def main() -> int:
    """Print every problem and return the exit code.

    Returns:
        int: 0 if valid, 1 otherwise.
    """
    problems = check()
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
