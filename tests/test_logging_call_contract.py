"""Guard: logger calls in src must only use keywords that stdlib logging accepts (structured fields go in extra=)."""

import ast
from pathlib import Path
from typing import List

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "arc_runner"
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
ALLOWED_KEYWORDS = {"exc_info", "stack_info", "stacklevel", "extra"}


def find_bad_logger_calls(source: str) -> List[str]:
    """Return 'line:keyword' for each logger.<level>() call that passes a keyword stdlib logging rejects.

    Args:
        source: Python source text.

    Returns:
        List[str]: Offending call sites as 'line:keyword' strings.
    """
    bad: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Name) and owner.id in {"logger", "log", "_logger"}):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in ALLOWED_KEYWORDS:
                bad.append(f"{node.lineno}:{keyword.arg}")
    return bad


def test_find_bad_logger_calls_flags_a_structured_keyword_negative_control_ok() -> None:
    """The guard is not vacuous: a call with device= is flagged, extra= and exc_info= are not."""
    assert find_bad_logger_calls('logger.info("x", device="GPU")') == ["1:device"]
    assert find_bad_logger_calls('logger.warning("x", extra={"device": "GPU"}, exc_info=True)') == []
    assert find_bad_logger_calls('other.info("x", device="GPU")') == []


def test_src_logger_calls_use_only_stdlib_logging_keywords_ok() -> None:
    """No module under src passes structured keywords straight to a stdlib logger (it raises TypeError once logging is configured)."""
    offenders = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        found = find_bad_logger_calls(path.read_text(encoding="utf-8"))
        if found:
            offenders[str(path.relative_to(SRC_ROOT))] = found
    assert offenders == {}, offenders
