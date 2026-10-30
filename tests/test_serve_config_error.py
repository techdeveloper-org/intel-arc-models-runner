"""`arc-runner serve` turns an invalid ARC_CONTEXT_WINDOW into a one-line error before any model load."""

import importlib
from typing import Any, List

import pytest
import uvicorn

from arc_runner.server.model_manager import ModelManager

cli_main = importlib.import_module("arc_runner.cli.main")


@pytest.fixture
def serve_spy(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Stub the server loop and record model loads.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        List[str]: Events recorded ("run", "load").
    """
    events: List[str] = []
    monkeypatch.setattr(cli_main, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, *a, **k: events.append("run"))

    def _load(self: Any, *a: Any, **k: Any) -> Any:
        events.append("load")
        raise RuntimeError("no model in tests")

    monkeypatch.setattr(ModelManager, "get_or_load_engine", _load)
    return events


@pytest.mark.parametrize("value", ["abc", "0", "8\n", "00000008", "9" * 5000, "-1", "1e3", " 8"])
def test_serve_invalid_context_window_exits_2_with_one_line(
    value: str, serve_spy: List[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A bad value prints one stderr line naming the variable and range, exits 2, loads nothing."""
    monkeypatch.setenv("ARC_CONTEXT_WINDOW", value)
    code = cli_main.main(["serve", "--default-model", "m"])
    captured = capsys.readouterr()
    assert code == 2
    assert serve_spy == []
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "ARC_CONTEXT_WINDOW" in lines[0]
    assert "1" in lines[0] and "2097152" in lines[0]
    assert "Traceback" not in captured.err + captured.out
    assert len(captured.err) < 400
    if len(value) > 20:
        assert value not in captured.err


@pytest.mark.parametrize("value", [None, "", "8", "08", "2097152"])
def test_serve_valid_or_empty_context_window_starts_server(
    value: Any, serve_spy: List[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Unset, empty and valid values reach the server loop with exit code 0."""
    if value is None:
        monkeypatch.delenv("ARC_CONTEXT_WINDOW", raising=False)
    else:
        monkeypatch.setenv("ARC_CONTEXT_WINDOW", value)
    code = cli_main.main(["serve"])
    assert code == 0
    assert serve_spy == ["run"]
    assert capsys.readouterr().err == ""
