"""Shared helpers for the live benchmark scripts."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from arc_runner.benchmark import harness_common
from arc_runner.benchmark.harness_common import default_result_path, optional_sysman, target_device


def test_default_result_path_is_deterministic_and_under_results_dir(tmp_path: Path) -> None:
    """The file name carries workload, model and a UTC timestamp."""
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    path = default_result_path(tmp_path, "audio_stt", "whisper-tiny", now=now)
    assert path == tmp_path / "docs" / "benchmarks" / "results" / "audio_stt_whisper-tiny_20260102T030405Z.v1.json"


def test_target_device_prefers_gpu_when_arc_present_and_flags_fallback_otherwise() -> None:
    """No Arc GPU means CPU and fallback_used=True so claim gates refuse the rows."""
    assert target_device(SimpleNamespace(has_intel_arc=True, has_intel_npu=False), "GPU") == ("GPU", False)
    assert target_device(SimpleNamespace(has_intel_arc=False, has_intel_npu=False), "GPU") == ("CPU", True)
    assert target_device(SimpleNamespace(has_intel_arc=False, has_intel_npu=True), "NPU") == ("NPU", False)
    assert target_device(SimpleNamespace(has_intel_arc=False, has_intel_npu=False), "NPU") == ("CPU", True)
    assert target_device(SimpleNamespace(has_intel_arc=False, has_intel_npu=False), "CPU") == ("CPU", False)


def test_optional_sysman_returns_none_when_level_zero_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Power measurement is optional; an unavailable loader gives None, not an error."""
    fake = SimpleNamespace(is_available=False, _devices=[])
    monkeypatch.setattr(harness_common, "LevelZeroSysman", lambda: fake)
    assert optional_sysman() is None


def test_optional_sysman_returns_instance_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Sysman with a GPU device is returned for power measurement."""
    fake = SimpleNamespace(is_available=True, _devices=[1])
    monkeypatch.setattr(harness_common, "LevelZeroSysman", lambda: fake)
    assert optional_sysman() is fake


def test_download_snapshot_skips_when_marker_present_and_otherwise_calls_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing model directories are not re-downloaded; elapsed seconds are measured otherwise."""
    dest = tmp_path / "m"
    dest.mkdir()
    (dest / "openvino_model.xml").write_text("x")
    assert harness_common.download_snapshot("repo/x", dest, ("openvino_model.xml",)) == 0.0
    calls: list = []

    def fake_snapshot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot))
    other = tmp_path / "n"
    assert harness_common.download_snapshot("repo/y", other, ("openvino_model.xml",)) >= 0.0
    assert calls and calls[0]["repo_id"] == "repo/y"
