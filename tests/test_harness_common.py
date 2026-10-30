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


def _keep_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser()
    harness_common.add_keep_args(parser)
    return parser


def test_add_keep_args_default_keeps_model_directory(tmp_path: Path) -> None:
    """Without flags the parsed namespace does not request deletion."""
    args = _keep_parser().parse_args([])
    model_dir = tmp_path / "alias"
    model_dir.mkdir()
    assert harness_common.reclaim_model_dir(args.delete_after, model_dir, tmp_path) is False
    assert model_dir.exists()


def test_add_keep_args_keep_flag_is_accepted_as_noop(tmp_path: Path) -> None:
    """The legacy --keep flag still parses and does not request deletion."""
    args = _keep_parser().parse_args(["--keep"])
    assert args.delete_after is False


def test_add_keep_args_keep_and_delete_after_together_is_argparse_error() -> None:
    """The two flags are mutually exclusive."""
    with pytest.raises(SystemExit):
        _keep_parser().parse_args(["--keep", "--delete-after"])


def test_reclaim_model_dir_delete_after_removes_only_alias_dir(tmp_path: Path) -> None:
    """Opting in deletes the alias directory and leaves siblings and the root alone."""
    models = tmp_path / "models"
    (models / "a").mkdir(parents=True)
    (models / "a" / "w.bin").write_bytes(b"x")
    (models / "b").mkdir()
    assert harness_common.reclaim_model_dir(True, models / "a", models) is True
    assert not (models / "a").exists()
    assert (models / "b").exists() and models.exists()


def test_reclaim_model_dir_outside_models_root_raises_and_keeps_files(tmp_path: Path) -> None:
    """A directory outside the models root, or the root itself, is refused."""
    from arc_runner.core.exceptions import ConfigurationError

    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ConfigurationError):
        harness_common.reclaim_model_dir(True, outside, models)
    with pytest.raises(ConfigurationError):
        harness_common.reclaim_model_dir(True, models, models)
    with pytest.raises(ConfigurationError):
        harness_common.reclaim_model_dir(True, models / ".." / "elsewhere", models)
    assert outside.exists() and models.exists()


def test_reclaim_model_dir_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    """Deleting a directory that is already gone returns False."""
    assert harness_common.reclaim_model_dir(True, tmp_path / "models" / "gone", tmp_path / "models") is False
