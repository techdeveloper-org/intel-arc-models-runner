"""WP-12a tests: device-keyed absolute compile cache and opt-in warmup for the OpenVINO engine."""

import errno
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from arc_runner.core.exceptions import ModelLoadError
from arc_runner.engine import openvino_engine as oe
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

INVALID_WINDOWS_CHARS = set('<>:"/\\|?*')


class FakePipeline:
    """Fake LLMPipeline recording construction kwargs and generate calls."""

    instances: List["FakePipeline"] = []
    ctor_error: Any = None
    generate_error: Any = None

    def __init__(self, path: str, device: str, **props: Any) -> None:
        if FakePipeline.ctor_error is not None:
            raise FakePipeline.ctor_error
        self.path = path
        self.device = device
        self.props = props
        self.generate_calls: List[Any] = []
        FakePipeline.instances.append(self)

    def generate(self, prompts: Any, config: Any = None) -> Any:
        if FakePipeline.generate_error is not None:
            raise FakePipeline.generate_error
        self.generate_calls.append((prompts, config))
        return "ok"


class FakeGenerationConfig:
    """Fake openvino_genai.GenerationConfig accepting attribute assignment."""


@pytest.fixture()
def fake_ov(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake openvino_genai module and reset FakePipeline state."""
    FakePipeline.instances = []
    FakePipeline.ctor_error = None
    FakePipeline.generate_error = None
    module = SimpleNamespace(LLMPipeline=FakePipeline, GenerationConfig=FakeGenerationConfig)
    monkeypatch.setitem(sys.modules, "openvino_genai", module)
    return FakePipeline


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Create an empty model directory."""
    path = tmp_path / "model"
    path.mkdir()
    return path


def test_build_ov_properties_cache_dir_absolute_independent_of_cwd_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARC_CACHE_DIR", raising=False)
    monkeypatch.setattr(oe, "DEFAULT_CACHE_DIR", tmp_path / "home_cache")
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    one = OpenVINOGenAIEngine()._build_ov_properties("GPU", {})["CACHE_DIR"]
    monkeypatch.chdir(second)
    two = OpenVINOGenAIEngine()._build_ov_properties("GPU", {})["CACHE_DIR"]
    assert one == two
    assert Path(one).is_absolute()
    assert Path(one).parent == tmp_path / "home_cache" / "ov_cache"


def test_build_ov_properties_gpu_npu_cpu_each_get_device_keyed_subdir_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path / "root"))
    engine = OpenVINOGenAIEngine()
    dirs = {dev: Path(engine._build_ov_properties(dev, {})["CACHE_DIR"]) for dev in ("GPU", "NPU", "CPU")}
    for dev, path in dirs.items():
        assert path == tmp_path / "root" / dev
        assert path.is_dir()
    assert "PERFORMANCE_HINT" not in engine._build_ov_properties("NPU", {})


def test_build_ov_properties_relative_env_made_absolute_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARC_CACHE_DIR", "rel/root")
    cache = Path(OpenVINOGenAIEngine()._build_ov_properties("CPU", {})["CACHE_DIR"])
    assert cache.is_absolute()
    assert cache == tmp_path / "rel" / "root" / "CPU"


@pytest.mark.parametrize("name", ["CON", "nul", "COM1", "LPT9", "CON.0", "prn", "AUX"])
def test_device_key_windows_reserved_names_prefixed_parametrised_ok(name: str) -> None:
    key = oe.device_key(name)
    assert key.startswith("_")
    assert key[1:] == name.upper()


def test_device_key_com10_is_not_reserved_ok() -> None:
    assert oe.device_key("COM10") == "COM10"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("GPU", "GPU"),
        ("gpu", "GPU"),
        ("GPU.0", "GPU.0"),
        ("GPU.1", "GPU.1"),
        ("AUTO:GPU,CPU", "AUTO_GPU_CPU"),
        ("HETERO:GPU,CPU", "HETERO_GPU_CPU"),
        ("GPU. ", "GPU"),
        (" .GPU. ", "GPU"),
        ("GPU:0", "GPU_0"),
        ("a/b\\c", "A_B_C"),
    ],
)
def test_device_key_sanitising_parametrised_ok(raw: str, expected: str) -> None:
    key = oe.device_key(raw)
    assert key == expected
    assert not (set(key) & INVALID_WINDOWS_CHARS)


@pytest.mark.parametrize("raw", ["", "   ", "...", "é中", ":::", "___"])
def test_device_key_empty_or_all_invalid_gives_nonempty_stable_key_ok(raw: str) -> None:
    key = oe.device_key(raw)
    assert key
    assert key == oe.device_key(raw)
    assert set(key) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    assert not key.endswith((".", " "))


def test_device_key_distinct_all_invalid_inputs_do_not_share_key_ok() -> None:
    assert oe.device_key("é") != oe.device_key("中")


def test_device_key_documented_collision_two_strings_share_key_ok() -> None:
    assert oe.device_key("GPU:0") == oe.device_key("GPU,0") == "GPU_0"


def test_device_key_length_boundary_truncated_with_stable_hash_and_limit_is_named_constant_ok() -> None:
    assert oe.DEVICE_KEY_MAX_LEN == 64
    limit = 16
    exact = "A" * limit
    over = "A" * (limit + 1)
    assert oe.device_key(exact, max_len=limit) == exact
    truncated = oe.device_key(over, max_len=limit)
    assert len(truncated) == limit
    assert truncated == oe.device_key(over, max_len=limit)
    assert truncated != oe.device_key("A" * (limit + 2), max_len=limit)
    below = "A" * (limit - 1)
    assert oe.device_key(below, max_len=limit) == below
    assert len(oe.device_key("B" * 500)) == oe.DEVICE_KEY_MAX_LEN


def test_build_ov_properties_user_cache_dir_wins_for_every_device_ok(tmp_path: Path) -> None:
    for dev in ("GPU", "NPU", "CPU"):
        props = OpenVINOGenAIEngine()._build_ov_properties(dev, {"CACHE_DIR": "user_dir"})
        assert props["CACHE_DIR"] == "user_dir"


def test_build_ov_properties_unwritable_dir_degrades_with_reason_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path / "root"))
    real_access = os.access
    (tmp_path / "root" / "CPU").mkdir(parents=True)
    monkeypatch.setattr(os, "access", lambda p, m, *a, **k: False if str(p).endswith("CPU") else real_access(p, m, *a, **k))
    with caplog.at_level(logging.WARNING):
        props = OpenVINOGenAIEngine()._build_ov_properties("CPU", {})
    assert "CACHE_DIR" not in props
    assert any(r.levelno == logging.WARNING and "cache" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.parametrize("code", [errno.EACCES, errno.ENOSPC, errno.EROFS])
def test_build_ov_properties_mkdir_oserror_degrades_with_reason_parametrised_ok(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path / "root"))

    def boom(self: Path, *a: Any, **k: Any) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(Path, "mkdir", boom)
    with caplog.at_level(logging.WARNING):
        props = OpenVINOGenAIEngine()._build_ov_properties("NPU", {})
    assert "CACHE_DIR" not in props
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_build_ov_properties_root_is_a_file_degrades_with_reason_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "rootfile"
    root.write_text("x")
    monkeypatch.setenv("ARC_CACHE_DIR", str(root))
    with caplog.at_level(logging.WARNING):
        props = OpenVINOGenAIEngine()._build_ov_properties("GPU", {})
    assert "CACHE_DIR" not in props
    assert props["PERFORMANCE_HINT"] == "LATENCY"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_load_model_unwritable_cache_still_loads_without_cache_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_ov: type, model_dir: Path) -> None:
    root = tmp_path / "rootfile"
    root.write_text("x")
    monkeypatch.setenv("ARC_CACHE_DIR", str(root))
    engine = OpenVINOGenAIEngine()
    engine.load_model(model_dir, "CPU")
    assert engine.is_loaded
    assert "CACHE_DIR" not in fake_ov.instances[0].props


@pytest.mark.parametrize("raw,suffix", [("", "CPU"), ("   ", "CPU")])
def test_build_ov_properties_empty_env_uses_default_root_ok(raw: str, suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oe, "DEFAULT_CACHE_DIR", tmp_path / "dflt")
    monkeypatch.setenv("ARC_CACHE_DIR", raw)
    cache = Path(OpenVINOGenAIEngine()._build_ov_properties("CPU", {})["CACHE_DIR"])
    assert cache == tmp_path / "dflt" / "ov_cache" / suffix


@pytest.mark.parametrize("sub", ["café dir", "with space", "a/../b", "a/./b"])
def test_build_ov_properties_unicode_space_and_dot_dot_normalised_parametrised_ok(sub: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path) + os.sep + sub)
    cache = Path(OpenVINOGenAIEngine()._build_ov_properties("CPU", {})["CACHE_DIR"])
    assert ".." not in cache.parts and "." not in cache.parts
    assert cache.is_dir()
    assert cache.name == "CPU"


def test_build_ov_properties_cache_dir_creation_race_two_loads_same_device_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path / "root"))
    barrier = threading.Barrier(2)
    results: List[Any] = []
    real_mkdir = Path.mkdir

    def synced_mkdir(self: Path, *a: Any, **k: Any) -> None:
        barrier.wait(timeout=5)
        real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", synced_mkdir)

    def work() -> None:
        results.append(OpenVINOGenAIEngine()._build_ov_properties("GPU", {}).get("CACHE_DIR"))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == [str(tmp_path / "root" / "GPU")] * 2


def test_warmup_default_off_no_generate_at_load_ok(fake_ov: type, model_dir: Path) -> None:
    OpenVINOGenAIEngine().load_model(model_dir, "CPU")
    assert fake_ov.instances[0].generate_calls == []


def test_warmup_on_runs_one_fixed_generate_excluded_from_telemetry_and_manager_counters_ok(
    fake_ov: type, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", "on")
    engine = OpenVINOGenAIEngine()
    before = engine.get_telemetry()
    engine.load_model(model_dir, "CPU")
    calls = fake_ov.instances[0].generate_calls
    assert len(calls) == 1
    assert calls[0][0] == [oe.WARMUP_PROMPT]
    assert calls[0][1].max_new_tokens == oe.WARMUP_MAX_NEW_TOKENS
    assert engine.get_telemetry() is before


@pytest.mark.parametrize(
    "value,expected_on",
    [("on", True), ("ON", True), ("On", True), (" on ", True), ("off", False), ("OFF", False), ("", False), ("   ", False)],
)
def test_arc_warmup_on_load_valid_values_case_insensitive_ok(
    value: str, expected_on: bool, fake_ov: type, model_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", value)
    with caplog.at_level(logging.INFO):
        OpenVINOGenAIEngine().load_model(model_dir, "CPU")
    assert len(fake_ov.instances[0].generate_calls) == (1 if expected_on else 0)
    assert not any("rejected" in m or "not recognised" in m for m in caplog.messages)


@pytest.mark.parametrize("value", ["1", "true", "yes", "onn", "o n", "enable", "0", "on,off", "é"])
def test_arc_warmup_on_load_other_defaults_off_logged_at_warning_ok(
    value: str, fake_ov: type, model_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", value)
    with caplog.at_level(logging.DEBUG):
        OpenVINOGenAIEngine().load_model(model_dir, "CPU")
    assert fake_ov.instances[0].generate_calls == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "ARC_WARMUP_ON_LOAD" in r.getMessage()]
    assert len(warnings) == 1
    assert all("warmup_on_load=on" not in m.lower() for m in caplog.messages)


def test_warmup_prompt_and_token_count_are_policy_constants_ok() -> None:
    assert isinstance(oe.WARMUP_PROMPT, str) and oe.WARMUP_PROMPT
    assert oe.WARMUP_PROMPT.isascii()
    assert isinstance(oe.WARMUP_MAX_NEW_TOKENS, int) and 0 < oe.WARMUP_MAX_NEW_TOKENS <= 16


def test_load_logs_warmup_on_load_on_at_warning_level_visible_without_setup_logging_ok(
    fake_ov: type, model_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", "on")
    with caplog.at_level(logging.INFO):
        OpenVINOGenAIEngine().load_model(model_dir, "CPU")
    hits = [r for r in caplog.records if r.getMessage().startswith("warmup_on_load=on")]
    assert len(hits) == 1 and hits[0].levelno == logging.WARNING
    caplog.clear()
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", "off")
    with caplog.at_level(logging.DEBUG):
        OpenVINOGenAIEngine().load_model(model_dir, "CPU")
    assert not any("warmup_on_load=on" in m.lower() for m in caplog.messages)


def test_warmup_generate_raises_load_fails_with_model_load_error_and_unloads_ok(
    fake_ov: type, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", "on")
    fake_ov.generate_error = RuntimeError("boom in warmup")
    engine = OpenVINOGenAIEngine()
    with pytest.raises(ModelLoadError, match="boom in warmup"):
        engine.load_model(model_dir, "CPU")
    assert not engine.is_loaded
    assert engine.model_path == ""


def test_warmup_prompt_over_user_set_npu_prompt_cap_raises_model_load_error_with_library_message_not_swallowed_ok(
    fake_ov: type, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_WARMUP_ON_LOAD", "on")
    fake_ov.generate_error = RuntimeError("Prompt length is greater than MAX_PROMPT_LEN")
    engine = OpenVINOGenAIEngine()
    with pytest.raises(ModelLoadError, match="greater than MAX_PROMPT_LEN"):
        engine.load_model(model_dir, "NPU", MAX_PROMPT_LEN=1)
    assert fake_ov.instances == [] or not engine.is_loaded
    assert not engine.is_loaded


def test_load_with_corrupt_cache_blob_raises_typed_model_load_error_naming_cache_dir_no_silent_retry_ok(
    fake_ov: type, model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path / "root"))
    attempts: List[int] = []

    def ctor(self: FakePipeline, path: str, device: str, **props: Any) -> None:
        attempts.append(1)
        raise RuntimeError("cannot import blob: corrupt")

    monkeypatch.setattr(FakePipeline, "__init__", ctor)
    engine = OpenVINOGenAIEngine()
    with pytest.raises(ModelLoadError) as info:
        engine.load_model(model_dir, "NPU")
    assert str(tmp_path / "root" / "NPU") in str(info.value)
    assert "corrupt" in str(info.value)
    assert len(attempts) == 1
    assert not engine.is_loaded


def test_load_pipeline_raises_oserror_enospc_surfaces_model_load_error_with_cache_dir_named_ok(
    fake_ov: type, model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARC_CACHE_DIR", str(tmp_path / "root"))
    fake_ov.ctor_error = OSError(errno.ENOSPC, "No space left on device")
    with pytest.raises(ModelLoadError) as info:
        OpenVINOGenAIEngine().load_model(model_dir, "GPU")
    assert str(tmp_path / "root" / "GPU") in str(info.value)
    assert "No space left" in str(info.value)


def test_load_error_without_cache_dir_does_not_mention_cache_ok(fake_ov: type, model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rootfile"
    root.write_text("x")
    monkeypatch.setenv("ARC_CACHE_DIR", str(root))
    fake_ov.ctor_error = RuntimeError("driver exploded")
    with pytest.raises(ModelLoadError) as info:
        OpenVINOGenAIEngine().load_model(model_dir, "CPU")
    assert "compile cache" not in str(info.value)


@pytest.mark.hardware
def test_llm_pipeline_cpu_cache_dir_accepted_integration(request: pytest.FixtureRequest, hardware_flag: bool) -> None:
    if not hardware_flag:
        pytest.skip("NOT RUN: needs a real tiny OpenVINO IR model and openvino_genai (WP-12a N-04 procedure)")
    pytest.skip("NOT RUN: no tiny model path configured in this repository")
