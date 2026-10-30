"""Context window resolution (env, model config, default), KV geometry and wiring tests. No real model."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from arc_runner.core.constants import DEFAULT_CONTEXT_WINDOW
from arc_runner.core.exceptions import ConfigurationError, ContextWindowExceededError
from arc_runner.engine.kv_cache import KVCacheManager
from arc_runner.engine.model_config import (
    CONTEXT_WINDOW_ENV,
    MAX_CONTEXT_WINDOW,
    MAX_MODEL_CONFIG_BYTES,
    parse_context_window_env,
    read_kv_geometry,
    resolve_context_window,
)
from arc_runner.server import model_manager as mm_module
from arc_runner.server.model_manager import ModelManager


def _model_dir(tmp_path: Path, config: Any = None, raw: Optional[bytes] = None, name: str = "m") -> Path:
    """Create a model directory with an optional fabricated config.json."""
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    if raw is not None:
        (directory / "config.json").write_bytes(raw)
    elif config is not None:
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


# ----- environment parsing -----


@pytest.mark.parametrize("value,expected", [("1", 1), ("8", 8), ("08", 8), ("4096", 4096), (str(MAX_CONTEXT_WINDOW), MAX_CONTEXT_WINDOW)])
def test_parse_context_window_env_valid_value_returns_int(value: str, expected: int) -> None:
    """ASCII digit strings inside 1..cap are accepted, including a leading zero."""
    assert parse_context_window_env({CONTEXT_WINDOW_ENV: value}) == expected


@pytest.mark.parametrize("env", [{}, {CONTEXT_WINDOW_ENV: ""}])
def test_parse_context_window_env_unset_or_empty_returns_none(env: Dict[str, str]) -> None:
    """Unset and empty both mean 'not configured'."""
    assert parse_context_window_env(env) is None


@pytest.mark.parametrize(
    "value",
    [
        "0", "-1", "+8", "8.0", "1e3", "abc", "8 ", " 8", "8\n", "\t8", "0x10", "1_000",
        str(MAX_CONTEXT_WINDOW + 1), "\u0668", "\uff18\uff19", "9" * 7, "1" * 5000, "0" * 8 + "1",
    ],
)
def test_parse_context_window_env_invalid_value_raises_configuration_error(value: str) -> None:
    """Zero, negative, signed, non-ASCII digits, whitespace, over-cap and over-long digit strings are rejected."""
    with pytest.raises(ConfigurationError) as excinfo:
        parse_context_window_env({CONTEXT_WINDOW_ENV: value})
    assert CONTEXT_WINDOW_ENV in str(excinfo.value)


def test_parse_context_window_env_error_does_not_echo_huge_value() -> None:
    """The message is bounded even for a megabyte of digits."""
    with pytest.raises(ConfigurationError) as excinfo:
        parse_context_window_env({CONTEXT_WINDOW_ENV: "7" * 1_000_000})
    assert len(str(excinfo.value)) < 400


# ----- model config -----


@pytest.mark.parametrize("value", [1, 2048, 131072, MAX_CONTEXT_WINDOW])
def test_resolve_context_window_config_value_is_used(tmp_path: Path, value: int) -> None:
    """A valid max_position_embeddings becomes the window."""
    assert resolve_context_window(_model_dir(tmp_path, {"max_position_embeddings": value}), env={}) == value


def test_resolve_context_window_config_above_cap_is_clamped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A model that advertises more than the cap is clamped to the cap, loudly (never silently)."""
    with caplog.at_level(logging.WARNING):
        got = resolve_context_window(_model_dir(tmp_path, {"max_position_embeddings": MAX_CONTEXT_WINDOW * 5}), env={})
    assert got == MAX_CONTEXT_WINDOW
    assert any(r.levelno == logging.WARNING and "clamp" in r.getMessage().lower() for r in caplog.records)


def _expect_default_with_log(model_dir: Path, caplog: pytest.LogCaptureFixture, fragment: str) -> None:
    """Assert the 4096 fallback and an INFO or WARNING record mentioning the reason."""
    with caplog.at_level(logging.INFO):
        assert resolve_context_window(model_dir, env={}) == DEFAULT_CONTEXT_WINDOW
    assert any(fragment in r.getMessage().lower() for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_resolve_context_window_no_config_file_falls_back_to_default_with_info(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Missing config.json: 4096 and an INFO record saying why."""
    _expect_default_with_log(_model_dir(tmp_path), caplog, "not found")
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_resolve_context_window_missing_model_dir_falls_back_to_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A model directory that does not exist is handled like a missing config."""
    _expect_default_with_log(tmp_path / "nope", caplog, "not found")


def test_resolve_context_window_config_is_a_directory_falls_back_to_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """config.json that is not a regular file is unreadable, not fatal."""
    directory = _model_dir(tmp_path)
    (directory / "config.json").mkdir()
    _expect_default_with_log(directory, caplog, "regular file")


def test_resolve_context_window_oversized_config_falls_back_without_parsing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A config.json above the size cap is never parsed."""
    big = b'{"max_position_embeddings": 9999, "pad": "' + b"x" * MAX_MODEL_CONFIG_BYTES + b'"}'
    _expect_default_with_log(_model_dir(tmp_path, raw=big), caplog, "larger than")


@pytest.mark.parametrize("raw", [b"{not json", b"[1, 2]", b'"text"', b"123", b"null", b"\xff\xfe\x00", b"", b"[" * 100000])
def test_resolve_context_window_malformed_or_non_object_config_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture, raw: bytes) -> None:
    """Invalid JSON, a non-object document, bad bytes and deep nesting all degrade to the default."""
    with caplog.at_level(logging.INFO):
        assert resolve_context_window(_model_dir(tmp_path, raw=raw), env={}) == DEFAULT_CONTEXT_WINDOW


def test_resolve_context_window_missing_key_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A config without max_position_embeddings is not guessed from other keys."""
    _expect_default_with_log(_model_dir(tmp_path, {"n_positions": 8192, "max_sequence_length": 8192}), caplog, "max_position_embeddings")


@pytest.mark.parametrize("value", [True, False, 0, -4096, 4096.0, 4096.5, "4096", None, [4096], {"v": 1}, float("nan")])
def test_resolve_context_window_bad_config_value_type_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture, value: Any) -> None:
    """Bool, non-positive, float, string, null and container values are rejected, not coerced."""
    model_dir = _model_dir(tmp_path, raw=json.dumps({"max_position_embeddings": value}).encode("utf-8"))
    with caplog.at_level(logging.INFO):
        assert resolve_context_window(model_dir, env={}) == DEFAULT_CONTEXT_WINDOW
    assert any("max_position_embeddings" in r.getMessage() for r in caplog.records)


def test_resolve_context_window_rope_scaling_is_ignored(tmp_path: Path) -> None:
    """rope_scaling does not change the number; only max_position_embeddings is read."""
    config = {"max_position_embeddings": 8192, "rope_scaling": {"type": "linear", "factor": 8.0}}
    assert resolve_context_window(_model_dir(tmp_path, config), env={}) == 8192


# ----- priority -----


def test_resolve_context_window_env_beats_model_config(tmp_path: Path) -> None:
    """ARC_CONTEXT_WINDOW wins over config.json."""
    model_dir = _model_dir(tmp_path, {"max_position_embeddings": 32768})
    assert resolve_context_window(model_dir, env={CONTEXT_WINDOW_ENV: "16384"}) == 16384


def test_resolve_context_window_env_beats_default_without_config(tmp_path: Path) -> None:
    """The env value applies even when the model has no config.json."""
    assert resolve_context_window(_model_dir(tmp_path), env={CONTEXT_WINDOW_ENV: "9000"}) == 9000


def test_resolve_context_window_invalid_env_raises_even_when_config_is_valid(tmp_path: Path) -> None:
    """A bad env value is an error, never a silent fall-through to the model config."""
    model_dir = _model_dir(tmp_path, {"max_position_embeddings": 32768})
    with pytest.raises(ConfigurationError):
        resolve_context_window(model_dir, env={CONTEXT_WINDOW_ENV: "0"})


def test_resolve_context_window_reads_process_environment_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With env=None the real environment is used."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "12345")
    assert resolve_context_window(_model_dir(tmp_path)) == 12345


# ----- KV geometry and footprint -----


def test_read_kv_geometry_full_config_returns_values(tmp_path: Path) -> None:
    """layers, kv heads and explicit head_dim are read as given."""
    cfg = {"num_hidden_layers": 32, "num_key_value_heads": 4, "head_dim": 96, "hidden_size": 4096, "num_attention_heads": 32}
    geometry = read_kv_geometry(_model_dir(tmp_path, cfg))
    assert (geometry.num_layers, geometry.num_kv_heads, geometry.head_dim) == (32, 4, 96)


def test_read_kv_geometry_derives_head_dim_from_hidden_size(tmp_path: Path) -> None:
    """head_dim = hidden_size / num_attention_heads when head_dim is absent and the division is exact."""
    cfg = {"num_hidden_layers": 28, "num_key_value_heads": 8, "hidden_size": 3072, "num_attention_heads": 24}
    assert read_kv_geometry(_model_dir(tmp_path, cfg)).head_dim == 128


def test_read_kv_geometry_missing_kv_heads_uses_attention_heads_mha_rule(tmp_path: Path) -> None:
    """Absent num_key_value_heads means multi-head attention in the Hugging Face convention."""
    cfg = {"num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    geometry = read_kv_geometry(_model_dir(tmp_path, cfg))
    assert (geometry.num_kv_heads, geometry.head_dim) == (12, 64)


@pytest.mark.parametrize(
    "cfg",
    [
        {"num_key_value_heads": 8, "head_dim": 128},
        {"num_hidden_layers": 28, "head_dim": 128},
        {"num_hidden_layers": 28, "num_key_value_heads": 8},
        {"num_hidden_layers": 28, "num_key_value_heads": 8, "hidden_size": 100, "num_attention_heads": 24},
        {"num_hidden_layers": True, "num_key_value_heads": 8, "head_dim": 128},
        {"num_hidden_layers": 28, "num_key_value_heads": 0, "head_dim": 128},
        {"num_hidden_layers": 28.0, "num_key_value_heads": 8, "head_dim": 128},
        {"num_hidden_layers": 28, "num_key_value_heads": 8, "head_dim": "128"},
        {"num_hidden_layers": 10**9, "num_key_value_heads": 8, "head_dim": 128},
    ],
)
def test_read_kv_geometry_incomplete_or_invalid_config_returns_none(tmp_path: Path, cfg: Dict[str, Any]) -> None:
    """Missing, inexact, bool, zero, float, string or absurd fields make the geometry unavailable (None)."""
    assert read_kv_geometry(_model_dir(tmp_path, cfg)) is None


def test_read_kv_geometry_without_config_returns_none(tmp_path: Path) -> None:
    """No config.json: nothing is invented."""
    assert read_kv_geometry(_model_dir(tmp_path)) is None


def test_kv_footprint_uses_config_geometry_and_precision(tmp_path: Path) -> None:
    """With geometry from config the estimate follows 2*L*H*D*T*bytes and is marked as model-derived."""
    cfg = {"num_hidden_layers": 16, "num_key_value_heads": 4, "head_dim": 64}
    manager = KVCacheManager(max_context_length=8192, geometry=read_kv_geometry(_model_dir(tmp_path, cfg)), kv_precision="u8")
    allocation = manager.reserve_allocation(prompt_tokens=1000, max_new_tokens=24)
    assert allocation.memory_footprint_mb == pytest.approx(2 * 16 * 4 * 64 * 1024 * 1 / (1024 * 1024))
    assert allocation.footprint_from_model_config is True


@pytest.mark.parametrize("precision,nbytes", [("", 2), ("f16", 2), ("bf16", 2), ("u8", 1), ("f32", 4)])
def test_kv_footprint_dtype_follows_kv_precision(precision: str, nbytes: int) -> None:
    """The byte width follows the KV precision setting; empty means the f16 runtime default."""
    geometry = read_kv_geometry_values(2, 2, 8)
    manager = KVCacheManager(max_context_length=100, geometry=geometry, kv_precision=precision)
    assert manager.reserve_allocation(10, 0).memory_footprint_mb == pytest.approx(2 * 2 * 2 * 8 * 10 * nbytes / (1024 * 1024))


def read_kv_geometry_values(layers: int, heads: int, dim: int) -> Any:
    """Build a KVGeometry directly."""
    from arc_runner.engine.model_config import KVGeometry

    return KVGeometry(num_layers=layers, num_kv_heads=heads, head_dim=dim)


def test_kv_footprint_without_geometry_falls_back_with_warning_and_flag(caplog: pytest.LogCaptureFixture) -> None:
    """No geometry: the legacy 28/8/128 assumption is used, flagged as not from the model, with a WARNING."""
    manager = KVCacheManager(max_context_length=4096)
    with caplog.at_level(logging.WARNING):
        allocation = manager.reserve_allocation(prompt_tokens=256, max_new_tokens=512)
    assert allocation.memory_footprint_mb == pytest.approx(2 * 28 * 8 * 128 * 768 * 2 / (1024 * 1024))
    assert allocation.footprint_from_model_config is False
    assert any(r.levelno == logging.WARNING and "assum" in r.getMessage().lower() for r in caplog.records)


def test_kv_footprint_unknown_precision_uses_two_bytes_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An unrecognised precision string is not guessed silently."""
    manager = KVCacheManager(max_context_length=100, geometry=read_kv_geometry_values(1, 1, 8), kv_precision="weird")
    with caplog.at_level(logging.WARNING):
        allocation = manager.reserve_allocation(10, 0)
    assert allocation.memory_footprint_mb == pytest.approx(2 * 1 * 1 * 8 * 10 * 2 / (1024 * 1024))
    assert any(r.levelno == logging.WARNING and "unrecognised" in r.getMessage().lower() and getattr(r, "kv_precision", "") == "weird" for r in caplog.records)


# ----- headroom at the new size -----


@pytest.mark.parametrize("window", [4096, 131072])
def test_kv_check_headroom_boundary_at_resolved_size(window: int) -> None:
    """window-1 leaves 1, window leaves 0, window+1 raises."""
    manager = KVCacheManager(max_context_length=window)
    assert manager.check_headroom(window - 1) == 1
    assert manager.check_headroom(window) == 0
    with pytest.raises(ContextWindowExceededError):
        manager.check_headroom(window + 1)


def test_engine_enforce_window_boundary_prompt_plus_max_new_tokens() -> None:
    """prompt + max_new_tokens: -1 and exact pass, +1 fails, at a window larger than 4096."""
    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine

    engine = OpenVINOGenAIEngine(kv_manager=KVCacheManager(max_context_length=32768))
    engine._enforce_context_window(30000, 2767)
    engine._enforce_context_window(30000, 2768)
    with pytest.raises(ContextWindowExceededError) as excinfo:
        engine._enforce_context_window(30000, 2769)
    assert excinfo.value.max_context == 32768


# ----- ModelManager wiring -----


class _FakePlain:
    """Stand-in for OpenVINOGenAIEngine recording its constructor arguments."""

    def __init__(self, lora_manager: Any = None, kv_manager: Any = None) -> None:
        self.kv_manager = kv_manager
        self.loaded: Any = None

    def load_model(self, path: Any, device: str = "GPU", **kwargs: Any) -> None:
        """Record the load call."""
        self.loaded = (str(path), device)

    def unload(self) -> None:
        """No-op."""


class _FakeSpec:
    """Stand-in for HybridSpeculativeEngine recording its constructor arguments."""

    def __init__(self, kv_cache_manager: Any = None) -> None:
        self.kv_manager = kv_cache_manager
        self.loaded: Any = None

    def load_model(self, path: Any, config: Any = None, device: str = "GPU", **kwargs: Any) -> None:
        """Record the load call."""
        self.loaded = (str(path), config)

    def unload(self) -> None:
        """No-op."""


@pytest.fixture
def fake_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both engine classes in model_manager and clear the context env."""
    monkeypatch.setattr(mm_module, "OpenVINOGenAIEngine", _FakePlain)
    monkeypatch.setattr(mm_module, "HybridSpeculativeEngine", _FakeSpec)
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)


def test_get_or_load_engine_plain_path_uses_model_config_window(tmp_path: Path, fake_engines: None) -> None:
    """The plain engine gets a KV manager sized from the model config, without loading anything real."""
    cfg = {"max_position_embeddings": 40960, "num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128}
    _model_dir(tmp_path, cfg, name="alpha")
    engine = ModelManager(model_dirs=[tmp_path]).get_or_load_engine("alpha")
    assert engine.kv_manager.max_context_length == 40960
    assert engine.kv_manager.geometry.num_layers == 36


def test_get_or_load_engine_plain_path_env_wins_over_config(tmp_path: Path, fake_engines: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """ARC_CONTEXT_WINDOW overrides the model's own value on the plain path."""
    _model_dir(tmp_path, {"max_position_embeddings": 40960}, name="alpha")
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "20000")
    assert ModelManager(model_dirs=[tmp_path]).get_or_load_engine("alpha").kv_manager.max_context_length == 20000


def test_get_or_load_engine_plain_path_without_config_uses_default(tmp_path: Path, fake_engines: None) -> None:
    """No config.json: 4096 and no geometry."""
    _model_dir(tmp_path, name="alpha")
    engine = ModelManager(model_dirs=[tmp_path]).get_or_load_engine("alpha")
    assert engine.kv_manager.max_context_length == DEFAULT_CONTEXT_WINDOW
    assert engine.kv_manager.geometry is None


def test_get_or_load_engine_speculative_path_uses_target_model_window(tmp_path: Path, fake_engines: None) -> None:
    """The speculative engine's KV manager follows the target model, not the draft."""
    _model_dir(tmp_path, {"max_position_embeddings": 65536}, name="target")
    _model_dir(tmp_path, {"max_position_embeddings": 2048}, name="draft")
    engine = ModelManager(model_dirs=[tmp_path]).get_or_load_engine("target", draft_model_id="draft")
    assert isinstance(engine, _FakeSpec)
    assert engine.kv_manager.max_context_length == 65536


def test_get_or_load_engine_speculative_path_env_applies(tmp_path: Path, fake_engines: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The env override also reaches the speculative engine."""
    _model_dir(tmp_path, name="target")
    _model_dir(tmp_path, name="draft")
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "7777")
    engine = ModelManager(model_dirs=[tmp_path]).get_or_load_engine("target", draft_model_id="draft")
    assert engine.kv_manager.max_context_length == 7777


def test_get_or_load_engine_kv_precision_env_reaches_manager(tmp_path: Path, fake_engines: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """ARC_KV_CACHE_PRECISION is carried to the KV manager so the footprint dtype matches the pipeline."""
    _model_dir(tmp_path, name="alpha")
    monkeypatch.setenv("ARC_KV_CACHE_PRECISION", "u8")
    assert ModelManager(model_dirs=[tmp_path]).get_or_load_engine("alpha").kv_manager.kv_precision == "u8"


@pytest.mark.parametrize("bad", ["0", "abc", "-5", str(MAX_CONTEXT_WINDOW + 1)])
def test_model_manager_init_invalid_env_raises_before_any_model_load(tmp_path: Path, fake_engines: None, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A bad ARC_CONTEXT_WINDOW fails at construction, so server start never reaches a model."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, bad)
    with pytest.raises(ConfigurationError):
        ModelManager(model_dirs=[tmp_path])


# ----- Anthropic route with a real engine and the real KV manager -----


def _route_client(tmp_path: Path, window: int, prompt_tokens: int) -> Any:
    """App whose cached engine is a real OpenVINOGenAIEngine over a fake pipeline of fixed prompt length."""
    from types import SimpleNamespace
    from unittest import mock

    from fastapi.testclient import TestClient

    from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
    from arc_runner.server.app import create_app
    from tests.anthropic_support import CACHE_KEY, MODEL

    pipeline = mock.MagicMock()
    pipeline.get_tokenizer.return_value.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(shape=(1, prompt_tokens)))
    pipeline.generate.return_value = SimpleNamespace(texts=["hi"], perf_metrics=None)
    engine = OpenVINOGenAIEngine(kv_manager=KVCacheManager(max_context_length=window))
    engine._pipeline = pipeline
    engine._is_loaded = True
    engine._model_path = MODEL
    engine._device = "CPU"
    _model_dir(tmp_path, name=MODEL)
    manager = ModelManager(model_dirs=[tmp_path])
    manager._cache[CACHE_KEY] = engine
    return TestClient(create_app(model_manager=manager, debug=True))


def test_anthropic_route_prompt_over_new_window_returns_400_prompt_is_too_long(tmp_path: Path) -> None:
    """Prompt + max_tokens above the NEW window gives 400 'prompt is too long' naming that window."""
    from tests.anthropic_support import body

    client = _route_client(tmp_path, window=65536, prompt_tokens=65000)
    response = client.post("/v1/messages", json=body(max_tokens=600))
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert message.startswith("prompt is too long")
    assert "65600" in message and "65536" in message


def test_anthropic_route_prompt_below_new_window_but_above_old_succeeds(tmp_path: Path) -> None:
    """A prompt far above 4096 that fits the new window is served (the old window would have refused it)."""
    from tests.anthropic_support import body

    client = _route_client(tmp_path, window=65536, prompt_tokens=30000)
    response = client.post("/v1/messages", json=body(max_tokens=600))
    assert response.status_code == 200, response.text
    assert response.json()["type"] == "message"


def test_anthropic_route_exact_fit_succeeds_and_one_over_fails(tmp_path: Path) -> None:
    """prompt + max_tokens == window passes; one more token fails with the prefix."""
    from tests.anthropic_support import body

    assert _route_client(tmp_path / "a", 10000, 9000).post("/v1/messages", json=body(max_tokens=1000)).status_code == 200
    response = _route_client(tmp_path / "b", 10000, 9001).post("/v1/messages", json=body(max_tokens=1000))
    assert response.status_code == 400 and response.json()["error"]["message"].startswith("prompt is too long")
