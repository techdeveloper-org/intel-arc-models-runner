"""Building native generation configs must fail loud instead of silently using defaults."""

import unittest.mock as mock
from typing import Any, Callable

import pytest

from arc_runner.core.exceptions import ConfigurationError
from arc_runner.engine.models import GenerationConfig
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine
from arc_runner.engine.speculative_engine import HybridSpeculativeEngine

ENGINES: list = [OpenVINOGenAIEngine, HybridSpeculativeEngine]


@pytest.fixture(params=ENGINES, ids=lambda c: c.__name__)
def build(request: pytest.FixtureRequest) -> Callable[[GenerationConfig], Any]:
    """Return the _build_ov_generation_config method of a fresh engine."""
    return request.param()._build_ov_generation_config


def test_build_ov_generation_config_valid_config_honours_max_new_tokens(build: Callable[[GenerationConfig], Any]) -> None:
    """max_new_tokens and sampling values are present on the built native config."""
    cfg = build(GenerationConfig(max_new_tokens=17, temperature=0.5, top_p=0.8, top_k=7))
    assert cfg is not None
    assert cfg.max_new_tokens == 17
    assert cfg.temperature == pytest.approx(0.5)
    assert cfg.top_p == pytest.approx(0.8)
    assert cfg.top_k == 7


def test_build_ov_generation_config_default_config_builds(build: Callable[[GenerationConfig], Any]) -> None:
    """An empty GenerationConfig() builds a native config with the documented defaults."""
    defaults = GenerationConfig()
    cfg = build(defaults)
    assert cfg.max_new_tokens == defaults.max_new_tokens


@pytest.mark.parametrize("field,value", [("temperature", -0.5), ("top_p", 0.0), ("top_p", 1.5), ("max_new_tokens", 0)])
def test_build_ov_generation_config_invalid_value_raises_configuration_error(
    build: Callable[[GenerationConfig], Any], field: str, value: Any
) -> None:
    """Values that bypass pydantic validation raise ConfigurationError, never return None."""
    bad = GenerationConfig.model_construct(**{**GenerationConfig().model_dump(), field: value})
    with pytest.raises(ConfigurationError) as info:
        build(bad)
    assert field in info.value.message


def test_build_ov_generation_config_library_type_error_raises_and_logs_warning(build: Callable[[GenerationConfig], Any]) -> None:
    """A TypeError from the native binding is logged at WARNING and re-raised as ConfigurationError."""
    bad = GenerationConfig.model_construct(**{**GenerationConfig().model_dump(), "top_k": "x"})
    with mock.patch("arc_runner.engine.openvino_engine.logger") as l1, mock.patch("arc_runner.engine.speculative_engine.logger") as l2:
        with pytest.raises(ConfigurationError):
            build(bad)
    assert l1.warning.called or l2.warning.called
