"""OpenVINO GenAI builds return perf durations as MeanStdPair objects, dicts or plain floats."""

from types import SimpleNamespace

import pytest

from arc_runner.engine import image_engine, video_engine


@pytest.mark.parametrize("module", [image_engine, video_engine])
@pytest.mark.parametrize(
    "value,expected",
    [(SimpleNamespace(mean=12.5, std=1.0), 12.5), ({"mean": 7.0, "std": 0.5}, 7.0), (3.25, 3.25), (4, 4.0), ({}, 0.0)],
)
def test_mean_accepts_object_dict_and_plain_number(module: object, value: object, expected: float) -> None:
    """The three shapes seen on real builds all give the mean in milliseconds."""
    assert module._mean(value) == expected
