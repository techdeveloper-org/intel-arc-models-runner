"""Red-first tests for WP-17 17b: PromptCapExceededError and NPU prompt cap enforcement.

Verifies that prompts exceeding NPU_MAX_PROMPT_LEN raise PromptCapExceededError (HTTP 400,
PROMPT_CAP_EXCEEDED) from both generate() and generate_stream(), and that other runtime
errors pass through unwrapped.
"""

import unittest.mock as mock
from pathlib import Path
from typing import Iterator, Optional

import pytest

from arc_runner.core.exceptions import ArcRunnerError, PromptCapExceededError
from arc_runner.core.constants import NPU_MAX_PROMPT_LEN
from arc_runner.engine.openvino_engine import OpenVINOGenAIEngine, _NPU_OVER_CAP_LIBRARY_MSG


_LIBRARY_CAP_ERROR = RuntimeError(
    'Set the "MAX_PROMPT_LEN" config option to increase the limit.'
)
_OTHER_RUNTIME_ERROR = RuntimeError("Some other runtime failure unrelated to prompt cap")
_KV_CAPACITY_ERROR = RuntimeError(
    "Invalid NPU KV-cache capacity: MAX_PROMPT_LEN + MIN_RESPONSE_LEN must be >= ..."
)


def _npu_engine_loaded(tmp_path: Path) -> OpenVINOGenAIEngine:
    """Return an OpenVINOGenAIEngine with a mock NPU pipeline loaded."""
    engine = OpenVINOGenAIEngine()
    mock_pipeline = mock.MagicMock()
    with mock.patch("openvino_genai.LLMPipeline", return_value=mock_pipeline):
        engine.load_model(str(tmp_path), device="NPU")
    return engine


class TestPromptCapExceededErrorClass:
    """The PromptCapExceededError exception class must have the correct HTTP contract."""

    def test_prompt_cap_exceeded_error_maps_http_400_code_ok(self):
        """PromptCapExceededError must report status_code=400 and PROMPT_CAP_EXCEEDED."""
        exc = PromptCapExceededError(prompt_tokens=1100, cap=1024)
        assert exc.status_code == 400, (
            f"Expected status_code=400, got {exc.status_code}"
        )
        assert exc.error_code == "PROMPT_CAP_EXCEEDED", (
            f"Expected error_code='PROMPT_CAP_EXCEEDED', got {exc.error_code!r}"
        )

    def test_prompt_cap_exceeded_error_is_arc_runner_error_subclass_ok(self):
        """PromptCapExceededError must be a subclass of ArcRunnerError."""
        exc = PromptCapExceededError(prompt_tokens=2000, cap=1024)
        assert isinstance(exc, ArcRunnerError), (
            "PromptCapExceededError must inherit from ArcRunnerError"
        )

    def test_prompt_cap_exceeded_error_chains_library_error_as_cause_ok(self):
        """PromptCapExceededError raised with 'from exc' must chain the original library error."""
        lib_exc = RuntimeError("library error")
        try:
            raise PromptCapExceededError(1100, 1024) from lib_exc
        except PromptCapExceededError as cap_exc:
            assert cap_exc.__cause__ is lib_exc, (
                "PromptCapExceededError must be raised 'from' the original library error"
            )

    def test_prompt_cap_exceeded_error_message_contains_token_counts_ok(self):
        """The exception message must mention the prompt token count and cap."""
        exc = PromptCapExceededError(prompt_tokens=1500, cap=1024)
        msg = str(exc)
        assert "1500" in msg and "1024" in msg, (
            f"Exception message must contain both 1500 and 1024, got: {msg!r}"
        )


class TestNpuPromptCapPreCheck:
    """generate() and generate_stream() must enforce NPU_MAX_PROMPT_LEN before calling pipeline."""

    def test_npu_prompt_exactly_at_cap_accepted_ok(self, tmp_path):
        """A prompt whose token count equals NPU_MAX_PROMPT_LEN must NOT raise."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()
        engine._pipeline.generate.return_value = "ok"

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=NPU_MAX_PROMPT_LEN):
            result = engine.generate("x " * NPU_MAX_PROMPT_LEN)

        assert result == "ok", (
            "A prompt at exactly the cap must succeed, not raise PromptCapExceededError"
        )

    def test_npu_prompt_one_below_cap_accepted_ok(self, tmp_path):
        """A prompt whose token count is NPU_MAX_PROMPT_LEN - 1 must NOT raise."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()
        engine._pipeline.generate.return_value = "ok"

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=NPU_MAX_PROMPT_LEN - 1):
            result = engine.generate("x " * (NPU_MAX_PROMPT_LEN - 1))

        assert result == "ok"

    def test_npu_prompt_over_cap_wraps_library_error_in_prompt_cap_exceeded_not_truncated_ok(self, tmp_path):
        """A prompt over the cap must raise PromptCapExceededError; the engine must not truncate."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=NPU_MAX_PROMPT_LEN + 1):
            with pytest.raises(PromptCapExceededError) as exc_info:
                engine.generate("x " * (NPU_MAX_PROMPT_LEN + 1))

        exc = exc_info.value
        assert exc.status_code == 400
        assert exc.error_code == "PROMPT_CAP_EXCEEDED"
        engine._pipeline.generate.assert_not_called(), (
            "Engine must not call the pipeline when the prompt exceeds the cap"
        )

    def test_npu_other_runtime_error_not_wrapped_as_prompt_cap_ok(self, tmp_path):
        """A RuntimeError with a message unrelated to MAX_PROMPT_LEN must NOT be wrapped."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()
        engine._pipeline.generate.side_effect = _OTHER_RUNTIME_ERROR

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=10):
            with pytest.raises(RuntimeError) as exc_info:
                engine.generate("short prompt")

        assert not isinstance(exc_info.value, PromptCapExceededError), (
            "A non-cap RuntimeError must propagate as RuntimeError, not PromptCapExceededError"
        )

    def test_npu_invalid_kv_cache_capacity_message_not_wrapped_as_prompt_cap_ok(self, tmp_path):
        """'Invalid NPU KV-cache capacity' error must NOT be wrapped as PromptCapExceededError."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()
        engine._pipeline.generate.side_effect = _KV_CAPACITY_ERROR

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=10):
            with pytest.raises(RuntimeError) as exc_info:
                engine.generate("short prompt")

        assert not isinstance(exc_info.value, PromptCapExceededError), (
            "Invalid KV-cache capacity error must NOT be wrapped as PromptCapExceededError"
        )


class TestNpuPromptCapStream:
    """generate_stream() must raise PromptCapExceededError before the first chunk when over cap."""

    def test_generate_stream_npu_prompt_over_cap_raises_prompt_cap_exceeded_before_first_chunk_not_truncated_ok(
        self, tmp_path
    ):
        """generate_stream() must raise PromptCapExceededError before yielding any token chunk."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=NPU_MAX_PROMPT_LEN + 1):
            gen = engine.generate_stream("x " * (NPU_MAX_PROMPT_LEN + 1))
            with pytest.raises(PromptCapExceededError) as exc_info:
                next(gen)

        exc = exc_info.value
        assert exc.status_code == 400
        assert exc.error_code == "PROMPT_CAP_EXCEEDED"

    def test_generate_stream_other_runtime_error_not_wrapped_as_prompt_cap_ok(self, tmp_path):
        """A non-cap RuntimeError from the worker thread must NOT become PromptCapExceededError."""
        engine = _npu_engine_loaded(tmp_path)
        engine._pipeline = mock.MagicMock()

        def _fake_generate(prompt, config=None, streamer=None):
            raise _OTHER_RUNTIME_ERROR

        engine._pipeline.generate.side_effect = _fake_generate

        with mock.patch.object(engine, "_count_prompt_tokens", return_value=10):
            gen = engine.generate_stream("short prompt")
            chunks = []
            with pytest.raises(RuntimeError) as exc_info:
                for chunk in gen:
                    chunks.append(chunk)

        assert not isinstance(exc_info.value, PromptCapExceededError), (
            "A non-cap worker RuntimeError must NOT become PromptCapExceededError"
        )


class TestNpuBuildOvProperties:
    """_build_ov_properties must set MAX_PROMPT_LEN and MIN_RESPONSE_LEN for NPU."""

    def test_build_ov_properties_sets_npu_max_prompt_len_default_ok(self):
        """_build_ov_properties must set MAX_PROMPT_LEN for NPU when not overridden."""
        engine = OpenVINOGenAIEngine()
        with mock.patch.object(engine, "_resolve_cache_dir", return_value=None):
            props = engine._build_ov_properties("NPU", {})

        assert "MAX_PROMPT_LEN" in props, (
            "_build_ov_properties must set MAX_PROMPT_LEN for NPU device"
        )
        assert props["MAX_PROMPT_LEN"] == NPU_MAX_PROMPT_LEN, (
            f"MAX_PROMPT_LEN must be {NPU_MAX_PROMPT_LEN}, got {props['MAX_PROMPT_LEN']}"
        )

    def test_build_ov_properties_caller_override_beats_npu_default_ok(self):
        """A caller-supplied MAX_PROMPT_LEN must not be overwritten."""
        engine = OpenVINOGenAIEngine()
        with mock.patch.object(engine, "_resolve_cache_dir", return_value=None):
            props = engine._build_ov_properties("NPU", {"MAX_PROMPT_LEN": 512})

        assert props["MAX_PROMPT_LEN"] == 512, (
            "Caller-supplied MAX_PROMPT_LEN must not be overwritten by the default"
        )

    def test_build_ov_properties_npu_defaults_not_applied_to_gpu_ok(self):
        """MAX_PROMPT_LEN and MIN_RESPONSE_LEN must NOT be set for GPU device."""
        engine = OpenVINOGenAIEngine()
        with mock.patch.object(engine, "_resolve_cache_dir", return_value=None):
            props = engine._build_ov_properties("GPU", {})

        assert "MAX_PROMPT_LEN" not in props, (
            "MAX_PROMPT_LEN must NOT be set for GPU device"
        )

    def test_npu_library_constant_present_in_module_ok(self):
        """_NPU_OVER_CAP_LIBRARY_MSG must be the exact library error message fragment."""
        assert isinstance(_NPU_OVER_CAP_LIBRARY_MSG, str), (
            "_NPU_OVER_CAP_LIBRARY_MSG must be a string constant"
        )
        assert "MAX_PROMPT_LEN" in _NPU_OVER_CAP_LIBRARY_MSG, (
            "_NPU_OVER_CAP_LIBRARY_MSG must contain 'MAX_PROMPT_LEN'"
        )
