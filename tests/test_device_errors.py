"""Red-first tests for WP-07: classify_device_error() and fallback handler behaviour.

Tests verify that:
  - classify_device_error() returns the correct category for each error class.
  - DEVICE_LOST wins over OOM when both markers are present in the cause chain.
  - The four fallback handlers in image/video/vision/embedding engines raise ModelLoadError
    on DEVICE_LOST and OOM, and fall back as before on DRIVER_ERROR and UNKNOWN.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest

from arc_runner.core.exceptions import ModelLoadError


class TestClassifyDeviceError:
    """Unit tests for arc_runner.engine.device_errors.classify_device_error."""

    def _fn(self):
        from arc_runner.engine.device_errors import classify_device_error
        return classify_device_error

    def test_cl_out_of_resources_returns_device_lost(self):
        """CL_OUT_OF_RESOURCES substring maps to DEVICE_LOST."""
        fn = self._fn()
        exc = RuntimeError("CL_OUT_OF_RESOURCES: GPU hang detected")
        assert fn(exc) == "DEVICE_LOST"

    def test_device_lost_substring_returns_device_lost(self):
        """'DEVICE_LOST' substring maps to DEVICE_LOST."""
        fn = self._fn()
        exc = RuntimeError("OpenCL: DEVICE_LOST after shader compile")
        assert fn(exc) == "DEVICE_LOST"

    def test_cl_mem_object_allocation_failure_returns_device_lost(self):
        """CL_MEM_OBJECT_ALLOCATION_FAILURE maps to DEVICE_LOST."""
        fn = self._fn()
        exc = RuntimeError("CL_MEM_OBJECT_ALLOCATION_FAILURE on buffer alloc")
        assert fn(exc) == "DEVICE_LOST"

    def test_cl_out_of_host_memory_returns_device_lost(self):
        """CL_OUT_OF_HOST_MEMORY maps to DEVICE_LOST."""
        fn = self._fn()
        exc = RuntimeError("CL_OUT_OF_HOST_MEMORY: host allocation failed")
        assert fn(exc) == "DEVICE_LOST"

    def test_memory_error_returns_oom(self):
        """MemoryError type maps to OOM."""
        fn = self._fn()
        exc = MemoryError("not enough memory")
        assert fn(exc) == "OOM"

    def test_device_not_found_is_not_registered_returns_driver_error(self):
        """DeviceNotFoundError-like text with 'is not registered' maps to DRIVER_ERROR."""
        fn = self._fn()
        exc = RuntimeError("DeviceNotFoundError: GPU is not registered in the system")
        assert fn(exc) == "DRIVER_ERROR"

    def test_not_supported_returns_driver_error(self):
        """'not supported' substring maps to DRIVER_ERROR."""
        fn = self._fn()
        exc = RuntimeError("Operation not supported on this device")
        assert fn(exc) == "DRIVER_ERROR"

    def test_unknown_exception_returns_unknown(self):
        """Generic exception with no matching substring returns UNKNOWN."""
        fn = self._fn()
        exc = ValueError("something unrelated happened")
        assert fn(exc) == "UNKNOWN"

    def test_device_lost_wins_over_oom_in_cause_chain(self):
        """DEVICE_LOST classification beats OOM when both are in the cause chain."""
        fn = self._fn()
        inner = MemoryError("OOM on device buffer")
        outer = RuntimeError("CL_OUT_OF_RESOURCES: context lost")
        outer.__cause__ = inner
        assert fn(outer) == "DEVICE_LOST"

    def test_cause_chain_is_walked(self):
        """Exception cause chain is walked to find DEVICE_LOST in a wrapped exception."""
        fn = self._fn()
        root = RuntimeError("CL_MEM_OBJECT_ALLOCATION_FAILURE")
        wrapper = RuntimeError("pipeline compile failed")
        wrapper.__cause__ = root
        assert fn(wrapper) == "DEVICE_LOST"

    def test_memory_error_in_cause_chain_returns_oom(self):
        """MemoryError buried as cause of a generic wrapper returns OOM."""
        fn = self._fn()
        inner = MemoryError("out of vram")
        outer = RuntimeError("driver error wrapper")
        outer.__cause__ = inner
        assert fn(outer) == "OOM"

    def test_pure_function_has_no_side_effects(self):
        """Calling classify_device_error twice with same exception returns same result."""
        fn = self._fn()
        exc = RuntimeError("CL_OUT_OF_RESOURCES")
        result1 = fn(exc)
        result2 = fn(exc)
        assert result1 == result2 == "DEVICE_LOST"


class TestImageEngineFallbackOnDeviceLost:
    """GPU->CPU fallback in ImageEngine raises ModelLoadError on DEVICE_LOST."""

    def test_device_lost_raises_model_load_error_not_fallback(self, tmp_path):
        """DEVICE_LOST on GPU must raise ModelLoadError instead of falling back to CPU."""
        (tmp_path / "model_index.json").write_text("{}")

        with mock.patch.dict("sys.modules", {"openvino_genai": mock.MagicMock(),
                                              "numpy": mock.MagicMock(),
                                              "PIL": mock.MagicMock(),
                                              "PIL.Image": mock.MagicMock()}):
            import openvino_genai as genai_mock
            bad_exc = RuntimeError("CL_OUT_OF_RESOURCES: GPU context lost")
            genai_mock.Text2ImagePipeline.side_effect = [bad_exc, mock.MagicMock()]

            from arc_runner.engine.image_engine import ImageEngine, ImageEngineConfig
            config = ImageEngineConfig(device="GPU")
            engine = ImageEngine.__new__(ImageEngine)
            engine._config = config
            engine._model_path = None
            engine._active_device = "GPU"
            engine._pipeline = None

            with pytest.raises(ModelLoadError):
                engine.load_model(str(tmp_path), device="GPU")

            assert genai_mock.Text2ImagePipeline.call_count == 1

    def test_oom_raises_model_load_error_not_fallback(self, tmp_path):
        """OOM on GPU must raise ModelLoadError instead of falling back to CPU."""
        (tmp_path / "model_index.json").write_text("{}")

        with mock.patch.dict("sys.modules", {"openvino_genai": mock.MagicMock(),
                                              "numpy": mock.MagicMock(),
                                              "PIL": mock.MagicMock(),
                                              "PIL.Image": mock.MagicMock()}):
            import openvino_genai as genai_mock
            genai_mock.Text2ImagePipeline.reset_mock()
            genai_mock.Text2ImagePipeline.side_effect = [MemoryError("OOM"), mock.MagicMock()]

            from arc_runner.engine.image_engine import ImageEngine, ImageEngineConfig
            config = ImageEngineConfig(device="GPU")
            engine = ImageEngine.__new__(ImageEngine)
            engine._config = config
            engine._model_path = None
            engine._active_device = "GPU"
            engine._pipeline = None

            with pytest.raises(ModelLoadError):
                engine.load_model(str(tmp_path), device="GPU")

            assert genai_mock.Text2ImagePipeline.call_count == 1

    def test_driver_error_still_falls_back_to_cpu(self, tmp_path):
        """DRIVER_ERROR / UNKNOWN on GPU still falls back to CPU as before WP-07."""
        (tmp_path / "model_index.json").write_text("{}")

        with mock.patch.dict("sys.modules", {"openvino_genai": mock.MagicMock(),
                                              "numpy": mock.MagicMock(),
                                              "PIL": mock.MagicMock(),
                                              "PIL.Image": mock.MagicMock()}):
            import openvino_genai as genai_mock
            genai_mock.Text2ImagePipeline.reset_mock()
            cpu_pipeline = mock.MagicMock()
            driver_exc = RuntimeError("not supported on this GPU model")
            genai_mock.Text2ImagePipeline.side_effect = [driver_exc, cpu_pipeline]

            from arc_runner.engine.image_engine import ImageEngine, ImageEngineConfig
            config = ImageEngineConfig(device="GPU")
            engine = ImageEngine.__new__(ImageEngine)
            engine._config = config
            engine._model_path = None
            engine._active_device = "GPU"
            engine._pipeline = None

            engine.load_model(str(tmp_path), device="GPU")

            assert engine._active_device == "CPU"
            assert engine._pipeline is cpu_pipeline


class TestVideoEngineFallbackOnDeviceLost:
    """GPU->CPU fallback in VideoEngine raises ModelLoadError on DEVICE_LOST."""

    def test_device_lost_raises_model_load_error_not_fallback(self, tmp_path):
        """DEVICE_LOST on GPU must raise ModelLoadError in VideoEngine."""
        (tmp_path / "model_index.json").write_text("{}")

        with mock.patch.dict("sys.modules", {"openvino_genai": mock.MagicMock(),
                                              "numpy": mock.MagicMock()}):
            import openvino_genai as genai_mock
            bad_exc = RuntimeError("CL_OUT_OF_RESOURCES: context lost")
            genai_mock.Text2VideoPipeline.side_effect = [bad_exc, mock.MagicMock()]

            from arc_runner.engine.video_engine import VideoEngine, VideoEngineConfig
            config = VideoEngineConfig(device="GPU")
            engine = VideoEngine.__new__(VideoEngine)
            engine._config = config
            engine._model_path = None
            engine._active_device = "GPU"
            engine._pipeline = None

            with pytest.raises(ModelLoadError):
                engine.load_model(str(tmp_path), device="GPU")

            assert genai_mock.Text2VideoPipeline.call_count == 1

    def test_oom_raises_model_load_error_not_fallback(self, tmp_path):
        """OOM on GPU must raise ModelLoadError in VideoEngine."""
        (tmp_path / "model_index.json").write_text("{}")

        with mock.patch.dict("sys.modules", {"openvino_genai": mock.MagicMock(),
                                              "numpy": mock.MagicMock()}):
            import openvino_genai as genai_mock
            genai_mock.Text2VideoPipeline.reset_mock()
            genai_mock.Text2VideoPipeline.side_effect = [MemoryError("OOM"), mock.MagicMock()]

            from arc_runner.engine.video_engine import VideoEngine, VideoEngineConfig
            config = VideoEngineConfig(device="GPU")
            engine = VideoEngine.__new__(VideoEngine)
            engine._config = config
            engine._model_path = None
            engine._active_device = "GPU"
            engine._pipeline = None

            with pytest.raises(ModelLoadError):
                engine.load_model(str(tmp_path), device="GPU")

    def test_driver_error_still_falls_back_to_cpu(self, tmp_path):
        """DRIVER_ERROR on GPU still falls back to CPU in VideoEngine."""
        (tmp_path / "model_index.json").write_text("{}")

        with mock.patch.dict("sys.modules", {"openvino_genai": mock.MagicMock(),
                                              "numpy": mock.MagicMock()}):
            import openvino_genai as genai_mock
            genai_mock.Text2VideoPipeline.reset_mock()
            cpu_pipeline = mock.MagicMock()
            driver_exc = RuntimeError("not supported on this hardware")
            genai_mock.Text2VideoPipeline.side_effect = [driver_exc, cpu_pipeline]

            from arc_runner.engine.video_engine import VideoEngine, VideoEngineConfig
            config = VideoEngineConfig(device="GPU")
            engine = VideoEngine.__new__(VideoEngine)
            engine._config = config
            engine._model_path = None
            engine._active_device = "GPU"
            engine._pipeline = None

            engine.load_model(str(tmp_path), device="GPU")

            assert engine._active_device == "CPU"


class TestVisionEngineFallbackOnDeviceLost:
    """GPU->CPU fallback in VisionEngine raises ModelLoadError on DEVICE_LOST."""

    def test_device_lost_raises_model_load_error_not_fallback(self, tmp_path):
        """DEVICE_LOST on GPU must raise ModelLoadError in VisionEngine."""
        (tmp_path / "openvino_model.xml").touch()

        with mock.patch.dict("sys.modules", {
            "openvino_genai": mock.MagicMock(),
            "PIL": mock.MagicMock(),
            "PIL.Image": mock.MagicMock(),
            "numpy": mock.MagicMock(),
        }):
            import openvino_genai as genai_mock
            bad_exc = RuntimeError("CL_OUT_OF_RESOURCES: GPU context lost")
            genai_mock.VLMPipeline.side_effect = [bad_exc, mock.MagicMock()]

            from arc_runner.engine import vision_engine
            engine = vision_engine.VisionEngine.__new__(vision_engine.VisionEngine)
            engine._model_path = tmp_path
            engine._device = "GPU"
            engine._patch_size = 14
            engine._pipeline = None

            with pytest.raises(ModelLoadError):
                engine._initialize_pipeline()

            assert genai_mock.VLMPipeline.call_count == 1

    def test_driver_error_still_falls_back_to_cpu(self, tmp_path):
        """DRIVER_ERROR on GPU still falls back to CPU in VisionEngine."""
        (tmp_path / "openvino_model.xml").touch()

        with mock.patch.dict("sys.modules", {
            "openvino_genai": mock.MagicMock(),
            "PIL": mock.MagicMock(),
            "PIL.Image": mock.MagicMock(),
            "numpy": mock.MagicMock(),
        }):
            import openvino_genai as genai_mock
            genai_mock.VLMPipeline.reset_mock()
            cpu_pipeline = mock.MagicMock()
            driver_exc = RuntimeError("not supported on this GPU")
            genai_mock.VLMPipeline.side_effect = [driver_exc, cpu_pipeline]

            from arc_runner.engine import vision_engine
            engine = vision_engine.VisionEngine.__new__(vision_engine.VisionEngine)
            engine._model_path = tmp_path
            engine._device = "GPU"
            engine._patch_size = 14
            engine._pipeline = None

            engine._initialize_pipeline()

            assert engine._device == "CPU"


class TestEmbeddingEngineFallbackOnDeviceLost:
    """NPU->GPU fallback in EmbeddingEngine raises ModelLoadError on DEVICE_LOST."""

    def test_device_lost_raises_model_load_error_not_fallback(self, tmp_path):
        """DEVICE_LOST on NPU must raise ModelLoadError in EmbeddingEngine."""
        (tmp_path / "openvino_model.xml").touch()

        mock_genai = mock.MagicMock()
        mock_genai.TextEmbeddingPipeline.Config.return_value = mock.MagicMock()
        mock_genai.TextEmbeddingPipeline.PoolingType.MEAN = "MEAN"

        bad_exc = RuntimeError("CL_OUT_OF_RESOURCES: NPU context lost")
        mock_genai.TextEmbeddingPipeline.side_effect = [bad_exc, mock.MagicMock()]

        with mock.patch.dict("sys.modules", {"openvino_genai": mock_genai}):
            from arc_runner.engine.embedding_engine import EmbeddingEngine
            engine = EmbeddingEngine.__new__(EmbeddingEngine)
            engine._model_path = tmp_path
            engine._device = "NPU"
            engine._normalize = True
            engine._max_length = 512
            engine._pipeline = None
            engine._dimension = 0

            with pytest.raises(ModelLoadError):
                engine._initialize_pipeline()

            assert mock_genai.TextEmbeddingPipeline.call_count == 1

    def test_oom_raises_model_load_error_not_fallback(self, tmp_path):
        """OOM on NPU must raise ModelLoadError in EmbeddingEngine."""
        (tmp_path / "openvino_model.xml").touch()

        mock_genai = mock.MagicMock()
        mock_genai.TextEmbeddingPipeline.Config.return_value = mock.MagicMock()
        mock_genai.TextEmbeddingPipeline.PoolingType.MEAN = "MEAN"
        mock_genai.TextEmbeddingPipeline.side_effect = [MemoryError("OOM"), mock.MagicMock()]

        with mock.patch.dict("sys.modules", {"openvino_genai": mock_genai}):
            from arc_runner.engine.embedding_engine import EmbeddingEngine
            engine = EmbeddingEngine.__new__(EmbeddingEngine)
            engine._model_path = tmp_path
            engine._device = "NPU"
            engine._normalize = True
            engine._max_length = 512
            engine._pipeline = None
            engine._dimension = 0

            with pytest.raises(ModelLoadError):
                engine._initialize_pipeline()

    def test_driver_error_still_falls_back_to_gpu(self, tmp_path):
        """DRIVER_ERROR on NPU still falls back to GPU in EmbeddingEngine."""
        (tmp_path / "openvino_model.xml").touch()

        mock_genai = mock.MagicMock()
        mock_genai.TextEmbeddingPipeline.Config.return_value = mock.MagicMock()
        mock_genai.TextEmbeddingPipeline.PoolingType.MEAN = "MEAN"
        gpu_pipeline = mock.MagicMock()
        gpu_pipeline.embed_query.return_value = [0.1, 0.2, 0.3]
        driver_exc = RuntimeError("NPU is not registered in this system")
        mock_genai.TextEmbeddingPipeline.side_effect = [driver_exc, gpu_pipeline]

        with mock.patch.dict("sys.modules", {"openvino_genai": mock_genai}):
            from arc_runner.engine.embedding_engine import EmbeddingEngine
            engine = EmbeddingEngine.__new__(EmbeddingEngine)
            engine._model_path = tmp_path
            engine._device = "NPU"
            engine._normalize = True
            engine._max_length = 512
            engine._pipeline = None
            engine._dimension = 0

            engine._initialize_pipeline()

            assert engine._device == "GPU"
            assert engine._pipeline is gpu_pipeline
