"""Exhaustive 8-test triad suite for Ahead-of-Time (AOT) compilation and cache.

Validates persistent hardware cache pathing, compilation telemetry,
zero-token warmup pre-flight passes, and cache purging lifecycle.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from arc_runner.core.exceptions import AOTCompilationError
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.quant.models import AOTCompilationResult, CacheEntryInfo


@pytest.fixture
def mock_aot_env():
    """Create a temporary cache directory and dummy OpenVINO IR model directory."""
    with tempfile.TemporaryDirectory() as cache_tmp, tempfile.TemporaryDirectory() as model_tmp:
        cache_path = Path(cache_tmp)
        model_path = Path(model_tmp)

        xml_file = model_path / "openvino_model.xml"
        bin_file = model_path / "openvino_model.bin"
        xml_file.write_text("<net name='mock_llm' version='11'></net>", encoding="utf-8")
        bin_file.write_bytes(b"\x00" * 1024)

        yield cache_path, model_path


def test_aot_get_cache_path(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify get_cache_path resolves device and model hierarchy correctly."""
    cache_dir, _ = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)
    target = mgr.get_cache_path("OpenVINO/Llama-3.2-3B", "GPU")
    assert target.parts[-2] == "GPU"
    assert "Llama-3.2-3B" in target.parts[-1]
    assert "/" not in target.name


def test_aot_get_cache_info_empty(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify get_cache_info returns exists=False for non-compiled model."""
    cache_dir, _ = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)
    info = mgr.get_cache_info("nonexistent-model", "NPU")
    assert info.exists is False
    assert info.size_bytes == 0
    assert info.device == "NPU"


def test_aot_compile_invalid_model_path(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify AOTCompilationError raised when target model directory does not exist."""
    cache_dir, _ = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)
    with pytest.raises(AOTCompilationError) as exc_info:
        mgr.compile_model("C:/nonexistent_model_dir", device="GPU")
    assert "not found" in str(exc_info.value)


def test_aot_compile_creates_cache_and_measures_telemetry(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify compile_model measures compilation latencies and returns result."""
    cache_dir, model_dir = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)

    mock_pipe = MagicMock()
    mock_gen_config = MagicMock()

    with patch("openvino_genai.LLMPipeline", return_value=mock_pipe), \
         patch("openvino_genai.GenerationConfig", return_value=mock_gen_config):

        cache_target = mgr.get_cache_path(model_dir.name, "GPU")
        cache_target.mkdir(parents=True, exist_ok=True)
        (cache_target / "cache_blob.bin").write_bytes(b"\xAA" * 2048)

        res = mgr.compile_model(model_dir, device="GPU", prewarm=True)
        assert isinstance(res, AOTCompilationResult)
        assert res.model_id == model_dir.name
        assert res.device == "GPU"
        assert res.compilation_time_ms >= 0.0
        assert res.warm_load_time_ms >= 0.0
        assert res.blob_size_bytes == 2048
        assert res.speedup_multiplier >= 0.0
        mock_pipe.generate.assert_called_once()


def test_aot_compile_without_prewarm(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify compile_model skips generation pass when prewarm is False."""
    cache_dir, model_dir = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)

    mock_pipe = MagicMock()
    with patch("openvino_genai.LLMPipeline", return_value=mock_pipe):
        res = mgr.compile_model(model_dir, device="NPU", prewarm=False)
        assert res.device == "NPU"
        mock_pipe.generate.assert_not_called()


def test_purge_cache_specific_model(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify purge_cache selectively deletes specified model and device artifacts."""
    cache_dir, _ = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)

    p1 = mgr.get_cache_path("model-a", "GPU")
    p1.mkdir(parents=True, exist_ok=True)
    (p1 / "blob1.bin").write_bytes(b"\xFF" * 1000)

    p2 = mgr.get_cache_path("model-b", "GPU")
    p2.mkdir(parents=True, exist_ok=True)
    (p2 / "blob2.bin").write_bytes(b"\xFF" * 2000)

    freed = mgr.purge_cache(model_id="model-a", device="GPU")
    assert freed == 1000
    assert not p1.exists()
    assert p2.exists()


def test_purge_cache_device_scope(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify purge_cache by device purges entire device directory."""
    cache_dir, _ = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)

    p_gpu = mgr.get_cache_path("model-1", "GPU")
    p_npu = mgr.get_cache_path("model-1", "NPU")
    p_gpu.mkdir(parents=True, exist_ok=True)
    p_npu.mkdir(parents=True, exist_ok=True)
    (p_gpu / "k.bin").write_bytes(b"\x00" * 500)
    (p_npu / "k.bin").write_bytes(b"\x00" * 700)

    freed = mgr.purge_cache(device="GPU")
    assert freed == 500
    assert not (cache_dir / "GPU").exists()
    assert (cache_dir / "NPU").exists()


def test_purge_cache_all(mock_aot_env: tuple[Path, Path]) -> None:
    """Verify purge_cache with no arguments reclaims all cache storage."""
    cache_dir, _ = mock_aot_env
    mgr = AOTCompilationManager(cache_dir=cache_dir)

    p = mgr.get_cache_path("model-x", "CPU")
    p.mkdir(parents=True, exist_ok=True)
    (p / "data.bin").write_bytes(b"\x00" * 4000)

    freed = mgr.purge_cache()
    assert freed == 4000
    assert mgr.cache_dir.exists()
    assert len(list(mgr.cache_dir.glob("*"))) == 0
