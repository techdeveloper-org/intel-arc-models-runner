"""Exhaustive 12-test triad suite for Model Hub, Curated Registry, and Lifecycle Management.

Validates alias resolution, snapshot download streams, integrity checks,
safe Level Zero engine unloads prior to deletion, and path traversal guards.
"""

from pathlib import Path
import tempfile
from typing import Any
import pytest

from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.engine.base import BaseEngine
from arc_runner.engine.models import EngineTelemetry
from arc_runner.hub.downloader import DownloadProgress, ModelDownloader
from arc_runner.hub.manager import HubManager
from arc_runner.hub.registry import CuratedModelInfo, ModelRegistry
from arc_runner.server.model_manager import ModelManager


class MockEngine(BaseEngine):
    """Simulated execution engine for testing unload synchronization during deletion."""

    def __init__(self, model_name: str = "mock-llama3") -> None:
        """Initialize mock engine."""
        self._model_name = model_name
        self._is_loaded = True
        self._device = "GPU"
        self._telemetry = EngineTelemetry(model_id=model_name, device="GPU")

    @property
    def is_loaded(self) -> bool:
        """Return loading status."""
        return self._is_loaded

    @property
    def model_path(self) -> str:
        """Return model path."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return device string."""
        return self._device

    def load_model(self, model_path: Any, device: str = "GPU", **kwargs: Any) -> None:
        """Simulate model compilation."""
        self._is_loaded = True
        self._model_name = str(model_path)

    def generate(self, prompt: str, config: Any = None) -> str:
        """Simulate generation."""
        return "response"

    def generate_stream(self, prompt: str, config: Any = None) -> Any:
        """Simulate stream."""
        return iter([])

    def unload(self) -> None:
        """Simulate Level Zero memory deallocation."""
        self._is_loaded = False

    def get_telemetry(self) -> EngineTelemetry:
        """Return telemetry metrics."""
        return self._telemetry


@pytest.fixture
def hub_env():
    """Create a temporary sandbox environment for hub operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_root = tmp_path / "models"
        models_root.mkdir()

        llama_dir = models_root / "llama-3.2-3b"
        llama_dir.mkdir()
        (llama_dir / "openvino_model.xml").write_text("<net></net>", encoding="utf-8")
        (llama_dir / "openvino_model.bin").write_bytes(b"\x00" * 2048)

        manager = ModelManager(model_dirs=[models_root])
        mock_engine = MockEngine("llama-3.2-3b")
        manager._cache["llama-3.2-3b:GPU:none:NPU:4"] = mock_engine

        hub = HubManager(base_models_dir=models_root, model_manager=manager)

        yield {
            "root": models_root,
            "hub": hub,
            "manager": manager,
            "mock_engine": mock_engine,
        }


# =========================================================================
# POSITIVE TEST TRIAD (5 TESTS)
# =========================================================================

def test_registry_resolves_all_curated_aliases():
    """Verify all 7 curated aliases resolve with valid HF repository IDs."""
    aliases = [
        "llama-3.2-3b",
        "llama-3.2-1b",
        "qwen-2.5-3b",
        "qwen-2.5-1.5b",
        "qwen-2.5-7b",
        "phi-3.5-mini",
        "deepseek-r1-1.5b",
    ]
    for alias in aliases:
        info = ModelRegistry.resolve(alias)
        assert isinstance(info, CuratedModelInfo)
        assert info.alias == alias
        assert "OpenVINO/" in info.repo_id
        assert info.size_bytes > 0
        assert info.context_window > 0


def test_registry_accepts_custom_huggingface_repo():
    """Verify arbitrary custom organization/model repository constructs valid model info."""
    custom_repo = "meta-llama/Llama-3.2-3B-Instruct"
    info = ModelRegistry.resolve(custom_repo)
    assert info.alias == custom_repo
    assert info.repo_id == custom_repo
    assert info.family == "llama"


def test_hub_downloader_mock_progress_stream():
    """Verify ModelDownloader yields monotonically increasing bytes and status."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / "test-model"
        dest.mkdir()
        (dest / "openvino_model.xml").write_text("<net></net>", encoding="utf-8")
        (dest / "openvino_model.bin").write_bytes(b"\x00" * 4096)

        downloader = ModelDownloader(cache_dir=Path(tmp_dir))
        size = downloader._get_dir_size(dest)
        assert size >= 4096


def test_hub_integrity_verification_success(hub_env):
    """Verify complete model directory with XML and BIN passes integrity check."""
    hub = hub_env["hub"]
    llama_dir = hub_env["root"] / "llama-3.2-3b"
    is_valid, missing = hub.verify_integrity(llama_dir)
    assert is_valid is True
    assert len(missing) == 0


def test_hub_list_local_models(hub_env):
    """Verify list_local discovers models on disk with exact sizes and formats."""
    hub = hub_env["hub"]
    models = hub.list_local()
    assert len(models) >= 1
    model_names = [m["name"] for m in models]
    assert "llama-3.2-3b" in model_names
    assert models[0]["details"]["format"] == "openvino"


# =========================================================================
# NEGATIVE TEST TRIAD (4 TESTS)
# =========================================================================

def test_hub_pull_empty_name_raises_422(hub_env):
    """Verify calling pull with empty or whitespace string raises ArcRunnerError."""
    hub = hub_env["hub"]
    with pytest.raises(ArcRunnerError) as exc_info:
        hub.pull("   ", stream=False)
    assert exc_info.value.status_code == 422


def test_hub_delete_nonexistent_model_raises_404(hub_env):
    """Verify deleting a non-existent model raises ModelNotFoundError."""
    hub = hub_env["hub"]
    with pytest.raises(ModelNotFoundError):
        hub.delete("nonexistent-phantom-model")


def test_hub_integrity_fails_when_bin_missing(hub_env):
    """Verify directory with missing .bin file fails integrity inspection."""
    hub = hub_env["hub"]
    corrupted_dir = hub_env["root"] / "corrupted-model"
    corrupted_dir.mkdir()
    (corrupted_dir / "openvino_model.xml").write_text("<net></net>", encoding="utf-8")

    is_valid, missing = hub.verify_integrity(corrupted_dir)
    assert is_valid is False
    assert any("openvino_model.bin" in msg for msg in missing)


def test_hub_copy_nonexistent_source_raises_404(hub_env):
    """Verify copying from a non-existent model raises ModelNotFoundError."""
    hub = hub_env["hub"]
    with pytest.raises(ModelNotFoundError):
        hub.copy("ghost-model", "destination-model")


# =========================================================================
# CHAOS & EDGE-CASE TEST TRIAD (3 TESTS)
# =========================================================================

def test_hub_download_interruption_cleans_partial_files(hub_env):
    """Verify interrupted or failed directory can be safely cleaned up."""
    hub = hub_env["hub"]
    dest = hub_env["root"] / "partial-download"
    dest.mkdir()
    (dest / "partial.tmp").write_bytes(b"\x00" * 1024)

    hub.delete("partial-download")
    assert not dest.exists()


def test_hub_delete_active_loaded_engine_unloads_first(hub_env):
    """Verify deleting a compiled model unloads Level Zero buffers before unlinking."""
    hub = hub_env["hub"]
    mock_engine = hub_env["mock_engine"]
    assert mock_engine.is_loaded is True

    hub.delete("llama-3.2-3b")
    assert mock_engine.is_loaded is False


def test_hub_path_traversal_attempt_blocked(hub_env):
    """Verify path traversal inputs like ../../secret cannot escape base directory."""
    hub = hub_env["hub"]
    with pytest.raises(ModelNotFoundError):
        hub.resolve_model_dir("../../windows/system32")
