"""Exhaustive 8-test triad suite for local Ollama model discovery and GGUF bridge.

Validates OCI/Docker distribution v2 manifest parsing, content-addressable
SHA-256 blob resolution, architectural mapping, and error resilience.
"""

import json
from pathlib import Path
import tempfile
import pytest

from arc_runner.hub.ollama_importer import OllamaImporter
from arc_runner.quant.models import OllamaModelManifest
from tests.test_gguf_parser import build_synthetic_gguf


@pytest.fixture
def mock_ollama_dir():
    """Create a temporary mock Ollama home directory structure with sample manifests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        manifests_dir = root / "models" / "manifests" / "registry.ollama.ai" / "library" / "llama3.2"
        blobs_dir = root / "models" / "blobs"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        blobs_dir.mkdir(parents=True, exist_ok=True)

        blob_hash = "74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
        blob_path = blobs_dir / f"sha256-{blob_hash}"
        gguf_bytes = build_synthetic_gguf()
        blob_path.write_bytes(gguf_bytes)

        manifest_data = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": f"sha256:{blob_hash}",
                    "size": len(gguf_bytes),
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                    "size": 128,
                },
            ],
        }

        tag_file = manifests_dir / "1b"
        tag_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        yield root


def test_discover_ollama_manifests(mock_ollama_dir: Path) -> None:
    """Verify OllamaImporter discovers manifests and resolves model layer blobs."""
    importer = OllamaImporter(mock_ollama_dir)
    assert importer.is_available is True

    models = importer.discover_models()
    assert len(models) == 1
    m = models[0]
    assert m.model_name == "llama3.2"
    assert m.tag == "1b"
    assert m.model_blob_path is not None
    assert Path(m.model_blob_path).exists()
    assert m.model_size_bytes > 0


def test_resolve_blob_path_from_digest(mock_ollama_dir: Path) -> None:
    """Verify conversion of SHA-256 digest string into physical content-addressable blob path."""
    importer = OllamaImporter(mock_ollama_dir)
    digest = "sha256:74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    path = importer._resolve_digest_path(digest)
    assert path is not None
    assert path.name == "sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    assert importer._resolve_digest_path("invalid-digest") is None


def test_map_ollama_name_to_openvino_profile(mock_ollama_dir: Path) -> None:
    """Verify mapping of Ollama model tags to official curated OpenVINO INT4 model profiles."""
    importer = OllamaImporter(mock_ollama_dir)

    p1 = importer.map_to_openvino_profile("llama-3.2-1b")
    assert p1 is not None
    assert p1.model_id == "llama-3.2-1b"

    p2 = importer.map_to_openvino_profile("qwen-2.5-7b")
    assert p2 is not None
    assert p2.model_id == "qwen-2.5-7b"

    p3 = importer.map_to_openvino_profile("phi-3.5-mini")
    assert p3 is not None
    assert p3.model_id == "phi-3.5-mini"


def test_missing_ollama_directory_handled_gracefully() -> None:
    """Verify importer returns empty catalog without error when directory is missing."""
    importer = OllamaImporter("C:/nonexistent_ollama_dir_path")
    assert importer.is_available is False
    assert importer.discover_models() == []
    assert importer.find_manifest("llama3.2") is None


def test_corrupted_manifest_json_skips_cleanly(mock_ollama_dir: Path) -> None:
    """Verify malformed manifest JSON files are safely skipped without aborting scan."""
    corrupt_dir = mock_ollama_dir / "models" / "manifests" / "registry.ollama.ai" / "library" / "badmodel"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    bad_file = corrupt_dir / "latest"
    bad_file.write_text("{ this is not valid json : [[", encoding="utf-8")

    importer = OllamaImporter(mock_ollama_dir)
    models = importer.discover_models()
    assert len(models) == 1
    assert models[0].model_name == "llama3.2"


def test_missing_model_layer_in_manifest(mock_ollama_dir: Path) -> None:
    """Verify manifest with no model layer results in None blob path."""
    no_model_dir = mock_ollama_dir / "models" / "manifests" / "registry.ollama.ai" / "library" / "empty"
    no_model_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {"schemaVersion": 2, "layers": [{"mediaType": "application/vnd.ollama.image.license"}]}
    (no_model_dir / "latest").write_text(json.dumps(manifest_data), encoding="utf-8")

    importer = OllamaImporter(mock_ollama_dir)
    m = importer.find_manifest("empty:latest")
    assert m is not None
    assert m.model_blob_path is None
    assert m.model_size_bytes == 0


def test_inspect_ollama_model(mock_ollama_dir: Path) -> None:
    """Verify binary GGUF inspection of a local Ollama model blob."""
    importer = OllamaImporter(mock_ollama_dir)
    summary = importer.inspect_ollama_model("llama3.2:1b")
    assert summary is not None
    assert summary.architecture == "llama"
    assert summary.context_length == 4096
    assert summary.total_parameters > 0


def test_find_manifest_tag_variations(mock_ollama_dir: Path) -> None:
    """Verify finding manifest with exact tag or default 'latest' fallback."""
    importer = OllamaImporter(mock_ollama_dir)
    m_exact = importer.find_manifest("llama3.2:1b")
    assert m_exact is not None
    assert m_exact.tag == "1b"

    m_none = importer.find_manifest("llama3.2:nonexistent")
    assert m_none is None
