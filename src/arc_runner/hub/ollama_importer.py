"""Local Ollama model discovery and GGUF bridge importer.

Parses local OCI/Docker distribution manifests and resolves layer digests
to content-addressable GGUF blobs for execution on Intel Arc GPUs and NPUs.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from arc_runner.core.constants import DEFAULT_OLLAMA_DIR
from arc_runner.core.logging import get_logger
from arc_runner.hub.registry import CuratedModelInfo, ModelRegistry
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.models import GGUFSummary, OllamaModelManifest

logger = get_logger(__name__)


class OllamaImporter:
    """Discovers, parses, and bridges local Ollama models into arc-runner."""

    def __init__(self, ollama_dir: Optional[Union[Path, str]] = None) -> None:
        """Initialize Ollama importer targeting local installation directory.

        Args:
            ollama_dir: Base directory containing Ollama models and manifests.
        """
        self.ollama_dir = Path(ollama_dir) if ollama_dir else DEFAULT_OLLAMA_DIR
        self.manifests_dir = self.ollama_dir / "models" / "manifests"
        self.blobs_dir = self.ollama_dir / "models" / "blobs"

    @property
    def is_available(self) -> bool:
        """Return True if local Ollama manifest directory exists on host."""
        return self.manifests_dir.exists() and self.manifests_dir.is_dir()

    def discover_models(self) -> list[OllamaModelManifest]:
        """Scan local Ollama manifests and return parsed model descriptors.

        Returns:
            List of OllamaModelManifest objects detailing models and blob paths.
        """
        if not self.is_available:
            logger.debug(f"Ollama manifest directory not found at {self.manifests_dir}")
            return []

        models: list[OllamaModelManifest] = []
        for file_path in self.manifests_dir.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                rel_parts = file_path.relative_to(self.manifests_dir).parts
                if len(rel_parts) < 2:
                    continue

                tag = rel_parts[-1]
                model_name = rel_parts[-2]

                content = file_path.read_text(encoding="utf-8")
                manifest_data = json.loads(content)
                layers = manifest_data.get("layers", [])

                model_digest = ""
                model_size = 0
                template_text = ""

                for layer in layers:
                    media_type = layer.get("mediaType", "")
                    if "model" in media_type:
                        model_digest = layer.get("digest", "")
                        model_size = layer.get("size", 0)
                    elif "template" in media_type:
                        template_digest = layer.get("digest", "")
                        template_blob = self._resolve_digest_path(template_digest)
                        if template_blob and template_blob.exists():
                            try:
                                template_text = template_blob.read_text(encoding="utf-8", errors="replace")
                            except Exception:
                                pass

                blob_path = self._resolve_digest_path(model_digest)
                blob_str = str(blob_path) if blob_path and blob_path.exists() else None

                models.append(
                    OllamaModelManifest(
                        model_name=model_name,
                        tag=tag,
                        manifest_path=str(file_path),
                        model_digest=model_digest,
                        model_blob_path=blob_str,
                        model_size_bytes=model_size,
                        system_template=template_text,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse Ollama manifest at '{file_path}': {e}")
                continue

        return models

    def _resolve_digest_path(self, digest: str) -> Optional[Path]:
        """Convert SHA-256 digest string into physical content-addressable blob path.

        Args:
            digest: Digest string formatted as 'sha256:<hash>'.

        Returns:
            Resolved Path inside blobs directory or None if invalid.
        """
        if not digest or not digest.startswith("sha256:"):
            return None
        hash_suffix = digest.split(":", 1)[1]
        return self.blobs_dir / f"sha256-{hash_suffix}"

    def find_manifest(self, model_identifier: str) -> Optional[OllamaModelManifest]:
        """Locate specific Ollama model manifest by name or name:tag string.

        Args:
            model_identifier: String formatted as 'name' or 'name:tag'.

        Returns:
            Matching OllamaModelManifest if discovered, else None.
        """
        target_name = model_identifier
        target_tag = "latest"
        if ":" in model_identifier:
            target_name, target_tag = model_identifier.split(":", 1)

        for m in self.discover_models():
            if m.model_name.lower() == target_name.lower():
                if m.tag.lower() == target_tag.lower() or target_tag == "latest":
                    return m
        return None

    def inspect_ollama_model(self, model_identifier: str) -> Optional[GGUFSummary]:
        """Inspect binary GGUF container metadata for a local Ollama model.

        Args:
            model_identifier: String formatted as 'name' or 'name:tag'.

        Returns:
            GGUFSummary detailing architectural specifications, or None if blob missing.
        """
        manifest = self.find_manifest(model_identifier)
        if not manifest or not manifest.model_blob_path:
            return None
        parser = GGUFParser(manifest.model_blob_path)
        return parser.get_summary()

    def map_to_openvino_profile(self, model_identifier: str) -> Optional[CuratedModelInfo]:
        """Map local Ollama model to an optimized official OpenVINO IR model profile.

        Args:
            model_identifier: Ollama model tag or identifier.

        Returns:
            CuratedModelInfo matching the model architecture, or None.
        """
        raw_clean = model_identifier.lower().split(":")[0]
        norm_name = raw_clean.replace(".", "").replace("-", "").replace("_", "")

        all_models = ModelRegistry.get_all()
        if raw_clean in all_models:
            return all_models[raw_clean]

        for slug, info in all_models.items():
            norm_slug = slug.lower().replace(".", "").replace("-", "").replace("_", "")
            if norm_slug in norm_name or norm_name in norm_slug:
                return info

        manifest = self.find_manifest(model_identifier)
        if manifest and manifest.model_blob_path:
            try:
                parser = GGUFParser(manifest.model_blob_path)
                arch = parser.metadata.get("general.architecture", "")
                if arch:
                    for slug, info in all_models.items():
                        if arch.lower() in info.family.lower():
                            return info
            except Exception:
                pass

        return None
