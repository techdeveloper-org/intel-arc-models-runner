"""Hub lifecycle manager orchestrating model acquisition, verification, and deletion.

Coordinates alias resolution, non-blocking downloads, SHA-256 integrity checks,
and synchronized Level Zero engine memory teardown prior to weight unlinking.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from arc_runner.core.exceptions import ArcRunnerError, ModelNotFoundError
from arc_runner.core.logging import get_logger
from arc_runner.core.path_policy import resolve_inside
from arc_runner.hub.downloader import DownloadProgress, ModelDownloader
from arc_runner.hub.registry import CuratedModelInfo, ModelRegistry
from arc_runner.server.model_manager import ModelManager

logger = get_logger("hub.manager")


class HubManager:
    """Manages local model repositories, automated downloads, and disk lifecycles."""

    def __init__(
        self,
        base_models_dir: Optional[Union[str, Path]] = None,
        model_manager: Optional[ModelManager] = None,
    ) -> None:
        """Initialize the hub manager with designated storage directory and engine cache.

        Args:
            base_models_dir: Root filesystem directory storing downloaded OpenVINO models.
            model_manager: Optional running ModelManager to synchronize engine unloads.
        """
        self._base_dir = Path(base_models_dir).resolve() if base_models_dir else Path("models").resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._model_manager = model_manager
        self._downloader = ModelDownloader(cache_dir=self._base_dir)
        self._lock = threading.Lock()

    @property
    def base_dir(self) -> Path:
        """Return the active base model storage directory path.

        Returns:
            Path: Resolved absolute path.
        """
        return self._base_dir

    def resolve_model_dir(self, model_id: str) -> Path:
        """Resolve a model name or alias to its local directory on disk.

        Args:
            model_id: Model alias name or path identifier.

        Returns:
            Path: Resolved directory path.

        Raises:
            ModelNotFoundError: If model does not exist locally.
        """
        clean_id = model_id.strip().split(":")[0]
        try:
            direct = resolve_inside(self._base_dir, clean_id)
        except ArcRunnerError:
            raise ModelNotFoundError(model_id, str(self._base_dir)) from None
        if direct.exists() and direct.is_dir():
            return direct

        alias_info = ModelRegistry.resolve(clean_id)
        candidate = resolve_inside(self._base_dir, alias_info.alias)
        if candidate.exists() and candidate.is_dir():
            return candidate

        raise ModelNotFoundError(model_id, str(self._base_dir))

    def verify_integrity(self, model_dir: Path) -> Tuple[bool, List[str]]:
        """Verify presence of essential OpenVINO IR files and tokenizers.

        Args:
            model_dir: Target directory path to inspect.

        Returns:
            Tuple[bool, List[str]]: (True, []) if valid, or (False, list of missing items).
        """
        if not model_dir.exists() or not model_dir.is_dir():
            return False, ["Model directory does not exist"]

        missing: List[str] = []
        xml_models = list(model_dir.glob("*.xml"))
        if not xml_models:
            missing.append("Missing OpenVINO IR graph definition (*.xml)")

        bin_models = list(model_dir.glob("*.bin"))
        if not bin_models:
            missing.append("Missing OpenVINO IR weights file (openvino_model.bin or *.bin)")

        return (len(missing) == 0, missing)

    def pull(
        self,
        model_id: str,
        stream: bool = True,
    ) -> Union[Iterator[DownloadProgress], Path]:
        """Download and register an OpenVINO model from Hugging Face.

        Args:
            model_id: Model alias or Hugging Face repository identifier.
            stream: If True, yields DownloadProgress; if False, blocks until complete.

        Returns:
            Iterator of DownloadProgress emissions, or resolved Path upon synchronous completion.

        Raises:
            ArcRunnerError: If model_id is empty or download fails.
        """
        clean_name = model_id.strip()
        if not clean_name:
            raise ArcRunnerError("Model identifier cannot be empty", "INVALID_ARGUMENT", 422)

        info: CuratedModelInfo = ModelRegistry.resolve(clean_name)
        target_dir = (self._base_dir / info.alias).resolve()

        if stream:
            return self._downloader.download_stream(
                repo_id=info.repo_id,
                destination_dir=target_dir,
                expected_size=info.size_bytes,
            )

        return self._downloader.download_sync(
            repo_id=info.repo_id,
            destination_dir=target_dir,
        )

    def delete(self, model_id: str) -> bool:
        """Unload active engine and delete model directory from local disk.

        Args:
            model_id: Target model alias or directory name.

        Returns:
            True if deletion was successful.

        Raises:
            ModelNotFoundError: If model directory does not exist.
        """
        target_dir = self.resolve_model_dir(model_id)

        with self._lock:
            if self._model_manager:
                try:
                    self._model_manager.unload_model(model_id)
                    self._model_manager.unload_model(target_dir.name)
                except Exception as exc:
                    logger.warning(f"Error unloading engine {model_id} during delete: {exc}")

            shutil.rmtree(target_dir)
            logger.info(f"Deleted model directory: {target_dir}")
            return True

    def list_local(self) -> List[Dict[str, Any]]:
        """List all valid and discoverable models in the base storage directory.

        Returns:
            List of dictionaries containing model metadata and on-disk size.
        """
        results: List[Dict[str, Any]] = []
        if not self._base_dir.exists():
            return results

        with self._lock:
            for item in self._base_dir.iterdir():
                if item.is_dir():
                    is_valid, _ = self.verify_integrity(item)
                    if is_valid:
                        total_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                        mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
                        digest = hashlib.sha256(item.name.encode("utf-8")).hexdigest()[:12]
                        info = ModelRegistry.resolve(item.name)
                        results.append({
                            "name": item.name,
                            "model": item.name,
                            "modified_at": mtime,
                            "size": total_bytes,
                            "digest": digest,
                            "details": {
                                "parent_model": "",
                                "format": "openvino",
                                "family": info.family,
                                "parameter_size": info.parameter_size,
                                "quantization_level": info.quantization,
                            },
                        })
        return results

    def copy(self, source: str, destination: str) -> bool:
        """Create a local copy of an existing model under a new alias.

        Args:
            source: Source model identifier.
            destination: New destination alias name.

        Returns:
            True if duplication was successful.

        Raises:
            ModelNotFoundError: If source does not exist.
            ArcRunnerError: If destination directory already exists.
        """
        src_dir = self.resolve_model_dir(source)
        dest_dir = resolve_inside(self._base_dir, destination)

        if dest_dir.exists():
            raise ArcRunnerError(f"Destination '{destination}' already exists", "RESOURCE_CONFLICT", 409)

        with self._lock:
            shutil.copytree(src_dir, dest_dir)
            logger.info(f"Copied model '{src_dir.name}' to '{dest_dir.name}'")
            return True
