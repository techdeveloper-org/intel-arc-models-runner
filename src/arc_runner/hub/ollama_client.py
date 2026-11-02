"""Direct HTTPS streaming client for registry.ollama.ai OCI distribution endpoints.

Enables downloading and caching arbitrary models directly from the Ollama model library
without requiring the local Ollama daemon or desktop client to be installed.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
import requests

from arc_runner.core.constants import DEFAULT_CACHE_DIR
from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("arc_runner.hub.ollama_client")


class OllamaClientError(ArcRunnerError):
    """Base exception for Ollama registry client failures."""

    def __init__(self, message: str, error_code: str = "OLLAMA_CLIENT_ERROR", status_code: int = 502) -> None:
        """Initialize Ollama client error.

        Args:
            message: Descriptive failure explanation.
            error_code: Unique error code token.
            status_code: HTTP status code.
        """
        super().__init__(message, error_code, status_code)


class ManifestFetchError(OllamaClientError):
    """Raised when fetching OCI manifest fails or returns non-200."""

    def __init__(self, model_tag: str, status_code: int, details: str = "") -> None:
        """Initialize manifest fetch error.

        Args:
            model_tag: Requested model identifier.
            status_code: HTTP response status code.
            details: Additional server response details.
        """
        super().__init__(
            f"Failed to fetch manifest for '{model_tag}' (HTTP {status_code}): {details}",
            "MANIFEST_FETCH_ERROR",
            status_code if status_code != 200 else 502,
        )
        self.model_tag = model_tag


class ModelNotFoundError(OllamaClientError):
    """Raised when requested model tag is not found on registry."""

    def __init__(self, model_tag: str) -> None:
        """Initialize model not found error.

        Args:
            model_tag: Missing model tag.
        """
        super().__init__(
            f"Model '{model_tag}' was not found on registry.ollama.ai.",
            "MODEL_NOT_FOUND",
            404,
        )
        self.model_tag = model_tag


class ChecksumMismatchError(OllamaClientError):
    """Raised when downloaded blob hash does not match expected digest."""

    def __init__(self, expected: str, actual: str) -> None:
        """Initialize checksum mismatch error.

        Args:
            expected: Expected SHA-256 digest string.
            actual: Computed SHA-256 digest string.
        """
        super().__init__(
            f"Checksum verification failed: expected {expected}, calculated {actual}.",
            "CHECKSUM_MISMATCH",
            422,
        )
        self.expected = expected
        self.actual = actual


class DownloadResumeError(OllamaClientError):
    """Raised when network stream drops during HTTP Range resumption."""

    def __init__(self, digest: str, reason: str) -> None:
        """Initialize download resume error.

        Args:
            digest: Layer blob digest string.
            reason: Explanation of resume failure.
        """
        super().__init__(
            f"Failed to resume blob download for {digest}: {reason}",
            "DOWNLOAD_RESUME_ERROR",
            502,
        )
        self.digest = digest


@dataclass(frozen=True)
class OllamaManifestLayerDto:
    """Descriptor DTO for an OCI content-addressable layer."""

    media_type: str
    digest: str
    size: int


@dataclass(frozen=True)
class OllamaManifestResponseDto:
    """Parsed response DTO from an Ollama OCI manifest endpoint."""

    schema_version: int
    media_type: str
    config: OllamaManifestLayerDto
    layers: List[OllamaManifestLayerDto]
    total_size_bytes: int


class OllamaRegistryClient:
    """Direct HTTPS streaming client for registry.ollama.ai OCI distribution endpoints."""

    REGISTRY_BASE_URL: str = "https://registry.ollama.ai"
    AUTH_BASE_URL: str = "https://ollama.com/auth/token"
    DEFAULT_CHUNK_SIZE: int = 1024 * 1024

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout_seconds: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize registry client with cache directory and request session.

        Args:
            cache_dir: Destination directory for downloaded blobs.
            timeout_seconds: Request timeout in seconds.
            session: Optional custom requests Session.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else (DEFAULT_CACHE_DIR / "ollama_blobs")
        self._timeout = timeout_seconds
        self._session = session or requests.Session()

    @property
    def cache_dir(self) -> Path:
        """Return destination blob cache directory."""
        return self._cache_dir

    @staticmethod
    def parse_model_tag(model_tag: str) -> Tuple[str, str, str]:
        """Parse raw model tag into namespace, model name, and version tag.

        Args:
            model_tag: Raw tag e.g. 'qwen2.5:3b', 'ollama:qwen2.5:3b', 'namespace/model:tag'.

        Returns:
            Tuple[str, str, str]: (namespace, model, tag).
        """
        cleaned = model_tag.strip()
        if cleaned.startswith("ollama:"):
            cleaned = cleaned[7:]

        if ":" in cleaned:
            name_part, tag = cleaned.split(":", 1)
        else:
            name_part, tag = cleaned, "latest"

        if "/" in name_part:
            namespace, model = name_part.split("/", 1)
        else:
            namespace, model = "library", name_part

        return namespace, model, tag

    def get_bearer_token(self, namespace: str, model: str) -> Optional[str]:
        """Obtain anonymous pull bearer token from Ollama auth service.

        Args:
            namespace: Repository namespace (e.g. 'library').
            model: Model repository name (e.g. 'qwen2.5').

        Returns:
            Optional[str]: Bearer token if granted; None if endpoint requires no auth.

        Raises:
            OllamaClientError: If auth endpoint returns an error.
        """
        url = f"{self.AUTH_BASE_URL}?service=registry.ollama.ai&scope=repository:{namespace}/{model}:pull"
        try:
            resp = self._session.get(url, timeout=self._timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("token")
            return None
        except requests.RequestException as exc:
            logger.warning(f"Failed to fetch bearer token for {namespace}/{model}: {exc}")
            return None

    def fetch_manifest(self, model_tag: str) -> OllamaManifestResponseDto:
        """Fetch OCI distribution manifest for specified model tag.

        Args:
            model_tag: Model tag e.g. 'qwen2.5:3b' or 'library/qwen2.5:3b'.

        Returns:
            OllamaManifestResponseDto: Structured manifest descriptor.

        Raises:
            ModelNotFoundError: If model does not exist on registry.
            ManifestFetchError: If request fails or returns non-200.
        """
        namespace, model, tag = self.parse_model_tag(model_tag)
        token = self.get_bearer_token(namespace, model)

        headers: Dict[str, str] = {
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        url = f"{self.REGISTRY_BASE_URL}/v2/{namespace}/{model}/manifests/{tag}"
        try:
            resp = self._session.get(url, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise ManifestFetchError(model_tag, 502, str(exc))

        if resp.status_code == 404:
            raise ModelNotFoundError(model_tag)
        if resp.status_code != 200:
            raise ManifestFetchError(model_tag, resp.status_code, resp.text[:200])

        try:
            data = resp.json()
        except ValueError as exc:
            raise ManifestFetchError(model_tag, 502, f"Invalid JSON in manifest payload: {exc}")

        schema_version = data.get("schemaVersion", 2)
        media_type = data.get("mediaType", "application/vnd.docker.distribution.manifest.v2+json")

        cfg_raw = data.get("config", {})
        config_dto = OllamaManifestLayerDto(
            media_type=cfg_raw.get("mediaType", ""),
            digest=cfg_raw.get("digest", ""),
            size=int(cfg_raw.get("size", 0)),
        )

        layers: List[OllamaManifestLayerDto] = []
        total_size = 0
        for lay in data.get("layers", []):
            sz = int(lay.get("size", 0))
            total_size += sz
            layers.append(
                OllamaManifestLayerDto(
                    media_type=lay.get("mediaType", ""),
                    digest=lay.get("digest", ""),
                    size=sz,
                )
            )

        return OllamaManifestResponseDto(
            schema_version=schema_version,
            media_type=media_type,
            config=config_dto,
            layers=layers,
            total_size_bytes=total_size,
        )

    def stream_blob(
        self,
        namespace: str,
        model: str,
        digest: str,
        destination_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Stream a content-addressable layer blob from registry with Range resumption and SHA-256 verification.

        Args:
            namespace: Repository namespace.
            model: Model repository name.
            digest: SHA-256 digest string (e.g. 'sha256:...').
            destination_path: Optional exact output file path.
            progress_callback: Optional callback invoked with (downloaded_bytes, total_bytes).

        Returns:
            Path: Path to verified local blob file.

        Raises:
            ChecksumMismatchError: If computed hash does not match expected digest.
            DownloadResumeError: If connection drops and cannot be resumed.
            OllamaClientError: If network or disk permissions fail.
        """
        clean_digest = digest.replace(":", "-")
        target_path = Path(destination_path) if destination_path else (self._cache_dir / clean_digest)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OllamaClientError(f"Permission denied creating cache directory '{target_path.parent}': {exc}", status_code=403)

        if target_path.exists() and self.verify_blob_checksum(target_path, digest):
            logger.debug(f"Layer blob {digest} already exists and is verified in cache.")
            return target_path

        token = self.get_bearer_token(namespace, model)
        headers: Dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        downloaded_bytes = 0
        file_mode = "wb"
        if target_path.exists():
            downloaded_bytes = target_path.stat().st_size
            if downloaded_bytes > 0:
                headers["Range"] = f"bytes={downloaded_bytes}-"
                file_mode = "ab"

        url = f"{self.REGISTRY_BASE_URL}/v2/{namespace}/{model}/blobs/{digest}"

        try:
            resp = self._session.get(url, headers=headers, stream=True, timeout=self._timeout)
        except requests.RequestException as exc:
            raise DownloadResumeError(digest, f"Network connection failed: {exc}")

        if resp.status_code == 416:
            file_mode = "wb"
            downloaded_bytes = 0
            headers.pop("Range", None)
            resp = self._session.get(url, headers=headers, stream=True, timeout=self._timeout)

        if resp.status_code not in (200, 206):
            raise OllamaClientError(f"Blob request returned HTTP {resp.status_code} for {digest}", status_code=resp.status_code)

        content_length_header = resp.headers.get("Content-Length")
        total_expected = (
            downloaded_bytes + int(content_length_header) if content_length_header else downloaded_bytes
        )

        try:
            with open(target_path, file_mode) as f:
                for chunk in resp.iter_content(chunk_size=self.DEFAULT_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_callback and total_expected > 0:
                            progress_callback(downloaded_bytes, total_expected)
        except OSError as exc:
            raise OllamaClientError(f"Failed writing blob to disk '{target_path}': {exc}", status_code=500)

        if not self.verify_blob_checksum(target_path, digest):
            computed = self._compute_sha256(target_path)
            target_path.unlink(missing_ok=True)
            raise ChecksumMismatchError(digest, computed)

        return target_path

    @classmethod
    def verify_blob_checksum(cls, file_path: Path, expected_digest: str) -> bool:
        """Verify whether an on-disk file matches an expected SHA-256 digest string.

        Args:
            file_path: Path to target file on disk.
            expected_digest: Expected digest string formatted as 'sha256:...'.

        Returns:
            bool: True if matching; False otherwise.
        """
        if not file_path.exists():
            return False

        computed = cls._compute_sha256(file_path)
        expected_norm = expected_digest.lower()
        if not expected_norm.startswith("sha256:"):
            expected_norm = f"sha256:{expected_norm}"

        return computed == expected_norm

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash formatted as 'sha256:...'.

        Args:
            file_path: Path to file.

        Returns:
            str: Prefixed SHA-256 string.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"
