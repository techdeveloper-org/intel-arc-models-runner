"""Asynchronous non-blocking model weight downloader wrapping huggingface_hub.

Implements EMA transfer rate smoothing, progress callbacks, and streaming NDJSON generator
bridges compliant with Ollama /api/pull wire specifications.
"""

import hashlib
from pathlib import Path
import queue
import threading
import time
from typing import Dict, Iterator, Optional, Tuple
from pydantic import BaseModel, Field

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger

logger = get_logger("hub.downloader")


class DownloadProgress(BaseModel):
    """Progress status container yielded during streaming model acquisition."""

    status: str = Field(description="Operational status message")
    digest: Optional[str] = Field(default=None, description="Model short hash digest")
    total: Optional[int] = Field(default=None, description="Total expected transfer size in bytes")
    completed: Optional[int] = Field(default=None, description="Accumulated bytes downloaded so far")
    percent: Optional[float] = Field(default=None, description="Completion percentage 0.0 to 100.0")
    speed_mbps: Optional[float] = Field(default=None, description="Smoothed bandwidth in Megabytes per second")
    eta_seconds: Optional[float] = Field(default=None, description="Estimated remaining seconds")
    verified: Optional[bool] = Field(default=None, description="True only when every file matched a remote hash; False when no comparison was possible; None while in progress")


class ModelDownloader:
    """Manages snapshot downloads from Hugging Face with progress tracking and rate smoothing."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the model downloader with optional base cache directory.

        Args:
            cache_dir: Optional root path for storing downloaded models.
        """
        self._cache_dir = Path(cache_dir).resolve() if cache_dir else Path("models").resolve()

    def _get_dir_size(self, path: Path) -> int:
        """Compute the total byte size of all files contained inside path.

        Args:
            path: Target directory to measure.

        Returns:
            int: Aggregated size in bytes.
        """
        if not path.exists():
            return 0
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except Exception:
            pass
        return total

    def _fetch_remote_file_hashes(self, repo_id: str) -> Dict[str, Tuple[str, str]]:
        """Fetch per-file content hashes published by the Hugging Face Hub.

        Files stored in LFS carry a sha256; other files carry a git blob object id (sha1).

        Args:
            repo_id: Canonical Hugging Face repository identifier.

        Returns:
            Dict[str, Tuple[str, str]]: Mapping of repository file name to (algorithm, hex digest),
            where algorithm is "sha256" or "git-sha1".

        Raises:
            ArcRunnerError: If the metadata cannot be fetched.
        """
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo_id, files_metadata=True)
        except Exception as exc:
            raise ArcRunnerError(f"Could not fetch file hashes for '{repo_id}': {exc}", "HUB_METADATA_UNAVAILABLE", 502) from exc
        hashes: Dict[str, Tuple[str, str]] = {}
        for sibling in info.siblings or []:
            lfs = getattr(sibling, "lfs", None)
            lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            blob_id = getattr(sibling, "blob_id", None)
            if lfs_sha:
                hashes[sibling.rfilename] = ("sha256", str(lfs_sha))
            elif blob_id:
                hashes[sibling.rfilename] = ("git-sha1", str(blob_id))
        return hashes

    @staticmethod
    def _hash_file(path: Path, algorithm: str) -> str:
        """Hash a file the way the Hub does.

        Args:
            path: File to hash.
            algorithm: "sha256" (plain content) or "git-sha1" (git blob object id).

        Returns:
            str: Lowercase hex digest.
        """
        size = path.stat().st_size
        digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1(b"blob %d\0" % size)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_snapshot(self, dest: Path, remote: Dict[str, Tuple[str, str]], repo_id: str) -> str:
        """Compare every downloaded file with the hash the Hub publishes for it.

        Args:
            dest: Resolved destination directory.
            remote: Mapping from file name to (algorithm, hex digest).
            repo_id: Repository identifier used in error messages.

        Returns:
            str: Short digest computed over the sorted (file name, verified hash) list.

        Raises:
            ArcRunnerError: With code CHECKSUM_MISMATCH if a file is missing, escapes dest, or differs.
        """
        combined = hashlib.sha256()
        for name in sorted(remote):
            algorithm, expected = remote[name]
            target = (dest / name).resolve()
            if dest != target and dest not in target.parents:
                raise ArcRunnerError(f"Remote file name '{name}' escapes the download directory", "CHECKSUM_MISMATCH", 502)
            if not target.is_file():
                raise ArcRunnerError(f"'{name}' is listed by '{repo_id}' but was not downloaded", "CHECKSUM_MISMATCH", 502)
            actual = self._hash_file(target, algorithm)
            if actual != expected.lower():
                raise ArcRunnerError(f"{algorithm} mismatch for '{name}' from '{repo_id}'", "CHECKSUM_MISMATCH", 502)
            combined.update(f"{name}\0{actual}\n".encode("utf-8"))
        return combined.hexdigest()[:12]

    def download_stream(
        self,
        repo_id: str,
        destination_dir: Path,
        expected_size: int = 1800000000,
    ) -> Iterator[DownloadProgress]:
        """Download model files and yield progress emissions as data is written to disk.

        Args:
            repo_id: Canonical Hugging Face repository identifier.
            destination_dir: Target directory on disk to write snapshot files.
            expected_size: Reference size in bytes used for initial progress scaling.

        Yields:
            DownloadProgress instances tracking transfer rate and completion.

        Raises:
            ArcRunnerError: If download fails or network error occurs.
        """
        dest = destination_dir.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        digest: Optional[str] = None

        yield DownloadProgress(
            status="pulling manifest",
            digest=digest,
            total=expected_size,
            completed=0,
            percent=0.0,
            speed_mbps=0.0,
            eta_seconds=None,
        )

        error_holder: queue.Queue[Exception] = queue.Queue()
        done_event = threading.Event()

        def download_worker() -> None:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(dest),
                    local_dir_use_symlinks=False,
                )
            except Exception as exc:
                error_holder.put(exc)
            finally:
                done_event.set()

        worker_thread = threading.Thread(target=download_worker, daemon=True)
        worker_thread.start()

        smoothed_rate = 0.0
        alpha = 0.2
        last_bytes = self._get_dir_size(dest)
        last_time = time.monotonic()

        while not done_event.is_set():
            time.sleep(0.3)
            current_bytes = self._get_dir_size(dest)
            current_time = time.monotonic()

            delta_bytes = max(0, current_bytes - last_bytes)
            delta_time = max(1e-4, current_time - last_time)
            instant_rate = delta_bytes / delta_time

            if smoothed_rate <= 0.0:
                smoothed_rate = instant_rate
            else:
                smoothed_rate = alpha * instant_rate + (1.0 - alpha) * smoothed_rate

            last_bytes = current_bytes
            last_time = current_time

            active_total = max(expected_size, current_bytes)
            pct = round(min(99.0, (current_bytes / active_total) * 100.0), 1)
            eta = round((active_total - current_bytes) / smoothed_rate, 1) if smoothed_rate > 1024.0 else None
            speed_mb = round(smoothed_rate / (1024.0 * 1024.0), 2)

            yield DownloadProgress(
                status=f"downloading {repo_id}",
                digest=digest,
                total=active_total,
                completed=current_bytes,
                percent=pct,
                speed_mbps=speed_mb,
                eta_seconds=eta,
            )

        if not error_holder.empty():
            err = error_holder.get()
            raise ArcRunnerError(f"Failed to download repository '{repo_id}': {err}", "DOWNLOAD_FAILED", 502)

        final_bytes = self._get_dir_size(dest)
        verified = False
        try:
            remote_hashes = self._fetch_remote_file_hashes(repo_id)
        except ArcRunnerError as exc:
            logger.warning("Checksum comparison unavailable", extra={"repo_id": repo_id, "reason": exc.message})
            remote_hashes = {}
        if remote_hashes:
            yield DownloadProgress(
                status="verifying checksum",
                digest=digest,
                total=final_bytes,
                completed=final_bytes,
                percent=99.9,
                speed_mbps=0.0,
                eta_seconds=0.0,
            )
            digest = self._verify_snapshot(dest, remote_hashes, repo_id)
            verified = True

        yield DownloadProgress(
            status="success",
            digest=digest,
            total=final_bytes,
            completed=final_bytes,
            percent=100.0,
            speed_mbps=0.0,
            eta_seconds=0.0,
            verified=verified,
        )

    def download_sync(self, repo_id: str, destination_dir: Path) -> Path:
        """Synchronously execute snapshot download and return the local path upon completion.

        Args:
            repo_id: Canonical Hugging Face repository identifier.
            destination_dir: Target directory path to write snapshot files.

        Returns:
            Path: Resolved absolute path to downloaded directory.
        """
        for _ in self.download_stream(repo_id, destination_dir):
            pass
        return destination_dir.resolve()
