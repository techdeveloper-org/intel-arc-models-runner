"""TM-10 tests: the downloader's checksum step is a real comparison against remote hashes, or it is not claimed."""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.hub.downloader import DownloadProgress, ModelDownloader

REPO = "org/model"
BIN_BYTES = b"weights" * 100
XML_BYTES = b"<net/>"


def _git_blob_sha1(data: bytes) -> str:
    """Return the git blob object id of data."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.fixture
def fake_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace snapshot_download with a writer of two small files."""
    import sys
    import types

    def writer(repo_id: str, local_dir: str, **_: object) -> str:
        (Path(local_dir) / "openvino_model.bin").write_bytes(BIN_BYTES)
        (Path(local_dir) / "openvino_model.xml").write_bytes(XML_BYTES)
        return local_dir

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.snapshot_download = writer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr("arc_runner.hub.downloader.time.sleep", lambda _s: None)


def _run(tmp_path: Path, remote: Dict[str, Tuple[str, str]] | None, monkeypatch: pytest.MonkeyPatch) -> List[DownloadProgress]:
    """Run a download with the given remote hash table (None means metadata unavailable)."""

    def fetch(self: ModelDownloader, repo_id: str) -> Dict[str, Tuple[str, str]]:
        if remote is None:
            raise ArcRunnerError("metadata unavailable", "HUB_METADATA_UNAVAILABLE", 502)
        return remote

    monkeypatch.setattr(ModelDownloader, "_fetch_remote_file_hashes", fetch, raising=False)
    return list(ModelDownloader(cache_dir=tmp_path).download_stream(REPO, tmp_path / "m", expected_size=1000))


def test_download_stream_hash_mismatch_raises_checksum_mismatch(tmp_path: Path, fake_snapshot: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file whose sha256 differs from the remote LFS hash aborts the pull."""
    remote = {"openvino_model.bin": ("sha256", "0" * 64), "openvino_model.xml": ("git-sha1", _git_blob_sha1(XML_BYTES))}
    with pytest.raises(ArcRunnerError) as info:
        _run(tmp_path, remote, monkeypatch)
    assert info.value.error_code == "CHECKSUM_MISMATCH"


def test_download_stream_missing_remote_file_raises_checksum_mismatch(tmp_path: Path, fake_snapshot: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file the remote lists but the snapshot did not produce is a failure, not a skip."""
    remote = {"absent.bin": ("sha256", "1" * 64)}
    with pytest.raises(ArcRunnerError) as info:
        _run(tmp_path, remote, monkeypatch)
    assert info.value.error_code == "CHECKSUM_MISMATCH"


def test_download_stream_traversal_name_in_remote_listing_is_rejected(tmp_path: Path, fake_snapshot: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote file name that escapes the destination is rejected rather than hashed."""
    remote = {"../outside.bin": ("sha256", "1" * 64)}
    with pytest.raises(ArcRunnerError) as info:
        _run(tmp_path, remote, monkeypatch)
    assert info.value.error_code == "CHECKSUM_MISMATCH"


def test_download_stream_matching_hashes_report_verified_with_content_digest(tmp_path: Path, fake_snapshot: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """When every listed hash matches, the final event says verified and carries a digest derived from content."""
    remote = {
        "openvino_model.bin": ("sha256", hashlib.sha256(BIN_BYTES).hexdigest()),
        "openvino_model.xml": ("git-sha1", _git_blob_sha1(XML_BYTES)),
    }
    events = _run(tmp_path, remote, monkeypatch)
    assert "verifying checksum" in [e.status for e in events]
    assert events[-1].status == "success"
    assert events[-1].verified is True
    assert events[-1].digest and events[-1].digest != hashlib.sha256(REPO.encode()).hexdigest()[:12]


def test_download_stream_unavailable_metadata_is_reported_as_unverified(tmp_path: Path, fake_snapshot: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """If remote hashes cannot be fetched the stream never claims a checksum step and ends unverified."""
    events = _run(tmp_path, None, monkeypatch)
    assert "verifying checksum" not in [e.status for e in events]
    assert events[-1].status == "success"
    assert events[-1].verified is False


def test_download_stream_never_labels_a_repo_id_hash_as_the_model_digest(tmp_path: Path, fake_snapshot: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The old digest was sha256(repo_id)[:12]; no event may carry it."""
    events = _run(tmp_path, None, monkeypatch)
    fake = hashlib.sha256(REPO.encode()).hexdigest()[:12]
    assert all(e.digest != fake for e in events)
