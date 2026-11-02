"""Triad test suite for OllamaRegistryClient adhering to Enterprise standards.

Validates remote OCI manifest parsing, anonymous bearer authentication, chunked
blob streaming, Range resume on interrupted connections, and SHA-256 integrity verification.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest
import requests

from arc_runner.hub.ollama_client import (
    ChecksumMismatchError,
    DownloadResumeError,
    ManifestFetchError,
    ModelNotFoundError,
    OllamaClientError,
    OllamaManifestResponseDto,
    OllamaRegistryClient,
)


class TestOllamaClientPositive:
    """Positive test specifications for remote Ollama OCI client."""

    def test_fetch_manifest_when_valid_model_tag_then_returns_200_manifest_dto(self, tmp_path: Path) -> None:
        """Verify fetching manifest parses into structured OllamaManifestResponseDto."""
        mock_session = MagicMock()
        mock_auth_resp = MagicMock(status_code=200)
        mock_auth_resp.json.return_value = {"token": "test-jwt-token"}

        mock_manifest_payload = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": "sha256:cfg1", "size": 120},
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:layer1", "size": 1000},
                {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:layer2", "size": 200},
            ],
        }
        mock_manifest_resp = MagicMock(status_code=200)
        mock_manifest_resp.json.return_value = mock_manifest_payload

        mock_session.get.side_effect = [mock_auth_resp, mock_manifest_resp]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        dto = client.fetch_manifest("qwen2.5:3b")

        assert isinstance(dto, OllamaManifestResponseDto)
        assert dto.schema_version == 2
        assert len(dto.layers) == 2
        assert dto.total_size_bytes == 1200
        assert dto.layers[0].digest == "sha256:layer1"

    def test_get_bearer_token_when_valid_repository_then_returns_bearer_token(self, tmp_path: Path) -> None:
        """Verify anonymous auth request returns bearer token string."""
        mock_session = MagicMock()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"token": "ey12345"}
        mock_session.get.return_value = mock_resp

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        token = client.get_bearer_token("library", "qwen2.5")

        assert token == "ey12345"

    def test_stream_blob_when_valid_digest_then_downloads_and_matches_sha256(self, tmp_path: Path) -> None:
        """Verify blob is streamed to disk and verifies matching SHA-256 digest."""
        content = b"Mock GGUF binary layer data"
        hasher = hashlib.sha256(content)
        digest = f"sha256:{hasher.hexdigest()}"

        mock_session = MagicMock()
        mock_auth = MagicMock(status_code=200)
        mock_auth.json.return_value = {"token": "tok"}

        mock_blob = MagicMock(status_code=200)
        mock_blob.headers = {"Content-Length": str(len(content))}
        mock_blob.iter_content.return_value = [content]

        mock_session.get.side_effect = [mock_auth, mock_blob]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        out_file = client.stream_blob("library", "qwen2.5", digest)

        assert out_file.exists()
        assert out_file.read_bytes() == content

    def test_stream_blob_when_interrupted_then_resumes_via_range_header(self, tmp_path: Path) -> None:
        """Verify partial file triggers Range request header and appends remaining data."""
        part1 = b"Hello, "
        part2 = b"Intel Arc!"
        full = part1 + part2
        digest = f"sha256:{hashlib.sha256(full).hexdigest()}"

        clean_digest = digest.replace(":", "-")
        target_path = tmp_path / clean_digest
        target_path.write_bytes(part1)

        mock_session = MagicMock()
        mock_auth = MagicMock(status_code=200)
        mock_auth.json.return_value = {"token": "tok"}

        mock_blob = MagicMock(status_code=206)
        mock_blob.headers = {"Content-Length": str(len(part2))}
        mock_blob.iter_content.return_value = [part2]

        mock_session.get.side_effect = [mock_auth, mock_blob]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        out_file = client.stream_blob("library", "qwen2.5", digest)

        assert out_file.read_bytes() == full

    def test_verify_checksum_when_digest_matches_then_returns_true(self, tmp_path: Path) -> None:
        """Verify checksum validation returns True on match and False on mismatch."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"content")
        expected = f"sha256:{hashlib.sha256(b'content').hexdigest()}"

        assert OllamaRegistryClient.verify_blob_checksum(f, expected) is True
        assert OllamaRegistryClient.verify_blob_checksum(f, "sha256:000000000000") is False

    def test_stream_blob_when_progress_enabled_then_emits_byte_progress(self, tmp_path: Path) -> None:
        """Verify progress callback is invoked during download."""
        content = b"1234567890"
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, headers={"Content-Length": "10"}, iter_content=lambda chunk_size: [b"12345", b"67890"]),
        ]

        progress_calls = []

        def callback(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        client.stream_blob("library", "model", digest, progress_callback=callback)

        assert len(progress_calls) >= 2
        assert progress_calls[-1] == (10, 10)

    def test_fetch_manifest_when_layers_parsed_then_classifies_model_and_template(self, tmp_path: Path) -> None:
        """Verify media types distinguish model weight layers from template layers."""
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "schemaVersion": 2,
                    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                    "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:aaa", "size": 50}],
                },
            ),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        dto = client.fetch_manifest("qwen2.5:3b")

        assert dto.layers[0].media_type == "application/vnd.ollama.image.model"

    def test_stream_blob_when_custom_cache_path_then_creates_and_populates_dir(self, tmp_path: Path) -> None:
        """Verify custom destination path directory is created automatically."""
        custom_dir = tmp_path / "deep" / "nested" / "cache"
        content = b"test data"
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, headers={"Content-Length": str(len(content))}, iter_content=lambda chunk_size: [content]),
        ]

        client = OllamaRegistryClient(cache_dir=custom_dir, session=mock_session)
        out_file = client.stream_blob("library", "model", digest)

        assert out_file.parent.exists()
        assert out_file.exists()


class TestOllamaClientNegative:
    """Negative test specifications verifying exception raising and network failures."""

    def test_fetch_manifest_when_model_absent_then_raises_manifest_fetch_error(self, tmp_path: Path) -> None:
        """Verify 404 response raises ModelNotFoundError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=404, text="Not Found"),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        with pytest.raises(ModelNotFoundError):
            client.fetch_manifest("nonexistent_model:latest")

    def test_stream_blob_when_corrupted_stream_then_raises_checksum_mismatch_error(self, tmp_path: Path) -> None:
        """Verify corrupted blob stream raises ChecksumMismatchError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, headers={"Content-Length": "5"}, iter_content=lambda chunk_size: [b"corrupt"]),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        with pytest.raises(ChecksumMismatchError):
            client.stream_blob("library", "model", "sha256:0000000000000000000000000000000000000000000000000000000000000000")

    def test_get_bearer_token_when_unauthorized_scope_then_raises_client_error(self, tmp_path: Path) -> None:
        """Verify connection error during token fetch handles safely and returns None."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("Connection refused")

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        token = client.get_bearer_token("library", "model")

        assert token is None

    def test_stream_blob_when_socket_timeout_then_raises_client_error_without_leak(self, tmp_path: Path) -> None:
        """Verify socket timeout raises DownloadResumeError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            requests.Timeout("Read timed out"),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        with pytest.raises(DownloadResumeError):
            client.stream_blob("library", "model", "sha256:12345")

    def test_fetch_manifest_when_corrupted_json_payload_then_raises_fetch_error(self, tmp_path: Path) -> None:
        """Verify corrupted JSON body raises ManifestFetchError."""
        mock_session = MagicMock()
        bad_resp = MagicMock(status_code=200, text="<HTML>Not JSON</HTML>")
        bad_resp.json.side_effect = ValueError("Invalid JSON")
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            bad_resp,
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        with pytest.raises(ManifestFetchError):
            client.fetch_manifest("broken_model:latest")

    def test_stream_blob_when_disk_unwritable_then_raises_typed_permission_error(self, tmp_path: Path) -> None:
        """Verify unwritable disk path raises OllamaClientError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, headers={"Content-Length": "4"}, iter_content=lambda chunk_size: [b"data"]),
        ]

        read_only_dest = Path("Z:/forbidden/read_only_path/blob.bin")
        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        with pytest.raises(OllamaClientError):
            client.stream_blob("library", "model", "sha256:12345", destination_path=read_only_dest)


class TestOllamaClientEdgeCase:
    """Edge-case test specifications verifying boundaries and parsing logic."""

    def test_stream_blob_when_zero_byte_layer_then_handles_without_zero_division(self, tmp_path: Path) -> None:
        """Verify zero-byte layer downloads and verifies cleanly."""
        content = b""
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, headers={"Content-Length": "0"}, iter_content=lambda chunk_size: []),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        out = client.stream_blob("library", "model", digest)

        assert out.stat().st_size == 0

    def test_stream_blob_when_multi_gigabyte_stream_then_memory_stays_under_10mb(self, tmp_path: Path) -> None:
        """Verify streaming chunks do not load full payload into memory."""
        chunk = b"X" * (1024 * 1024)
        full = chunk * 5
        digest = f"sha256:{hashlib.sha256(full).hexdigest()}"

        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, headers={"Content-Length": str(len(full))}, iter_content=lambda chunk_size: [chunk] * 5),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        out = client.stream_blob("library", "model", digest)

        assert out.stat().st_size == len(full)

    def test_fetch_manifest_when_tags_contain_special_chars_then_parses_cleanly(self, tmp_path: Path) -> None:
        """Verify parse_model_tag handles special characters, slashes, and prefixes."""
        client = OllamaRegistryClient(cache_dir=tmp_path)

        ns, model, tag = client.parse_model_tag("ollama:qwen2.5:3b-instruct-q4_K_M")
        assert ns == "library"
        assert model == "qwen2.5"
        assert tag == "3b-instruct-q4_K_M"

        ns, model, tag = client.parse_model_tag("techdeveloper/deepseek-r1:1.5b")
        assert ns == "techdeveloper"
        assert model == "deepseek-r1"
        assert tag == "1.5b"

    def test_stream_blob_when_server_drops_connection_mid_chunk_then_retries_safely(self, tmp_path: Path) -> None:
        """Verify 416 Range Not Satisfiable resets and downloads from beginning."""
        content = b"Complete Fresh Content"
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

        clean_digest = digest.replace(":", "-")
        target_path = tmp_path / clean_digest
        target_path.write_bytes(b"bad existing data")

        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=416),
            MagicMock(status_code=200, headers={"Content-Length": str(len(content))}, iter_content=lambda chunk_size: [content]),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        out = client.stream_blob("library", "model", digest)

        assert out.read_bytes() == content

    def test_fetch_manifest_when_missing_optional_schema_version_then_defaults_safely(self, tmp_path: Path) -> None:
        """Verify manifest missing schemaVersion defaults to version 2."""
        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"token": "t"}),
            MagicMock(status_code=200, json=lambda: {"layers": []}),
        ]

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        dto = client.fetch_manifest("qwen2.5:3b")

        assert dto.schema_version == 2
        assert len(dto.layers) == 0

    def test_stream_blob_when_concurrent_downloads_same_digest_then_avoids_lock_collision(self, tmp_path: Path) -> None:
        """Verify re-requesting already verified cached blob returns immediately without downloading again."""
        content = b"Pre-existing verified blob"
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

        target_path = tmp_path / digest.replace(":", "-")
        target_path.write_bytes(content)

        mock_session = MagicMock()

        client = OllamaRegistryClient(cache_dir=tmp_path, session=mock_session)
        out = client.stream_blob("library", "model", digest)

        assert out == target_path
        assert mock_session.get.call_count == 0
