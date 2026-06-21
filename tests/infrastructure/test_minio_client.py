"""Tests for infrastructure.minio_client."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from infrastructure.minio_client import (
    download_artifact_sources,
    ensure_bucket,
    list_artifact_objects,
    upload_file,
)


class TestListArtifactObjects:
    def test_filters_supported_suffixes(self) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = True
        client.list_objects.return_value = [
            SimpleNamespace(object_name="a.pdf"),
            SimpleNamespace(object_name="b.png"),
            SimpleNamespace(object_name="nested/c.txt"),
        ]

        result = list_artifact_objects(bucket="cvs", client=client)

        assert result == ["a.pdf"]
        client.list_objects.assert_called_once_with("cvs", prefix="", recursive=True)

    def test_rejects_missing_bucket(self) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="MinIO bucket does not exist"):
            list_artifact_objects(bucket="missing", client=client)


class TestDownloadArtifactSources:
    def test_returns_local_path_and_minio_uri(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = True
        client.list_objects.return_value = [
            SimpleNamespace(object_name="nested/a.pdf"),
        ]

        result = download_artifact_sources(
            bucket="cvs-test",
            target_dir=tmp_path,
            client=client,
        )

        assert len(result) == 1
        assert result[0].local_path == tmp_path / "a.pdf"
        assert result[0].object_name == "nested/a.pdf"
        assert result[0].uri == "minio://cvs-test/nested/a.pdf"
        client.fget_object.assert_called_once_with(
            "cvs-test",
            "nested/a.pdf",
            str(tmp_path / "a.pdf"),
        )

    def test_handles_duplicate_basenames(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = True
        client.list_objects.return_value = [
            SimpleNamespace(object_name="nested/a.pdf"),
            SimpleNamespace(object_name="other/a.pdf"),
        ]

        result = download_artifact_sources(
            bucket="cvs-test",
            target_dir=tmp_path,
            client=client,
        )

        assert [artifact.local_path for artifact in result] == [
            tmp_path / "a.pdf",
            tmp_path / "other__a.pdf",
        ]
        assert [artifact.uri for artifact in result] == [
            "minio://cvs-test/nested/a.pdf",
            "minio://cvs-test/other/a.pdf",
        ]


class TestEnsureBucket:
    def test_creates_bucket_when_missing(self) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = False

        bucket_name = ensure_bucket(client=client, bucket="cvs-test")

        assert bucket_name == "cvs-test"
        client.make_bucket.assert_called_once_with("cvs-test")

    def test_skips_creation_when_bucket_exists(self) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = True

        ensure_bucket(client=client, bucket="cvs-test")

        client.make_bucket.assert_not_called()


class TestUploadFile:
    def test_uploads_and_returns_minio_uri(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = True
        path = tmp_path / "cv.pdf"
        path.write_bytes(b"%PDF-1.4\n")

        uri = upload_file(
            path,
            object_name="generated/cv.pdf",
            content_type="application/pdf",
            client=client,
            bucket="cvs-test",
        )

        assert uri == "minio://cvs-test/generated/cv.pdf"
        client.fput_object.assert_called_once_with(
            bucket_name="cvs-test",
            object_name="generated/cv.pdf",
            file_path=str(path),
            content_type="application/pdf",
        )

    def test_creates_bucket_before_uploading(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = False
        path = tmp_path / "cv.pdf"
        path.write_bytes(b"%PDF-1.4\n")

        upload_file(path, object_name="cv.pdf", client=client, bucket="cvs-test")

        client.make_bucket.assert_called_once_with("cvs-test")
        client.fput_object.assert_called_once()

    def test_omits_content_type_when_not_given(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.bucket_exists.return_value = True
        path = tmp_path / "cv.pdf"
        path.write_bytes(b"%PDF-1.4\n")

        upload_file(path, object_name="cv.pdf", client=client, bucket="cvs-test")

        assert "content_type" not in client.fput_object.call_args.kwargs
