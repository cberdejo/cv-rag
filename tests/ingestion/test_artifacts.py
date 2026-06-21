"""Tests for cv_ingestion.artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.artifacts import ArtifactStorage
from cv_ingestion.artifacts import (
    ResolvedIngestionInput,
    resolve_ingestion_input,
)
from cv_ingestion.exceptions import CVIngestionConfigurationError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_minio_artifact(
    local_path: Path,
    uri: str,
    object_name: str | None = None,
):
    """Return a fake DownloadedArtifact."""
    artifact = MagicMock()
    artifact.local_path = local_path
    artifact.object_name = object_name or local_path.name
    artifact.uri = uri
    return artifact


# ─────────────────────────────────────────────────────────────────────────────
# resolve_ingestion_input — LOCAL storage
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveIngestionInputLocal:
    def test_yields_original_input_dir(self, tmp_path):
        (tmp_path / "cv.pdf").write_bytes(b"fake")

        with resolve_ingestion_input(
            input_dir=tmp_path,
            storage=ArtifactStorage.LOCAL,
            bucket=None,
        ) as resolved:
            assert resolved.input_dir == tmp_path

    def test_source_uri_by_local_path_empty_for_local(self, tmp_path):
        (tmp_path / "cv.pdf").write_bytes(b"fake")

        with resolve_ingestion_input(
            input_dir=tmp_path,
            storage=ArtifactStorage.LOCAL,
            bucket=None,
        ) as resolved:
            assert resolved.source_uri_by_local_path == {}

    def test_yields_resolved_ingestion_input_type(self, tmp_path):
        (tmp_path / "cv.pdf").write_bytes(b"fake")

        with resolve_ingestion_input(
            input_dir=tmp_path,
            storage=ArtifactStorage.LOCAL,
            bucket=None,
        ) as resolved:
            assert isinstance(resolved, ResolvedIngestionInput)

    def test_raises_if_input_dir_does_not_exist(self, tmp_path):
        missing = tmp_path / "nonexistent"

        with pytest.raises(CVIngestionConfigurationError, match="nonexistent"):
            with resolve_ingestion_input(
                input_dir=missing,
                storage=ArtifactStorage.LOCAL,
                bucket=None,
            ):
                pass


# ─────────────────────────────────────────────────────────────────────────────
# resolve_ingestion_input — MINIO storage
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveIngestionInputMinio:
    def test_raises_when_bucket_is_none(self, tmp_path):
        with pytest.raises(CVIngestionConfigurationError, match="--bucket"):
            with resolve_ingestion_input(
                input_dir=tmp_path,
                storage=ArtifactStorage.MINIO,
                bucket=None,
            ):
                pass

    def test_downloads_artifacts_from_minio(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        cv_file = staged / "cv.pdf"
        cv_file.write_bytes(b"fake")

        artifact = _make_minio_artifact(cv_file, "minio://cvs/cv.pdf")

        with patch(
            "cv_ingestion.artifacts.download_artifact_sources",
            return_value=[artifact],
        ):
            with resolve_ingestion_input(
                input_dir=tmp_path,
                storage=ArtifactStorage.MINIO,
                bucket="cvs",
            ) as resolved:
                uri = resolved.source_uri_by_local_path.get(cv_file.resolve())

        assert uri == "minio://cvs/cv.pdf"

    def test_source_uri_by_local_path_maps_resolved_paths(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        cv_file = staged / "cv.pdf"
        cv_file.write_bytes(b"fake")

        artifact = _make_minio_artifact(cv_file, "minio://cvs/cv.pdf")

        with patch(
            "cv_ingestion.artifacts.download_artifact_sources",
            return_value=[artifact],
        ):
            with resolve_ingestion_input(
                input_dir=tmp_path,
                storage=ArtifactStorage.MINIO,
                bucket="cvs",
            ) as resolved:
                keys = list(resolved.source_uri_by_local_path.keys())

        assert all(k == k.resolve() for k in keys)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_ingestion_input — BOTH storage
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveIngestionInputBoth:
    def test_raises_when_bucket_is_none(self, tmp_path):
        with pytest.raises(CVIngestionConfigurationError, match="--bucket"):
            with resolve_ingestion_input(
                input_dir=tmp_path,
                storage=ArtifactStorage.BOTH,
                bucket=None,
            ):
                pass

    def test_raises_if_input_dir_does_not_exist(self, tmp_path):
        missing = tmp_path / "nonexistent"

        with pytest.raises(CVIngestionConfigurationError):
            with resolve_ingestion_input(
                input_dir=missing,
                storage=ArtifactStorage.BOTH,
                bucket="cvs",
            ):
                pass

    def test_combines_local_and_minio_without_duplicate_downloads(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Local CV
        (input_dir / "local_cv.pdf").write_bytes(b"local fake")

        def fake_download_artifact_sources(**kwargs):
            minio_file = kwargs["target_dir"] / "minio__minio_cv.pdf"
            minio_file.write_bytes(b"minio fake")
            return [_make_minio_artifact(minio_file, "minio://cvs/minio_cv.pdf")]

        with patch(
            "cv_ingestion.artifacts.download_artifact_sources",
            side_effect=fake_download_artifact_sources,
        ) as download_artifact_sources:
            with resolve_ingestion_input(
                input_dir=input_dir,
                storage=ArtifactStorage.BOTH,
                bucket="cvs",
            ) as resolved:
                all_files = list(resolved.input_dir.rglob("*.pdf"))
                source_uri_by_local_path = resolved.source_uri_by_local_path

        assert (
            download_artifact_sources.call_args.kwargs["filename_prefix"] == "minio__"
        )
        names = {f.name for f in all_files}
        assert "local_cv.pdf" in names
        assert "minio__minio_cv.pdf" in names
        assert len(all_files) == 2
        local_path = next(path for path in all_files if path.name == "local_cv.pdf")
        assert source_uri_by_local_path[local_path.resolve()] == str(
            (input_dir / "local_cv.pdf").resolve()
        )

    def test_minio_file_that_matches_local_name_gets_distinct_name(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Same filename in both local and MinIO
        (input_dir / "cv.pdf").write_bytes(b"local fake")

        def fake_download_artifact_sources(**kwargs):
            minio_file = kwargs["target_dir"] / "minio__cv.pdf"
            minio_file.write_bytes(b"minio fake")
            return [
                _make_minio_artifact(
                    minio_file,
                    "minio://cvs/cv.pdf",
                    object_name="cv.pdf",
                )
            ]

        with patch(
            "cv_ingestion.artifacts.download_artifact_sources",
            side_effect=fake_download_artifact_sources,
        ):
            with resolve_ingestion_input(
                input_dir=input_dir,
                storage=ArtifactStorage.BOTH,
                bucket="cvs",
            ) as resolved:
                pdf_files = list(resolved.input_dir.rglob("*.pdf"))
                source_uri_by_local_path = resolved.source_uri_by_local_path

        names = [f.name for f in pdf_files]
        assert names.count("cv.pdf") == 1
        assert "minio__cv.pdf" in names
        minio_path = next(path for path in pdf_files if path.name == "minio__cv.pdf")
        assert source_uri_by_local_path[minio_path.resolve()] == "minio://cvs/cv.pdf"
