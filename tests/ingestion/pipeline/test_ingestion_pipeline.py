"""Tests for cv_ingestion.pipeline.pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cv_ingestion.exceptions import (
    CVIngestionConfigurationError,
)
from cv_ingestion.models.normalized_cv import (
    ContactInfo,
    ExperienceEntry,
    NormalizedCV,
)
from cv_ingestion.pipeline.pipeline import (
    _discover_cv_paths,
    ingest_cv,
    ingest_directory,
)
from cv_ingestion.pipeline.result import BatchIngestResult, CVIngestResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_normalized_cv(source_path: str = "test.pdf") -> NormalizedCV:
    return NormalizedCV(
        source_path=source_path,
        contact=ContactInfo(
            name="Ana García",
            email="ana.garcia01@example.com",
            phone="+34600123456",
        ),
        skills=["Python", "FastAPI"],
        experience=[
            ExperienceEntry(
                title="Backend Developer",
                company="Acme",
                start_date=date(2021, 1, 1),
                end_date=date(2023, 12, 31),
            )
        ],
        current_title="Backend Developer",
        current_company="Acme",
    )


def _make_indexer(index_return: int = 3) -> MagicMock:
    indexer = MagicMock()
    indexer.delete_cv = AsyncMock(return_value=0)
    indexer.index = AsyncMock(return_value=index_return)
    indexer.create_collection = AsyncMock()
    return indexer


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


# ─────────────────────────────────────────────────────────────────────────────
# _discover_cv_paths
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverCvPaths:
    def test_finds_pdf_files(self, tmp_path):
        (tmp_path / "cv1.pdf").write_bytes(b"fake")
        (tmp_path / "cv2.pdf").write_bytes(b"fake")
        result = _discover_cv_paths(tmp_path)
        assert len(result) == 2

    def test_ignores_txt_files(self, tmp_path):
        (tmp_path / "cv.txt").write_text("content")
        result = _discover_cv_paths(tmp_path)
        assert result == []

    def test_ignores_unsupported_extensions(self, tmp_path):
        (tmp_path / "cv.docx").write_bytes(b"fake")
        (tmp_path / "cv.png").write_bytes(b"fake")
        (tmp_path / "cv.xlsx").write_bytes(b"fake")
        result = _discover_cv_paths(tmp_path)
        assert result == []

    def test_recursive_discovery(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "cv.pdf").write_bytes(b"fake")
        result = _discover_cv_paths(tmp_path)
        assert len(result) == 1

    def test_returns_sorted_paths(self, tmp_path):
        (tmp_path / "c.pdf").write_bytes(b"fake")
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.pdf").write_bytes(b"fake")
        result = _discover_cv_paths(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert _discover_cv_paths(tmp_path) == []

    def test_case_insensitive_pdf(self, tmp_path):
        (tmp_path / "cv.PDF").write_bytes(b"fake")
        result = _discover_cv_paths(tmp_path)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ingest_cv — happy path
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestCvHappyPath:
    @pytest.fixture()
    def cv_file(self, tmp_path) -> Path:
        p = tmp_path / "ana_garcia.pdf"
        p.write_bytes(b"%PDF-1.4 fake content")
        return p

    @pytest.mark.anyio
    async def test_returns_success_result(self, cv_file):
        normalized = _make_normalized_cv(str(cv_file.resolve()))
        indexer = _make_indexer(index_return=4)

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text content",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock() for _ in range(4)],
            ),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert isinstance(result, CVIngestResult)
        assert result.success is True

    @pytest.mark.anyio
    async def test_result_contains_candidate_name(self, cv_file):
        normalized = _make_normalized_cv(str(cv_file.resolve()))
        indexer = _make_indexer()

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.candidate_name == "Ana García"

    @pytest.mark.anyio
    async def test_result_contains_chunk_count(self, cv_file):
        normalized = _make_normalized_cv()
        indexer = _make_indexer(index_return=5)

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock() for _ in range(5)],
            ),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.chunks_indexed == 5

    @pytest.mark.anyio
    async def test_delete_existing_calls_delete_cv(self, cv_file):
        normalized = _make_normalized_cv()
        indexer = _make_indexer()

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            await ingest_cv(cv_file, indexer=indexer, delete_existing=True)

        indexer.delete_cv.assert_called_once()

    @pytest.mark.anyio
    async def test_delete_existing_false_skips_delete_cv(self, cv_file):
        normalized = _make_normalized_cv()
        indexer = _make_indexer()

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            await ingest_cv(cv_file, indexer=indexer, delete_existing=False)

        indexer.delete_cv.assert_not_called()

    @pytest.mark.anyio
    async def test_custom_source_uri_is_stored(self, cv_file):
        normalized = _make_normalized_cv("minio://cvs/ana.pdf")
        indexer = _make_indexer()

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            result = await ingest_cv(
                cv_file,
                indexer=indexer,
                source_uri="minio://cvs/ana.pdf",
            )

        assert result.source_path == "minio://cvs/ana.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# ingest_cv — error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestCvErrors:
    @pytest.fixture()
    def cv_file(self, tmp_path) -> Path:
        p = tmp_path / "cv.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        return p

    @pytest.mark.anyio
    async def test_extraction_failure_returns_failed_result(self, cv_file):
        indexer = _make_indexer()

        with patch(
            "cv_ingestion.pipeline.pipeline.extract_text",
            side_effect=RuntimeError("PDF corrupt"),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.success is False
        assert result.chunks_indexed == 0

    @pytest.mark.anyio
    async def test_empty_text_returns_failed_result(self, cv_file):
        indexer = _make_indexer()

        with patch(
            "cv_ingestion.pipeline.pipeline.extract_text",
            return_value="   ",
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.anyio
    async def test_normalization_failure_returns_failed_result(self, cv_file):
        indexer = _make_indexer()

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                side_effect=Exception("LLM failed"),
            ),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.success is False

    @pytest.mark.anyio
    async def test_indexing_failure_returns_failed_result(self, cv_file):
        normalized = _make_normalized_cv()
        indexer = _make_indexer()
        indexer.index = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.success is False

    @pytest.mark.anyio
    async def test_error_message_is_stored(self, cv_file):
        indexer = _make_indexer()

        with patch(
            "cv_ingestion.pipeline.pipeline.extract_text",
            side_effect=RuntimeError("something went wrong"),
        ):
            result = await ingest_cv(cv_file, indexer=indexer)

        assert result.error is not None
        assert len(result.error) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ingest_directory
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestDirectory:
    @pytest.mark.anyio
    async def test_raises_when_directory_does_not_exist(self, tmp_path):
        indexer = _make_indexer()
        missing = tmp_path / "nonexistent"

        with pytest.raises(CVIngestionConfigurationError):
            await ingest_directory(missing, indexer=indexer)

    @pytest.mark.anyio
    async def test_empty_directory_returns_zero_total(self, tmp_path):
        indexer = _make_indexer()

        result = await ingest_directory(tmp_path, indexer=indexer)

        assert isinstance(result, BatchIngestResult)
        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0

    @pytest.mark.anyio
    async def test_single_cv_success(self, tmp_path):
        (tmp_path / "cv.pdf").write_bytes(b"%PDF-1.4 fake")
        indexer = _make_indexer()
        normalized = _make_normalized_cv()

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            result = await ingest_directory(tmp_path, indexer=indexer)

        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0

    @pytest.mark.anyio
    async def test_failed_cv_does_not_abort_batch(self, tmp_path):
        (tmp_path / "good.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 fake")
        indexer = _make_indexer()
        normalized = _make_normalized_cv()

        call_count = 0

        def extract_text_side_effect(path):
            nonlocal call_count
            call_count += 1
            if "bad" in str(path):
                raise RuntimeError("corrupt PDF")
            return "CV text"

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                side_effect=extract_text_side_effect,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                return_value=normalized,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            result = await ingest_directory(tmp_path, indexer=indexer)

        assert result.total == 2
        assert result.succeeded == 1
        assert result.failed == 1

    @pytest.mark.anyio
    async def test_batch_result_errors_property(self, tmp_path):
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 fake")
        indexer = _make_indexer()

        with patch(
            "cv_ingestion.pipeline.pipeline.extract_text",
            side_effect=RuntimeError("failed"),
        ):
            result = await ingest_directory(tmp_path, indexer=indexer)

        assert len(result.errors) == 1
        assert result.errors[0].success is False

    @pytest.mark.anyio
    async def test_create_collection_is_called(self, tmp_path):
        indexer = _make_indexer()

        await ingest_directory(tmp_path, indexer=indexer)

        indexer.create_collection.assert_called_once()

    @pytest.mark.anyio
    async def test_recreate_collection_passed_through(self, tmp_path):
        indexer = _make_indexer()

        await ingest_directory(tmp_path, indexer=indexer, recreate_collection=True)

        indexer.create_collection.assert_called_once_with(recreate=True)

    @pytest.mark.anyio
    async def test_source_uri_by_local_path_overrides_local_paths(self, tmp_path):
        cv_path = tmp_path / "cv.pdf"
        cv_path.write_bytes(b"%PDF-1.4 fake")
        minio_uri = "minio://cvs/cv.pdf"
        normalized = _make_normalized_cv(minio_uri)
        indexer = _make_indexer()

        captured_source_path = []

        def capture_normalizer(source_path, text):
            captured_source_path.append(source_path)
            return normalized

        with (
            patch(
                "cv_ingestion.pipeline.pipeline.extract_text",
                return_value="CV text",
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.extract_normalized_cv",
                side_effect=capture_normalizer,
            ),
            patch(
                "cv_ingestion.pipeline.pipeline.build_chunks",
                return_value=[MagicMock()],
            ),
        ):
            await ingest_directory(
                tmp_path,
                indexer=indexer,
                source_uri_by_local_path={cv_path: minio_uri},
            )

        assert captured_source_path[0] == minio_uri
