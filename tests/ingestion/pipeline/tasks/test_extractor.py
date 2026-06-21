"""Tests for cv_ingestion.pipeline.tasks.extractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cv_ingestion.exceptions import UnsupportedCVFormatError
from cv_ingestion.pipeline.tasks.extractor import extract_text


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_pdf_reader(pages: list[str | None]):
    """Return a mock PdfReader whose pages yield the given text values."""
    reader = MagicMock()
    reader.pages = [MagicMock(extract_text=MagicMock(return_value=p)) for p in pages]
    return reader


# ─────────────────────────────────────────────────────────────────────────────
# .pdf extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractTextPdf:
    def test_single_page_returns_page_text(self, tmp_path):
        pdf_path = tmp_path / "cv.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "cv_ingestion.pipeline.tasks.extractor.PdfReader",
            return_value=_fake_pdf_reader(["Page one content"]),
        ):
            result = extract_text(pdf_path)

        assert result == "Page one content"

    def test_multiple_pages_joined_with_double_newline(self, tmp_path):
        pdf_path = tmp_path / "cv.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "cv_ingestion.pipeline.tasks.extractor.PdfReader",
            return_value=_fake_pdf_reader(["Page 1", "Page 2", "Page 3"]),
        ):
            result = extract_text(pdf_path)

        assert result == "Page 1\n\nPage 2\n\nPage 3"

    def test_empty_pages_are_skipped(self, tmp_path):
        pdf_path = tmp_path / "cv.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "cv_ingestion.pipeline.tasks.extractor.PdfReader",
            return_value=_fake_pdf_reader(["Page 1", "", None, "Page 3"]),
        ):
            result = extract_text(pdf_path)

        assert result == "Page 1\n\nPage 3"

    def test_whitespace_only_pages_are_skipped(self, tmp_path):
        pdf_path = tmp_path / "cv.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "cv_ingestion.pipeline.tasks.extractor.PdfReader",
            return_value=_fake_pdf_reader(["Real content", "   \n\t  "]),
        ):
            result = extract_text(pdf_path)

        assert result == "Real content"

    def test_all_empty_pages_returns_empty_string(self, tmp_path):
        pdf_path = tmp_path / "cv.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "cv_ingestion.pipeline.tasks.extractor.PdfReader",
            return_value=_fake_pdf_reader([None, "", "   "]),
        ):
            result = extract_text(pdf_path)

        assert result == ""

    def test_case_insensitive_suffix(self, tmp_path):
        pdf_path = tmp_path / "cv.PDF"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "cv_ingestion.pipeline.tasks.extractor.PdfReader",
            return_value=_fake_pdf_reader(["Content"]),
        ):
            result = extract_text(pdf_path)

        assert result == "Content"


# ─────────────────────────────────────────────────────────────────────────────
# Unsupported formats
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractTextUnsupportedFormat:
    @pytest.mark.parametrize(
        "suffix",
        [".txt", ".docx", ".png", ".jpg", ".xlsx", ".html"],
    )
    def test_unsupported_suffix_raises(self, tmp_path, suffix):
        path = tmp_path / f"cv{suffix}"
        path.write_bytes(b"fake content")

        with pytest.raises(UnsupportedCVFormatError, match=suffix):
            extract_text(path)

    def test_error_message_includes_suffix(self, tmp_path):
        path = tmp_path / "cv.docx"
        path.write_bytes(b"fake")

        with pytest.raises(UnsupportedCVFormatError, match=r"\.docx"):
            extract_text(path)
