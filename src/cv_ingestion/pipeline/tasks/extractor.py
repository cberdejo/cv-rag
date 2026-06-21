"""Text extraction helpers for ingestion sources."""

from __future__ import annotations

from pathlib import Path

from cv_ingestion.exceptions import UnsupportedCVFormatError
from pypdf import PdfReader


def _extract_pdf(path: Path) -> str:
    """Extract text from all non-empty pages of a PDF file."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page for page in pages if page.strip())


def extract_text(path: Path) -> str:
    """Extract plain text from a CV artifact."""
    match path.suffix.lower():
        case ".pdf":
            return _extract_pdf(path)
        case suffix:
            raise UnsupportedCVFormatError(f"Unsupported CV format: {suffix}")


__all__ = ["extract_text"]
