"""Tests for Chainlit CV preview helpers."""

from __future__ import annotations

from pathlib import Path

import chainlit as cl
import pytest

from rag.frontend import chainlit_app
from rag.assistant import AssistantResponse, CVSource

pytestmark = pytest.mark.anyio


async def test_parse_minio_uri_extracts_bucket_and_object() -> None:
    result = chainlit_app.parse_minio_uri("minio://cvs/generated/ana%20garcia.pdf")

    assert result == ("cvs", "generated/ana garcia.pdf")


async def test_resolve_cv_preview_uses_existing_local_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ana.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    preview = await chainlit_app.resolve_cv_preview(str(pdf_path))

    assert preview is not None
    assert preview.path == pdf_path.resolve()
    assert preview.filename == "ana.pdf"


async def test_resolve_cv_preview_downloads_minio_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeMinioClient:
        def fget_object(self, bucket: str, object_name: str, target_path: str) -> None:
            calls.append((bucket, object_name, target_path))
            Path(target_path).write_bytes(b"%PDF-1.4\n")

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(chainlit_app, "CV_PREVIEW_CACHE_DIR", tmp_path)
    monkeypatch.setattr(chainlit_app, "get_minio_client", lambda: FakeMinioClient())

    preview = await chainlit_app.resolve_cv_preview("minio://cvs/ana.pdf")

    assert preview is not None
    assert preview.path.exists()
    assert preview.filename == "ana.pdf"
    assert calls == [("cvs", "ana.pdf", str(preview.path))]


async def test_build_cv_source_elements_adds_page_pdf_preview(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "ana.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    response = AssistantResponse(
        answer="Ana is a good match for the search.",
        sources=[
            CVSource(
                id="source-1",
                candidate_name="Ana Garcia",
                source_path=str(pdf_path),
                section="experience",
                score=0.91,
                snippet="Python experience.",
            )
        ],
    )

    elements = await chainlit_app.build_cv_source_elements(
        response, thread_id="test-thread"
    )

    assert [type(element) for element in elements] == [cl.Pdf]
    assert elements[0].name == "CV - Ana Garcia"
    assert elements[0].path == str(pdf_path.resolve())
    assert elements[0].display == "page"
    assert elements[0].page == 1


async def test_append_cv_source_links_references_pdf_element_names() -> None:
    elements = [
        cl.Pdf(
            name="CV - Ana Garcia",
            path="/tmp/ana.pdf",
            display="page",
            page=1,
            thread_id="test-thread",
        ),
        cl.Text(
            name="CVs without preview",
            content="missing",
            display="inline",
            thread_id="test-thread",
        ),
    ]

    content = chainlit_app.append_cv_source_links("Ana is a good match.", elements)

    assert content == "Ana is a good match.\n\nDocuments:\n- CV - Ana Garcia"


async def test_build_cv_source_elements_falls_back_to_text_for_missing_pdf() -> None:
    response = AssistantResponse(
        answer="No preview is available.",
        sources=[
            CVSource(
                id="source-1",
                candidate_name=None,
                source_path="minio://cvs/missing.txt",
                section=None,
                score=None,
                snippet="Retrieved content.",
            )
        ],
    )

    elements = await chainlit_app.build_cv_source_elements(
        response, thread_id="test-thread"
    )

    assert len(elements) == 1
    assert isinstance(elements[0], cl.Text)
    assert elements[0].name == "CVs without preview"
    assert elements[0].display == "inline"
    assert "CV not available for preview" in elements[0].content
