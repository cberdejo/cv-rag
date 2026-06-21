"""
CV chunking.

Strategy: one chunk per meaningful section, each enriched with a compact
context header that summarises the rest of the CV.
"""

from __future__ import annotations

from cv_ingestion.models.chunking import CVChunk, CVSection
from cv_ingestion.models.normalized_cv import NormalizedCV

INDEXED_SECTIONS = (
    CVSection.CONTACT,
    CVSection.ABOUT,
    CVSection.EXPERIENCE,
    CVSection.EDUCATION,
    CVSection.SKILLS,
    CVSection.CERTIFICATIONS,
    CVSection.LANGUAGES,
    CVSection.PROJECTS,
    CVSection.LINKS,
    CVSection.ADDITIONAL,
)


def build_chunks(cv: NormalizedCV) -> list[CVChunk]:
    """Return one enriched CVChunk per meaningful section of *cv*."""
    context = cv.context_summary()
    shared_metadata = cv.shared_metadata()

    chunks: list[CVChunk] = []
    for section in INDEXED_SECTIONS:
        section_text = cv.section_text(section)
        if not section_text.strip():
            continue

        chunks.append(
            CVChunk(
                page_content=CVChunk.compose_page_content(
                    context=context,
                    section=section,
                    section_text=section_text,
                ),
                metadata=shared_metadata.model_copy(
                    update={
                        "section": section,
                        "chunk_index": len(chunks),
                    }
                ),
            )
        )

    return chunks


__all__ = ["INDEXED_SECTIONS", "build_chunks"]
