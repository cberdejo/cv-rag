"""Tests for cv_ingestion.pipeline.tasks.chunker."""

from __future__ import annotations

from datetime import date

from cv_ingestion.models.chunking import CVSection
from cv_ingestion.models.normalized_cv import (
    AdditionalSection,
    CertificationEntry,
    ContactInfo,
    EducationEntry,
    ExperienceEntry,
    LanguageEntry,
    LinkEntry,
    NormalizedCV,
    ProjectEntry,
)
from cv_ingestion.pipeline.tasks.chunker import INDEXED_SECTIONS, build_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _full_cv() -> NormalizedCV:
    return NormalizedCV(
        source_path="minio://cvs/juan-garcia.pdf",
        contact=ContactInfo(
            name="Juan García",
            email="juan.garcia01@example.com",
            phone="+34612345678",
        ),
        about="Experienced backend engineer.",
        experience=[
            ExperienceEntry(
                title="Senior Backend Engineer",
                company="Accenture",
                start_date=date(2020, 1, 1),
                is_current=True,
                description="Built FastAPI services.",
            )
        ],
        education=[
            EducationEntry(
                degree="Bachelor's Degree in Computer Engineering",
                institution="UPM",
                start_date=date(2016, 9, 1),
                end_date=date(2020, 6, 30),
            )
        ],
        skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        certifications=[CertificationEntry(name="AWS Certified", issuer="Amazon")],
        languages=[LanguageEntry(language="English", proficiency="Native")],
        projects=[ProjectEntry(name="CVault", description="CV ingestion pipeline.")],
        links=[LinkEntry(label="GitHub", url="https://github.com/juan")],
        additional_sections=[
            AdditionalSection(title="Awards", content="Best Hackathon 2022.")
        ],
        current_title="Senior Backend Engineer",
        current_company="Accenture",
        years_of_experience=4,
    )


def _minimal_cv() -> NormalizedCV:
    """CV with only contact info — other sections empty."""
    return NormalizedCV(
        source_path="test.pdf",
        contact=ContactInfo(name="Ana García", email="ana@example.com"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# INDEXED_SECTIONS
# ─────────────────────────────────────────────────────────────────────────────


class TestIndexedSections:
    def test_contains_contact(self):
        assert CVSection.CONTACT in INDEXED_SECTIONS

    def test_contains_experience(self):
        assert CVSection.EXPERIENCE in INDEXED_SECTIONS

    def test_contains_education(self):
        assert CVSection.EDUCATION in INDEXED_SECTIONS

    def test_contains_skills(self):
        assert CVSection.SKILLS in INDEXED_SECTIONS

    def test_does_not_contain_unknown(self):
        assert CVSection.UNKNOWN not in INDEXED_SECTIONS

    def test_contains_certifications(self):
        assert CVSection.CERTIFICATIONS in INDEXED_SECTIONS

    def test_contains_languages(self):
        assert CVSection.LANGUAGES in INDEXED_SECTIONS

    def test_contains_projects(self):
        assert CVSection.PROJECTS in INDEXED_SECTIONS

    def test_contains_links(self):
        assert CVSection.LINKS in INDEXED_SECTIONS

    def test_contains_additional(self):
        assert CVSection.ADDITIONAL in INDEXED_SECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# build_chunks — basic structure
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildChunksStructure:
    def test_returns_list(self):
        chunks = build_chunks(_full_cv())
        assert isinstance(chunks, list)

    def test_full_cv_produces_multiple_chunks(self):
        chunks = build_chunks(_full_cv())
        assert len(chunks) >= 4  # contact, about, experience, education, skills

    def test_empty_sections_are_skipped(self):
        cv = _minimal_cv()  # no experience, education, skills, about
        chunks = build_chunks(cv)
        sections = {c.metadata.section for c in chunks}
        assert CVSection.EXPERIENCE not in sections
        assert CVSection.EDUCATION not in sections
        assert CVSection.SKILLS not in sections

    def test_chunk_indexes_are_sequential(self):
        chunks = build_chunks(_full_cv())
        indexes = [c.metadata.chunk_index for c in chunks]
        assert indexes == list(range(len(chunks)))

    def test_each_chunk_has_unique_section(self):
        chunks = build_chunks(_full_cv())
        sections = [c.metadata.section for c in chunks]
        assert len(sections) == len(set(sections))

    def test_no_chunks_when_cv_is_completely_empty(self):
        cv = NormalizedCV(
            source_path="empty.pdf",
            contact=ContactInfo(),
        )
        chunks = build_chunks(cv)
        assert chunks == []


# ─────────────────────────────────────────────────────────────────────────────
# build_chunks — page_content format
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildChunksPageContent:
    def test_page_content_contains_context_header(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert "[CONTEXT]" in chunk.page_content

    def test_page_content_contains_section_marker(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert "[SECTION:" in chunk.page_content

    def test_experience_chunk_contains_section_text(self):
        chunks = build_chunks(_full_cv())
        exp_chunk = next(
            c for c in chunks if c.metadata.section == CVSection.EXPERIENCE
        )
        assert "Accenture" in exp_chunk.page_content
        assert "Senior Backend Engineer" in exp_chunk.page_content

    def test_contact_chunk_contains_name(self):
        chunks = build_chunks(_full_cv())
        contact_chunk = next(
            c for c in chunks if c.metadata.section == CVSection.CONTACT
        )
        assert "Juan García" in contact_chunk.page_content

    def test_skills_chunk_lists_skills(self):
        chunks = build_chunks(_full_cv())
        skills_chunk = next(c for c in chunks if c.metadata.section == CVSection.SKILLS)
        assert "Python" in skills_chunk.page_content
        assert "FastAPI" in skills_chunk.page_content

    def test_additional_section_chunk_contains_title_and_content(self):
        chunks = build_chunks(_full_cv())
        additional_chunk = next(
            c for c in chunks if c.metadata.section == CVSection.ADDITIONAL
        )
        assert "Awards" in additional_chunk.page_content
        assert "Best Hackathon 2022." in additional_chunk.page_content

    def test_context_contains_candidate_name(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert "Juan García" in chunk.page_content

    def test_context_contains_current_title(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert "Senior Backend Engineer" in chunk.page_content

    def test_full_page_content_structure(self):
        """Verify the exact format produced by compose_page_content."""
        cv = NormalizedCV(
            source_path="/tmp/juan-garcia.pdf",
            contact=ContactInfo(
                name="Juan Garcia",
                email="juan@example.com",
                phone="+34612345678",
            ),
            experience=[
                ExperienceEntry(
                    title="Senior Backend Engineer",
                    company="Accenture",
                    start_date=date(2020, 1, 1),
                    is_current=True,
                    description="Built FastAPI services.",
                )
            ],
            education=[
                EducationEntry(
                    degree="Bachelor's Degree in Computer Engineering",
                    institution="UPM",
                )
            ],
            skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
            current_title="Senior Backend Engineer",
            current_company="Accenture",
        )
        chunks = build_chunks(cv)
        exp_chunk = next(
            c for c in chunks if c.metadata.section == CVSection.EXPERIENCE
        )

        assert exp_chunk.page_content == (
            "[CONTEXT]\n"
            "Candidate: Juan Garcia | Senior Backend Engineer @ Accenture\n"
            "Skills: Python, FastAPI, Docker, PostgreSQL\n"
            "Education: Bachelor's Degree in Computer Engineering (UPM)\n"
            "Email: juan@example.com\n"
            "---\n"
            "[SECTION: experience]\n"
            "Senior Backend Engineer \u2014 Accenture (Jan 2020 \u2013 Present)\n"
            "Built FastAPI services."
        )


# ─────────────────────────────────────────────────────────────────────────────
# build_chunks — metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildChunksMetadata:
    def test_skills_propagated_to_all_chunks(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert chunk.metadata.skills == [
                "Python",
                "FastAPI",
                "Docker",
                "PostgreSQL",
            ]

    def test_source_path_propagated_to_all_chunks(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert chunk.metadata.source_path == "minio://cvs/juan-garcia.pdf"

    def test_candidate_name_propagated(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert chunk.metadata.candidate_name == "Juan García"

    def test_current_title_propagated(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert chunk.metadata.current_title == "Senior Backend Engineer"

    def test_companies_propagated(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert "Accenture" in chunk.metadata.companies

    def test_degrees_propagated(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert "Bachelor's Degree in Computer Engineering" in chunk.metadata.degrees

    def test_email_propagated(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert chunk.metadata.email == "juan.garcia01@example.com"

    def test_section_matches_chunk_content(self):
        chunks = build_chunks(_full_cv())
        for chunk in chunks:
            assert f"[SECTION: {chunk.metadata.section.value}]" in chunk.page_content
