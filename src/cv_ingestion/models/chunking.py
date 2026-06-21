"""Models for CV chunking and vector-index payload metadata."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CVSection(StrEnum):
    """Canonical section identifiers used for chunk routing and metadata."""

    CONTACT = "contact"
    ABOUT = "about"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"
    PROJECTS = "projects"
    LINKS = "links"
    ADDITIONAL = "additional"
    UNKNOWN = "unknown"


class CVChunkMetadata(BaseModel):
    """
    Qdrant payload metadata for a CV chunk.

    Each chunk stores both its section origin and CV-level fields so retrieval
    filters can match candidate identity, skills, education, companies, and
    contact data without reloading the original CV.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity
    candidate_name: str | None = None
    source_path: str

    # Section origin
    section: CVSection
    chunk_index: int = Field(
        ge=0,
        description="Position of this chunk among all chunks for this CV.",
    )

    # Skill index
    skills: list[str] = Field(
        default_factory=list,
        description="All skills from the full CV (not only this section).",
    )

    # Role
    current_title: str | None = None
    current_company: str | None = None
    years_of_experience: int | None = None

    # Education
    degrees: list[str] = Field(
        default_factory=list,
        description="Degree titles across all education entries.",
    )
    institutions: list[str] = Field(
        default_factory=list,
        description="Institution names across all education entries.",
    )

    # Companies
    companies: list[str] = Field(
        default_factory=list,
        description="All companies the candidate has worked at.",
    )

    # Certifications and languages
    certifications: list[str] = Field(
        default_factory=list,
        description="Certification names across all certification entries.",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Language names across all language entries.",
    )

    # Contact
    email: str | None = None
    phone: str | None = None


class CVChunk(BaseModel):
    """
    A single indexable unit: one CV section enriched with cross-section context.

    The ``page_content`` fed to the embedding model contains:
    - The section text (primary signal)
    - A compact CV context header (secondary signal, lower weight)

    This lets semantic search find a chunk both by section content
    ("experience with Kubernetes") and by CV-level facts
    ("Juan Garcia, Senior DevOps").
    """

    model_config = ConfigDict(extra="forbid")

    page_content: str = Field(
        description="Text that will be embedded and indexed.",
    )
    metadata: CVChunkMetadata

    @staticmethod
    def compose_page_content(
        *,
        context: str,
        section: CVSection,
        section_text: str,
    ) -> str:
        """Compose the embedded chunk text from shared context and section text."""
        return f"[CONTEXT]\n{context}\n---\n[SECTION: {section.value}]\n{section_text}"


__all__ = ["CVChunk", "CVChunkMetadata", "CVSection"]
