"""Models for structured, normalized CV data."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cv_ingestion.models.chunking import CVChunkMetadata, CVSection


# E.164-ish: optional leading +, then digits/spaces/dashes/parens, 7-20 chars
_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,20}$")


NormalizedPhone = Annotated[
    str,
    Field(
        description="Phone normalised to digits-only with optional leading +.",
        examples=["+34612345678", "612345678"],
    ),
]


def normalize_phone(raw: str) -> str | None:
    """Strip formatting from a raw phone string. Returns None if unrecognisable."""
    cleaned = raw.strip()
    if not _PHONE_RE.match(cleaned):
        return None
    # Keep leading + if present; strip everything else
    has_plus = cleaned.startswith("+")
    digits = re.sub(r"\D", "", cleaned)
    return ("+" if has_plus else "") + digits


def parse_cv_date(value: Any) -> date | None:
    """
    Parse CV date values into a date using the first day when only year/month
    precision is available.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        if re.fullmatch(r"\d{4}", text):
            return date(int(text), 1, 1)
        if re.fullmatch(r"\d{4}-\d{2}", text):
            year, month = text.split("-")
            return date(int(year), int(month), 1)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return date.fromisoformat(text)
    except ValueError:
        return None

    return None


def format_date_range(
    start_date: date | None,
    end_date: date | None,
    is_current: bool = False,
) -> str:
    """Render a compact date range for CV section text."""
    parts = []
    if start_date:
        parts.append(start_date.strftime("%b %Y"))
    if is_current:
        parts.append("Present")
    elif end_date:
        parts.append(end_date.strftime("%b %Y"))
    return " – ".join(parts)


def _clean_optional_text(value: Any) -> str | None:
    """Strip optional text and convert blank values to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_required_text(value: Any) -> str:
    """Strip required text and convert missing values to an empty string."""
    if value is None:
        return ""
    return str(value).strip()


def _normalise_skills(value: Any) -> list[str]:
    """Normalize, deduplicate, and preserve the original casing of skills."""
    if value is None:
        return []

    raw_items = [value] if isinstance(value, str) else value
    if not isinstance(raw_items, list):
        return []

    seen: set[str] = set()
    result: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            continue
        skill = item.strip()
        key = skill.lower()
        if skill and key not in seen:
            seen.add(key)
            result.append(skill)
    return result


class ContactInfo(BaseModel):
    """Normalized contact block extracted from a CV."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, description="Candidate full name.")
    email: str | None = Field(default=None, description="Candidate email address.")
    phone: NormalizedPhone | None = Field(
        default=None,
        description="Candidate phone number normalized by ``normalize_phone``.",
    )
    location: str | None = Field(
        default=None,
        description="Candidate location text when present in the source CV.",
    )

    @field_validator("name", "email", "location", mode="before")
    @classmethod
    def _clean_text(cls, v: Any) -> str | None:
        """Clean optional contact text fields before validation."""
        return _clean_optional_text(v)

    @field_validator("phone", mode="before")
    @classmethod
    def _normalise_phone(cls, v: Any) -> str | None:
        """Normalize phone values before validation."""
        if v is None:
            return None
        return normalize_phone(str(v))

    def to_text(self) -> str:
        """Return contact fields as a clean plain-text block."""
        return "\n".join(
            item for item in [self.name, self.email, self.phone, self.location] if item
        )


class ExperienceEntry(BaseModel):
    """One normalized work experience record from LLM extraction."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", description="Role title.")
    company: str = Field(default="", description="Employer name.")
    start_date: date | None = Field(default=None, description="Role start date.")
    end_date: date | None = Field(default=None, description="Role end date.")
    is_current: bool = Field(
        default=False,
        description="Whether the role is ongoing even when no end date is present.",
    )
    description: str | None = Field(
        default=None,
        description="Responsibilities and achievements text from the CV.",
    )

    @field_validator("title", "company", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean required experience text fields before validation."""
        return _clean_required_text(v)

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, v: Any) -> str | None:
        """Clean optional experience descriptions before validation."""
        return _clean_optional_text(v)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_dates(cls, value: Any) -> date | None:
        """Parse date-like LLM values before Pydantic date validation."""
        return parse_cv_date(value)

    def to_text(self) -> str:
        """Render this experience entry as plain text."""
        header = " — ".join(part for part in [self.title, self.company] if part)
        date_range = format_date_range(
            self.start_date,
            self.end_date,
            self.is_current,
        )
        if date_range:
            header = f"{header} ({date_range})" if header else date_range

        lines = [part for part in [header, self.description] if part]
        return "\n".join(lines)


class EducationEntry(BaseModel):
    """One normalized education record from LLM extraction."""

    model_config = ConfigDict(extra="forbid")

    degree: str = Field(default="", description="Degree or credential title.")
    institution: str = Field(default="", description="Institution name.")
    start_date: date | None = Field(default=None, description="Study start date.")
    end_date: date | None = Field(default=None, description="Study end date.")
    description: str | None = Field(
        default=None,
        description="Additional education details from the CV.",
    )

    @field_validator("degree", "institution", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean required education text fields before validation."""
        return _clean_required_text(v)

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, v: Any) -> str | None:
        """Clean optional education descriptions before validation."""
        return _clean_optional_text(v)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_dates(cls, value: Any) -> date | None:
        """Parse date-like LLM values before Pydantic date validation."""
        return parse_cv_date(value)

    def to_text(self) -> str:
        """Render this education entry as plain text."""
        header = " — ".join(part for part in [self.degree, self.institution] if part)
        date_range = format_date_range(self.start_date, self.end_date)
        if date_range:
            header = f"{header} ({date_range})" if header else date_range

        lines = [part for part in [header, self.description] if part]
        return "\n".join(lines)


class CertificationEntry(BaseModel):
    """One normalized certification record from LLM extraction."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="Certification or credential name.")
    issuer: str | None = Field(default=None, description="Issuing organization.")
    issue_date: date | None = Field(default=None, description="Date the credential was issued.")
    description: str | None = Field(
        default=None,
        description="Additional certification details from the CV.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean the required certification name before validation."""
        return _clean_required_text(v)

    @field_validator("issuer", "description", mode="before")
    @classmethod
    def _clean_optional_fields(cls, v: Any) -> str | None:
        """Clean optional certification text fields before validation."""
        return _clean_optional_text(v)

    @field_validator("issue_date", mode="before")
    @classmethod
    def parse_dates(cls, value: Any) -> date | None:
        """Parse date-like LLM values before Pydantic date validation."""
        return parse_cv_date(value)

    def to_text(self) -> str:
        """Render this certification entry as plain text."""
        header = " — ".join(part for part in [self.name, self.issuer] if part)
        if self.issue_date:
            date_str = self.issue_date.strftime("%b %Y")
            header = f"{header} ({date_str})" if header else date_str

        lines = [part for part in [header, self.description] if part]
        return "\n".join(lines)


class LanguageEntry(BaseModel):
    """One normalized language record from LLM extraction."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(default="", description="Language name.")
    proficiency: str | None = Field(
        default=None,
        description="Proficiency level as stated in the CV (e.g. 'Native', 'B2').",
    )

    @field_validator("language", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean the required language name before validation."""
        return _clean_required_text(v)

    @field_validator("proficiency", mode="before")
    @classmethod
    def _clean_proficiency(cls, v: Any) -> str | None:
        """Clean the optional proficiency text before validation."""
        return _clean_optional_text(v)

    def to_text(self) -> str:
        """Render this language entry as plain text."""
        if self.proficiency:
            return f"{self.language} ({self.proficiency})"
        return self.language


class ProjectEntry(BaseModel):
    """One normalized project record from LLM extraction."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="Project name.")
    description: str | None = Field(default=None, description="Project description.")
    technologies: list[str] = Field(
        default_factory=list,
        description="Technologies or tools used in the project.",
    )
    url: str | None = Field(default=None, description="Project URL, if present.")

    @field_validator("name", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean the required project name before validation."""
        return _clean_required_text(v)

    @field_validator("description", "url", mode="before")
    @classmethod
    def _clean_optional_fields(cls, v: Any) -> str | None:
        """Clean optional project text fields before validation."""
        return _clean_optional_text(v)

    @field_validator("technologies", mode="before")
    @classmethod
    def _dedupe_technologies(cls, v: Any) -> list[str]:
        """Normalize and deduplicate technologies before validation."""
        return _normalise_skills(v)

    def to_text(self) -> str:
        """Render this project entry as plain text."""
        lines = [part for part in [self.name, self.description] if part]
        if self.technologies:
            lines.append(f"Technologies: {', '.join(self.technologies)}")
        if self.url:
            lines.append(self.url)
        return "\n".join(lines)


class LinkEntry(BaseModel):
    """One normalized link/profile record from LLM extraction."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", description="Link label, e.g. LinkedIn, GitHub, Portfolio.")
    url: str = Field(default="", description="URL of the link.")

    @field_validator("label", "url", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean required link fields before validation."""
        return _clean_required_text(v)

    def to_text(self) -> str:
        """Render this link entry as plain text."""
        if self.label:
            return f"{self.label}: {self.url}"
        return self.url


class AdditionalSection(BaseModel):
    """A free-form CV section that does not map to a canonical field.

    Captures content the LLM cannot confidently place elsewhere (e.g.
    volunteer work, publications, awards, references) instead of forcing it
    into an unrelated section or discarding it.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", description="Original section heading from the CV.")
    content: str = Field(default="", description="Raw text content of the section.")

    @field_validator("title", mode="before")
    @classmethod
    def _clean_required_text(cls, v: Any) -> str:
        """Clean the required section title before validation."""
        return _clean_required_text(v)

    @field_validator("content", mode="before")
    @classmethod
    def _clean_content(cls, v: Any) -> str:
        """Clean the required section content before validation."""
        return _clean_required_text(v)

    def to_text(self) -> str:
        """Render this additional section as plain text."""
        if self.title:
            return f"{self.title}\n{self.content}"
        return self.content


class NormalizedCVLLMOutput(BaseModel):
    """Strict structured payload expected from the normalization LLM."""

    model_config = ConfigDict(extra="forbid")

    contact: ContactInfo = Field(description="Extracted candidate contact details.")
    about: str | None = Field(
        default=None,
        description="Professional summary extracted from the CV.",
    )
    experience: list[ExperienceEntry] = Field(
        default_factory=list,
        description="Extracted work experience entries.",
    )
    education: list[EducationEntry] = Field(
        default_factory=list,
        description="Extracted education entries.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Deduplicated skills extracted from the CV.",
    )
    certifications: list[CertificationEntry] = Field(
        default_factory=list,
        description="Extracted certifications and credentials.",
    )
    languages: list[LanguageEntry] = Field(
        default_factory=list,
        description="Extracted spoken/written languages and proficiency.",
    )
    projects: list[ProjectEntry] = Field(
        default_factory=list,
        description="Extracted personal or professional projects.",
    )
    links: list[LinkEntry] = Field(
        default_factory=list,
        description="Extracted profile or portfolio links (LinkedIn, GitHub, etc.).",
    )
    additional_sections: list[AdditionalSection] = Field(
        default_factory=list,
        description=(
            "Any other titled CV section that does not map to a field above "
            "(e.g. awards, publications, volunteering, references)."
        ),
    )

    @field_validator("about", mode="before")
    @classmethod
    def _clean_about(cls, v: Any) -> str | None:
        """Clean optional summary text before validation."""
        return _clean_optional_text(v)

    @field_validator("skills", mode="before")
    @classmethod
    def _dedupe_skills(cls, v: Any) -> list[str]:
        """Normalize and deduplicate skills before validation."""
        return _normalise_skills(v)


class NormalizedCV(BaseModel):
    """Fully normalized CV enriched with source and derived chunk metadata."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(description="Durable source URI or local path.")
    contact: ContactInfo = Field(description="Normalized contact details.")
    about: str | None = Field(default=None, description="Professional summary.")
    experience: list[ExperienceEntry] = Field(
        default_factory=list,
        description="Normalized work experience entries.",
    )
    education: list[EducationEntry] = Field(
        default_factory=list,
        description="Normalized education entries.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Deduplicated skill names.",
    )
    certifications: list[CertificationEntry] = Field(
        default_factory=list,
        description="Normalized certifications and credentials.",
    )
    languages: list[LanguageEntry] = Field(
        default_factory=list,
        description="Normalized spoken/written languages and proficiency.",
    )
    projects: list[ProjectEntry] = Field(
        default_factory=list,
        description="Normalized personal or professional projects.",
    )
    links: list[LinkEntry] = Field(
        default_factory=list,
        description="Normalized profile or portfolio links.",
    )
    additional_sections: list[AdditionalSection] = Field(
        default_factory=list,
        description="Other titled CV sections not covered by a canonical field.",
    )

    years_of_experience: int | None = Field(
        default=None,
        description="Derived total experience in rounded years.",
    )
    current_title: str | None = Field(
        default=None,
        description="Derived title from current or latest experience.",
    )
    current_company: str | None = Field(
        default=None,
        description="Derived company from current or latest experience.",
    )

    @field_validator("about", "current_title", "current_company", mode="before")
    @classmethod
    def _clean_optional_fields(cls, v: Any) -> str | None:
        """Clean optional derived CV metadata fields before validation."""
        return _clean_optional_text(v)

    @field_validator("skills", mode="before")
    @classmethod
    def _dedupe_skills(cls, v: Any) -> list[str]:
        """Normalize and deduplicate final CV skills before validation."""
        return _normalise_skills(v)

    @classmethod
    def from_llm_payload(
        cls,
        data: dict[str, Any],
        *,
        source_path: str,
    ) -> "NormalizedCV":
        """Validate an LLM payload and attach source and derived metadata.

        Args:
            data: Raw JSON-like object returned by the normalization LLM.
            source_path: Durable source identity to store in chunk metadata.

        Returns:
            NormalizedCV with current role and years of experience populated.
        """
        output = NormalizedCVLLMOutput.model_validate(data)
        cv = cls(
            source_path=source_path,
            contact=output.contact,
            about=output.about,
            experience=output.experience,
            education=output.education,
            skills=output.skills,
            certifications=output.certifications,
            languages=output.languages,
            projects=output.projects,
            links=output.links,
            additional_sections=output.additional_sections,
        )
        current = cv._current_experience()
        cv.current_title = current.title if current else None
        cv.current_company = current.company if current else None
        cv.years_of_experience = cv.calculate_years_of_experience()
        return cv

    def calculate_years_of_experience(self) -> int | None:
        """Estimate total work experience in years from dated entries."""
        if not self.experience:
            return None

        today = date.today()
        total_days = 0

        for entry in self.experience:
            if entry.start_date is None:
                continue
            end = (
                entry.end_date
                if entry.end_date is not None and not entry.is_current
                else today
            )
            delta = (end - entry.start_date).days
            if delta > 0:
                total_days += delta

        return max(1, round(total_days / 365)) if total_days > 0 else None

    def context_summary(self) -> str:
        """
        Build a compact cross-section context string for chunk enrichment.

        The summary is prepended to each embedded chunk so a section can be
        retrieved by important CV-level facts such as candidate name, skills,
        latest education, and current role.
        """
        parts: list[str] = []

        name = self.contact.name or "Unknown Candidate"
        identity = name
        if self.current_title:
            identity += f" | {self.current_title}"
        if self.current_company:
            identity += f" @ {self.current_company}"
        parts.append(f"Candidate: {identity}")

        if self.years_of_experience:
            parts.append(f"Level: {self.years_of_experience}y exp")

        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills)}")

        if self.education:
            latest = self.education[0]
            parts.append(f"Education: {latest.degree} ({latest.institution})")

        if self.contact.email:
            parts.append(f"Email: {self.contact.email}")

        return "\n".join(parts)

    def section_text(self, section: CVSection) -> str:
        """Render one canonical CV section as plain text."""
        if section == CVSection.CONTACT:
            return self.contact.to_text()
        if section == CVSection.ABOUT:
            return self.about or ""
        if section == CVSection.EXPERIENCE:
            return "\n\n".join(entry.to_text() for entry in self.experience)
        if section == CVSection.EDUCATION:
            return "\n\n".join(entry.to_text() for entry in self.education)
        if section == CVSection.SKILLS:
            return ", ".join(self.skills)
        if section == CVSection.CERTIFICATIONS:
            return "\n\n".join(entry.to_text() for entry in self.certifications)
        if section == CVSection.LANGUAGES:
            return "\n".join(entry.to_text() for entry in self.languages)
        if section == CVSection.PROJECTS:
            return "\n\n".join(entry.to_text() for entry in self.projects)
        if section == CVSection.LINKS:
            return "\n".join(entry.to_text() for entry in self.links)
        if section == CVSection.ADDITIONAL:
            return "\n\n".join(entry.to_text() for entry in self.additional_sections)
        return ""

    def shared_metadata(self) -> CVChunkMetadata:
        """Build CV-level metadata shared by all chunks for this CV."""
        return CVChunkMetadata(
            candidate_name=self.contact.name,
            source_path=self.source_path,
            section=CVSection.UNKNOWN,
            chunk_index=0,
            skills=self.skills,
            current_title=self.current_title,
            current_company=self.current_company,
            years_of_experience=self.years_of_experience,
            degrees=[entry.degree for entry in self.education],
            institutions=[entry.institution for entry in self.education],
            companies=[entry.company for entry in self.experience],
            certifications=[entry.name for entry in self.certifications],
            languages=[entry.language for entry in self.languages],
            email=self.contact.email,
            phone=self.contact.phone,
        )

    def _current_experience(self) -> ExperienceEntry | None:
        """Return the current experience, or the latest listed experience."""
        return next(
            (entry for entry in self.experience if entry.is_current),
            self.experience[0] if self.experience else None,
        )


__all__ = [
    "AdditionalSection",
    "CertificationEntry",
    "ContactInfo",
    "EducationEntry",
    "ExperienceEntry",
    "LanguageEntry",
    "LinkEntry",
    "NormalizedCV",
    "NormalizedCVLLMOutput",
    "NormalizedPhone",
    "ProjectEntry",
    "format_date_range",
    "normalize_phone",
    "parse_cv_date",
]
