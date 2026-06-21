"""
Pydantic models for CV generation and validation.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class Gender(str, Enum):
    """Supported gender values for generated candidate identity data."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


PhoneStr = Annotated[
    str,
    Field(
        pattern=r"^\+?[0-9\s\-\(\)]{7,20}$",
        description="Mobile phone number; accepts international format.",
        examples=["+34 612 345 678", "612345678"],
    ),
]

MIN_YEAR: int = 1950
MAX_YEAR: int = date.today().year


def _is_bachelor_degree(title: str) -> bool:
    """Return whether an education title looks like a bachelor degree."""
    return "bachelor" in title.lower()


def _is_master_degree(title: str) -> bool:
    """Return whether an education title looks like a master degree."""
    normalized_title = title.lower().strip()
    return "master" in normalized_title or normalized_title == "mba"


class Period(BaseModel):
    """
    Represents a period with a required start date and an optional end date.

    The end date can be None if the activity is ongoing.

    Invariant:
    - start_date <= end_date when end_date is not None.
    """

    start_date: date = Field(description="Start date of the period.")
    end_date: date | None = Field(
        default=None,
        description="End date. None indicates that the period is currently active.",
    )

    @model_validator(mode="after")
    def end_after_start(self) -> "Period":
        """Ensure an ended period does not finish before it starts."""
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(
                f"The end date ({self.end_date}) cannot be earlier "
                f"than the start date ({self.start_date})."
            )
        return self

    @property
    def current(self) -> bool:
        """Returns True if the period is currently active."""
        return self.end_date is None


class Experience(BaseModel):
    """
    Professional experience entry in a generated CV.

    The pipeline keeps these entries ordered from most recent to oldest and
    validates overlap at the parent ``CV`` level.
    """

    title: str = Field(
        min_length=2,
        max_length=120,
        description="Job title, e.g. 'Senior Backend Engineer'.",
    )
    company: str = Field(
        min_length=1,
        max_length=120,
        description="Name of the company or organization.",
    )
    description: str = Field(
        min_length=10,
        max_length=2000,
        description="Description of responsibilities and achievements in the role.",
    )
    period: Period = Field(description="Employment start and end period.")


class Education(BaseModel):
    """
    Academic education entry in a generated CV.

    The parent ``CV`` validates non-overlap and requires bachelor-level study
    before master-level study when a master's entry is present.
    """

    institution: str = Field(
        min_length=2,
        max_length=200,
        description="Name of the educational institution or university.",
    )
    title: str = Field(
        min_length=2,
        max_length=200,
        description="Degree or qualification obtained, e.g. 'Bachelor's Degree in Computer Engineering'.",
    )
    period: Period = Field(description="Start and end period of the studies.")
    gpa: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Grade point average on a 0–10 scale. Omit if not applicable.",
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Final projects, honors, notable activities, etc.",
    )


class CV(BaseModel):
    """
    Complete generated Curriculum Vitae.

    Validators enforce realistic chronology for work and education, reject
    duplicate skills case-insensitively, and keep skill values non-empty.
    """

    name: str = Field(
        min_length=2,
        max_length=120,
        description="Candidate full name.",
    )
    email: EmailStr = Field(description="Professional email address.")
    phone: PhoneStr = Field(description="Mobile phone number.")
    gender: Gender = Field(description="Candidate gender.")

    about: str = Field(
        min_length=50,
        max_length=2000,
        description="Introduction paragraph / professional summary.",
    )
    experience: list[Experience] = Field(
        default_factory=list,
        description="List of work experiences, ordered from most recent to oldest.",
    )
    education: list[Education] = Field(
        default_factory=list,
        description="Academic background, ordered from most recent to oldest.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and soft skills.",
    )

    @field_validator("skills", mode="before")
    @classmethod
    def validate_non_empty_skills(cls, value: list[str]) -> list[str]:
        """Strip skills and reject empty or whitespace-only values."""
        cleaned_skills = [skill.strip() for skill in value if skill.strip()]

        if len(cleaned_skills) != len(value):
            raise ValueError("Skills cannot be empty or whitespace-only strings.")

        return cleaned_skills

    @model_validator(mode="after")
    def validate_experience(self) -> "CV":
        """
        Rules for work experience:

        1. At most one position can be ongoing.
        2. Periods must not overlap with each other.
        3. Simultaneous jobs are not allowed.
        """

        jobs = self.experience

        if not jobs:
            return self

        current_jobs = [job for job in jobs if job.period.current]

        if len(current_jobs) > 1:
            companies = ", ".join(job.company for job in current_jobs)
            raise ValueError(
                f"There can only be one ongoing job at a time. "
                f"There are {len(current_jobs)} jobs without an end date: {companies}."
            )

        today = date.today()

        ranges: list[tuple[date, date, str]] = [
            (
                job.period.start_date,
                job.period.end_date if job.period.end_date is not None else today,
                job.company,
            )
            for job in jobs
        ]

        for i, (start_a, end_a, company_a) in enumerate(ranges):
            for j, (start_b, end_b, company_b) in enumerate(ranges):
                if i >= j:
                    continue

                if max(start_a, start_b) < min(end_a, end_b):
                    raise ValueError(
                        f"Employment overlap detected between '{company_a}' "
                        f"({start_a} – {end_a}) and '{company_b}' ({start_b} – {end_b}). "
                        "Simultaneous employment is not allowed."
                    )

        return self

    @model_validator(mode="after")
    def validate_education(self) -> "CV":
        """
        Rules for education:

        1. At most one education entry can be ongoing.
        2. Periods must not overlap.
        """

        studies = self.education

        if not studies:
            return self

        current_studies = [study for study in studies if study.period.current]

        if len(current_studies) > 1:
            titles = ", ".join(study.title for study in current_studies)
            raise ValueError(
                f"There can only be one ongoing education entry at a time. "
                f"There are {len(current_studies)} entries without an end date: {titles}."
            )

        today = date.today()

        ranges: list[tuple[date, date, str]] = [
            (
                study.period.start_date,
                study.period.end_date if study.period.end_date is not None else today,
                study.title,
            )
            for study in studies
        ]

        for i, (start_a, end_a, title_a) in enumerate(ranges):
            for j, (start_b, end_b, title_b) in enumerate(ranges):
                if i >= j:
                    continue

                if max(start_a, start_b) < min(end_a, end_b):
                    raise ValueError(
                        f"Education overlap detected between '{title_a}' "
                        f"({start_a} – {end_a}) and '{title_b}' ({start_b} – {end_b})."
                    )

        bachelor_end_dates = [
            study.period.end_date
            for study in studies
            if _is_bachelor_degree(study.title) and study.period.end_date is not None
        ]
        master_studies = [study for study in studies if _is_master_degree(study.title)]

        for master in master_studies:
            has_previous_bachelor = any(
                end_date < master.period.start_date for end_date in bachelor_end_dates
            )
            if not has_previous_bachelor:
                raise ValueError(
                    f"Master education '{master.title}' requires a previously "
                    "completed bachelor education."
                )

        return self

    @model_validator(mode="after")
    def validate_unique_skills(self) -> "CV":
        """
        There must not be two skills with the same name, case-insensitively.
        """

        normalized_skills = [skill.lower() for skill in self.skills]
        seen_skills: set[str] = set()

        for skill in normalized_skills:
            if skill in seen_skills:
                raise ValueError(f"Duplicate skill: '{skill}'.")

            seen_skills.add(skill)

        return self


__all__ = [
    "CV",
    "Education",
    "Experience",
    "Gender",
    "Period",
    "PhoneStr",
]
