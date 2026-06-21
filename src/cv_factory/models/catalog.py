"""Pydantic models for the built-in career profile catalog."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Seniority(str, Enum):
    """Supported job seniority levels used to shape career progression."""

    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class SkillCategory(str, Enum):
    """Skill buckets used when selecting realistic profile capabilities."""

    CORE = "core"
    SECONDARY = "secondary"
    SOFT = "soft"
    TOOL = "tool"


class Skill(BaseModel):
    """Weighted skill candidate attached to a career profile."""

    name: str = Field(
        min_length=2,
        max_length=80,
        description="Display name used in generated CV skill lists.",
    )
    category: SkillCategory = Field(description="Role of this skill in the profile.")
    weight: int = Field(
        ge=1,
        le=5,
        description="Importance of the skill for this profile.",
    )


class EducationOption(BaseModel):
    """Education path that can be sampled for one career profile."""

    title: str = Field(
        min_length=2,
        max_length=200,
        description="Degree, credential, or learning path title.",
    )
    institution_type: Literal[
        "university",
        "vocational_training",
        "bootcamp",
        "professional_certification",
        "self_taught",
    ]
    level: Literal[
        "certificate",
        "vocational",
        "bachelor",
        "master",
        "phd",
        "bootcamp",
    ]
    relevance: int = Field(
        ge=1,
        le=5,
        description="How typical this education option is for the profile.",
    )


class JobTitleOption(BaseModel):
    """Job title that may be sampled at a specific seniority level."""

    title: str = Field(min_length=2, max_length=120, description="Job title text.")
    seniority: Seniority = Field(description="Seniority represented by this title.")
    min_years_experience: int = Field(
        ge=0,
        le=40,
        description="Minimum plausible experience before using this title.",
    )


class CareerProfile(BaseModel):
    """Static catalog entry used to build coherent CV skeletons."""

    key: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="Internal identifier, e.g. backend_developer.",
    )
    label: str = Field(description="Human-readable profile label.")
    family: str = Field(
        description="Professional family, e.g. software, data, design, healthcare.",
    )
    titles: list[JobTitleOption] = Field(description="Allowed job titles.")
    education: list[EducationOption] = Field(description="Allowed education paths.")
    skills: list[Skill] = Field(description="Skills available for generated CVs.")
    possible_industries: list[str] = Field(
        default_factory=list,
        description="Optional industries that make generated experience more specific.",
    )

    @model_validator(mode="after")
    def validate_profile(self) -> "CareerProfile":
        """Validate that a profile has enough data to generate realistic CVs."""
        if not self.titles:
            raise ValueError("A career profile must contain at least one job title.")

        if not self.education:
            raise ValueError(
                "A career profile must contain at least one education option."
            )

        core_skills = [
            skill for skill in self.skills if skill.category == SkillCategory.CORE
        ]
        if len(core_skills) < 3:
            raise ValueError(
                "A career profile must contain at least three core skills."
            )

        skill_names = [skill.name.lower().strip() for skill in self.skills]
        if len(skill_names) != len(set(skill_names)):
            raise ValueError("Duplicate skills are not allowed.")

        return self


__all__ = [
    "CareerProfile",
    "EducationOption",
    "JobTitleOption",
    "Seniority",
    "Skill",
    "SkillCategory",
]
