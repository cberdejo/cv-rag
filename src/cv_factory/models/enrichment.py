"""Pydantic models for structured LLM CV enrichment output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DescriptionOutput(BaseModel):
    """LLM-generated description for one indexed CV section entry."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(
        ge=0,
        description="Zero-based index of the experience entry in the input CV.",
    )
    description: str = Field(
        min_length=80,
        max_length=900,
        description="Professional role description.",
    )


class CVEnrichmentOutput(BaseModel):
    """Structured narrative fields returned by the enrichment LLM."""

    model_config = ConfigDict(extra="forbid")

    about: str = Field(
        min_length=80,
        max_length=900,
        description="Professional summary for the candidate.",
    )
    experience: list[DescriptionOutput] = Field(
        description="Descriptions keyed by zero-based experience indexes."
    )
    education: list[DescriptionOutput] = Field(
        description="Descriptions keyed by zero-based education indexes."
    )

    @model_validator(mode="after")
    def validate_unique_indexes(self) -> "CVEnrichmentOutput":
        """Ensure each enriched experience and education index appears once."""
        experience_indexes = [item.index for item in self.experience]
        education_indexes = [item.index for item in self.education]

        if len(experience_indexes) != len(set(experience_indexes)):
            raise ValueError("Duplicate experience indexes in LLM output.")

        if len(education_indexes) != len(set(education_indexes)):
            raise ValueError("Duplicate education indexes in LLM output.")

        return self
