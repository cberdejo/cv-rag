"""Models and graph state for the CV RAG assistant."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class IntentRoute(StrEnum):
    """Routes supported by the assistant graph."""

    OUT_OF_SCOPE = "out_of_scope"
    METADATA = "metadata"
    HYBRID = "hybrid"
    HYBRID_FILTERED = "hybrid_filtered"


class MetadataFilters(BaseModel):
    """Metadata filters extracted from a user request."""

    candidate_name: str | None = None
    skills: list[str] = Field(default_factory=list)
    current_title: str | None = None
    current_company: str | None = None
    companies: list[str] = Field(default_factory=list)
    degrees: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    email: str | None = None
    section: str | None = None
    min_years_of_experience: int | None = None
    max_years_of_experience: int | None = None

    def has_any(self) -> bool:
        """Return whether any meaningful filter is set."""
        return any(
            [
                self.candidate_name,
                self.skills,
                self.current_title,
                self.current_company,
                self.companies,
                self.degrees,
                self.institutions,
                self.certifications,
                self.languages,
                self.email,
                self.section,
                self.min_years_of_experience is not None,
                self.max_years_of_experience is not None,
            ]
        )


class QueryRewrite(BaseModel):
    """Structured output from the query rewrite node."""

    standalone_query: str = Field(
        description="Self-contained user query, rewritten only if history is needed."
    )


class IntentDecision(BaseModel):
    """Structured output from the intent routing LLM call."""

    route: IntentRoute
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)


class RetrievedChunk(BaseModel):
    """Normalized search result returned by retrieval methods."""

    id: str
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class CVSource(BaseModel):
    """Source object that the frontend can use for preview/download actions."""

    id: str
    candidate_name: str | None = None
    source_path: str
    section: str | None = None
    score: float | None = None
    snippet: str = ""


class AssistantResponse(BaseModel):
    """Public response contract for the CV assistant service."""

    answer: str
    sources: list[CVSource] = Field(default_factory=list)


class AssistantState(TypedDict, total=False):
    """LangGraph state for a single assistant conversation."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    standalone_query: str
    route: IntentRoute
    metadata_filters: MetadataFilters
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    sources: list[CVSource]


__all__ = [
    "AssistantResponse",
    "AssistantState",
    "CVSource",
    "IntentDecision",
    "IntentRoute",
    "MetadataFilters",
    "QueryRewrite",
    "RetrievedChunk",
]
