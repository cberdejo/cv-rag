"""Pydantic schemas for the v1 HTTP API."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from rag.assistant import CVSource


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    app: str
    version: str


class ChatRequest(BaseModel):
    """Request body for assistant chat calls."""

    message: str = Field(..., min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        """Reject empty or whitespace-only messages."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty")
        return stripped


class ChatResponse(BaseModel):
    """Assistant chat response returned by the API."""

    answer: str
    thread_id: str
    sources: list[CVSource] = Field(default_factory=list)


__all__ = ["ChatRequest", "ChatResponse", "HealthResponse"]
