"""Shared service factories for the cv-rag API and Chainlit UI."""

from __future__ import annotations

from functools import lru_cache

from rag.assistant import CVAssistant


@lru_cache
def get_assistant() -> CVAssistant:
    """Return the process-wide CV assistant instance."""
    return CVAssistant()


__all__ = ["get_assistant"]
