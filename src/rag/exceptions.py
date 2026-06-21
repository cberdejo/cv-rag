"""Exception hierarchy for CV RAG retrieval and assistant workflows."""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for CV RAG errors."""


class RAGConfigurationError(RAGError):
    """Raised when required RAG configuration is missing or invalid."""


class RAGVectorStoreError(RAGError):
    """Raised when the vector store dependency fails."""


class RAGIndexingError(RAGVectorStoreError):
    """Raised when chunks cannot be indexed or the collection cannot be prepared."""


class RAGRetrievalError(RAGVectorStoreError):
    """Raised when retrieval from the vector store fails."""


class RAGAssistantError(RAGError):
    """Raised when the assistant graph cannot complete a request."""


class RAGLLMServiceError(RAGAssistantError):
    """Raised when an LLM call required by the assistant fails."""


class RAGStructuredOutputError(RAGAssistantError):
    """Raised when assistant LLM output cannot be parsed or validated."""


__all__ = [
    "RAGAssistantError",
    "RAGConfigurationError",
    "RAGError",
    "RAGIndexingError",
    "RAGLLMServiceError",
    "RAGRetrievalError",
    "RAGStructuredOutputError",
    "RAGVectorStoreError",
]
