"""Exception hierarchy for CV ingestion, normalization, chunking, and indexing."""

from __future__ import annotations


class CVIngestionError(Exception):
    """Base exception for CV ingestion errors."""


class CVIngestionConfigurationError(CVIngestionError):
    """Raised when ingestion configuration or input options are invalid."""


class CVArtifactStagingError(CVIngestionError):
    """Raised when CV artifacts cannot be resolved or staged for ingestion."""


class CVExtractionError(CVIngestionError):
    """Raised when source text cannot be extracted."""


class EmptyCVTextError(CVExtractionError):
    """Raised when extracted CV text is empty."""


class UnsupportedCVFormatError(CVExtractionError):
    """Raised when a CV artifact format is not supported."""


class CVNormalizationError(CVIngestionError):
    """Raised when CV text cannot be normalized into a structured model."""


class CVLLMServiceError(CVNormalizationError):
    """Raised when the LLM service cannot produce a response."""


class CVStructuredOutputError(CVNormalizationError):
    """Raised when LLM output cannot be parsed or validated."""


class CVChunkingError(CVIngestionError):
    """Raised when no valid chunks can be produced."""


class CVIndexingError(CVIngestionError):
    """Raised when chunks cannot be indexed."""


__all__ = [
    "CVArtifactStagingError",
    "CVChunkingError",
    "CVExtractionError",
    "CVIndexingError",
    "CVIngestionConfigurationError",
    "CVIngestionError",
    "CVLLMServiceError",
    "CVNormalizationError",
    "CVStructuredOutputError",
    "EmptyCVTextError",
    "UnsupportedCVFormatError",
]
