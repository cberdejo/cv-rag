"""Exception hierarchy for CV factory generation, enrichment, and storage."""

from __future__ import annotations


class CVFactoryError(Exception):
    """Base exception for CV Factory errors."""


class CVFactoryConfigurationError(CVFactoryError):
    """Raised when required CV Factory configuration is missing or invalid."""


class CVFactoryExternalServiceError(CVFactoryError):
    """Raised when an external service dependency fails."""


class LLMEnrichmentServiceError(CVFactoryError):
    """Raised when the LLM enrichment service cannot produce a valid output."""


class LLMStructuredOutputError(LLMEnrichmentServiceError):
    """Raised when the LLM response cannot be parsed or validated."""


class LLMEnrichmentShapeError(LLMEnrichmentServiceError):
    """Raised when the LLM output does not match the input CV structure."""


class ProfileImageConfigurationError(CVFactoryConfigurationError):
    """Raised when profile image generation is missing required configuration."""


class ProfileImageAPIError(CVFactoryExternalServiceError):
    """Raised when the profile image provider returns an unsuccessful response."""


__all__ = [
    "CVFactoryConfigurationError",
    "CVFactoryError",
    "CVFactoryExternalServiceError",
    "LLMEnrichmentServiceError",
    "LLMEnrichmentShapeError",
    "LLMStructuredOutputError",
    "ProfileImageAPIError",
    "ProfileImageConfigurationError",
]
