"""Tests for the CV factory exception hierarchy."""

import pytest

from cv_factory.exceptions import (
    CVFactoryConfigurationError,
    CVFactoryError,
    CVFactoryExternalServiceError,
    LLMEnrichmentServiceError,
    LLMEnrichmentShapeError,
    LLMStructuredOutputError,
    ProfileImageAPIError,
    ProfileImageConfigurationError,
)


class TestExceptionHierarchy:
    """All custom CV factory exceptions inherit correctly."""

    def test_cv_factory_error_is_exception(self):
        assert issubclass(CVFactoryError, Exception)

    def test_configuration_error_inherits_cv_factory_error(self):
        assert issubclass(CVFactoryConfigurationError, CVFactoryError)

    def test_external_service_error_inherits_cv_factory_error(self):
        assert issubclass(CVFactoryExternalServiceError, CVFactoryError)

    def test_llm_enrichment_service_error_inherits_cv_factory_error(self):
        assert issubclass(LLMEnrichmentServiceError, CVFactoryError)

    def test_structured_output_error_inherits_llm_enrichment_service_error(self):
        assert issubclass(LLMStructuredOutputError, LLMEnrichmentServiceError)
        assert issubclass(LLMStructuredOutputError, CVFactoryError)

    def test_enrichment_shape_error_inherits_llm_enrichment_service_error(self):
        assert issubclass(LLMEnrichmentShapeError, LLMEnrichmentServiceError)
        assert issubclass(LLMEnrichmentShapeError, CVFactoryError)

    def test_profile_image_configuration_error_inherits_configuration_error(self):
        assert issubclass(ProfileImageConfigurationError, CVFactoryConfigurationError)
        assert issubclass(ProfileImageConfigurationError, CVFactoryError)

    def test_profile_image_api_error_inherits_external_service_error(self):
        assert issubclass(ProfileImageAPIError, CVFactoryExternalServiceError)
        assert issubclass(ProfileImageAPIError, CVFactoryError)

    def test_exceptions_can_be_raised_with_message(self):
        with pytest.raises(LLMEnrichmentShapeError, match="index mismatch"):
            raise LLMEnrichmentShapeError("index mismatch")

    def test_exceptions_preserve_cause(self):
        cause = ValueError("original cause")
        with pytest.raises(LLMStructuredOutputError) as exc_info:
            raise LLMStructuredOutputError("invalid output") from cause
        assert exc_info.value.__cause__ is cause
