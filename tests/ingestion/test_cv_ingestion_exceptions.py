"""Tests for the CV ingestion exception hierarchy."""

import pytest

from cv_ingestion.exceptions import (
    CVArtifactStagingError,
    CVChunkingError,
    CVExtractionError,
    CVIndexingError,
    CVIngestionConfigurationError,
    CVIngestionError,
    CVLLMServiceError,
    CVNormalizationError,
    CVStructuredOutputError,
    EmptyCVTextError,
    UnsupportedCVFormatError,
)


class TestExceptionHierarchy:
    """All custom CV ingestion exceptions inherit correctly."""

    def test_cv_ingestion_error_is_exception(self):
        assert issubclass(CVIngestionError, Exception)

    def test_configuration_error_inherits_cv_ingestion_error(self):
        assert issubclass(CVIngestionConfigurationError, CVIngestionError)

    def test_artifact_staging_error_inherits_cv_ingestion_error(self):
        assert issubclass(CVArtifactStagingError, CVIngestionError)

    def test_extraction_error_inherits_cv_ingestion_error(self):
        assert issubclass(CVExtractionError, CVIngestionError)

    def test_empty_text_error_inherits_extraction_error(self):
        assert issubclass(EmptyCVTextError, CVExtractionError)
        assert issubclass(EmptyCVTextError, CVIngestionError)

    def test_unsupported_format_error_inherits_extraction_error(self):
        assert issubclass(UnsupportedCVFormatError, CVExtractionError)
        assert issubclass(UnsupportedCVFormatError, CVIngestionError)

    def test_normalization_error_inherits_cv_ingestion_error(self):
        assert issubclass(CVNormalizationError, CVIngestionError)

    def test_llm_service_error_inherits_normalization_error(self):
        assert issubclass(CVLLMServiceError, CVNormalizationError)
        assert issubclass(CVLLMServiceError, CVIngestionError)

    def test_structured_output_error_inherits_normalization_error(self):
        assert issubclass(CVStructuredOutputError, CVNormalizationError)
        assert issubclass(CVStructuredOutputError, CVIngestionError)

    def test_chunking_error_inherits_cv_ingestion_error(self):
        assert issubclass(CVChunkingError, CVIngestionError)

    def test_indexing_error_inherits_cv_ingestion_error(self):
        assert issubclass(CVIndexingError, CVIngestionError)

    def test_exceptions_can_be_raised_with_message(self):
        with pytest.raises(CVChunkingError, match="no valid chunks"):
            raise CVChunkingError("no valid chunks")

    def test_exceptions_preserve_cause(self):
        cause = ValueError("original cause")
        with pytest.raises(CVLLMServiceError) as exc_info:
            raise CVLLMServiceError("llm failed") from cause
        assert exc_info.value.__cause__ is cause
