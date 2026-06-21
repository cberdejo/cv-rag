"""Tests for the RAG exception hierarchy."""

import pytest

from rag.exceptions import (
    RAGAssistantError,
    RAGConfigurationError,
    RAGError,
    RAGIndexingError,
    RAGLLMServiceError,
    RAGRetrievalError,
    RAGStructuredOutputError,
    RAGVectorStoreError,
)


class TestExceptionHierarchy:
    """All custom RAG exceptions inherit correctly."""

    def test_rag_error_is_exception(self):
        assert issubclass(RAGError, Exception)

    def test_configuration_error_inherits_rag_error(self):
        assert issubclass(RAGConfigurationError, RAGError)

    def test_vector_store_error_inherits_rag_error(self):
        assert issubclass(RAGVectorStoreError, RAGError)

    def test_indexing_error_inherits_vector_store_error(self):
        assert issubclass(RAGIndexingError, RAGVectorStoreError)
        assert issubclass(RAGIndexingError, RAGError)

    def test_retrieval_error_inherits_vector_store_error(self):
        assert issubclass(RAGRetrievalError, RAGVectorStoreError)
        assert issubclass(RAGRetrievalError, RAGError)

    def test_assistant_error_inherits_rag_error(self):
        assert issubclass(RAGAssistantError, RAGError)

    def test_llm_service_error_inherits_assistant_error(self):
        assert issubclass(RAGLLMServiceError, RAGAssistantError)
        assert issubclass(RAGLLMServiceError, RAGError)

    def test_structured_output_error_inherits_assistant_error(self):
        assert issubclass(RAGStructuredOutputError, RAGAssistantError)
        assert issubclass(RAGStructuredOutputError, RAGError)

    def test_exceptions_can_be_raised_with_message(self):
        with pytest.raises(RAGIndexingError, match="collection failed"):
            raise RAGIndexingError("collection failed")

    def test_exceptions_preserve_cause(self):
        cause = ValueError("original cause")
        with pytest.raises(RAGLLMServiceError) as exc_info:
            raise RAGLLMServiceError("llm failed") from cause
        assert exc_info.value.__cause__ is cause
