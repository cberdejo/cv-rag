"""Tests for AssistantNodes async methods using mocked LLM and indexer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from rag.assistant.models import (
    AssistantState,
    IntentRoute,
    MetadataFilters,
    RetrievedChunk,
)
from rag.assistant.nodes import AssistantNodes
from rag.assistant.nodes import stream_token_callback_var
from rag.exceptions import RAGLLMServiceError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nodes(*, search_results: list | None = None) -> AssistantNodes:
    """Build AssistantNodes with mocked indexer and LLM settings."""
    indexer = MagicMock()
    indexer.search_hybrid = AsyncMock(return_value=search_results or [])
    indexer.search_metadata = AsyncMock(return_value=search_results or [])
    indexer.search_metadata_then_rerank = AsyncMock(return_value=search_results or [])

    settings = MagicMock()
    settings.default_model = "test-model"
    settings.quality_model = "test-quality-model"

    return AssistantNodes(indexer=indexer, settings=settings, client=MagicMock())


def _make_state(**overrides) -> AssistantState:
    base: AssistantState = {
        "messages": [HumanMessage(content="Who knows Python?")],
        "question": "Who knows Python?",
        "standalone_query": "Who knows Python?",
        "route": IntentRoute.HYBRID,
        "metadata_filters": MetadataFilters(),
        "retrieved_chunks": [],
        "sources": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_successful_rewrite(self):
        nodes = _make_nodes()
        rewrite_response = json.dumps({"standalone_query": "Who knows Python?"})

        with patch(
            "rag.assistant.nodes.llm_client.chat_json_schema_content",
            new_callable=AsyncMock,
            return_value=rewrite_response,
        ):
            state = _make_state()
            result = await nodes.rewrite_query(state)

        assert result["question"] == "Who knows Python?"
        assert result["standalone_query"] == "Who knows Python?"

    @pytest.mark.asyncio
    async def test_rewrite_raises_rag_error_on_llm_failure(self):
        nodes = _make_nodes()

        with patch(
            "rag.assistant.nodes.llm_client.chat_json_schema_content",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            with pytest.raises(RAGLLMServiceError):
                await nodes.rewrite_query(_make_state())

    @pytest.mark.asyncio
    async def test_rewrite_falls_back_to_original_if_empty_response(self):
        nodes = _make_nodes()

        with patch(
            "rag.assistant.nodes.llm_client.chat_json_schema_content",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await nodes.rewrite_query(_make_state())

        # fallback: original question used as standalone_query
        assert result["standalone_query"] == "Who knows Python?"


# ---------------------------------------------------------------------------
# route_intent
# ---------------------------------------------------------------------------


class TestRouteIntent:
    @pytest.mark.asyncio
    async def test_routes_to_hybrid(self):
        nodes = _make_nodes()
        content = json.dumps({"route": "hybrid", "metadata_filters": {}})

        with patch(
            "rag.assistant.nodes.llm_client.chat_json_schema_content",
            new_callable=AsyncMock,
            return_value=content,
        ):
            result = await nodes.route_intent(_make_state())

        assert result["route"] == IntentRoute.HYBRID

    @pytest.mark.asyncio
    async def test_routes_to_metadata_with_skills(self):
        nodes = _make_nodes()
        content = json.dumps(
            {
                "route": "metadata",
                "metadata_filters": {
                    "skills": ["Python"],
                    "candidate_name": None,
                    "current_title": None,
                    "current_company": None,
                    "companies": [],
                    "degrees": [],
                    "institutions": [],
                    "email": None,
                    "section": None,
                    "min_years_of_experience": None,
                    "max_years_of_experience": None,
                },
            }
        )

        with patch(
            "rag.assistant.nodes.llm_client.chat_json_schema_content",
            new_callable=AsyncMock,
            return_value=content,
        ):
            result = await nodes.route_intent(_make_state())

        assert result["route"] == IntentRoute.METADATA
        assert "Python" in result["metadata_filters"].skills

    @pytest.mark.asyncio
    async def test_raises_rag_error_on_llm_failure(self):
        nodes = _make_nodes()

        with patch(
            "rag.assistant.nodes.llm_client.chat_json_schema_content",
            new_callable=AsyncMock,
            side_effect=Exception("connection refused"),
        ):
            with pytest.raises(RAGLLMServiceError):
                await nodes.route_intent(_make_state())


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_returns_chunks_and_sources(self):
        raw_result = {
            "id": "point-1",
            "page_content": "Expert in Python and Docker.",
            "metadata": {
                "source_path": "/cvs/test.pdf",
                "candidate_name": "Test Candidate",
                "section": "experience",
            },
            "score": 0.88,
        }
        nodes = _make_nodes(search_results=[raw_result])

        result = await nodes.hybrid_search(_make_state())

        assert len(result["retrieved_chunks"]) == 1
        assert (
            result["retrieved_chunks"][0].page_content == "Expert in Python and Docker."
        )
        assert len(result["sources"]) == 1
        assert result["sources"][0].source_path == "/cvs/test.pdf"

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_collections(self):
        nodes = _make_nodes(search_results=[])

        result = await nodes.hybrid_search(_make_state())

        assert result["retrieved_chunks"] == []
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_uses_standalone_query(self):
        nodes = _make_nodes()
        state = _make_state(standalone_query="senior DevOps experience")

        await nodes.hybrid_search(state)

        nodes.indexer.search_hybrid.assert_called_once()
        call_args = nodes.indexer.search_hybrid.call_args
        assert call_args[0][0] == "senior DevOps experience"


# ---------------------------------------------------------------------------
# metadata_search
# ---------------------------------------------------------------------------


class TestMetadataSearch:
    @pytest.mark.asyncio
    async def test_returns_deduplicated_sources(self):
        results = [
            {
                "id": f"p{i}",
                "page_content": f"content-{i}",
                "metadata": {"source_path": "/cvs/cv.pdf", "candidate_name": "X"},
                "score": 0.5,
            }
            for i in range(3)
        ]
        nodes = _make_nodes(search_results=results)

        result = await nodes.metadata_search(
            _make_state(metadata_filters=MetadataFilters(skills=["Python"]))
        )

        # Three chunks returned but deduplicated to 1 unique source
        assert len(result["retrieved_chunks"]) == 3
        assert len(result["sources"]) == 1


# ---------------------------------------------------------------------------
# hybrid_filtered_search
# ---------------------------------------------------------------------------


class TestHybridFilteredSearch:
    @pytest.mark.asyncio
    async def test_uses_query_and_filters(self):
        nodes = _make_nodes()
        filters = MetadataFilters(skills=["Python"])
        state = _make_state(
            standalone_query="senior Python engineer",
            metadata_filters=filters,
        )

        await nodes.hybrid_filtered_search(state)

        nodes.indexer.search_metadata_then_rerank.assert_called_once_with(
            "senior Python engineer",
            filters,
            limit=5,
            scroll_limit=100,
        )


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_generates_answer_from_context(self):
        nodes = _make_nodes()
        chunks = [
            RetrievedChunk(
                id="c1",
                page_content="Ana García has 5 years of Python experience.",
                metadata={
                    "candidate_name": "Ana García",
                    "source_path": "/cvs/ana.pdf",
                },
                score=0.9,
            )
        ]

        with patch(
            "rag.assistant.nodes.llm_client.chat_completion_content",
            new_callable=AsyncMock,
            return_value="Ana García is a strong Python candidate.",
        ):
            result = await nodes.generate_answer(
                _make_state(retrieved_chunks=chunks, sources=[])
            )

        assert result["answer"] == "Ana García is a strong Python candidate."

    @pytest.mark.asyncio
    async def test_raises_rag_error_on_llm_failure(self):
        nodes = _make_nodes()

        with patch(
            "rag.assistant.nodes.llm_client.chat_completion_content",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM timeout"),
        ):
            with pytest.raises(RAGLLMServiceError):
                await nodes.generate_answer(_make_state())

    @pytest.mark.asyncio
    async def test_empty_response_returns_default_message(self):
        nodes = _make_nodes()

        with patch(
            "rag.assistant.nodes.llm_client.chat_completion_content",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = await nodes.generate_answer(_make_state())

        assert "could not generate" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_streams_answer_tokens_when_callback_is_set(self):
        async def stream_tokens(**kwargs):
            del kwargs
            for token in ["Ana", " knows", " Python"]:
                yield token

        streamed_tokens: list[str] = []

        async def on_token(token: str) -> None:
            streamed_tokens.append(token)

        nodes = _make_nodes()
        callback_token = stream_token_callback_var.set(on_token)
        try:
            with patch(
                "rag.assistant.nodes.llm_client.stream_chat_completion_content",
                stream_tokens,
            ):
                result = await nodes.generate_answer(_make_state())
        finally:
            stream_token_callback_var.reset(callback_token)

        assert result["answer"] == "Ana knows Python"
        assert streamed_tokens == ["Ana", " knows", " Python"]


# ---------------------------------------------------------------------------
# out_of_scope
# ---------------------------------------------------------------------------


class TestOutOfScope:
    @pytest.mark.asyncio
    async def test_returns_fixed_answer(self):
        from rag.assistant.prompts import OUT_OF_SCOPE_ANSWER

        nodes = _make_nodes()
        result = await nodes.out_of_scope(_make_state())

        assert result["answer"] == OUT_OF_SCOPE_ANSWER
        assert result["sources"] == []
