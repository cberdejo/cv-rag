"""Tests for the CVAssistant public service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from rag.assistant.assistant import CVAssistant
from rag.assistant.models import AssistantResponse, CVSource
from rag.assistant.nodes import stream_token_callback_var
from rag.exceptions import RAGAssistantError


def _make_assistant() -> CVAssistant:
    """Build a CVAssistant with a real MemorySaver and mocked indexer/client."""
    indexer = MagicMock()
    settings = MagicMock()
    settings.default_model = "test-model"
    settings.quality_model = "test-quality-model"
    return CVAssistant(
        indexer=indexer,
        settings=settings,
        client=MagicMock(),
        memory=MemorySaver(),  # LangGraph requires a real BaseCheckpointSaver
    )


class TestCVAssistantAsk:
    @pytest.mark.asyncio
    async def test_returns_assistant_response(self):
        assistant = _make_assistant()
        source = CVSource(id="s1", source_path="/cvs/ana.pdf", candidate_name="Ana")

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "answer": "Ana García is a Python developer.",
                "sources": [source],
            }
        )
        assistant._graph = mock_graph

        response = await assistant.ask("Who knows Python?", thread_id="thread-1")

        assert isinstance(response, AssistantResponse)
        assert response.answer == "Ana García is a Python developer."
        assert len(response.sources) == 1
        assert response.sources[0].candidate_name == "Ana"

    @pytest.mark.asyncio
    async def test_raises_rag_error_on_graph_failure(self):
        assistant = _make_assistant()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph crashed"))
        assistant._graph = mock_graph

        with pytest.raises(RAGAssistantError):
            await assistant.ask("test query", thread_id="t1")

    @pytest.mark.asyncio
    async def test_passes_thread_id_to_graph(self):
        assistant = _make_assistant()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"answer": "ok", "sources": []})
        assistant._graph = mock_graph

        await assistant.ask("hello", thread_id="my-thread-123")

        call_kwargs = mock_graph.ainvoke.call_args[1]
        assert call_kwargs["config"]["configurable"]["thread_id"] == "my-thread-123"

    @pytest.mark.asyncio
    async def test_empty_answer_in_state_returns_empty_string(self):
        assistant = _make_assistant()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"sources": []})
        assistant._graph = mock_graph

        response = await assistant.ask("anything", thread_id="t1")

        assert response.answer == ""

    @pytest.mark.asyncio
    async def test_missing_sources_key_returns_empty_list(self):
        assistant = _make_assistant()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"answer": "done"})
        assistant._graph = mock_graph

        response = await assistant.ask("anything", thread_id="t1")

        assert response.sources == []

    @pytest.mark.asyncio
    async def test_rag_error_propagates_without_wrapping(self):
        """RAGError subclasses should not be re-wrapped as RAGAssistantError."""
        from rag.exceptions import RAGLLMServiceError

        assistant = _make_assistant()
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RAGLLMServiceError("llm down"))
        assistant._graph = mock_graph

        with pytest.raises(RAGLLMServiceError):
            await assistant.ask("anything", thread_id="t1")

    @pytest.mark.asyncio
    async def test_ask_stream_sets_and_resets_token_callback(self):
        assistant = _make_assistant()

        async def on_token(token: str) -> None:
            del token

        async def graph_invoke(*args, **kwargs):
            del args, kwargs
            assert stream_token_callback_var.get() is on_token
            return {"answer": "done", "sources": []}

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=graph_invoke)
        assistant._graph = mock_graph

        response = await assistant.ask_stream(
            "anything",
            thread_id="t1",
            on_token=on_token,
        )

        assert response.answer == "done"
        assert stream_token_callback_var.get() is None


class TestCVAssistantInit:
    def test_creates_with_defaults(self):
        with (
            patch("rag.assistant.assistant.QdrantCVStore") as mock_store,
            patch("rag.assistant.assistant.get_qdrant_settings"),
            patch("rag.assistant.assistant.get_cv_ingestion_settings"),
        ):
            mock_store.return_value = MagicMock()
            assistant = CVAssistant()
            assert assistant.memory is not None
            assert assistant._graph is not None

    def test_accepts_injected_indexer(self):
        indexer = MagicMock()
        assistant = CVAssistant(indexer=indexer)
        assert assistant.indexer is indexer

    def test_default_memory_is_memory_saver(self):
        indexer = MagicMock()
        assistant = CVAssistant(indexer=indexer)
        assert isinstance(assistant.memory, MemorySaver)
