"""Tests for rag/assistant/nodes.py — pure helper functions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag.assistant.models import (
    AssistantState,
    CVSource,
    IntentRoute,
    MetadataFilters,
    RetrievedChunk,
)
from rag.assistant.nodes import (
    _parse_intent_decision,
    _parse_query_rewrite,
    _snippet,
    chunks_to_sources,
    format_context,
    format_history,
    route_after_intent,
)


# ---------------------------------------------------------------------------
# format_history
# ---------------------------------------------------------------------------


class TestFormatHistory:
    def test_single_message_returns_no_history(self):
        messages = [HumanMessage(content="Hello")]
        result = format_history(messages)
        assert result == "No previous chat history."

    def test_empty_messages_returns_no_history(self):
        result = format_history([])
        assert result == "No previous chat history."

    def test_two_messages_formats_first(self):
        messages = [
            HumanMessage(content="Who knows Python?"),
            AIMessage(content="Ana García does."),
        ]
        # latest message is excluded
        result = format_history(messages)
        assert "User: Who knows Python?" in result

    def test_alternating_roles(self):
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
        ]
        result = format_history(messages)
        assert "User: Q1" in result
        assert "Assistant: A1" in result
        # Q2 is the latest message and should be excluded
        assert "Q2" not in result

    def test_caps_at_six_previous_messages(self):
        messages = [HumanMessage(content=f"msg-{i}") for i in range(10)]
        result = format_history(messages)
        # msg-9 is the latest (excluded), msgs 3–8 should appear (last 6 of first 9)
        assert "msg-9" not in result
        assert "msg-8" in result


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_chunks_returns_no_results(self):
        result = format_context([], [])
        assert result == "No relevant CVs were retrieved."

    def test_single_chunk_renders_fields(self):
        chunk = RetrievedChunk(
            id="c1",
            page_content="Experienced in Python and FastAPI.",
            metadata={
                "candidate_name": "Ana García",
                "source_path": "/cvs/ana.pdf",
                "section": "experience",
            },
            score=0.9,
        )
        source = CVSource(id="source-1", source_path="/cvs/ana.pdf")
        result = format_context([chunk], [source])
        assert "Ana García" in result
        assert "Python" in result
        assert "source-1" in result
        assert "0.9" in result

    def test_multiple_chunks_separated_by_divider(self):
        chunks = [
            RetrievedChunk(id=f"c{i}", page_content=f"content-{i}", metadata={})
            for i in range(3)
        ]
        result = format_context(chunks, [])
        assert result.count("---") >= 2

    def test_unknown_candidate_shown(self):
        chunk = RetrievedChunk(id="c1", page_content="text", metadata={})
        result = format_context([chunk], [])
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# chunks_to_sources
# ---------------------------------------------------------------------------


class TestChunksToSources:
    def test_empty_chunks_returns_empty(self):
        assert chunks_to_sources([]) == []

    def test_single_chunk_produces_one_source(self):
        chunk = RetrievedChunk(
            id="c1",
            page_content="Python developer with 5 years of experience.",
            metadata={"source_path": "/cvs/juan.pdf", "candidate_name": "Juan"},
            score=0.8,
        )
        sources = chunks_to_sources([chunk])
        assert len(sources) == 1
        assert sources[0].source_path == "/cvs/juan.pdf"
        assert sources[0].candidate_name == "Juan"
        assert sources[0].score == pytest.approx(0.8)

    def test_chunks_from_same_source_deduplicated(self):
        chunks = [
            RetrievedChunk(
                id=f"c{i}",
                page_content=f"content-{i}",
                metadata={"source_path": "/cvs/shared.pdf"},
            )
            for i in range(5)
        ]
        sources = chunks_to_sources(chunks)
        assert len(sources) == 1

    def test_chunks_without_source_path_excluded(self):
        chunk = RetrievedChunk(id="c1", page_content="text", metadata={})
        sources = chunks_to_sources([chunk])
        assert sources == []

    def test_source_ids_are_sequential(self):
        chunks = [
            RetrievedChunk(
                id=f"c{i}",
                page_content="text",
                metadata={"source_path": f"/cvs/cv{i}.pdf"},
            )
            for i in range(3)
        ]
        sources = chunks_to_sources(chunks)
        ids = [s.id for s in sources]
        assert ids == ["source-1", "source-2", "source-3"]

    def test_snippet_is_truncated(self):
        long_text = "word " * 100
        chunk = RetrievedChunk(
            id="c1",
            page_content=long_text,
            metadata={"source_path": "/cv.pdf"},
        )
        sources = chunks_to_sources([chunk])
        assert len(sources[0].snippet) <= 244  # 240 chars + "..."


# ---------------------------------------------------------------------------
# _snippet helper
# ---------------------------------------------------------------------------


class TestSnippet:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert _snippet(text) == "Short text."

    def test_long_text_truncated_with_ellipsis(self):
        text = "word " * 100
        result = _snippet(text)
        assert result.endswith("...")
        assert len(result) <= 244

    def test_whitespace_normalised(self):
        text = "a  b\n\nc"
        assert _snippet(text) == "a b c"


# ---------------------------------------------------------------------------
# _parse_query_rewrite
# ---------------------------------------------------------------------------


class TestParseQueryRewrite:
    def test_valid_json(self):
        content = '{"standalone_query": "Who knows Docker?"}'
        result = _parse_query_rewrite(content, "fallback")
        assert result.standalone_query == "Who knows Docker?"

    def test_invalid_json_falls_back_to_stripped_content(self):
        result = _parse_query_rewrite("plain text query", "fallback")
        assert result.standalone_query == "plain text query"

    def test_empty_content_uses_fallback(self):
        result = _parse_query_rewrite("", "fallback query")
        assert result.standalone_query == "fallback query"

    def test_whitespace_only_uses_fallback(self):
        result = _parse_query_rewrite("   ", "my fallback")
        assert result.standalone_query == "my fallback"


# ---------------------------------------------------------------------------
# _parse_intent_decision
# ---------------------------------------------------------------------------


class TestParseIntentDecision:
    def test_valid_hybrid_json(self):
        content = '{"route": "hybrid", "metadata_filters": {}}'
        result = _parse_intent_decision(content)
        assert result.route == IntentRoute.HYBRID

    def test_valid_metadata_json_with_filters(self):
        content = (
            '{"route": "metadata", '
            '"metadata_filters": {"skills": ["Python"], "candidate_name": null}}'
        )
        result = _parse_intent_decision(content)
        assert result.route == IntentRoute.METADATA
        assert "Python" in result.metadata_filters.skills

    def test_out_of_scope_route(self):
        content = '{"route": "out_of_scope", "metadata_filters": {}}'
        result = _parse_intent_decision(content)
        assert result.route == IntentRoute.OUT_OF_SCOPE

    def test_invalid_json_falls_back_to_hybrid(self):
        result = _parse_intent_decision("not json at all")
        assert result.route == IntentRoute.HYBRID
        assert not result.metadata_filters.has_any()

    def test_unknown_route_in_json_falls_back_to_hybrid(self):
        result = _parse_intent_decision('{"route": "nonsense", "metadata_filters": {}}')
        assert result.route == IntentRoute.HYBRID


# ---------------------------------------------------------------------------
# route_after_intent
# ---------------------------------------------------------------------------


class TestRouteAfterIntent:
    def _make_state(
        self, route: IntentRoute, filters: MetadataFilters | None = None
    ) -> AssistantState:
        state: AssistantState = {
            "messages": [],
            "route": route,
            "metadata_filters": filters or MetadataFilters(),
        }
        return state

    def test_out_of_scope_goes_to_out_of_scope(self):
        state = self._make_state(IntentRoute.OUT_OF_SCOPE)
        assert route_after_intent(state) == "out_of_scope"

    def test_hybrid_route_goes_to_hybrid_search(self):
        state = self._make_state(IntentRoute.HYBRID)
        assert route_after_intent(state) == "hybrid_search"

    def test_metadata_with_filters_goes_to_metadata_search(self):
        state = self._make_state(
            IntentRoute.METADATA,
            MetadataFilters(skills=["Kubernetes"]),
        )
        assert route_after_intent(state) == "metadata_search"

    def test_metadata_without_filters_falls_back_to_hybrid(self):
        state = self._make_state(IntentRoute.METADATA, MetadataFilters())
        assert route_after_intent(state) == "hybrid_search"

    def test_hybrid_filtered_with_filters_goes_to_hybrid_filtered_search(self):
        state = self._make_state(
            IntentRoute.HYBRID_FILTERED,
            MetadataFilters(skills=["Python"]),
        )
        assert route_after_intent(state) == "hybrid_filtered_search"

    def test_hybrid_filtered_without_filters_falls_back_to_hybrid(self):
        state = self._make_state(IntentRoute.HYBRID_FILTERED, MetadataFilters())
        assert route_after_intent(state) == "hybrid_search"

    def test_missing_route_defaults_to_hybrid(self):
        state: AssistantState = {"messages": []}  # type: ignore[typeddict-item]
        assert route_after_intent(state) == "hybrid_search"
