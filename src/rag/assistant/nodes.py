"""LangGraph node implementations for the CV RAG assistant."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import AsyncOpenAI

from core.logging import get_logger
from core.settings import LLMSettings, get_llm_settings
from infrastructure import llm_client
from rag.exceptions import RAGLLMServiceError
from rag.assistant.models import (
    AssistantState,
    CVSource,
    IntentDecision,
    IntentRoute,
    MetadataFilters,
    QueryRewrite,
    RetrievedChunk,
)
from rag.assistant.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
    OUT_OF_SCOPE_ANSWER,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT,
)
from rag.retrieval.indexer import QdrantCVStore

logger = get_logger(__name__)

StreamTokenCallback = Callable[[str], Awaitable[None]]
stream_token_callback_var: ContextVar[StreamTokenCallback | None] = ContextVar(
    "stream_token_callback",
    default=None,
)


class AssistantNodes:
    """Dependency-bound node methods used by the assistant graph."""

    def __init__(
        self,
        *,
        indexer: QdrantCVStore,
        settings: LLMSettings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Bind shared retrieval and LLM dependencies for all graph nodes."""
        self.indexer = indexer
        self.settings = settings or get_llm_settings()
        self.client = client

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    async def rewrite_query(self, state: AssistantState) -> dict[str, Any]:
        """Rewrite the latest user message into a standalone query."""
        question = _latest_user_text(state["messages"])
        chat_history = format_history(state["messages"])
        logger.info(
            "Assistant node=rewrite_query started | messages={} | question={}",
            len(state["messages"]),
            _preview(question),
        )
        logger.debug(
            "Assistant node=rewrite_query context | history_chars={} | question_chars={}",
            len(chat_history),
            len(question),
        )

        try:
            content = await llm_client.chat_json_schema_content(
                client=self.client,
                settings=self.settings,
                model=self.settings.default_model,
                temperature=0.0,
                schema=QueryRewrite.model_json_schema(),
                schema_name="cv_assistant_query_rewrite",
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": REWRITE_USER_PROMPT.format(
                            chat_history=chat_history,
                            question=question,
                        ),
                    },
                ],
            )
        except Exception as exc:
            logger.exception("Assistant node=rewrite_query failed")
            raise RAGLLMServiceError("Could not rewrite the assistant query.") from exc

        rewrite = _parse_query_rewrite(content, question)
        standalone_query = rewrite.standalone_query or question
        logger.info(
            "Assistant node=rewrite_query completed | rewritten={} | standalone_query={}",
            standalone_query != question,
            _preview(standalone_query),
        )
        return {
            "question": question,
            "standalone_query": standalone_query,
        }

    # ------------------------------------------------------------------
    # Intent routing
    # ------------------------------------------------------------------

    async def route_intent(self, state: AssistantState) -> dict[str, Any]:
        """Classify the rewritten query and extract metadata filters."""
        standalone_query = state.get("standalone_query") or state.get("question") or ""
        logger.info(
            "Assistant node=route_intent started | query={}",
            _preview(standalone_query),
        )

        try:
            content = await llm_client.chat_json_schema_content(
                client=self.client,
                settings=self.settings,
                model=self.settings.default_model,
                temperature=0.0,
                schema=IntentDecision.model_json_schema(),
                schema_name="cv_assistant_intent_router",
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": ROUTER_USER_PROMPT.format(question=standalone_query),
                    },
                ],
            )
        except Exception as exc:
            logger.exception("Assistant node=route_intent failed")
            raise RAGLLMServiceError("Could not route the assistant intent.") from exc

        decision = _parse_intent_decision(content)
        logger.info(
            "Assistant node=route_intent completed | route={} | has_filters={}",
            decision.route,
            decision.metadata_filters.has_any(),
        )
        logger.debug(
            "Assistant node=route_intent filters | filters={}",
            _filters_for_log(decision.metadata_filters),
        )
        return {
            "route": decision.route,
            "metadata_filters": decision.metadata_filters,
        }

    # ------------------------------------------------------------------
    # Search nodes
    # ------------------------------------------------------------------

    async def metadata_search(self, state: AssistantState) -> dict[str, Any]:
        """
        Pure metadata scroll — no vector scoring.

        Used when the query is a hard lookup with no free-text intent:
        "list all candidates", "show me Ana García's CV", "who has a PhD?".
        Results are deduplicated by source_path and returned as-is.
        """
        filters = state.get("metadata_filters") or MetadataFilters()
        logger.info(
            "Assistant node=metadata_search started | filters={}",
            _filters_for_log(filters),
        )
        results = await self.indexer.search_metadata(filters, limit=20)
        chunks = [_result_to_chunk(r) for r in results]
        sources = chunks_to_sources(chunks)
        logger.info(
            "Assistant node=metadata_search completed | chunks={} | sources={}",
            len(chunks),
            len(sources),
        )
        logger.debug(
            "Assistant node=metadata_search sources | sources={}",
            _sources_for_log(sources),
        )
        return {
            "retrieved_chunks": chunks,
            "sources": sources,
        }

    async def hybrid_search(self, state: AssistantState) -> dict[str, Any]:
        """
        Pure dense+sparse RRF search — no metadata filter.

        Used when the query is fully open-ended and no structured constraints
        were extracted: "who has strong leadership experience?",
        "find someone with a background in distributed systems".
        """
        query = state.get("standalone_query") or state.get("question") or ""
        logger.info(
            "Assistant node=hybrid_search started | query={}",
            _preview(query),
        )
        results = await self.indexer.search_hybrid(
            query,
            limit=5,
            rerank=True,
        )
        chunks = [_result_to_chunk(r) for r in results]
        sources = chunks_to_sources(chunks)
        logger.info(
            "Assistant node=hybrid_search completed | chunks={} | sources={}",
            len(chunks),
            len(sources),
        )
        logger.debug(
            "Assistant node=hybrid_search sources | sources={}",
            _sources_for_log(sources),
        )
        return {
            "retrieved_chunks": chunks,
            "sources": sources,
        }

    async def hybrid_filtered_search(self, state: AssistantState) -> dict[str, Any]:
        """
        Metadata hard filter → cross-encoder rerank against the full query.

        Used when the query combines structured constraints with a free-text,
        semantic, or comparative intent: "Python developers with 5+ years who
        have worked at a startup", "among candidates who know React with 3+
        years, who is the strongest?".

        Phase 1 scrolls all chunks that pass the filter (up to 100) so no
        candidate is dropped just because of a low initial vector score.
        Phase 2 reranks them with the cross-encoder against the full query,
        which captures both plain semantic relevance and "who's the best"
        judgements.
        """
        query = state.get("standalone_query") or state.get("question") or ""
        filters = state.get("metadata_filters") or MetadataFilters()
        logger.info(
            "Assistant node=hybrid_filtered_search started | query={} | filters={}",
            _preview(query),
            _filters_for_log(filters),
        )
        results = await self.indexer.search_metadata_then_rerank(
            query,
            filters,
            limit=5,
            scroll_limit=100,
        )
        chunks = [_result_to_chunk(r) for r in results]
        sources = chunks_to_sources(chunks)
        logger.info(
            "Assistant node=hybrid_filtered_search completed | chunks={} | sources={}",
            len(chunks),
            len(sources),
        )
        logger.debug(
            "Assistant node=hybrid_filtered_search sources | sources={}",
            _sources_for_log(sources),
        )
        return {
            "retrieved_chunks": chunks,
            "sources": sources,
        }

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    async def generate_answer(self, state: AssistantState) -> dict[str, Any]:
        """Generate the final user-facing answer from retrieved context."""
        messages = build_answer_messages(state)
        stream_token_callback = stream_token_callback_var.get()
        retrieved_chunks = state.get("retrieved_chunks", [])
        sources = state.get("sources", [])
        logger.info(
            "Assistant node=generate_answer started | chunks={} | sources={} | streaming={}",
            len(retrieved_chunks),
            len(sources),
            stream_token_callback is not None,
        )
        logger.debug(
            "Assistant node=generate_answer source_context | sources={}",
            _sources_for_log(sources),
        )
        try:
            if stream_token_callback is None:
                content = await llm_client.chat_completion_content(
                    client=self.client,
                    settings=self.settings,
                    model=self.settings.quality_model,
                    temperature=0.2,
                    messages=messages,
                )
            else:
                tokens: list[str] = []
                async for token in llm_client.stream_chat_completion_content(
                    client=self.client,
                    settings=self.settings,
                    model=self.settings.quality_model,
                    temperature=0.2,
                    messages=messages,
                ):
                    tokens.append(token)
                    await stream_token_callback(token)
                content = "".join(tokens)
        except Exception as exc:
            logger.exception("Assistant node=generate_answer failed")
            raise RAGLLMServiceError(
                "Could not generate the assistant answer."
            ) from exc

        answer = (
            content.strip() or "I could not generate an answer from the available CVs."
        )
        logger.info(
            "Assistant node=generate_answer completed | answer_chars={}",
            len(answer),
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    async def out_of_scope(self, state: AssistantState) -> dict[str, Any]:
        """Return the fixed assistant-scope answer for irrelevant requests."""
        logger.info(
            "Assistant node=out_of_scope completed | question={}",
            _preview(state.get("question", "")),
        )
        return {
            "answer": OUT_OF_SCOPE_ANSWER,
            "sources": [],
            "messages": [AIMessage(content=OUT_OF_SCOPE_ANSWER)],
        }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_intent(state: AssistantState) -> str:
    """
    Choose the next graph node after intent classification.

    Decision table
    --------------
    out_of_scope
        The query is unrelated to CVs.  Return immediately.

    metadata
        The LLM extracted filters but found no free-text intent.
        → If filters are present: metadata_search (pure scroll, no vectors).
        → If no filters were extracted: fall back to hybrid_search.

    hybrid_filtered
        The LLM found structured constraints combined with a free-text or
        ranking/comparison intent ("who is the best among…").
        → If filters are present: hybrid_filtered_search (scroll filtered set
          → cross-encoder rerank against the full query).
        → If no filters: hybrid_search (open-ended semantic search).

    hybrid (default / fallback)
        Open-ended semantic query with no meaningful filters.
        → hybrid_search.
    """
    route = state.get("route", IntentRoute.HYBRID)
    filters: MetadataFilters = state.get("metadata_filters") or MetadataFilters()
    has_filters = filters.has_any()

    if route == IntentRoute.OUT_OF_SCOPE:
        logger.info(
            "Assistant route_after_intent selected | route={} | has_filters={} | next_node=out_of_scope",
            route,
            has_filters,
        )
        return "out_of_scope"

    if route == IntentRoute.METADATA:
        next_node = "metadata_search" if has_filters else "hybrid_search"
        logger.info(
            "Assistant route_after_intent selected | route={} | has_filters={} | next_node={}",
            route,
            has_filters,
            next_node,
        )
        return next_node

    if route == IntentRoute.HYBRID_FILTERED:
        next_node = "hybrid_filtered_search" if has_filters else "hybrid_search"
        logger.info(
            "Assistant route_after_intent selected | route={} | has_filters={} | next_node={}",
            route,
            has_filters,
            next_node,
        )
        return next_node

    # IntentRoute.HYBRID or any unrecognised value
    logger.info(
        "Assistant route_after_intent selected | route={} | has_filters={} | next_node=hybrid_search",
        route,
        has_filters,
    )
    return "hybrid_search"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def format_history(messages: list[BaseMessage]) -> str:
    """Format recent conversation history excluding the latest user message."""
    if len(messages) <= 1:
        return "No previous chat history."

    lines: list[str] = []
    for message in messages[:-1][-6:]:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{role}: {_message_content(message)}")
    return "\n".join(lines) if lines else "No previous chat history."


def format_context(chunks: list[RetrievedChunk], sources: list[CVSource]) -> str:
    """Render retrieved chunks for the answer prompt."""
    if not chunks:
        return "No relevant CVs were retrieved."

    source_by_path = {source.source_path: source for source in sources}
    rendered: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        source = source_by_path.get(metadata.get("source_path", ""))
        source_id = source.id if source else f"source-{index}"
        rendered.append(
            "\n".join(
                [
                    f"Source: {source_id}",
                    f"Candidate: {metadata.get('candidate_name') or 'Unknown'}",
                    f"Origin: {metadata.get('source_path') or 'Not available'}",
                    f"Section: {metadata.get('section') or 'Not available'}",
                    f"Score: {chunk.score if chunk.score is not None else 'N/A'}",
                    "Content:",
                    chunk.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(rendered)


def build_answer_messages(state: AssistantState) -> list[dict[str, str]]:
    """Build the final answer prompt messages from assistant state."""
    context = format_context(
        state.get("retrieved_chunks", []),
        state.get("sources", []),
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_USER_PROMPT.format(
                question=state.get("question", ""),
                standalone_query=state.get("standalone_query", ""),
                context=context,
            ),
        },
    ]


def chunks_to_sources(chunks: list[RetrievedChunk]) -> list[CVSource]:
    """Build frontend-friendly source records from retrieved chunks."""
    sources: list[CVSource] = []
    seen_paths: set[str] = set()
    for chunk in chunks:
        source_path = chunk.metadata.get("source_path")
        if not source_path or source_path in seen_paths:
            continue
        seen_paths.add(source_path)
        sources.append(
            CVSource(
                id=f"source-{len(sources) + 1}",
                candidate_name=chunk.metadata.get("candidate_name"),
                source_path=source_path,
                section=chunk.metadata.get("section"),
                score=chunk.score,
                snippet=_snippet(chunk.page_content),
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_query_rewrite(content: str, fallback_query: str) -> QueryRewrite:
    """Parse query rewrite output and fall back to raw text when needed."""
    try:
        return QueryRewrite.model_validate_json(content)
    except Exception:
        try:
            return QueryRewrite.model_validate(json.loads(content))
        except Exception:
            stripped = content.strip()
            return QueryRewrite(standalone_query=stripped or fallback_query)


def _parse_intent_decision(content: str) -> IntentDecision:
    """Parse router output and fall back to a hybrid search decision."""
    try:
        return IntentDecision.model_validate_json(content)
    except Exception:
        try:
            return IntentDecision.model_validate(json.loads(content))
        except Exception:
            return IntentDecision(
                route=IntentRoute.HYBRID,
                metadata_filters=MetadataFilters(),
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the content of the latest message in the conversation."""
    if not messages:
        return ""
    return _message_content(messages[-1])


def _message_content(message: BaseMessage) -> str:
    """Convert LangChain message content to a plain string."""
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _result_to_chunk(result: dict[str, Any]) -> RetrievedChunk:
    """Convert a retrieval result dictionary into assistant chunk state."""
    return RetrievedChunk(
        id=str(result.get("id") or ""),
        page_content=str(result.get("page_content") or ""),
        metadata=dict(result.get("metadata") or {}),
        score=result.get("score"),
    )


def _filters_for_log(filters: MetadataFilters) -> dict[str, Any]:
    """Return non-empty metadata filters in a log-friendly dictionary."""
    data = filters.model_dump(exclude_none=True)
    return {key: value for key, value in data.items() if value not in ("", [])}


def _sources_for_log(sources: list[CVSource]) -> list[dict[str, Any]]:
    """Return source metadata without snippets for concise logging."""
    return [
        {
            "id": source.id,
            "candidate_name": source.candidate_name,
            "source_path": source.source_path,
            "section": source.section,
            "score": source.score,
        }
        for source in sources
    ]


def _preview(text: str, *, max_chars: int = 160) -> str:
    """Collapse whitespace and truncate text for log previews."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 1].rstrip()}..."


def _snippet(text: str, *, max_chars: int = 240) -> str:
    """Collapse whitespace and truncate retrieved content for source snippets."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 1].rstrip()}..."


__all__ = [
    "AssistantNodes",
    "build_answer_messages",
    "chunks_to_sources",
    "format_context",
    "format_history",
    "route_after_intent",
    "stream_token_callback_var",
]
