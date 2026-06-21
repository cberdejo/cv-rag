"""Public service for the CV RAG assistant."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from core.settings import LLMSettings, get_cv_ingestion_settings, get_qdrant_settings
from rag.exceptions import RAGAssistantError, RAGError
from rag.assistant.models import AssistantResponse, AssistantState
from rag.assistant.nodes import (
    AssistantNodes,
    route_after_intent,
    stream_token_callback_var,
)
from rag.retrieval.indexer import QdrantCVStore

StreamTokenCallback = Callable[[str], Awaitable[None]]


class CVAssistant:
    """Async service for asking questions over indexed CVs."""

    def __init__(
        self,
        *,
        indexer: QdrantCVStore | None = None,
        settings: LLMSettings | None = None,
        client: AsyncOpenAI | None = None,
        memory: MemorySaver | None = None,
    ) -> None:
        """Initialize assistant dependencies and compile the LangGraph workflow.

        Args:
            indexer: Optional retrieval store. When omitted, a Qdrant-backed store is
                created from application settings.
            settings: Optional LLM settings override used by assistant nodes.
            client: Optional OpenAI-compatible async client for tests or custom
                LiteLLM routing.
            memory: Optional LangGraph checkpointer used to persist thread state.
        """
        if indexer is None:
            qdrant_settings = get_qdrant_settings()
            ingestion_settings = get_cv_ingestion_settings()
            indexer = QdrantCVStore(
                url=qdrant_settings.url,
                collection_name=ingestion_settings.collection_name,
            )

        self.indexer = indexer
        self.settings = settings
        self.client = client
        self.memory = memory or MemorySaver()
        self._graph = self._build_graph()

    async def ask(self, message: str, *, thread_id: str) -> AssistantResponse:
        """Ask the assistant a question within a memory thread."""
        return await self._invoke(message, thread_id=thread_id)

    async def ask_stream(
        self,
        message: str,
        *,
        thread_id: str,
        on_token: StreamTokenCallback,
    ) -> AssistantResponse:
        """Ask the assistant and stream generated answer tokens through a callback."""
        token = stream_token_callback_var.set(on_token)
        try:
            return await self._invoke(message, thread_id=thread_id)
        finally:
            stream_token_callback_var.reset(token)

    async def _invoke(self, message: str, *, thread_id: str) -> AssistantResponse:
        """Run the assistant graph within a memory thread."""
        try:
            final_state = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except RAGError:
            raise
        except Exception as exc:
            raise RAGAssistantError("The assistant graph failed.") from exc

        return AssistantResponse(
            answer=final_state.get("answer", ""),
            sources=final_state.get("sources", []),
        )

    def _build_graph(self):
        """Build and compile the assistant's LangGraph state machine."""
        nodes = AssistantNodes(
            indexer=self.indexer,
            settings=self.settings,
            client=self.client,
        )
        workflow = StateGraph(AssistantState)

        # --- nodes -------------------------------------------------------
        workflow.add_node("rewrite_query", nodes.rewrite_query)
        workflow.add_node("route_intent", nodes.route_intent)
        workflow.add_node("metadata_search", nodes.metadata_search)
        workflow.add_node("hybrid_search", nodes.hybrid_search)
        workflow.add_node("hybrid_filtered_search", nodes.hybrid_filtered_search)
        workflow.add_node("generate_answer", nodes.generate_answer)
        workflow.add_node("out_of_scope", nodes.out_of_scope)

        # --- edges -------------------------------------------------------
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "route_intent")
        workflow.add_conditional_edges(
            "route_intent",
            route_after_intent,
            {
                "metadata_search": "metadata_search",
                "hybrid_search": "hybrid_search",
                "hybrid_filtered_search": "hybrid_filtered_search",
                "out_of_scope": "out_of_scope",
            },
        )
        workflow.add_edge("metadata_search", "generate_answer")
        workflow.add_edge("hybrid_search", "generate_answer")
        workflow.add_edge("hybrid_filtered_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("out_of_scope", END)

        return workflow.compile(checkpointer=self.memory)


__all__ = ["CVAssistant"]
