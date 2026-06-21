"""Tests for Qdrant retrieval helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Iterable

import pytest

from rag.exceptions import RAGIndexingError, RAGRetrievalError
from rag.retrieval.indexer import QdrantCVStore


class FakeQdrantClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self.points = points
        self.query_points_kwargs = {}

    async def query_points(self, **kwargs):
        self.query_points_kwargs = kwargs
        return SimpleNamespace(points=self.points)


class FailingQdrantClient:
    async def collection_exists(self, collection_name: str) -> bool:
        raise RuntimeError(f"cannot inspect {collection_name}")

    async def query_points(self, **kwargs):
        raise RuntimeError("query failed")

    async def scroll(self, **kwargs):
        raise RuntimeError("scroll failed")


class StaticReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]:
        document_list = list(documents)
        self.calls.append((query, document_list))
        return self.scores


def test_search_hybrid_reranks_prefetched_candidates() -> None:
    async def run() -> None:
        points = [
            _point("first", 0.99, "generic backend developer"),
            _point("best", 0.55, "python machine learning engineer"),
            _point("second", 0.70, "python data engineer"),
        ]
        client = FakeQdrantClient(points)
        reranker = StaticReranker([0.10, 0.95, 0.60])
        store = QdrantCVStore(
            url="http://qdrant.test",
            collection_name="cv_chunks",
            reranker=reranker,
        )
        store._client = client

        results = await store.search_hybrid(
            "python engineer",
            limit=2,
            prefetch_limit=3,
            rerank=True,
        )

        assert [result["id"] for result in results] == ["best", "second"]
        assert [result["score"] for result in results] == [0.95, 0.60]
        assert [result["rerank_score"] for result in results] == [0.95, 0.60]
        assert [result["hybrid_score"] for result in results] == [0.55, 0.70]
        assert reranker.calls == [
            (
                "python engineer",
                [
                    "generic backend developer",
                    "python machine learning engineer",
                    "python data engineer",
                ],
            )
        ]
        assert client.query_points_kwargs["limit"] == 3
        assert [
            prefetch.limit for prefetch in client.query_points_kwargs["prefetch"]
        ] == [3, 3]

    asyncio.run(run())


def test_create_collection_wraps_vector_store_errors() -> None:
    async def run() -> None:
        store = QdrantCVStore(
            url="http://qdrant.test",
            collection_name="cv_chunks",
        )
        store._client = FailingQdrantClient()

        with pytest.raises(RAGIndexingError, match="cv_chunks") as exc_info:
            await store.create_collection()

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    asyncio.run(run())


def test_search_hybrid_wraps_vector_store_errors() -> None:
    async def run() -> None:
        store = QdrantCVStore(
            url="http://qdrant.test",
            collection_name="cv_chunks",
        )
        store._client = FailingQdrantClient()

        with pytest.raises(RAGRetrievalError, match="Hybrid search failed") as exc_info:
            await store.search_hybrid("python engineer")

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    asyncio.run(run())


def test_search_metadata_wraps_vector_store_errors() -> None:
    async def run() -> None:
        store = QdrantCVStore(
            url="http://qdrant.test",
            collection_name="cv_chunks",
        )
        store._client = FailingQdrantClient()

        with pytest.raises(RAGRetrievalError, match="Metadata search failed"):
            await store.search_metadata({"candidate_name": "Ada"})

    asyncio.run(run())


def _point(point_id: str, score: float, page_content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id,
        score=score,
        payload={
            "page_content": page_content,
            "metadata": {"source_path": f"{point_id}.pdf"},
        },
    )
