"""Tests for rag/retrieval/indexer.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from cv_ingestion.models.chunking import CVChunk, CVChunkMetadata, CVSection
from rag.assistant.models import MetadataFilters
from rag.exceptions import RAGIndexingError, RAGRetrievalError
from rag.retrieval.indexer import (
    QdrantCVStore,
    _chunk_to_point,
    build_metadata_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    source_path: str = "/cvs/test.pdf",
    candidate_name: str = "Test Candidate",
    section: CVSection = CVSection.EXPERIENCE,
    page_content: str = "Python developer with 5 years of experience.",
) -> CVChunk:
    metadata = CVChunkMetadata(
        candidate_name=candidate_name,
        source_path=source_path,
        section=section,
        chunk_index=0,
        skills=["Python", "FastAPI"],
        current_title="Backend Developer",
        current_company="Acme Corp",
        years_of_experience=5,
        degrees=["Bachelor in CS"],
        institutions=["Tech University"],
        companies=["Acme Corp", "Initech"],
        email="test@example.com",
    )
    return CVChunk(page_content=page_content, metadata=metadata)


def _make_store() -> QdrantCVStore:
    store = QdrantCVStore.__new__(QdrantCVStore)
    store.collection_name = "test_collection"
    store._client = MagicMock()
    store.reranker_model = "test-reranker"
    store._reranker = None
    return store


class StaticReranker:
    """Cross-encoder stub that returns a fixed score per call, in order."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def rerank(self, query: str, documents: list[str]):
        return self.scores


# ---------------------------------------------------------------------------
# _chunk_to_point
# ---------------------------------------------------------------------------


class TestChunkToPoint:
    def test_returns_point_struct(self):
        chunk = _make_chunk()
        point = _chunk_to_point(chunk)
        assert isinstance(point, models.PointStruct)

    def test_id_is_uuid_string(self):
        chunk = _make_chunk()
        point = _chunk_to_point(chunk)
        import uuid

        uuid.UUID(str(point.id))  # raises if not a valid UUID

    def test_payload_contains_page_content(self):
        chunk = _make_chunk(page_content="Experienced in Docker.")
        point = _chunk_to_point(chunk)
        assert point.payload["page_content"] == "Experienced in Docker."

    def test_metadata_section_is_string(self):
        chunk = _make_chunk(section=CVSection.SKILLS)
        point = _chunk_to_point(chunk)
        # Enum must be serialised to its string value for Qdrant
        assert point.payload["metadata"]["section"] == "skills"

    def test_dense_and_sparse_vectors_present(self):
        from rag.retrieval.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

        chunk = _make_chunk()
        point = _chunk_to_point(chunk)
        assert DENSE_VECTOR_NAME in point.vector
        assert SPARSE_VECTOR_NAME in point.vector

    def test_dense_vector_gets_e5_passage_prefix_sparse_does_not(self):
        from rag.retrieval.indexer import (
            DENSE_VECTOR_NAME,
            E5_PASSAGE_PREFIX,
            SPARSE_VECTOR_NAME,
        )

        chunk = _make_chunk(page_content="Experienced in Docker.")
        point = _chunk_to_point(chunk)

        assert point.vector[DENSE_VECTOR_NAME].text == (
            E5_PASSAGE_PREFIX + "Experienced in Docker."
        )
        assert point.vector[SPARSE_VECTOR_NAME].text == "Experienced in Docker."


# ---------------------------------------------------------------------------
# build_metadata_filter
# ---------------------------------------------------------------------------


class TestBuildMetadataFilter:
    def test_none_filter_returns_none(self):
        f = MetadataFilters()
        assert build_metadata_filter(f) is None

    def test_candidate_name_filter(self):
        f = MetadataFilters(candidate_name="Ana García")
        result = build_metadata_filter(f)
        assert result is not None
        keys = [c.key for c in result.must]
        assert "metadata.candidate_name" in keys

    def test_skills_filter_adds_one_condition_per_skill(self):
        f = MetadataFilters(skills=["Python", "Docker"])
        result = build_metadata_filter(f)
        assert result is not None
        skill_conditions = [c for c in result.must if c.key == "metadata.skills"]
        assert len(skill_conditions) == 2

    def test_companies_filter(self):
        f = MetadataFilters(companies=["Acme", "Initech"])
        result = build_metadata_filter(f)
        company_conditions = [c for c in result.must if c.key == "metadata.companies"]
        assert len(company_conditions) == 2

    def test_years_of_experience_range(self):
        f = MetadataFilters(min_years_of_experience=3, max_years_of_experience=8)
        result = build_metadata_filter(f)
        range_conditions = [
            c for c in result.must if c.key == "metadata.years_of_experience"
        ]
        assert len(range_conditions) == 1
        r = range_conditions[0].range
        assert r.gte == 3
        assert r.lte == 8

    def test_min_years_only(self):
        f = MetadataFilters(min_years_of_experience=5)
        result = build_metadata_filter(f)
        range_cond = next(
            c for c in result.must if c.key == "metadata.years_of_experience"
        )
        assert range_cond.range.gte == 5
        assert range_cond.range.lte is None

    def test_degrees_filter(self):
        f = MetadataFilters(degrees=["Bachelor in CS"])
        result = build_metadata_filter(f)
        degree_conds = [c for c in result.must if c.key == "metadata.degrees"]
        assert len(degree_conds) == 1
        assert degree_conds[0].match.text == "bachelor in cs"

    def test_empty_string_skills_ignored(self):
        f = MetadataFilters(skills=["Python", ""])
        result = build_metadata_filter(f)
        skill_conditions = [c for c in result.must if c.key == "metadata.skills"]
        # "" should be filtered out
        assert len(skill_conditions) == 1

    def test_section_filter(self):
        f = MetadataFilters(section="experience")
        result = build_metadata_filter(f)
        section_conds = [c for c in result.must if c.key == "metadata.section"]
        assert len(section_conds) == 1
        assert section_conds[0].match.value == "experience"

    def test_certifications_filter_adds_one_condition_per_item(self):
        f = MetadataFilters(certifications=["AWS Certified Solutions Architect"])
        result = build_metadata_filter(f)
        certification_conds = [
            c for c in result.must if c.key == "metadata.certifications"
        ]
        assert len(certification_conds) == 1
        assert certification_conds[0].match.text == "aws certified solutions architect"

    def test_languages_filter_adds_one_condition_per_item(self):
        f = MetadataFilters(languages=["German", "Spanish"])
        result = build_metadata_filter(f)
        language_conds = [c for c in result.must if c.key == "metadata.languages"]
        assert len(language_conds) == 2

    def test_mapping_input_works(self):
        f = {"candidate_name": "Juan", "skills": [], "companies": []}
        result = build_metadata_filter(f)
        assert result is not None
        name_conds = [c for c in result.must if c.key == "metadata.candidate_name"]
        assert len(name_conds) == 1


# ---------------------------------------------------------------------------
# QdrantCVStore.index (mocked)
# ---------------------------------------------------------------------------


class TestQdrantCVStoreIndex:
    @pytest.mark.asyncio
    async def test_index_returns_chunk_count(self):
        store = _make_store()
        store._client.upsert = AsyncMock()
        chunks = [_make_chunk() for _ in range(3)]

        indexed = await store.index(chunks)

        assert indexed == 3

    @pytest.mark.asyncio
    async def test_index_calls_upsert(self):
        store = _make_store()
        store._client.upsert = AsyncMock()
        chunks = [_make_chunk()]

        await store.index(chunks)

        store._client.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_index_raises_indexing_error_on_failure(self):
        store = _make_store()
        store._client.upsert = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        with pytest.raises(RAGIndexingError):
            await store.index([_make_chunk()])

    @pytest.mark.asyncio
    async def test_index_batches_large_collections(self):
        store = _make_store()
        store._client.upsert = AsyncMock()
        chunks = [_make_chunk() for _ in range(130)]

        await store.index(chunks, batch_size=64)

        # 130 chunks / 64 per batch → 3 batches
        assert store._client.upsert.call_count == 3


# ---------------------------------------------------------------------------
# QdrantCVStore.delete_cv (mocked)
# ---------------------------------------------------------------------------


class TestQdrantCVStoreDeleteCV:
    @pytest.mark.asyncio
    async def test_delete_calls_qdrant_delete(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.result = 3
        store._client.delete = AsyncMock(return_value=mock_result)

        deleted = await store.delete_cv("/cvs/old.pdf")

        store._client.delete.assert_called_once()
        assert deleted == 3

    @pytest.mark.asyncio
    async def test_delete_raises_indexing_error_on_failure(self):
        store = _make_store()
        store._client.delete = AsyncMock(side_effect=Exception("connection error"))

        with pytest.raises(RAGIndexingError):
            await store.delete_cv("/cvs/missing.pdf")


# ---------------------------------------------------------------------------
# QdrantCVStore.search_hybrid (mocked)
# ---------------------------------------------------------------------------


class TestQdrantCVStoreSearchHybrid:
    def _make_mock_point(self, candidate: str, source: str) -> MagicMock:
        point = MagicMock()
        point.id = "uuid-1"
        point.score = 0.85
        point.payload = {
            "page_content": f"Content for {candidate}",
            "metadata": {"candidate_name": candidate, "source_path": source},
        }
        return point

    @pytest.mark.asyncio
    async def test_returns_results(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.points = [self._make_mock_point("Ana", "/cvs/ana.pdf")]
        store._client.query_points = AsyncMock(return_value=mock_result)

        results = await store.search_hybrid("Python developer", limit=5)

        assert len(results) == 1
        assert results[0]["metadata"]["candidate_name"] == "Ana"

    @pytest.mark.asyncio
    async def test_dense_prefetch_gets_e5_query_prefix_sparse_does_not(self):
        from rag.retrieval.indexer import E5_QUERY_PREFIX

        store = _make_store()
        mock_result = MagicMock()
        mock_result.points = []
        store._client.query_points = AsyncMock(return_value=mock_result)

        await store.search_hybrid("Python developer", limit=5)

        prefetch = store._client.query_points.call_args.kwargs["prefetch"]
        dense_prefetch, sparse_prefetch = prefetch
        assert dense_prefetch.query.text == E5_QUERY_PREFIX + "Python developer"
        assert sparse_prefetch.query.text == "Python developer"

    @pytest.mark.asyncio
    async def test_raises_retrieval_error_on_failure(self):
        store = _make_store()
        store._client.query_points = AsyncMock(side_effect=RuntimeError("query failed"))

        with pytest.raises(RAGRetrievalError):
            await store.search_hybrid("test query")

    @pytest.mark.asyncio
    async def test_does_not_deduplicate_by_source_path_without_rerank(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.points = [
            self._make_mock_point("Ana - experience", "/cvs/ana.pdf"),
            self._make_mock_point("Ana - skills", "/cvs/ana.pdf"),
            self._make_mock_point("Beto", "/cvs/beto.pdf"),
        ]
        store._client.query_points = AsyncMock(return_value=mock_result)

        results = await store.search_hybrid("python developer", limit=5)

        # Each CV is chunked one-per-section, so two sections of the same
        # candidate can legitimately both appear in the result set.
        assert len(results) == 3
        assert [r["metadata"]["source_path"] for r in results] == [
            "/cvs/ana.pdf",
            "/cvs/ana.pdf",
            "/cvs/beto.pdf",
        ]

    @pytest.mark.asyncio
    async def test_does_not_deduplicate_by_source_path_with_rerank(self):
        store = _make_store()
        mock_result = MagicMock()
        mock_result.points = [
            self._make_mock_point("Ana - experience", "/cvs/ana.pdf"),
            self._make_mock_point("Ana - skills", "/cvs/ana.pdf"),
            self._make_mock_point("Beto", "/cvs/beto.pdf"),
        ]
        store._client.query_points = AsyncMock(return_value=mock_result)
        store._reranker = StaticReranker([0.40, 0.90, 0.50])

        results = await store.search_hybrid(
            "python developer", limit=5, rerank=True
        )

        # Both Ana chunks rank above Beto's and both are kept: the top match
        # can contribute more than one section as LLM context.
        assert len(results) == 3
        assert [r["page_content"] for r in results] == [
            "Content for Ana - skills",
            "Content for Beto",
            "Content for Ana - experience",
        ]


# ---------------------------------------------------------------------------
# QdrantCVStore.search_metadata (mocked)
# ---------------------------------------------------------------------------


class TestQdrantCVStoreSearchMetadata:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_filters(self):
        store = _make_store()
        results = await store.search_metadata(MetadataFilters())
        assert results == []

    @pytest.mark.asyncio
    async def test_deduplicates_by_source_path(self):
        store = _make_store()

        def _make_point(i):
            p = MagicMock()
            p.id = f"id-{i}"
            p.score = 0.5
            p.payload = {
                "page_content": f"chunk-{i}",
                "metadata": {"source_path": "/cvs/same.pdf", "candidate_name": "X"},
            }
            return p

        store._client.scroll = AsyncMock(
            return_value=([_make_point(i) for i in range(5)], None)
        )

        results = await store.search_metadata(
            MetadataFilters(candidate_name="X"),
            limit=10,
        )

        # All 5 share the same source_path, so only 1 deduplicated result
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_prefers_experience_section_for_companies_filter(self):
        # Metadata filters are CV-level facts denormalised onto every chunk
        # (see CVChunkMetadata), so "companies" matches every section of a
        # CV that worked at that company — but only the "experience" chunk's
        # page_content actually mentions it. Dedup must not arbitrarily keep
        # whichever chunk the scroll happens to return first.
        store = _make_store()

        def _make_point(section: str, content: str):
            p = MagicMock()
            p.id = f"id-{section}"
            p.score = 0.5
            p.payload = {
                "page_content": content,
                "metadata": {
                    "source_path": "/cvs/hugo.pdf",
                    "candidate_name": "Hugo",
                    "section": section,
                    "companies": ["promociones del noroeste s.c.p"],
                },
            }
            return p

        store._client.scroll = AsyncMock(
            return_value=(
                [
                    _make_point("skills", "Python, FastAPI"),
                    _make_point("experience", "Worked at Promociones del Noroeste S.C.P"),
                ],
                None,
            )
        )

        results = await store.search_metadata(
            MetadataFilters(companies=["Promociones del Noroeste S.C.P"]),
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["metadata"]["section"] == "experience"
        assert "Promociones" in results[0]["page_content"]

    @pytest.mark.asyncio
    async def test_raises_retrieval_error_on_failure(self):
        store = _make_store()
        store._client.scroll = AsyncMock(side_effect=RuntimeError("scroll failed"))

        with pytest.raises(RAGRetrievalError):
            await store.search_metadata(MetadataFilters(candidate_name="X"))


# ---------------------------------------------------------------------------
# QdrantCVStore.create_collection (mocked)
# ---------------------------------------------------------------------------


class TestQdrantCVStoreCreateCollection:
    @pytest.mark.asyncio
    async def test_creates_collection_if_not_exists(self):
        store = _make_store()
        store._client.collection_exists = AsyncMock(return_value=False)
        store._client.get_embedding_size = MagicMock(return_value=384)
        store._client.create_collection = AsyncMock()
        store._client.create_payload_index = AsyncMock()

        await store.create_collection()

        store._client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_indexes_for_certifications_and_languages(self):
        from rag.retrieval.indexer import _FULLTEXT_FIELDS, _KEYWORD_FIELDS

        store = _make_store()
        store._client.collection_exists = AsyncMock(return_value=False)
        store._client.get_embedding_size = MagicMock(return_value=384)
        store._client.create_collection = AsyncMock()
        store._client.create_payload_index = AsyncMock()

        await store.create_collection()

        indexed_fields = {
            call.kwargs["field_name"]
            for call in store._client.create_payload_index.call_args_list
        }
        # certifications uses fuzzy MatchText (punctuation/abbreviation drift,
        # e.g. versioned cert names); languages stays exact (single tokens).
        assert "metadata.certifications" in _FULLTEXT_FIELDS
        assert "metadata.languages" in _KEYWORD_FIELDS
        assert (_FULLTEXT_FIELDS | _KEYWORD_FIELDS) <= indexed_fields

    @pytest.mark.asyncio
    async def test_skips_creation_if_already_exists(self):
        store = _make_store()
        store._client.collection_exists = AsyncMock(return_value=True)
        store._client.create_collection = AsyncMock()

        await store.create_collection()

        store._client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_recreates_when_flag_set(self):
        store = _make_store()
        store._client.collection_exists = AsyncMock(return_value=True)
        store._client.delete_collection = AsyncMock()
        store._client.get_embedding_size = MagicMock(return_value=384)
        store._client.create_collection = AsyncMock()
        store._client.create_payload_index = AsyncMock()

        await store.create_collection(recreate=True)

        store._client.delete_collection.assert_called_once()
        store._client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_indexing_error_on_failure(self):
        store = _make_store()
        store._client.collection_exists = AsyncMock(
            side_effect=RuntimeError("Qdrant unreachable")
        )

        with pytest.raises(RAGIndexingError):
            await store.create_collection()
