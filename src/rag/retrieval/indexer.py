"""
Qdrant indexer and retrieval for CV chunks.

Search modes
------------
search_hybrid
    Pure semantic + keyword fusion (RRF). Optional metadata pre-filter.
    Supports reranking.

search_metadata
    Scroll-based metadata filter with fuzzy text matching.
    Deduplicates by source_path. No vector scoring.

search_metadata_then_rerank
    First collects all candidates that pass a metadata filter (scroll),
    then re-scores them with the cross-encoder against a free-text query.
    Ideal when the hard constraints matter most but you still want the
    best-matching chunks at the top.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Mapping
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import AsyncQdrantClient, models

from core.logging import get_logger
from cv_ingestion.models.chunking import CVChunk, CVSection
from rag.exceptions import RAGError, RAGIndexingError, RAGRetrievalError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Collection schema constants
# ---------------------------------------------------------------------------

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# intfloat/multilingual-e5-large is an *asymmetric* retrieval model: it was
# trained with these literal prefixes distinguishing queries from passages,
# and drops noticeably in quality without them. They must NOT be applied to
# the sparse (BM25) text, which matches on raw terms. See README
# "Retrieval Model and Reranking Decisions".
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Oversampling factor: how many candidates to prefetch per vector branch
# relative to the final `limit` when reranking is enabled. See README
# "Retrieval Model and Reranking Decisions" for the reasoning.
RERANK_PREFETCH_LIMIT = 50

# Fields that support MatchText (Qdrant full-text index) for fuzzy-like matching.
# These must also have a full-text payload index created at collection setup.
# companies/degrees/institutions/certifications are free-text-ish values prone
# to punctuation and abbreviation drift between CVs (e.g. "S.C.P" vs "S.C.P.",
# "B.Sc." vs "BSc") — MatchText's word tokenisation absorbs that variance.
_FULLTEXT_FIELDS = {
    "metadata.candidate_name",
    "metadata.current_title",
    "metadata.current_company",
    "metadata.companies",
    "metadata.degrees",
    "metadata.institutions",
    "metadata.certifications",
}

# Fields that stay as exact KEYWORD indexes (low-cardinality, single-token
# values where fuzzy/partial matching would only add false positives).
_KEYWORD_FIELDS = {
    "metadata.section",
    "metadata.email",
    "metadata.skills",
    "metadata.languages",
}

# Array-valued metadata fields: build_metadata_filter emits one MatchValue
# condition per list item for these (Qdrant evaluates that as "array
# contains X"). Keep in sync with CVChunkMetadata's list fields.
_ARRAY_FILTER_FIELDS = (
    "skills",
    "companies",
    "degrees",
    "institutions",
    "certifications",
    "languages",
)

# Metadata filter fields are CV-level facts denormalised onto every chunk
# (see CVChunkMetadata), but their actual text only lives in one section.
# search_metadata dedupes by source_path and must prefer that section —
# otherwise it can return e.g. the "skills" chunk for a "companies" filter,
# whose page_content never mentions the matched company.
_FIELD_SECTION_PRIORITY: dict[str, CVSection] = {
    "companies": CVSection.EXPERIENCE,
    "current_company": CVSection.EXPERIENCE,
    "current_title": CVSection.EXPERIENCE,
    "degrees": CVSection.EDUCATION,
    "institutions": CVSection.EDUCATION,
    "certifications": CVSection.CERTIFICATIONS,
    "languages": CVSection.LANGUAGES,
    "skills": CVSection.SKILLS,
}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class QdrantCVStore:
    """
    Indexes CVChunk objects into a Qdrant collection using hybrid vectors
    (dense + sparse) so both semantic and keyword queries work.

    All public search methods accept an optional ``rerank`` flag.  When True
    the cross-encoder is used to re-score candidates before returning the
    final top-K.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        reranker_model: str = RERANKER_MODEL,
        reranker: Any | None = None,
    ) -> None:
        """Create a Qdrant-backed CV store and defer reranker loading.

        Args:
            url: Qdrant HTTP endpoint.
            collection_name: Collection used to store CV chunks.
            reranker_model: Cross-encoder model loaded when reranking is requested.
            reranker: Optional injected reranker instance for tests or reuse.
        """
        self.collection_name = collection_name
        self._client = AsyncQdrantClient(url=url)
        self.reranker_model = reranker_model
        self._reranker = reranker

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collection(self, *, recreate: bool = False) -> None:
        """Create the Qdrant collection.  Optionally drop and recreate it."""
        try:
            exists = await self._client.collection_exists(self.collection_name)

            if recreate and exists:
                await self._client.delete_collection(self.collection_name)
                exists = False

            if not exists:
                embedding_size = self._client.get_embedding_size(DENSE_MODEL)
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=embedding_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams()
                    },
                )
                await self._create_payload_indexes()
                logger.info(
                    "Collection '{}' created with embedding_size={}",
                    self.collection_name,
                    embedding_size,
                )
        except RAGError:
            raise
        except Exception as exc:
            raise RAGIndexingError(
                f"Could not prepare Qdrant collection '{self.collection_name}'."
            ) from exc

    async def _create_payload_indexes(self) -> None:
        """
        Create payload indexes.

        Full-text indexes (see ``_FULLTEXT_FIELDS``) enable Qdrant's MatchText
        operator, which does tokenised substring matching — a lightweight
        "fuzzy" alternative to exact MatchValue.

        Keyword indexes on the remaining fields support exact/array lookups.
        Integer index on years_of_experience supports range queries.
        """
        try:
            for field in _FULLTEXT_FIELDS:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.TextIndexParams(
                        type=models.TextIndexType.TEXT,
                        tokenizer=models.TokenizerType.WORD,
                        lowercase=True,
                    ),
                )

            for field in _KEYWORD_FIELDS:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

            await self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.years_of_experience",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

            # source_path: keyword index used by delete_cv
            await self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source_path",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except RAGError:
            raise
        except Exception as exc:
            raise RAGIndexingError(
                f"Could not create payload indexes for '{self.collection_name}'."
            ) from exc

        logger.info("Payload indexes created for '{}'", self.collection_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, chunks: list[CVChunk], *, batch_size: int = 64) -> int:
        """
        Upsert all chunks into Qdrant.

        Returns the number of successfully indexed chunks.
        """
        try:
            points = [_chunk_to_point(chunk) for chunk in chunks]
            total = 0

            batch_ranges = range(0, len(points), batch_size)
            for i in batch_ranges:
                batch = points[i : i + batch_size]
                await self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                total += len(batch)
                logger.debug(
                    "Indexed batch {}/{} ({} points)",
                    i // batch_size + 1,
                    -(-len(points) // batch_size),
                    len(batch),
                )

            return total
        except RAGError:
            raise
        except Exception as exc:
            raise RAGIndexingError(
                f"Could not index chunks into '{self.collection_name}'."
            ) from exc

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete_cv(self, source_path: str) -> int:
        """
        Remove all chunks for a given CV identified by source_path.

        Returns the number of deleted points.
        """
        try:
            result = await self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source_path",
                                match=models.MatchValue(value=source_path),
                            )
                        ]
                    )
                ),
            )
            deleted = result.result if hasattr(result, "result") else 0
            logger.info("Deleted {} points for source_path='{}'", deleted, source_path)
            return deleted
        except RAGError:
            raise
        except Exception as exc:
            raise RAGIndexingError(
                f"Could not delete CV chunks for source_path='{source_path}'."
            ) from exc

    # ------------------------------------------------------------------
    # Search — hybrid (vector-based)
    # ------------------------------------------------------------------

    async def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        prefetch_limit: int = RERANK_PREFETCH_LIMIT,
        query_filter: models.Filter | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Dense + sparse RRF fusion search.

        ``query_filter`` is forwarded directly to Qdrant so you can pass an
        already-built filter (e.g. from ``build_metadata_filter``).
        When ``rerank`` is True the cross-encoder re-scores the candidate set.

        Returns the top ``limit`` chunks by score, without deduplicating by
        candidate: a CV is split one chunk per section, so the strongest
        match can legitimately contribute more than one chunk (e.g. its
        "experience" and "skills" sections both ranking high). Deduplicating
        the citation list shown to the user happens separately, in
        ``chunks_to_sources`` (see README "Retrieval Model and Reranking
        Decisions").
        """
        try:
            candidate_limit = max(prefetch_limit, limit) if rerank else limit
            search_result = await self._client.query_points(
                collection_name=self.collection_name,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                prefetch=[
                    models.Prefetch(
                        query=models.Document(
                            text=E5_QUERY_PREFIX + query, model=DENSE_MODEL
                        ),
                        using=DENSE_VECTOR_NAME,
                        limit=candidate_limit,
                    ),
                    models.Prefetch(
                        query=models.Document(text=query, model=SPARSE_MODEL),
                        using=SPARSE_VECTOR_NAME,
                        limit=candidate_limit,
                    ),
                ],
                query_filter=query_filter,
                limit=candidate_limit,
                with_payload=True,
            )
            if rerank:
                return _rerank_points(
                    query=query,
                    points=search_result.points,
                    reranker=self._get_reranker(),
                    limit=limit,
                )
            return [_point_to_result(p) for p in search_result.points]
        except RAGError:
            raise
        except Exception as exc:
            raise RAGRetrievalError(
                f"Hybrid search failed for collection '{self.collection_name}'."
            ) from exc

    # ------------------------------------------------------------------
    # Search — metadata-only scroll (no vectors)
    # ------------------------------------------------------------------

    async def search_metadata(
        self,
        filters: Mapping[str, Any] | object,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Scroll-based metadata filter, deduplicated by source_path.

        Among a source's matching chunks, prefers the section that actually
        contains the filtered fact (see ``_FIELD_SECTION_PRIORITY``) so the
        chunk handed to answer generation isn't an unrelated section that
        happens to carry the same denormalised CV-level metadata.

        No vector scoring — use when you need hard filtered candidate sets
        without a free-text query (e.g. "list all ML Engineers at Accenture").
        """
        try:
            query_filter = build_metadata_filter(filters)
            if query_filter is None:
                return []

            points, _ = await self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=max(limit * 5, 100),
                with_payload=True,
                with_vectors=False,
            )

            preferred_sections = _preferred_sections(filters)

            by_source: dict[str, dict[str, Any]] = {}
            order: list[str] = []
            for point in points:
                result = _point_to_result(point)
                key = result.get("metadata", {}).get("source_path") or str(result["id"])
                if key not in by_source:
                    by_source[key] = result
                    order.append(key)
                    continue
                current_section = by_source[key].get("metadata", {}).get("section")
                if (
                    current_section not in preferred_sections
                    and result.get("metadata", {}).get("section") in preferred_sections
                ):
                    by_source[key] = result

            return [by_source[key] for key in order[:limit]]
        except RAGError:
            raise
        except Exception as exc:
            raise RAGRetrievalError(
                f"Metadata search failed for collection '{self.collection_name}'."
            ) from exc

    # ------------------------------------------------------------------
    # Search — metadata filter → rerank with free-text query
    # ------------------------------------------------------------------

    async def search_metadata_then_rerank(
        self,
        query: str,
        filters: Mapping[str, Any] | object,
        *,
        limit: int = 5,
        scroll_limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Two-phase retrieval: metadata hard filter → cross-encoder rerank.

        Phase 1 — scroll all chunks that satisfy ``filters`` (up to
        ``scroll_limit`` points; every chunk is visible to the reranker,
        including several chunks from the same CV).

        Phase 2 — the cross-encoder scores every candidate against ``query``
        and the top ``limit`` chunks are returned, without deduplicating by
        candidate (see ``search_hybrid`` for why).

        Use this when hard constraints dominate ("must have worked at Google,
        must know Kubernetes") but you still want the most relevant chunks at
        the top rather than arbitrary scroll order.
        """
        try:
            query_filter = build_metadata_filter(filters)
            if query_filter is None:
                # No filter provided — fall back to pure hybrid search
                logger.warning(
                    "search_metadata_then_rerank called with no filter; "
                    "falling back to search_hybrid with rerank=True"
                )
                return await self.search_hybrid(
                    query, limit=limit, prefetch_limit=scroll_limit, rerank=True
                )

            points, _ = await self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=scroll_limit,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                return []

            return _rerank_points(
                query=query,
                points=list(points),
                reranker=self._get_reranker(),
                limit=limit,
            )
        except RAGError:
            raise
        except Exception as exc:
            raise RAGRetrievalError(
                f"Metadata-then-rerank search failed for '{self.collection_name}'."
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reranker(self) -> TextCrossEncoder:
        """Load and cache the cross-encoder reranker on first use."""
        if self._reranker is None:
            self._reranker = TextCrossEncoder(model_name=self.reranker_model)
        return self._reranker


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------


def build_metadata_filter(filters: Mapping[str, Any] | object) -> models.Filter | None:
    """
    Build a Qdrant Filter from metadata field values.

    Text fields (candidate_name, current_title, current_company) use
    ``MatchText`` so that Qdrant's tokenised full-text index handles
    case-insensitive, partial-word matches — effectively fuzzy search
    without a separate embedding pass.

    All values are normalised (lowercased, accent-stripped, whitespace
    collapsed) before building the condition so minor typos and encoding
    differences do not break queries.

    Array fields emit one condition *per item*, which Qdrant evaluates as
    "array contains X". companies/degrees/institutions/certifications use
    ``MatchText`` (same fuzzy-ish tokenised matching as the text fields
    above); skills/languages use exact ``MatchValue``.

    Numeric range: min/max_years_of_experience maps to a Range condition.
    """
    conditions: list[Any] = []

    # --- fuzzy text fields (MatchText via full-text index) ---------------
    _add_text_condition(
        conditions, "metadata.candidate_name", _value(filters, "candidate_name")
    )
    _add_text_condition(
        conditions, "metadata.current_title", _value(filters, "current_title")
    )
    _add_text_condition(
        conditions, "metadata.current_company", _value(filters, "current_company")
    )

    # --- exact keyword fields --------------------------------------------
    _add_keyword_condition(conditions, "metadata.email", _value(filters, "email"))
    _add_keyword_condition(conditions, "metadata.section", _value(filters, "section"))

    # --- array fields (one condition per item) ---------------------------
    # companies/degrees/institutions/certifications use MatchText (fuzzy-ish,
    # tokenised) since stored values have free-text-like punctuation drift.
    # skills/languages stay exact MatchValue (low-cardinality, single-token).
    for field_name in _ARRAY_FILTER_FIELDS:
        key = f"metadata.{field_name}"
        add_condition = _add_text_condition if key in _FULLTEXT_FIELDS else _add_keyword_condition
        for item in _list_value(filters, field_name):
            add_condition(conditions, key, item)

    # --- numeric range ---------------------------------------------------
    min_years = _value(filters, "min_years_of_experience")
    max_years = _value(filters, "max_years_of_experience")
    if min_years is not None or max_years is not None:
        range_kwargs: dict[str, int] = {}
        if min_years is not None:
            range_kwargs["gte"] = int(min_years)
        if max_years is not None:
            range_kwargs["lte"] = int(max_years)
        conditions.append(
            models.FieldCondition(
                key="metadata.years_of_experience",
                range=models.Range(**range_kwargs),
            )
        )

    if not conditions:
        return None
    return models.Filter(must=conditions)


# ---------------------------------------------------------------------------
# Internal helpers — conditions
# ---------------------------------------------------------------------------


def _add_text_condition(
    conditions: list[Any],
    key: str,
    value: Any,
) -> None:
    """
    Add a MatchText condition for fields with a full-text payload index.

    MatchText tokenises both the stored value and the query string and checks
    for token-level containment.  This gives us:
      • case-insensitive matching
      • partial-word matching  ("engineer" matches "Senior Engineer")
      • accent/diacritic normalisation (applied by _normalise before the call)
    """
    if value is None:
        return
    text = _normalise(str(value))
    if not text:
        return
    conditions.append(
        models.FieldCondition(
            key=key,
            match=models.MatchText(text=text),
        )
    )


def _add_keyword_condition(
    conditions: list[Any],
    key: str,
    value: Any,
) -> None:
    """Add an exact MatchValue condition after normalising the value."""
    if value is None:
        return
    if isinstance(value, str):
        normalised = _normalise(value)
        if not normalised:
            return
        conditions.append(
            models.FieldCondition(key=key, match=models.MatchValue(value=normalised))
        )
    else:
        conditions.append(
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
        )


# ---------------------------------------------------------------------------
# Internal helpers — normalisation
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """
    Normalise a filter value for consistent matching.

    Steps:
    1. Unicode NFC normalisation
    2. Strip diacritics (é → e, ü → u) so accent typos don't break filters
    3. Lowercase
    4. Collapse whitespace

    This does *not* do stemming — that is handled implicitly by the dense
    model during hybrid search, and by MatchText tokenisation for text fields.
    """
    # NFC first, then decompose to strip combining diacritical marks
    nfd = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    lowered = stripped.lower()
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


# ---------------------------------------------------------------------------
# Internal helpers — accessors
# ---------------------------------------------------------------------------


def _value(filters: Mapping[str, Any] | object, key: str) -> Any:
    """Read a metadata filter value from either a mapping or an object."""
    if isinstance(filters, Mapping):
        return filters.get(key)
    return getattr(filters, key, None)


def _list_value(filters: Mapping[str, Any] | object, key: str) -> list[Any]:
    """Return a cleaned list-valued filter for mapping or object inputs."""
    value = _value(filters, key)
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    return [value]


def _preferred_sections(filters: Mapping[str, Any] | object) -> set[CVSection]:
    """Sections likely to contain the text behind the active filter fields."""
    return {
        section
        for field_name, section in _FIELD_SECTION_PRIORITY.items()
        if _list_value(filters, field_name)
    }


# ---------------------------------------------------------------------------
# Internal helpers — indexing
# ---------------------------------------------------------------------------


def _chunk_to_point(chunk: CVChunk) -> models.PointStruct:
    """Convert a CVChunk to a Qdrant PointStruct with lazy embedded vectors."""
    meta_dict = chunk.metadata.model_dump(mode="json")
    meta_dict["section"] = chunk.metadata.section.value

    # Normalise stored text fields so they match normalised query values.
    for field in ("candidate_name", "current_title", "current_company"):
        if field in meta_dict and isinstance(meta_dict[field], str):
            meta_dict[field] = _normalise(meta_dict[field])

    # Normalise stored array fields too — build_metadata_filter normalises
    # query values to lowercase, and Qdrant's KEYWORD index match is
    # case-sensitive, so the stored values must match that normalisation.
    for field in _ARRAY_FILTER_FIELDS:
        if field in meta_dict and isinstance(meta_dict[field], list):
            meta_dict[field] = [
                _normalise(item) if isinstance(item, str) else item
                for item in meta_dict[field]
            ]

    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            DENSE_VECTOR_NAME: models.Document(
                text=E5_PASSAGE_PREFIX + chunk.page_content,
                model=DENSE_MODEL,
            ),
            SPARSE_VECTOR_NAME: models.Document(
                text=chunk.page_content,
                model=SPARSE_MODEL,
            ),
        },
        payload={
            "page_content": chunk.page_content,
            "metadata": meta_dict,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers — reranking
# ---------------------------------------------------------------------------


def _rerank_points(
    *,
    query: str,
    points: list[Any],
    reranker: TextCrossEncoder,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Re-score Qdrant points with a cross-encoder and return the top-K results.

    The original hybrid/RRF score is preserved as ``hybrid_score`` so callers
    can inspect both scoring dimensions.
    """
    if not points:
        return []

    documents = [_point_page_content(p) for p in points]
    rerank_scores = list(reranker.rerank(query, documents))

    ranked = sorted(
        zip(points, rerank_scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for point, rerank_score in ranked[:limit]:
        result = _point_to_result(point)
        result["hybrid_score"] = result.get("score")
        result["rerank_score"] = float(rerank_score)
        result["score"] = float(rerank_score)
        results.append(result)
    return results


def _point_page_content(point: Any) -> str:
    """Extract page content from a Qdrant point payload."""
    payload = point.payload or {}
    return payload.get("page_content") or payload.get("page-content") or ""


def _point_to_result(point: Any) -> dict[str, Any]:
    """Convert a Qdrant point into the retrieval result dictionary shape."""
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "score": getattr(point, "score", None),
        "page_content": _point_page_content(point),
        "metadata": payload.get("metadata") or {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "QdrantCVStore",
    "build_metadata_filter",
]
