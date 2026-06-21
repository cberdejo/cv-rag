"""
CV ingestion pipeline.

Orchestrates the full sequence for one or many CV files:

    1. extract_text           — CV artifact → plain text
    2. extract_normalized_cv  — plain text → NormalizedCV via LLM
    3. build_chunks           — NormalizedCV → list[CVChunk]
    4. QdrantCVStore.index        — CVChunk list → Qdrant

The pipeline is intentionally resilient: a single bad CV logs the error and
continues rather than aborting the whole batch.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from tqdm import tqdm

from core.logging import get_logger
from core.settings import get_cv_ingestion_settings, get_qdrant_settings
from cv_ingestion.exceptions import (
    CVChunkingError,
    CVExtractionError,
    CVIndexingError,
    CVIngestionConfigurationError,
    EmptyCVTextError,
)
from cv_ingestion.models.chunking import CVChunk
from cv_ingestion.models.normalized_cv import NormalizedCV
from cv_ingestion.pipeline.result import BatchIngestResult, CVIngestResult
from cv_ingestion.pipeline.tasks.chunker import build_chunks
from cv_ingestion.pipeline.tasks.extractor import extract_text
from cv_ingestion.pipeline.tasks.normalizer import extract_normalized_cv
from infrastructure.minio_client import SUPPORTED_CV_SUFFIXES
from rag.retrieval.indexer import QdrantCVStore


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-CV pipeline
# ---------------------------------------------------------------------------


async def ingest_cv(
    path: Path,
    *,
    indexer: QdrantCVStore,
    delete_existing: bool = True,
    source_uri: str | None = None,
) -> CVIngestResult:
    """
    Run the full pipeline for a single CV file.

    Args:
        path:             Path to the local CV file (.pdf).
        indexer:          Pre-configured QdrantCVStore instance.
        delete_existing:  If True, remove any existing chunks for this CV
                          before re-indexing (idempotent re-ingestion).
        source_uri:       Durable source identity stored in chunk metadata.
                          Defaults to the resolved local path.

    Returns:
        CVIngestResult with outcome details.
    """
    local_path = str(path.resolve())
    source_path = source_uri or local_path
    logger.info("Ingesting CV | path={} | source={}", local_path, source_path)

    try:
        # 1. Extract
        try:
            full_text: str = await asyncio.to_thread(extract_text, path)
        except CVExtractionError:
            raise
        except Exception as exc:
            raise CVExtractionError(
                f"Could not extract CV text from '{source_path}'."
            ) from exc

        if not (full_text or "").strip():
            raise EmptyCVTextError("Extracted CV text is empty.")

        # 2. Normalize
        normalized: NormalizedCV = await extract_normalized_cv(source_path, full_text)
        logger.debug("Normalised CV | {!r}", normalized)

        # 3. Chunk
        chunks: list[CVChunk] = build_chunks(normalized)
        if not chunks:
            raise CVChunkingError("No CV chunks produced; all sections were empty.")
        for chunk in chunks:
            logger.debug("CV chunk | source={} | {!r}", source_path, chunk)

        # 4. Index
        try:
            if delete_existing:
                await indexer.delete_cv(source_path)

            indexed = await indexer.index(chunks)
        except Exception as exc:
            raise CVIndexingError(
                f"Could not index CV chunks for '{source_path}'."
            ) from exc
        logger.info(
            "Indexed CV | name={} | chunks={} | path={}",
            normalized.contact.name,
            indexed,
            source_path,
        )

        return CVIngestResult(
            source_path=source_path,
            candidate_name=normalized.contact.name,
            chunks_indexed=indexed,
            success=True,
        )

    except Exception as exc:
        logger.exception(
            "Failed to ingest CV | path={} | source={}", local_path, source_path
        )
        return CVIngestResult(
            source_path=source_path,
            candidate_name=None,
            chunks_indexed=0,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------


async def ingest_directory(
    input_dir: Path,
    *,
    indexer: QdrantCVStore,
    recreate_collection: bool = False,
    delete_existing: bool = True,
    source_uri_by_local_path: Mapping[Path | str, str] | None = None,
    max_workers: int = 4,
) -> BatchIngestResult:
    """
    Ingest all supported CV files found (recursively) in *input_dir*.

    Args:
        input_dir:            Root directory to scan.
        indexer:              Pre-configured QdrantCVStore instance.
        recreate_collection:  Drop and recreate the Qdrant collection.
        delete_existing:      Remove stale chunks before re-indexing each CV.
        source_uri_by_local_path:
            Optional mapping from staged local paths to durable source URIs,
            such as MinIO object URIs.
        max_workers:          Upper bound for concurrent ingestion workers.

    Returns:
        BatchIngestResult with per-file outcomes.
    """
    if not input_dir.exists():
        raise CVIngestionConfigurationError(f"Input directory not found: {input_dir}")

    try:
        await indexer.create_collection(recreate=recreate_collection)
    except Exception as exc:
        raise CVIndexingError("Could not prepare CV index collection.") from exc

    cv_paths = _discover_cv_paths(input_dir)

    if not cv_paths:
        logger.warning("No supported CV files found in '{}'", input_dir)
        return BatchIngestResult(total=0, succeeded=0, failed=0)

    logger.info("Found {} CV files to ingest | dir={}", len(cv_paths), input_dir)

    normalized_source_uri_by_local_path = _normalize_source_uri_map(
        source_uri_by_local_path
    )
    workers = max(1, min(max_workers, len(cv_paths)))
    semaphore = asyncio.Semaphore(workers)

    async def run_one(index: int, path: Path) -> tuple[int, CVIngestResult]:
        """Ingest one CV file under the shared worker semaphore."""
        async with semaphore:
            result = await ingest_cv(
                path,
                indexer=indexer,
                delete_existing=delete_existing,
                source_uri=normalized_source_uri_by_local_path.get(path.resolve()),
            )
            return index, result

    tasks = [
        asyncio.create_task(run_one(index, path))
        for index, path in enumerate(cv_paths)
    ]

    indexed_results: list[tuple[int, CVIngestResult]] = []
    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Ingesting CVs",
        disable=len(tasks) == 1,
    ):
        indexed_results.append(await task)

    results = [result for _, result in sorted(indexed_results, key=lambda item: item[0])]

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded

    logger.info(
        "Batch ingestion complete | total={} succeeded={} failed={}",
        len(results),
        succeeded,
        failed,
    )

    return BatchIngestResult(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


def run_ingest_pipeline(
    input_dir: Path,
    *,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    recreate_collection: bool = False,
    delete_existing: bool = True,
    source_uri_by_local_path: Mapping[Path | str, str] | None = None,
    max_workers: int = 4,
) -> BatchIngestResult:
    """Run the async ingestion pipeline from synchronous entry points."""
    ingestion_settings = get_cv_ingestion_settings()
    qdrant_url = qdrant_url or get_qdrant_settings().url
    collection = collection_name or ingestion_settings.collection_name
    indexer = QdrantCVStore(url=qdrant_url, collection_name=collection)
    return asyncio.run(
        ingest_directory(
            input_dir,
            indexer=indexer,
            recreate_collection=recreate_collection,
            delete_existing=delete_existing,
            source_uri_by_local_path=source_uri_by_local_path,
            max_workers=max_workers,
        )
    )


def _discover_cv_paths(input_dir: Path) -> list[Path]:
    """Find supported CV source files under an input directory."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_CV_SUFFIXES
    )


def _normalize_source_uri_map(
    source_uri_by_local_path: Mapping[Path | str, str] | None,
) -> dict[Path, str]:
    """Resolve source URI mapping keys to absolute local paths."""
    if not source_uri_by_local_path:
        return {}

    return {
        Path(local_path).resolve(): source_uri
        for local_path, source_uri in source_uri_by_local_path.items()
    }


__all__ = [
    "BatchIngestResult",
    "CVIngestResult",
    "ingest_cv",
    "ingest_directory",
    "run_ingest_pipeline",
]
