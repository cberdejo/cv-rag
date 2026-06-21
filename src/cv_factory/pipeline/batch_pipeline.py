"""Async batch orchestration for synthetic CV generation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tqdm import tqdm

from core.artifacts import ArtifactStorage
from core.logging import get_logger
from cv_factory.pipeline.pipeline import run_pipeline
from cv_factory.pipeline.result import BatchResult

logger = get_logger(__name__)


async def run_batch_pipeline(
    *,
    count: int,
    output_dir: Path | None = None,
    base_seed: int | None = None,
    skip_image: bool = False,
    max_workers: int = 4,
    artifact_storage: ArtifactStorage = ArtifactStorage.LOCAL,
    minio_bucket: str | None = None,
    minio_endpoint: str | None = None,
) -> BatchResult:
    """Generate multiple CVs concurrently and collect failure details.

    Args:
        count: Number of CVs to generate.
        output_dir: Optional local artifact output directory.
        base_seed: First random seed; later jobs use ``base_seed + index``.
        skip_image: Skip profile image generation for each CV.
        max_workers: Upper bound for concurrent pipeline workers.
        artifact_storage: Where generated PDF artifacts are persisted.
        minio_bucket: Bucket used when ``artifact_storage`` includes MinIO.
        minio_endpoint: MinIO endpoint override. Defaults to MINIO_ENDPOINT.

    Returns:
        BatchResult with total, success, failure, and exception details.
    """
    workers = max(1, min(max_workers, count))
    semaphore = asyncio.Semaphore(workers)
    seeds = [None if base_seed is None else base_seed + i for i in range(count)]

    async def run_one(index: int, seed: int | None) -> tuple[int, Exception | None]:
        """Run one CV generation job and return its index plus any failure."""
        async with semaphore:
            try:
                await run_pipeline(
                    seed=seed,
                    output_dir=output_dir,
                    skip_image=skip_image,
                    artifact_storage=artifact_storage,
                    minio_bucket=minio_bucket,
                    minio_endpoint=minio_endpoint,
                )
            except Exception as exc:
                logger.exception("CV generation failed | index={}", index)
                return index, exc
            return index, None

    tasks = [
        asyncio.create_task(run_one(index, seed))
        for index, seed in enumerate(seeds, start=1)
    ]

    errors: list[tuple[int, Exception]] = []
    for task in tqdm(
        asyncio.as_completed(tasks),
        total=count,
        desc="Generating CVs",
        disable=count == 1,
    ):
        index, error = await task
        if error is not None:
            errors.append((index, error))

    return BatchResult(
        total=count,
        succeeded=count - len(errors),
        failed=len(errors),
        errors=errors,
    )
