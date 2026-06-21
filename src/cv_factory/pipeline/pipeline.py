"""
CV generation pipeline.

Orchestrates the full sequence:
  1. build_cv_base          — generate a validated CV skeleton from the catalog
  2. enrich_cv              — enrich narrative fields via LLM
  3. generate_profile_image — produce a profile image
  4. render_cv              — render the PDF artifact

Each step receives the output of the previous one; no global state is used
except application settings.
"""

from __future__ import annotations

import asyncio
from datetime import date
import re
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

from core.artifacts import ArtifactStorage
from core.logging import get_logger
from core.settings import get_cv_factory_settings
from infrastructure.minio_client import upload_file
from cv_factory.models.cv import CV
from cv_factory.pipeline.result import PipelineResult, StoredArtifact
from cv_factory.pipeline.tasks.build_cv_base.main import build_cv_base
from cv_factory.pipeline.tasks.enrich_cv.main import enrich_cv
from cv_factory.pipeline.tasks.generate_profile_image.main import generate_profile_image
from cv_factory.pipeline.tasks.render_cv.main import render_cv

logger = get_logger(__name__)


async def run_pipeline(
    *,
    seed: int | None = None,
    output_dir: Path | None = None,
    skip_image: bool = False,
    artifact_storage: ArtifactStorage = ArtifactStorage.LOCAL,
    minio_bucket: str | None = None,
    minio_endpoint: str | None = None,
) -> PipelineResult:
    """Execute the full CV generation pipeline and return its outputs.

    Args:
        seed: Random seed for reproducible output.
        output_dir: Root directory for generated files.
            Falls back to ``CVFactorySettings.output_dir``.
        skip_image: Skip profile image generation.
        artifact_storage: Where to persist rendered CV artifacts.
        minio_bucket: Bucket used when persisting artifacts to MinIO.
        minio_endpoint: MinIO endpoint override. Defaults to MINIO_ENDPOINT.

    Returns:
        PipelineResult containing the enriched CV and generated artifact paths.
    """
    settings = get_cv_factory_settings()
    root = (output_dir or settings.output_dir).expanduser()

    logger.info(
        "Building CV skeleton | seed={}",
        seed,
    )
    cv: CV = build_cv_base(seed=seed)

    logger.info("Enriching CV with LLM | name={}", cv.name)
    cv_enriched: CV = await enrich_cv(cv)

    photo_url: str | None = None

    if skip_image:
        logger.info("Profile image generation skipped")
    else:
        logger.info(
            "Generating profile image | gender={}",
            cv_enriched.gender.value,
        )
        photo_url = await generate_profile_image(cv_enriched.gender)

    if artifact_storage == ArtifactStorage.MINIO:
        with TemporaryDirectory(prefix="cv-factory-") as temp_dir:
            artifacts = await asyncio.to_thread(
                _render_and_store,
                cv_enriched,
                output_dir=Path(temp_dir),
                photo_url=photo_url,
                seed=seed,
                storage=artifact_storage,
                minio_bucket=minio_bucket,
                minio_endpoint=minio_endpoint,
            )
    else:
        root.mkdir(parents=True, exist_ok=True)
        artifacts = await asyncio.to_thread(
            _render_and_store,
            cv_enriched,
            output_dir=root,
            photo_url=photo_url,
            seed=seed,
            storage=artifact_storage,
            minio_bucket=minio_bucket,
            minio_endpoint=minio_endpoint,
        )

    return PipelineResult(
        cv=cv_enriched,
        artifacts=artifacts,
    )


def _render_and_store(
    cv: CV,
    *,
    output_dir: Path,
    photo_url: str | None,
    seed: int | None,
    storage: ArtifactStorage,
    minio_bucket: str | None = None,
    minio_endpoint: str | None = None,
    today: date | None = None,
) -> tuple[StoredArtifact, ...]:
    """Render a CV and persist the PDF according to the storage mode.

    Args:
        cv: Enriched CV model to render.
        output_dir: Local render directory. For MinIO-only runs this may be a
            temporary directory.
        photo_url: Optional profile photo (typically a base64 data URI)
            embedded directly in templates that use it.
        seed: Template randomization seed passed to the renderer.
        storage: Target storage mode for the generated PDF.
        minio_bucket: Bucket used when uploading to MinIO.
        minio_endpoint: MinIO endpoint override. Defaults to MINIO_ENDPOINT.
        today: Optional date override for deterministic object names in tests.

    Returns:
        Tuple containing the stored PDF artifact metadata.
    """
    logger.info(
        "Rendering CV artifacts | output_dir={} | storage={}",
        output_dir,
        storage.value,
    )
    rendered = render_cv(
        cv,
        output_dir=output_dir,
        photo_url=photo_url,
        seed=seed,
    )
    local_path = rendered.pdf_path

    if storage == ArtifactStorage.LOCAL:
        return (StoredArtifact(kind="pdf", local_path=local_path, minio_uri=None),)

    object_name = minio_pdf_object_name(cv=cv, today=today)
    minio_uri = upload_file(
        rendered.pdf_path,
        object_name=object_name,
        content_type="application/pdf",
        bucket=minio_bucket,
        endpoint=minio_endpoint,
    )
    logger.info("Uploaded artifact to MinIO | uri={}", minio_uri)

    return (
        StoredArtifact(
            kind="pdf",
            local_path=local_path if storage == ArtifactStorage.BOTH else None,
            minio_uri=minio_uri,
        ),
    )


def minio_pdf_object_name(*, cv: CV, today: date | None = None) -> str:
    """Return the root-level MinIO PDF object name for a CV."""
    run_date = today or date.today()
    return f"{run_date.isoformat()}-{_safe_object_stem(cv.name)}.pdf"


def _safe_object_stem(value: str) -> str:
    """Normalize a human name into a safe lowercase object-name stem."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    candidate = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return candidate or "cv"


__all__ = [
    "ArtifactStorage",
    "PipelineResult",
    "StoredArtifact",
    "run_pipeline",
]
