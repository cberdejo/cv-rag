"""CLI commands for CV ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from core.artifacts import ArtifactStorage
from core.logging import setup_logger
from core.settings import get_cv_ingestion_settings
from cv_ingestion.artifacts import resolve_ingestion_input
from cv_ingestion.exceptions import (
    CVArtifactStagingError,
    CVIngestionConfigurationError,
)
from cv_ingestion.pipeline.pipeline import run_ingest_pipeline


def ingest_command(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory containing local CV artifacts. Ignored when --storage=minio.",
        ),
    ] = Path("outputs"),
    collection: Annotated[
        str | None,
        typer.Option(
            "--collection",
            "-c",
            help="Qdrant collection name.",
            show_default=False,
        ),
    ] = None,
    max_workers: Annotated[int, typer.Option("--max-workers")] = 4,
    storage: Annotated[
        ArtifactStorage,
        typer.Option(
            "--storage",
            help="Where CV artifacts are read from: local, minio, or both.",
        ),
    ] = ArtifactStorage.LOCAL,
    bucket: Annotated[
        str | None,
        typer.Option(
            "--bucket",
            help="MinIO bucket for CV artifacts. Required for minio or both storage.",
            show_default=False,
        ),
    ] = None,
    minio_endpoint: Annotated[
        str | None,
        typer.Option(
            "--minio-endpoint",
            help="MinIO endpoint (host:port) override. Defaults to MINIO_ENDPOINT.",
            show_default=False,
        ),
    ] = None,
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate",
            help="Drop and recreate the Qdrant collection before ingestion.",
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR or CRITICAL.",
        ),
    ] = "INFO",
    log_to_file: Annotated[
        bool,
        typer.Option(
            "--log-to-file",
            help="Write logs to files in addition to the console.",
        ),
    ] = False,
) -> None:
    """Ingest generated CV artifacts into the Qdrant index.

    Args:
        input_dir: Directory containing local CV artifacts.
        collection: Optional Qdrant collection override.
        max_workers: Maximum number of concurrent ingestion workers.
        storage: Artifact source: local filesystem, MinIO, or both.
        bucket: MinIO bucket required when storage includes MinIO.
        minio_endpoint: MinIO endpoint override. Defaults to MINIO_ENDPOINT.
        recreate: Drop and recreate the target Qdrant collection first.
        log_level: Console/file logging threshold.
        log_to_file: Also write logs under the configured log directory.
    """
    setup_logger(level=log_level, log_to_file=log_to_file)
    collection_name = collection or get_cv_ingestion_settings().collection_name

    try:
        with resolve_ingestion_input(
            input_dir=input_dir,
            storage=storage,
            bucket=bucket,
            minio_endpoint=minio_endpoint,
        ) as resolved:
            result = run_ingest_pipeline(
                resolved.input_dir,
                collection_name=collection_name,
                recreate_collection=recreate,
                source_uri_by_local_path=resolved.source_uri_by_local_path,
                max_workers=max_workers,
            )
    except CVIngestionConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except CVArtifactStagingError as exc:
        typer.echo(f"Could not prepare CV artifacts for ingestion: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"Ingestion complete: {result.succeeded}/{result.total} CVs indexed successfully."
    )

    if result.failed:
        typer.echo(f"Failed: {result.failed} CVs could not be ingested.")
        for err in result.errors:
            typer.echo(f"  - {err.source_path}: {err.error}")


__all__ = ["ingest_command"]
