"""CLI commands for cv-factory."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from core.artifacts import ArtifactStorage
from core.logging import setup_logger
from cv_factory.pipeline.batch_pipeline import run_batch_pipeline


def generate_cv_command(
    count: Annotated[int, typer.Option("--count", "-n")] = 1,
    max_workers: Annotated[int, typer.Option("--max-workers")] = 4,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducible CV generation.",
            show_default=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where generated artifacts will be written.",
            show_default=False,
        ),
    ] = None,
    storage: Annotated[
        ArtifactStorage,
        typer.Option(
            "--storage",
            help="Where generated artifacts are saved: local, minio, or both.",
        ),
    ] = ArtifactStorage.LOCAL,
    bucket: Annotated[
        str | None,
        typer.Option(
            "--bucket",
            help="MinIO bucket for generated artifacts. Required for minio or both storage.",
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
    skip_image: Annotated[
        bool,
        typer.Option(
            "--skip-image/--no-skip-image",
            help="Skip profile image generation.",
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
    """Run the full CV generation pipeline from the CLI.

    Args:
        count: Number of synthetic CVs to generate.
        max_workers: Maximum number of concurrent generation workers.
        seed: Optional base random seed for reproducible CVs.
        output_dir: Local directory for generated artifacts.
        storage: Artifact target: local filesystem, MinIO, or both.
        bucket: MinIO bucket required when storage includes MinIO.
        minio_endpoint: MinIO endpoint override. Defaults to MINIO_ENDPOINT.
        skip_image: Skip external profile image generation.
        log_level: Console/file logging threshold.
        log_to_file: Also write logs under the configured log directory.
    """
    setup_logger(level=log_level, log_to_file=log_to_file)

    if storage in {ArtifactStorage.MINIO, ArtifactStorage.BOTH} and not bucket:
        raise typer.BadParameter(
            "--bucket is required when --storage is minio or both."
        )

    result = asyncio.run(
        run_batch_pipeline(
            count=count,
            output_dir=output_dir,
            base_seed=seed,
            skip_image=skip_image,
            max_workers=max_workers,
            artifact_storage=storage,
            minio_bucket=bucket,
            minio_endpoint=minio_endpoint,
        )
    )
    typer.echo(
        f"Completed: {result.succeeded}/{result.total} CVs generated successfully."
    )

    if result.failed:
        typer.echo(f"Failed: {result.failed} CVs failed.")
