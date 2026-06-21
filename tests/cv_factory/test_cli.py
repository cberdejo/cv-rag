"""Tests for cv_factory.cli."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import typer

from cv_factory.cli import generate_cv_command
from cv_factory.pipeline.pipeline import ArtifactStorage


def test_generate_cv_command_requires_bucket_for_minio() -> None:
    with (
        patch("cv_factory.cli.setup_logger"),
        pytest.raises(typer.BadParameter, match="--bucket is required"),
    ):
        generate_cv_command(storage=ArtifactStorage.MINIO)


def test_generate_cv_command_passes_bucket_to_batch_pipeline(
    tmp_path: Path,
) -> None:
    with (
        patch("cv_factory.cli.setup_logger"),
        patch(
            "cv_factory.cli.run_batch_pipeline",
            new_callable=AsyncMock,
        ) as run_batch_pipeline,
    ):
        generate_cv_command(
            output_dir=tmp_path,
            storage=ArtifactStorage.BOTH,
            bucket="cvs-test",
        )

    assert (
        run_batch_pipeline.call_args.kwargs["artifact_storage"] == ArtifactStorage.BOTH
    )
    assert run_batch_pipeline.call_args.kwargs["minio_bucket"] == "cvs-test"
