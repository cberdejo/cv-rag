"""Tests for cv_ingestion.cli."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from core.artifacts import ArtifactStorage
from cv_ingestion.artifacts import ResolvedIngestionInput
from cv_ingestion.cli import ingest_command
from cv_ingestion.exceptions import (
    CVArtifactStagingError,
    CVIngestionConfigurationError,
)
from cv_ingestion.pipeline.result import BatchIngestResult, CVIngestResult


@contextmanager
def _resolved(input_dir: Path):
    yield ResolvedIngestionInput(input_dir=input_dir, source_uri_by_local_path={})


def test_ingest_command_converts_configuration_error_to_bad_parameter(
    tmp_path: Path,
) -> None:
    with (
        patch("cv_ingestion.cli.setup_logger"),
        patch(
            "cv_ingestion.cli.resolve_ingestion_input",
            side_effect=CVIngestionConfigurationError("--bucket is required"),
        ),
        pytest.raises(typer.BadParameter, match="--bucket is required"),
    ):
        ingest_command(input_dir=tmp_path, storage=ArtifactStorage.MINIO)


def test_ingest_command_exits_with_status_1_on_staging_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch("cv_ingestion.cli.setup_logger"),
        patch(
            "cv_ingestion.cli.resolve_ingestion_input",
            side_effect=CVArtifactStagingError("could not stage artifacts"),
        ),
        pytest.raises(typer.Exit) as exc_info,
    ):
        ingest_command(input_dir=tmp_path)

    assert exc_info.value.exit_code == 1
    assert "could not stage artifacts" in capsys.readouterr().err


def test_ingest_command_uses_explicit_collection_name(tmp_path: Path) -> None:
    with (
        patch("cv_ingestion.cli.setup_logger"),
        patch(
            "cv_ingestion.cli.resolve_ingestion_input",
            side_effect=lambda **kwargs: _resolved(tmp_path),
        ),
        patch(
            "cv_ingestion.cli.run_ingest_pipeline",
            return_value=BatchIngestResult(total=1, succeeded=1, failed=0),
        ) as run_pipeline_mock,
    ):
        ingest_command(input_dir=tmp_path, collection="custom-collection")

    assert run_pipeline_mock.call_args.kwargs["collection_name"] == "custom-collection"


def test_ingest_command_falls_back_to_settings_collection_name(
    tmp_path: Path,
) -> None:
    with (
        patch("cv_ingestion.cli.setup_logger"),
        patch(
            "cv_ingestion.cli.resolve_ingestion_input",
            side_effect=lambda **kwargs: _resolved(tmp_path),
        ),
        patch(
            "cv_ingestion.cli.run_ingest_pipeline",
            return_value=BatchIngestResult(total=1, succeeded=1, failed=0),
        ) as run_pipeline_mock,
    ):
        ingest_command(input_dir=tmp_path, collection=None)

    assert run_pipeline_mock.call_args.kwargs["collection_name"] == "cvs_collection"


def test_ingest_command_reports_per_cv_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    failed_result = BatchIngestResult(
        total=2,
        succeeded=1,
        failed=1,
        results=[
            CVIngestResult(
                source_path="bad.pdf",
                candidate_name=None,
                chunks_indexed=0,
                success=False,
                error="extraction failed",
            ),
        ],
    )
    with (
        patch("cv_ingestion.cli.setup_logger"),
        patch(
            "cv_ingestion.cli.resolve_ingestion_input",
            side_effect=lambda **kwargs: _resolved(tmp_path),
        ),
        patch("cv_ingestion.cli.run_ingest_pipeline", return_value=failed_result),
    ):
        ingest_command(input_dir=tmp_path)

    out = capsys.readouterr().out
    assert "1/2 CVs indexed successfully" in out
    assert "bad.pdf: extraction failed" in out
