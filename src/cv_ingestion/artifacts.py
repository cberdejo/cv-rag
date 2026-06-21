"""Artifact staging helpers for CV ingestion."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from core.artifacts import ArtifactStorage
from cv_ingestion.exceptions import (
    CVArtifactStagingError,
    CVIngestionConfigurationError,
)
from infrastructure.minio_client import SUPPORTED_CV_SUFFIXES, download_artifact_sources


@dataclass(frozen=True)
class ResolvedIngestionInput:
    """Local staging directory plus durable source URIs for staged files."""

    input_dir: Path
    source_uri_by_local_path: dict[Path, str]


@contextmanager
def resolve_ingestion_input(
    *,
    input_dir: Path,
    storage: ArtifactStorage,
    bucket: str | None,
    minio_endpoint: str | None = None,
) -> Iterator[ResolvedIngestionInput]:
    """
    Resolve local and/or MinIO CV artifacts into a local directory.

    The ingestion pipeline only knows how to process local files. This function
    normalizes all supported storage modes into a local input directory while
    preserving the durable source URI for staged files.

    Storage modes:
    - LOCAL: use ``input_dir`` directly.
    - MINIO: download MinIO artifacts into a temporary directory.
    - BOTH: copy local artifacts and download MinIO artifacts into one temporary
      directory, keeping source identities in ``source_uri_by_local_path``.
    """
    _validate_storage_options(input_dir=input_dir, storage=storage, bucket=bucket)

    if storage == ArtifactStorage.LOCAL:
        yield ResolvedIngestionInput(
            input_dir=input_dir,
            source_uri_by_local_path={},
        )
        return

    with TemporaryDirectory(prefix="cv-rag-ingest-") as temp_dir:
        staging_dir = Path(temp_dir) / "input"
        source_uri_by_local_path: dict[Path, str] = {}

        if storage == ArtifactStorage.BOTH:
            source_uri_by_local_path.update(
                _stage_local_artifacts(
                    input_dir=input_dir,
                    target_dir=staging_dir,
                )
            )

        if storage in {ArtifactStorage.MINIO, ArtifactStorage.BOTH}:
            source_uri_by_local_path.update(
                _stage_minio_artifacts(
                    bucket=_require_bucket(bucket),
                    target_dir=staging_dir,
                    filename_prefix=(
                        "minio__" if storage == ArtifactStorage.BOTH else ""
                    ),
                    endpoint=minio_endpoint,
                )
            )

        yield ResolvedIngestionInput(
            input_dir=staging_dir,
            source_uri_by_local_path=source_uri_by_local_path,
        )


def _validate_storage_options(
    *,
    input_dir: Path,
    storage: ArtifactStorage,
    bucket: str | None,
) -> None:
    """Validate ingestion source options before staging artifacts."""
    if storage in {ArtifactStorage.LOCAL, ArtifactStorage.BOTH}:
        if not input_dir.exists():
            raise CVIngestionConfigurationError(
                f"Input directory does not exist: {input_dir}"
            )

        if not input_dir.is_dir():
            raise CVIngestionConfigurationError(
                f"Input path is not a directory: {input_dir}"
            )

    if storage in {ArtifactStorage.MINIO, ArtifactStorage.BOTH} and not bucket:
        raise CVIngestionConfigurationError(
            "--bucket is required when --storage is minio or both."
        )


def _stage_local_artifacts(
    *,
    input_dir: Path,
    target_dir: Path,
) -> dict[Path, str]:
    """
    Copy supported local CV artifacts into ``target_dir``.

    Returns:
        Mapping from staged local paths to original resolved filesystem paths.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CVArtifactStagingError(
            f"Could not create CV artifact staging directory: {target_dir}"
        ) from exc

    source_uri_by_local_path: dict[Path, str] = {}
    used_names: set[str] = set()

    for source_path in _discover_supported_cv_paths(input_dir):
        staged_filename = _unique_staged_filename(
            source_path.name,
            used_names,
        )
        staged_path = target_dir / staged_filename

        try:
            copy2(source_path, staged_path)
        except OSError as exc:
            raise CVArtifactStagingError(
                f"Could not stage local CV artifact '{source_path}' to '{staged_path}'."
            ) from exc

        source_uri_by_local_path[staged_path.resolve()] = str(source_path.resolve())

    return source_uri_by_local_path


def _stage_minio_artifacts(
    *,
    bucket: str,
    target_dir: Path,
    filename_prefix: str = "",
    endpoint: str | None = None,
) -> dict[Path, str]:
    """
    Download supported MinIO CV artifacts into ``target_dir``.

    Returns:
        Mapping from downloaded local paths to durable ``minio://`` URIs.
    """
    try:
        downloaded = download_artifact_sources(
            bucket=bucket,
            target_dir=target_dir,
            supported_suffixes=set(SUPPORTED_CV_SUFFIXES),
            filename_prefix=filename_prefix,
            endpoint=endpoint,
        )
    except Exception as exc:
        raise CVArtifactStagingError(
            f"Could not stage MinIO CV artifacts from bucket '{bucket}'."
        ) from exc

    return {artifact.local_path.resolve(): artifact.uri for artifact in downloaded}


def _discover_supported_cv_paths(input_dir: Path) -> list[Path]:
    """Return supported CV files under ``input_dir`` recursively."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_CV_SUFFIXES
    )


def _unique_staged_filename(
    raw_name: str,
    used_names: set[str],
    *,
    prefix: str = "",
) -> str:
    """Return a unique flat filename for a staged artifact."""
    candidate = Path(f"{prefix}{raw_name}")
    name = candidate.name
    suffix = candidate.suffix
    stem = candidate.stem
    counter = 1

    while name in used_names:
        name = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(name)
    return name


def _require_bucket(bucket: str | None) -> str:
    """Return a validated bucket value after option validation."""
    if not bucket:
        raise CVIngestionConfigurationError(
            "--bucket is required when --storage is minio or both."
        )

    return bucket


__all__ = [
    "ResolvedIngestionInput",
    "resolve_ingestion_input",
]
