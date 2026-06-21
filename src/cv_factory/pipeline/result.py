"""Dataclasses returned by CV factory pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cv_factory.models.cv import CV


@dataclass(frozen=True)
class StoredArtifact:
    """Local and/or remote location for one generated artifact."""

    kind: str
    local_path: Path | None
    minio_uri: str | None


@dataclass(frozen=True)
class PipelineResult:
    """Outputs produced by one successful single-CV pipeline execution."""

    cv: CV
    artifacts: tuple[StoredArtifact, ...]


@dataclass(frozen=True)
class BatchResult:
    """Aggregate success and failure counts for a batch generation run."""

    total: int
    succeeded: int
    failed: int
    errors: list[tuple[int, Exception]]


__all__ = [
    "BatchResult",
    "PipelineResult",
    "StoredArtifact",
]
