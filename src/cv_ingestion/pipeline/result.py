"""Dataclasses returned by CV ingestion pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CVIngestResult:
    """Result of ingesting one source CV into the vector index."""

    source_path: str
    candidate_name: str | None
    chunks_indexed: int
    success: bool
    error: str | None = None


@dataclass
class BatchIngestResult:
    """Aggregated result of ingesting all discovered CV files."""

    total: int
    succeeded: int
    failed: int
    results: list[CVIngestResult] = field(default_factory=list)

    @property
    def errors(self) -> list[CVIngestResult]:
        """Return only failed per-CV ingestion results."""
        return [r for r in self.results if not r.success]


__all__ = [
    "BatchIngestResult",
    "CVIngestResult",
]
