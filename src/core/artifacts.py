"""Shared artifact-related types."""

from __future__ import annotations

from enum import StrEnum


class ArtifactStorage(StrEnum):
    """Supported artifact storage targets."""

    LOCAL = "local"
    MINIO = "minio"
    BOTH = "both"
