"""System endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from core.settings import get_settings
from cv_rag.api.v1.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return application health metadata."""
    settings = get_settings()
    return HealthResponse(
        app=settings.app.name,
        version=settings.app.version,
    )


__all__ = ["router"]
