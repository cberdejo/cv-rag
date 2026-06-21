"""Router aggregation for cv-rag v1."""

from __future__ import annotations

from fastapi import APIRouter

from cv_rag.api.v1.endpoints import assistant, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(assistant.router)


__all__ = ["api_router"]
