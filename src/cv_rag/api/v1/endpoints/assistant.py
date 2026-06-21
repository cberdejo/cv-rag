"""Assistant chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from core.logging import get_logger
from cv_rag.api.v1.schemas import ChatRequest, ChatResponse
from cv_rag.api.v1.services import get_assistant
from rag.assistant import CVAssistant

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["assistant"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    assistant: CVAssistant = Depends(get_assistant),
) -> ChatResponse:
    """Ask the CV assistant a question."""
    try:
        response = await assistant.ask(
            request.message,
            thread_id=request.thread_id,
        )
    except Exception as exc:
        logger.exception("Assistant request failed")
        raise HTTPException(
            status_code=503,
            detail="The CV assistant is unavailable.",
        ) from exc

    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        thread_id=request.thread_id,
    )


__all__ = ["router"]
