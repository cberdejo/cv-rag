"""Shared async OpenAI-compatible LLM client helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from core.settings import LLMSettings, get_llm_settings

DEFAULT_LITELLM_API_KEY = "sk-litellm"


@lru_cache
def get_llm_client() -> AsyncOpenAI:
    """Return a cached async OpenAI-compatible client for the LiteLLM proxy."""
    settings = get_llm_settings()
    return AsyncOpenAI(
        base_url=f"{settings.litellm_base_url.rstrip('/')}/v1",
        api_key=DEFAULT_LITELLM_API_KEY,
    )


async def chat_completion_content(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
    client: AsyncOpenAI | None = None,
    settings: LLMSettings | None = None,
) -> str:
    """Call async chat completions and return the first message content."""
    resolved_settings = settings or get_llm_settings()
    resolved_client = client or get_llm_client()
    response = await resolved_client.chat.completions.create(
        model=model or resolved_settings.model,
        messages=messages,
        temperature=temperature,
        response_format=response_format,
    )
    return response.choices[0].message.content or ""


async def stream_chat_completion_content(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
    client: AsyncOpenAI | None = None,
    settings: LLMSettings | None = None,
) -> AsyncIterator[str]:
    """Stream async chat completion content tokens."""
    resolved_settings = settings or get_llm_settings()
    resolved_client = client or get_llm_client()
    stream = await resolved_client.chat.completions.create(
        model=model or resolved_settings.model,
        messages=messages,
        temperature=temperature,
        response_format=response_format,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


async def chat_json_schema_content(
    *,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
    temperature: float = 0.0,
    client: AsyncOpenAI | None = None,
    settings: LLMSettings | None = None,
) -> str:
    """Call async chat completions with OpenAI JSON Schema structured output."""
    return await chat_completion_content(
        messages=messages,
        model=model,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
        client=client,
        settings=settings,
    )


async def chat_json_object_content(
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    client: AsyncOpenAI | None = None,
    settings: LLMSettings | None = None,
) -> str:
    """Call async chat completions with JSON object response mode."""
    return await chat_completion_content(
        messages=messages,
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        client=client,
        settings=settings,
    )


__all__ = [
    "get_llm_client",
    "chat_completion_content",
    "chat_json_object_content",
    "chat_json_schema_content",
    "stream_chat_completion_content",
]
