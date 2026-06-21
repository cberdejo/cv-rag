"""Tests for infrastructure.llm_client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from infrastructure import llm_client


def test_get_llm_client_uses_litellm_proxy_base_url() -> None:
    settings = SimpleNamespace(litellm_base_url="http://litellm:4000/")

    llm_client.get_llm_client.cache_clear()
    with (
        patch("infrastructure.llm_client.get_llm_settings", return_value=settings),
        patch("infrastructure.llm_client.AsyncOpenAI") as openai_cls,
    ):
        llm_client.get_llm_client()

    openai_cls.assert_called_once_with(
        base_url="http://litellm:4000/v1",
        api_key=llm_client.DEFAULT_LITELLM_API_KEY,
    )
    llm_client.get_llm_client.cache_clear()


def test_chat_json_object_content_passes_json_object_response_format() -> None:
    async def run() -> str:
        client = _mock_async_client("content")
        settings = SimpleNamespace(model="test-model")

        result = await llm_client.chat_json_object_content(
            client=client,
            settings=settings,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
        )

        client.chat.completions.create.assert_awaited_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return result

    assert asyncio.run(run()) == "content"


def test_chat_json_schema_content_passes_schema_response_format() -> None:
    async def run() -> dict:
        client = _mock_async_client("{}")
        settings = SimpleNamespace(model="test-model")
        schema = {"type": "object"}

        await llm_client.chat_json_schema_content(
            client=client,
            settings=settings,
            messages=[{"role": "user", "content": "hello"}],
            schema=schema,
            schema_name="test_schema",
        )

        return client.chat.completions.create.call_args.kwargs["response_format"]

    assert asyncio.run(run()) == {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {"type": "object"},
            "strict": True,
        },
    }


def test_chat_completion_content_passes_model() -> None:
    async def run() -> str:
        client = _mock_async_client("async content")
        settings = SimpleNamespace(model="fallback-model")
        result = await llm_client.chat_completion_content(
            client=client,
            settings=settings,
            model="explicit-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
        )
        client.chat.completions.create.assert_awaited_once_with(
            model="explicit-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            response_format=None,
        )
        return result

    assert asyncio.run(run()) == "async content"


def test_stream_chat_completion_content_yields_delta_tokens() -> None:
    async def run() -> list[str]:
        client = MagicMock()
        settings = SimpleNamespace(model="fallback-model")
        client.chat.completions.create = AsyncMock(
            return_value=_async_chunks(["Hello", "", " world"])
        )

        tokens: list[str] = []
        async for token in llm_client.stream_chat_completion_content(
            client=client,
            settings=settings,
            model="explicit-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
        ):
            tokens.append(token)

        client.chat.completions.create.assert_awaited_once_with(
            model="explicit-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            response_format=None,
            stream=True,
        )
        return tokens

    assert asyncio.run(run()) == ["Hello", " world"]


def _mock_async_client(content: str) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _async_chunks(tokens: list[str]):
    class AsyncChunks:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not tokens:
                raise StopAsyncIteration
            delta = MagicMock()
            delta.content = tokens.pop(0)
            return MagicMock(choices=[MagicMock(delta=delta)])

    return AsyncChunks()
