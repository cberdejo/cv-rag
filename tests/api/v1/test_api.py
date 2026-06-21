"""Tests for the FastAPI v1 application."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from core.settings import get_settings
from cv_rag.api.application import APP_IMPORT_PATH, create_app, run_application
from cv_rag.api.v1.services import get_assistant
from rag.assistant import AssistantResponse, CVSource


class DummyAssistant:
    """Assistant test double with the same async ask contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def ask(self, message: str, *, thread_id: str) -> AssistantResponse:
        self.calls.append((message, thread_id))
        return AssistantResponse(
            answer=f"Response for: {message}",
            sources=[
                CVSource(
                    id="source-1",
                    candidate_name="Ana Garcia",
                    source_path="outputs/ana.pdf",
                    section="experience",
                    score=0.91,
                    snippet="Python and FastAPI experience.",
                )
            ],
        )


class FailingAssistant:
    async def ask(self, message: str, *, thread_id: str) -> AssistantResponse:
        raise RuntimeError("backend unavailable")


@pytest.fixture()
def client_and_assistant() -> tuple[TestClient, DummyAssistant]:
    assistant = DummyAssistant()
    app = create_app(mount_frontend=False)
    app.dependency_overrides[get_assistant] = lambda: assistant
    return TestClient(app), assistant


def test_health_returns_application_metadata(
    client_and_assistant: tuple[TestClient, DummyAssistant],
) -> None:
    client, _ = client_and_assistant

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "RAG CV Platform",
        "version": "0.1.0",
    }


def test_chat_delegates_to_assistant(
    client_and_assistant: tuple[TestClient, DummyAssistant],
) -> None:
    client, assistant = client_and_assistant

    response = client.post(
        "/api/v1/chat",
        json={"message": "  Who knows FastAPI?  ", "thread_id": "thread-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Response for: Who knows FastAPI?",
        "thread_id": "thread-1",
        "sources": [
            {
                "id": "source-1",
                "candidate_name": "Ana Garcia",
                "source_path": "outputs/ana.pdf",
                "section": "experience",
                "score": 0.91,
                "snippet": "Python and FastAPI experience.",
            }
        ],
    }
    assert assistant.calls == [("Who knows FastAPI?", "thread-1")]


def test_chat_rejects_blank_messages(
    client_and_assistant: tuple[TestClient, DummyAssistant],
) -> None:
    client, _ = client_and_assistant

    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_returns_503_when_assistant_fails() -> None:
    app = create_app(mount_frontend=False)
    app.dependency_overrides[get_assistant] = lambda: FailingAssistant()
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "The CV assistant is unavailable."}


def test_create_app_uses_cors_origins_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CORS_ORIGINS", "http://localhost:5173, https://app.example")
    get_settings.cache_clear()
    try:
        app = create_app(mount_frontend=False)
    finally:
        get_settings.cache_clear()

    cors_middleware = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )

    assert cors_middleware.kwargs["allow_origins"] == [
        "http://localhost:5173",
        "https://app.example",
    ]


def test_chainlit_pdf_worker_asset_is_served_from_root_assets() -> None:
    app = create_app(mount_frontend=True)
    client = TestClient(app)

    response = client.get("/assets/pdf.worker.min-qwK7q_zL.mjs")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/javascript")


def test_run_application_uses_host_and_port_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_HOST", "0.0.0.0")
    monkeypatch.setenv("RAG_PORT", "8123")
    get_settings.cache_clear()
    try:
        with patch("cv_rag.api.application.uvicorn.run") as uvicorn_run:
            run_application()
    finally:
        get_settings.cache_clear()

    uvicorn_run.assert_called_once_with(
        APP_IMPORT_PATH,
        host="0.0.0.0",
        port=8123,
        reload=False,
    )


def test_run_application_prefers_explicit_host_and_port_over_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_HOST", "0.0.0.0")
    monkeypatch.setenv("RAG_PORT", "8123")
    get_settings.cache_clear()
    try:
        with patch("cv_rag.api.application.uvicorn.run") as uvicorn_run:
            run_application(host="127.0.0.1", port=9000, reload=True)
    finally:
        get_settings.cache_clear()

    uvicorn_run.assert_called_once_with(
        APP_IMPORT_PATH,
        host="127.0.0.1",
        port=9000,
        reload=True,
    )
