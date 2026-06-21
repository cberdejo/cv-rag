"""Tests for the cv-rag CLI."""

from __future__ import annotations

from unittest.mock import patch

from cv_rag.api.application import serve_command


def test_serve_command_delegates_defaults_to_application_settings() -> None:
    with patch("cv_rag.api.application.run_application") as run_application:
        serve_command()

    run_application.assert_called_once_with(
        host=None,
        port=None,
        reload=False,
    )


def test_serve_command_passes_explicit_overrides() -> None:
    with patch("cv_rag.api.application.run_application") as run_application:
        serve_command(host="0.0.0.0", port=9000, reload=True)

    run_application.assert_called_once_with(
        host="0.0.0.0",
        port=9000,
        reload=True,
    )
