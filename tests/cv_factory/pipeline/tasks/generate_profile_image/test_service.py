"""Tests for cv_factory.pipeline.tasks.generate_profile_image.service."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cv_factory.exceptions import ProfileImageAPIError, ProfileImageConfigurationError
from cv_factory.models.cv import Gender
from cv_factory.pipeline.tasks.generate_profile_image.service import (
    ProfileImageGeneratorService,
    _detect_image_extension,
    _resolve_hf_token,
    _to_data_uri,
)

pytestmark = pytest.mark.anyio

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"rest-of-image-bytes"
_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"rest-of-image-bytes"


def _settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        hf_api_token="hf_configured_token",
        width=512,
        height=512,
        guidance=3.5,
        steps=28,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FakeResponse:
    def __init__(
        self, *, status_code: int, content: bytes = b"", content_type: str = ""
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.headers = {"content-type": content_type}

    def json(self) -> dict:
        return {"estimated_time": 12}


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self._response = response

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def post(self, *args, **kwargs) -> _FakeResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestResolveHfToken:
    def test_prefers_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_API_TOKEN", "env-token")
        settings = SimpleNamespace(hf_api_token="settings-token")

        assert _resolve_hf_token(settings) == "env-token"

    def test_falls_back_to_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        settings = SimpleNamespace(hf_api_token="settings-token")

        assert _resolve_hf_token(settings) == "settings-token"

    def test_raises_when_neither_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        settings = SimpleNamespace(hf_api_token=None)

        with pytest.raises(ProfileImageConfigurationError):
            _resolve_hf_token(settings)


class TestDetectImageExtension:
    def test_accepts_valid_png_signature(self) -> None:
        assert _detect_image_extension(_PNG_BYTES) == "png"

    def test_accepts_valid_jpeg_signature(self) -> None:
        assert _detect_image_extension(_JPEG_BYTES) == "jpg"

    def test_rejects_unrecognized_bytes(self) -> None:
        with pytest.raises(ProfileImageAPIError):
            _detect_image_extension(b"not an image")


class TestToDataUri:
    def test_encodes_png_with_matching_mime_type(self) -> None:
        data_uri = _to_data_uri(_PNG_BYTES)

        assert data_uri.startswith("data:image/png;base64,")
        encoded = data_uri.removeprefix("data:image/png;base64,")
        assert base64.b64decode(encoded) == _PNG_BYTES

    def test_encodes_jpeg_with_matching_mime_type(self) -> None:
        data_uri = _to_data_uri(_JPEG_BYTES)

        assert data_uri.startswith("data:image/jpeg;base64,")
        encoded = data_uri.removeprefix("data:image/jpeg;base64,")
        assert base64.b64decode(encoded) == _JPEG_BYTES

    def test_rejects_invalid_image_bytes(self) -> None:
        with pytest.raises(ProfileImageAPIError):
            _to_data_uri(b"not an image")


class TestProfileImageGeneratorServiceInit:
    def test_uses_explicit_token_over_settings(self) -> None:
        service = ProfileImageGeneratorService(
            settings=_settings(hf_api_token=None),
            hf_token="explicit-token",
        )

        assert service._token == "explicit-token"

    def test_raises_when_no_token_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)

        with pytest.raises(ProfileImageConfigurationError):
            ProfileImageGeneratorService(settings=_settings(hf_api_token=None))


class TestBuildPrompt:
    @pytest.mark.parametrize(
        "gender, expected",
        [(Gender.MALE, "man"), (Gender.FEMALE, "woman"), (Gender.OTHER, "person")],
    )
    def test_includes_gender_description(
        self, gender: Gender, expected: str
    ) -> None:
        service = ProfileImageGeneratorService(settings=_settings())

        assert expected in service._build_prompt(gender)


class TestCallHfApi:
    async def test_returns_image_bytes_on_success(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())
        response = _FakeResponse(
            status_code=200, content=_PNG_BYTES, content_type="image/png"
        )

        with patch("httpx.AsyncClient", return_value=_FakeAsyncClient(response)):
            result = await service._call_hf_api("a prompt")

        assert result == _PNG_BYTES

    async def test_sends_a_random_seed_on_each_call(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())
        response = _FakeResponse(
            status_code=200, content=_PNG_BYTES, content_type="image/png"
        )
        seen_payloads = []

        class _RecordingClient(_FakeAsyncClient):
            async def post(self, *args, **kwargs):
                seen_payloads.append(kwargs["json"])
                return await super().post(*args, **kwargs)

        with patch("httpx.AsyncClient", return_value=_RecordingClient(response)):
            await service._call_hf_api("a prompt")
            await service._call_hf_api("a prompt")

        seeds = [payload["parameters"]["seed"] for payload in seen_payloads]
        assert seeds[0] != seeds[1]

    async def test_raises_on_http_error(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())

        with patch(
            "httpx.AsyncClient",
            return_value=_FakeAsyncClient(httpx.ConnectError("boom")),
        ):
            with pytest.raises(ProfileImageAPIError):
                await service._call_hf_api("a prompt")

    async def test_raises_when_model_is_loading(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())
        response = _FakeResponse(status_code=503)

        with patch("httpx.AsyncClient", return_value=_FakeAsyncClient(response)):
            with pytest.raises(ProfileImageAPIError, match="still loading"):
                await service._call_hf_api("a prompt")

    async def test_raises_on_non_200_status(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())
        response = _FakeResponse(status_code=500, content=b"server error")

        with patch("httpx.AsyncClient", return_value=_FakeAsyncClient(response)):
            with pytest.raises(ProfileImageAPIError, match="500"):
                await service._call_hf_api("a prompt")

    async def test_raises_on_unexpected_content_type(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())
        response = _FakeResponse(
            status_code=200, content=b"{}", content_type="application/json"
        )

        with patch("httpx.AsyncClient", return_value=_FakeAsyncClient(response)):
            with pytest.raises(ProfileImageAPIError, match="content-type"):
                await service._call_hf_api("a prompt")


class TestGenerate:
    async def test_generates_and_returns_data_uri(self) -> None:
        service = ProfileImageGeneratorService(settings=_settings())

        with patch.object(
            service, "_call_hf_api", new=AsyncMock(return_value=_PNG_BYTES)
        ):
            data_uri = await service.generate(Gender.FEMALE)

        assert data_uri == _to_data_uri(_PNG_BYTES)
