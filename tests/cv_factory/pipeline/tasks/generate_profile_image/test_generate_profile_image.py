"""Tests for cv_factory.pipeline.tasks.generate_profile_image.main."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cv_factory.models.cv import Gender
from cv_factory.pipeline.tasks.generate_profile_image.main import (
    generate_profile_image,
)

pytestmark = pytest.mark.anyio


async def test_delegates_to_provided_service() -> None:
    expected_data_uri = "data:image/png;base64,abc123"
    service = AsyncMock()
    service.generate.return_value = expected_data_uri

    result = await generate_profile_image(Gender.FEMALE, service)

    assert result == expected_data_uri
    service.generate.assert_awaited_once_with(Gender.FEMALE)
