"""Generate a profile image for a CV candidate."""

from __future__ import annotations

from cv_factory.models.cv import Gender
from cv_factory.pipeline.tasks.generate_profile_image.service import (
    ProfileImageGeneratorService,
)


async def generate_profile_image(
    gender: Gender,
    image_service: ProfileImageGeneratorService | None = None,
) -> str:
    """Generate a profile image for the given gender as a base64 data URI."""
    service = image_service or ProfileImageGeneratorService()
    return await service.generate(gender)


__all__ = [
    "generate_profile_image",
]
