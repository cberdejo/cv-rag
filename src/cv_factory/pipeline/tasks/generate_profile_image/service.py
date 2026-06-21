"""
Profile image generator for CV Factory.

Calls the Hugging Face Inference Providers router (FLUX.1-schnell via the
hf-inference provider) to produce a realistic professional headshot for a
generated candidate. Images live in memory only and are handed back as a
base64 data URI for direct embedding in the rendered CV's HTML — nothing is
written to local disk or MinIO.

"""

from __future__ import annotations

import base64
import os
import secrets

import httpx

from core.logging import get_logger
from core.settings import ImageGenerationSettings, get_image_generation_settings
from cv_factory.exceptions import (
    ProfileImageAPIError,
    ProfileImageConfigurationError,
)
from cv_factory.models.cv import Gender

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
)


def _resolve_hf_token(settings: ImageGenerationSettings) -> str:
    """Return the Hugging Face API token from the environment.

    Checks HF_API_TOKEN first, then CV_HF_API_TOKEN through settings.
    Raises ProfileImageConfigurationError if neither is set.
    """
    token = os.getenv("HF_API_TOKEN") or settings.hf_api_token
    if not token:
        raise ProfileImageConfigurationError(
            "No Hugging Face API token found. "
            "Set HF_API_TOKEN or CV_HF_API_TOKEN in your environment or .env file."
        )
    return token


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Professional corporate headshot of a {gender_desc} candidate, "
    "natural studio lighting, neutral background, business attire, "
    "sharp focus, photorealistic, high resolution, 35mm portrait lens"
)

_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, anime, 3d render, "
    "low quality, blurry, watermark, text, signature, frame, border, "
    "deformed, extra limbs, multiple people"
)

_GENDER_DESC: dict[Gender, str] = {
    Gender.MALE: "man",
    Gender.FEMALE: "woman",
    Gender.OTHER: "person",
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ProfileImageGeneratorService:
    """Generate a professional headshot through HF Inference, in memory only."""

    def __init__(
        self,
        *,
        settings: ImageGenerationSettings | None = None,
        hf_token: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Create a profile image generation service.

        Args:
            settings: Optional image generation settings override.
            hf_token: Explicit Hugging Face token; falls back to settings/env.
            timeout: HTTP client timeout in seconds for model generation.
        """
        self.settings = settings or get_image_generation_settings()
        self._token = hf_token or _resolve_hf_token(self.settings)
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, gender: Gender) -> str:
        """Generate a profile image for *gender* and return it as a data URI.

        Args:
            gender: Candidate gender, used to personalise the prompt.

        Returns:
            A ``data:image/...;base64,...`` URI ready to embed directly in
            an ``<img src="">`` tag. The image is never written to disk.
        """
        prompt = self._build_prompt(gender)
        logger.info(
            "Generating profile image | gender={} | model=FLUX.1-schnell",
            gender.value,
        )

        image_bytes = await self._call_hf_api(prompt)
        data_uri = _to_data_uri(image_bytes)

        logger.info("Profile image generated | bytes={}", len(image_bytes))
        return data_uri

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(self, gender: Gender) -> str:
        """Return a FLUX-optimised prompt for the given gender."""
        gender_desc = _GENDER_DESC.get(gender, "person")
        return _PROMPT_TEMPLATE.format(gender_desc=gender_desc)

    async def _call_hf_api(self, prompt: str) -> bytes:
        """POST to the HF Inference API and return raw PNG bytes.

        Args:
            prompt: Positive prompt sent as the model input.

        Returns:
            PNG image bytes returned by the API.

        Raises:
            ProfileImageAPIError: If the request fails, the model is loading,
                the response is not successful, or the content is not an image.
        """
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "X-Wait-For-Model": "true",  # wait instead of returning 503
        }
        payload: dict = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": _NEGATIVE_PROMPT,
                "width": self.settings.width,
                "height": self.settings.height,
                "guidance_scale": self.settings.guidance,
                "num_inference_steps": self.settings.steps,
                "num_images_per_prompt": 1,
                # A fresh random seed on every call. Diffusion models have no
                # "temperature" knob — without an explicit seed, the HF
                # Inference router can return the same cached image for an
                # identical prompt+parameters payload.
                "seed": secrets.randbelow(2**32),
            },
        }

        logger.debug("Calling HF Inference API | url={}", _HF_API_URL)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(_HF_API_URL, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ProfileImageAPIError("HF Inference API request failed.") from exc

        if response.status_code == 503:
            # Model is loading — the X-Wait-For-Model header should prevent
            # this, but handle it gracefully just in case.
            try:
                estimated_wait = response.json().get("estimated_time", "?")
            except ValueError:
                estimated_wait = "?"
            raise ProfileImageAPIError(
                f"FLUX.1-dev model is still loading (estimated wait: {estimated_wait}s). "
                "Retry in a few seconds, or set X-Wait-For-Model=true (already set)."
            )

        if response.status_code != 200:
            raise ProfileImageAPIError(
                f"HF Inference API returned {response.status_code}: {response.text}"
            )

        # Validate that the response is actually an image
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            raise ProfileImageAPIError(
                f"Unexpected content-type from HF API: {content_type!r}. "
                f"Body snippet: {response.text[:200]}"
            )

        return response.content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_MIME_TYPES = {"png": "image/png", "jpg": "image/jpeg"}


def _to_data_uri(image_bytes: bytes) -> str:
    """Encode raw image bytes as a base64 ``data:`` URI for inline embedding."""
    extension = _detect_image_extension(image_bytes)
    mime_type = _MIME_TYPES[extension]
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _detect_image_extension(data: bytes) -> str:
    """Return the file extension matching *data*'s magic bytes.

    HF providers serving FLUX.1-schnell/dev  returns
    either PNG or JPEG bytes for the same request, so both are accepted.

    Raises:
        ProfileImageAPIError: If *data* is neither a PNG nor a JPEG.
    """
    if data.startswith(_PNG_SIGNATURE):
        return "png"
    if data.startswith(_JPEG_SIGNATURE):
        return "jpg"
    raise ProfileImageAPIError(
        "The bytes returned by the API are not a valid PNG or JPEG image. "
        f"First 16 bytes: {data[:16]!r}"
    )
