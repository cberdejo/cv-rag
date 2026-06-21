"""Enrich validated base CV payloads with LLM-generated narrative fields."""

from __future__ import annotations

import json

from openai import AsyncOpenAI
from pydantic import ValidationError

from core.logging import get_logger
from core.settings import LLMSettings, get_llm_settings
from cv_factory.exceptions import LLMEnrichmentShapeError, LLMStructuredOutputError
from cv_factory.models.cv import CV, Education, Experience
from cv_factory.models.enrichment import CVEnrichmentOutput
from cv_factory.pipeline.tasks.enrich_cv.prompts import build_enrichment_messages
from infrastructure.llm_client import (
    chat_json_object_content,
    chat_json_schema_content,
    get_llm_client,
)

logger = get_logger(__name__)


async def enrich_cv(
    plain_cv: CV,
    *,
    client: AsyncOpenAI | None = None,
    settings: LLMSettings | None = None,
) -> CV:
    """Return a copy of ``plain_cv`` with enriched narrative fields only."""
    resolved_settings = settings or get_llm_settings()
    resolved_client = client or get_llm_client()

    enrichment = await _generate_enrichment(
        plain_cv,
        client=resolved_client,
        settings=resolved_settings,
    )
    _validate_shape(cv=plain_cv, enrichment=enrichment)
    return _merge_enrichment(cv=plain_cv, enrichment=enrichment)


async def _generate_enrichment(
    cv: CV,
    *,
    client: AsyncOpenAI,
    settings: LLMSettings,
) -> CVEnrichmentOutput:
    """Generate CV enrichment, falling back when JSON Schema mode is unsupported."""
    messages = build_enrichment_messages(cv)

    try:
        return await _call_with_json_schema(
            messages,
            client=client,
            settings=settings,
        )
    except LLMStructuredOutputError as exc:
        logger.warning(
            "JSON schema structured output failed. Falling back to JSON object mode. error={}",
            repr(exc),
        )
    except Exception as exc:
        logger.warning(
            "JSON schema request failed. Falling back to JSON object mode. error={}",
            repr(exc),
        )

    return await _call_with_json_object(
        messages,
        client=client,
        settings=settings,
    )


async def _call_with_json_schema(
    messages: list[dict[str, str]],
    *,
    client: AsyncOpenAI,
    settings: LLMSettings,
) -> CVEnrichmentOutput:
    """Request strict JSON Schema output from the quality model and validate it."""
    content = await chat_json_schema_content(
        client=client,
        settings=settings,
        model=settings.quality_model,
        messages=messages,
        temperature=0.4,
        schema=CVEnrichmentOutput.model_json_schema(),
        schema_name="cv_enrichment_output",
    )

    if not content:
        raise LLMStructuredOutputError("The LLM returned an empty response.")

    try:
        return CVEnrichmentOutput.model_validate_json(content)
    except ValidationError as exc:
        raise LLMStructuredOutputError(
            "The LLM returned JSON that does not match CVEnrichmentOutput."
        ) from exc


async def _call_with_json_object(
    messages: list[dict[str, str]],
    *,
    client: AsyncOpenAI,
    settings: LLMSettings,
) -> CVEnrichmentOutput:
    """Request JSON object output and validate it as a fallback enrichment path."""
    fallback_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                "Return only valid JSON. Do not include markdown fences, comments, "
                "or explanatory text."
            ),
        },
    ]

    content = await chat_json_object_content(
        client=client,
        settings=settings,
        model=settings.quality_model,
        messages=fallback_messages,
        temperature=0.3,
    )

    if not content:
        raise LLMStructuredOutputError("The LLM returned an empty response.")

    try:
        payload = json.loads(content)
        return CVEnrichmentOutput.model_validate(payload)
    except json.JSONDecodeError as exc:
        raise LLMStructuredOutputError("The LLM did not return valid JSON.") from exc
    except ValidationError as exc:
        raise LLMStructuredOutputError(
            "The LLM returned JSON that does not match CVEnrichmentOutput."
        ) from exc


def _validate_shape(*, cv: CV, enrichment: CVEnrichmentOutput) -> None:
    """Ensure enrichment entries exactly match the input experience and education indexes."""
    expected_experience_indexes = set(range(len(cv.experience)))
    expected_education_indexes = set(range(len(cv.education)))

    received_experience_indexes = {item.index for item in enrichment.experience}
    received_education_indexes = {item.index for item in enrichment.education}

    if received_experience_indexes != expected_experience_indexes:
        raise LLMEnrichmentShapeError(
            "LLM output does not match input experience indexes. "
            f"expected={sorted(expected_experience_indexes)} "
            f"received={sorted(received_experience_indexes)}"
        )

    if received_education_indexes != expected_education_indexes:
        raise LLMEnrichmentShapeError(
            "LLM output does not match input education indexes. "
            f"expected={sorted(expected_education_indexes)} "
            f"received={sorted(received_education_indexes)}"
        )


def _merge_enrichment(*, cv: CV, enrichment: CVEnrichmentOutput) -> CV:
    """Return a CV copy with generated narrative fields merged into fixed positions."""
    experience_descriptions = {
        item.index: item.description for item in enrichment.experience
    }
    education_descriptions = {
        item.index: item.description for item in enrichment.education
    }

    enriched_experience: list[Experience] = [
        item.model_copy(update={"description": experience_descriptions[index]})
        for index, item in enumerate(cv.experience)
    ]

    enriched_education: list[Education] = [
        item.model_copy(update={"description": education_descriptions[index]})
        for index, item in enumerate(cv.education)
    ]

    return cv.model_copy(
        update={
            "about": enrichment.about,
            "experience": enriched_experience,
            "education": enriched_education,
        }
    )


__all__ = ["enrich_cv"]
