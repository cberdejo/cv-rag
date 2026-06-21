"""LLM-backed CV normalisation."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from cv_ingestion.exceptions import CVLLMServiceError, CVStructuredOutputError
from cv_ingestion.models.normalized_cv import NormalizedCV, NormalizedCVLLMOutput
from infrastructure.llm_client import (
    chat_json_object_content,
    chat_json_schema_content,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a CV/resume parser. Extract all structured information from the CV text
provided by the user and return it as a single JSON object.

Rules:
- Extract the candidate's full name, email address, phone number, and location
  when present.
- Extract `about`: the candidate's professional summary — a short paragraph
  describing who they are professionally, their profile, or career
  objective. It is often untitled and appears right after the contact
  details and before the first work experience entry, but may also appear
  under a heading such as "About", "About Me", "Summary", "Professional
  Summary", "Profile", "Objective", or similar in other languages. Do not confuse it
  with a section title alone (e.g. a job title under the candidate's name) —
  it must be descriptive prose about the candidate. If no such summary or
  introductory paragraph exists anywhere in the CV, leave `about` as null.
- For each work experience entry extract: job title, company name, start date,
  end date (null if ongoing), whether it is the current role, and a short
  description of responsibilities if present in the CV.
- For each education entry extract: degree/qualification title, institution
  name, start date, end date, and any description present.
- Extract all skills mentioned anywhere in the CV as a flat list of strings.
- For each certification/credential entry extract: name, issuing
  organization, issue date, and any description present.
- For each language entry extract: language name and proficiency level as
  stated in the CV (e.g. "Native", "B2", "Fluent").
- For each project entry extract: name, description, technologies/tools used,
  and URL if present.
- For each link/profile entry (LinkedIn, GitHub, portfolio, personal website,
  etc.) extract: a short label and the URL.
- If the CV contains any other clearly titled section that does not match one
  of the categories above (e.g. awards, publications, volunteering,
  references, hobbies), put it in `additional_sections` with its original
  title and the section's raw text content. Never discard information and
  never force content into a section it does not belong to.
- Dates must use the format "YYYY-MM" (e.g. "2022-01"). Use null when a date
  is not present or cannot be determined.
- Set is_current to true only for roles with no end date that appear to be
  the candidate's current position.
- Preserve the original text of descriptions — do not paraphrase.
- Do not invent information that is not in the CV.
- Many CVs omit section headings entirely. When a heading is missing, infer
  the section from its content and position: a short first-person or
  third-person narrative paragraph near the top is the `about` summary; a
  list of short noun phrases is `skills`; an entry with a title, employer,
  and date range is `experience`; an entry with a degree/qualification and
  institution is `education`. Use content and position together, not just
  position alone.
- Return ONLY the JSON object. No markdown fences, no explanations.
"""

_USER_TEMPLATE = """\
Parse the following CV:

--- CV START ---
{full_text}
--- CV END ---
"""


async def extract_normalized_cv(source_path: str, full_text: str) -> NormalizedCV:
    """
    Extract and normalise a CV in a single LLM call.

    Args:
        source_path: Original file path or object URI, stored as metadata.
        full_text: Plain text content of the CV document.

    Returns:
        A fully validated NormalizedCV ready for chunking.

    Raises:
        CVStructuredOutputError: If the LLM returns unusable structured output.
        CVLLMServiceError: If the LLM service call fails.
    """
    messages = _build_messages(full_text)
    raw_data = await _call_llm(messages, source_path)

    try:
        return NormalizedCV.from_llm_payload(raw_data, source_path=source_path)
    except ValidationError as exc:
        raise CVStructuredOutputError(
            f"Could not build NormalizedCV for '{source_path}': {exc}"
        ) from exc


def _build_messages(full_text: str) -> list[dict[str, str]]:
    """Build chat messages that instruct the LLM to extract normalized CV data."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(full_text=full_text)},
    ]


async def _call_llm(
    messages: list[dict[str, str]],
    source_path: str,
) -> dict[str, Any]:
    """Call the LLM and parse the normalization payload.

    Args:
        messages: Chat messages containing the extraction instructions and CV text.
        source_path: Durable source identity used in error messages and logs.

    Returns:
        Parsed JSON object returned by the model.

    Raises:
        CVLLMServiceError: If both structured-output attempts fail or return
            empty content.
        CVStructuredOutputError: If the fallback response is not valid JSON.
    """
    try:
        content = await chat_json_schema_content(
            messages=messages,
            schema=_llm_output_schema(),
            schema_name="cv_extraction_output",
            temperature=0,
        )
        if content:
            return json.loads(content)
    except Exception as exc:
        logger.warning(
            "JSON schema extraction failed for '%s', falling back. error=%r",
            source_path,
            exc,
        )

    fallback_messages = [
        *messages,
        {
            "role": "user",
            "content": (
                "Return only valid JSON matching the schema. "
                "No markdown fences, no explanatory text."
            ),
        },
    ]
    try:
        content = await chat_json_object_content(
            messages=fallback_messages,
            temperature=0,
        )
    except Exception as exc:
        raise CVLLMServiceError(
            f"LLM extraction request failed for '{source_path}'."
        ) from exc

    if not content:
        raise CVLLMServiceError(f"LLM returned an empty response for '{source_path}'.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise CVStructuredOutputError(
            f"LLM returned non-JSON for '{source_path}':\n{content[:300]}"
        ) from exc


def _llm_output_schema() -> dict[str, Any]:
    """Return the strict JSON schema expected from the normalization LLM."""
    return _make_strict_json_schema(NormalizedCVLLMOutput.model_json_schema())


def _make_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt Pydantic's generated schema for strict JSON-schema LLM mode.

    The structure still comes from Pydantic; this only removes unsupported
    defaults and marks object properties as required so nullable fields are
    emitted as explicit nulls.
    """
    _strip_defaults_and_require_properties(schema)
    return schema


def _strip_defaults_and_require_properties(node: Any) -> None:
    """Recursively remove defaults and require all object properties in a schema."""
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties)
            node["additionalProperties"] = False
        for value in node.values():
            _strip_defaults_and_require_properties(value)
    elif isinstance(node, list):
        for item in node:
            _strip_defaults_and_require_properties(item)


__all__ = ["extract_normalized_cv"]
