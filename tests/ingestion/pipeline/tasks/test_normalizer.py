"""Tests for cv_ingestion.pipeline.tasks.normalizer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from cv_ingestion.exceptions import CVNormalizationError
from cv_ingestion.models.normalized_cv import NormalizedCV
from cv_ingestion.pipeline.tasks.normalizer import (
    _make_strict_json_schema,
    _strip_defaults_and_require_properties,
    extract_normalized_cv,
)

pytestmark = pytest.mark.anyio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _minimal_llm_payload() -> dict[str, Any]:
    """Return the smallest valid LLM payload accepted by NormalizedCVLLMOutput."""
    return {
        "contact": {
            "name": "Ana García",
            "email": "ana.garcia01@example.com",
            "phone": "+34612345678",
            "location": "Madrid",
        },
        "about": None,
        "experience": [],
        "education": [],
        "skills": [],
    }


def _full_llm_payload() -> dict[str, Any]:
    return {
        "contact": {
            "name": "Pedro López",
            "email": "pedro.lopez22@example.com",
            "phone": "+34699000111",
            "location": None,
        },
        "about": "Experienced backend developer.",
        "experience": [
            {
                "title": "Backend Developer",
                "company": "Acme",
                "start_date": "2021-01",
                "end_date": "2023-12",
                "is_current": False,
                "description": "Built REST APIs.",
            }
        ],
        "education": [
            {
                "degree": "Bachelor's Degree in Computer Engineering",
                "institution": "UPM",
                "start_date": "2017-09",
                "end_date": "2021-06",
                "description": None,
            }
        ],
        "skills": ["Python", "FastAPI", "PostgreSQL"],
    }


def _patch_llm(json_schema_result=None, json_object_result=None, raise_on_schema=False):
    """Context manager that patches both LLM call helpers."""
    schema_side_effect = Exception("not supported") if raise_on_schema else None
    schema_return = json_schema_result if not raise_on_schema else None

    return (
        patch(
            "cv_ingestion.pipeline.tasks.normalizer.chat_json_schema_content",
            side_effect=schema_side_effect,
            return_value=schema_return,
        ),
        patch(
            "cv_ingestion.pipeline.tasks.normalizer.chat_json_object_content",
            return_value=json_object_result,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# extract_normalized_cv — happy path (json_schema succeeds)
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractNormalizedCvHappyPath:
    async def test_returns_normalized_cv_instance(self):
        payload = _full_llm_payload()
        schema_patch, object_patch = _patch_llm(json_schema_result=json.dumps(payload))

        with schema_patch, object_patch:
            result = await extract_normalized_cv("test.pdf", "CV full text here")

        assert isinstance(result, NormalizedCV)

    async def test_contact_name_is_extracted(self):
        payload = _full_llm_payload()
        schema_patch, object_patch = _patch_llm(json_schema_result=json.dumps(payload))

        with schema_patch, object_patch:
            result = await extract_normalized_cv("test.pdf", "CV text")

        assert result.contact.name == "Pedro López"

    async def test_skills_are_extracted(self):
        payload = _full_llm_payload()
        schema_patch, object_patch = _patch_llm(json_schema_result=json.dumps(payload))

        with schema_patch, object_patch:
            result = await extract_normalized_cv("test.pdf", "CV text")

        assert result.skills == ["Python", "FastAPI", "PostgreSQL"]

    async def test_source_path_is_stored(self):
        payload = _minimal_llm_payload()
        schema_patch, object_patch = _patch_llm(json_schema_result=json.dumps(payload))

        with schema_patch, object_patch:
            result = await extract_normalized_cv("minio://cvs/pedro.pdf", "CV text")

        assert result.source_path == "minio://cvs/pedro.pdf"

    async def test_current_title_derived_from_experience(self):
        payload = _full_llm_payload()
        # Mark as current
        payload["experience"][0]["is_current"] = True
        payload["experience"][0]["end_date"] = None
        schema_patch, object_patch = _patch_llm(json_schema_result=json.dumps(payload))

        with schema_patch, object_patch:
            result = await extract_normalized_cv("test.pdf", "CV text")

        assert result.current_title == "Backend Developer"
        assert result.current_company == "Acme"

    async def test_years_of_experience_is_computed(self):
        payload = _full_llm_payload()
        schema_patch, object_patch = _patch_llm(json_schema_result=json.dumps(payload))

        with schema_patch, object_patch:
            result = await extract_normalized_cv("test.pdf", "CV text")

        assert result.years_of_experience is not None
        assert result.years_of_experience >= 1


# ─────────────────────────────────────────────────────────────────────────────
# extract_normalized_cv — fallback to json_object
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractNormalizedCvFallback:
    async def test_falls_back_when_json_schema_raises(self):
        payload = _full_llm_payload()
        schema_patch, object_patch = _patch_llm(
            raise_on_schema=True,
            json_object_result=json.dumps(payload),
        )

        with schema_patch as mock_schema, object_patch as mock_object:
            result = await extract_normalized_cv("test.pdf", "CV text")

        assert isinstance(result, NormalizedCV)
        mock_schema.assert_called_once()
        mock_object.assert_called_once()

    async def test_falls_back_when_json_schema_returns_none(self):
        payload = _full_llm_payload()
        schema_patch, object_patch = _patch_llm(
            json_schema_result=None,
            json_object_result=json.dumps(payload),
        )

        with schema_patch, object_patch as mock_object:
            result = await extract_normalized_cv("test.pdf", "CV text")

        assert isinstance(result, NormalizedCV)
        mock_object.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# extract_normalized_cv — error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractNormalizedCvErrors:
    async def test_raises_when_both_calls_return_empty(self):
        schema_patch, object_patch = _patch_llm(
            raise_on_schema=True,
            json_object_result=None,
        )

        with schema_patch, object_patch:
            with pytest.raises(CVNormalizationError, match="empty response"):
                await extract_normalized_cv("test.pdf", "CV text")

    async def test_raises_when_json_object_returns_invalid_json(self):
        schema_patch, object_patch = _patch_llm(
            raise_on_schema=True,
            json_object_result="this is not json {{{",
        )

        with schema_patch, object_patch:
            with pytest.raises(CVNormalizationError):
                await extract_normalized_cv("test.pdf", "CV text")

    async def test_raises_when_payload_fails_validation(self):
        invalid_payload = {"contact": None}  # contact is required, None is invalid
        schema_patch, object_patch = _patch_llm(
            raise_on_schema=True,
            json_object_result=json.dumps(invalid_payload),
        )

        with schema_patch, object_patch:
            with pytest.raises(CVNormalizationError):
                await extract_normalized_cv("test.pdf", "CV text")

    async def test_error_message_includes_source_path(self):
        schema_patch, object_patch = _patch_llm(
            raise_on_schema=True,
            json_object_result=None,
        )

        with schema_patch, object_patch:
            with pytest.raises(CVNormalizationError, match="my_cv.pdf"):
                await extract_normalized_cv("my_cv.pdf", "CV text")


# ─────────────────────────────────────────────────────────────────────────────
# _make_strict_json_schema / _strip_defaults_and_require_properties
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeStrictJsonSchema:
    async def test_removes_top_level_defaults(self):
        schema = {"type": "object", "default": "something", "properties": {}}
        result = _make_strict_json_schema(schema)
        assert "default" not in result

    async def test_adds_required_for_all_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "default": 0},
            },
        }
        result = _make_strict_json_schema(schema)
        assert set(result["required"]) == {"name", "age"}

    async def test_adds_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        result = _make_strict_json_schema(schema)
        assert result["additionalProperties"] is False

    async def test_removes_defaults_recursively(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "default": "x",
                    "properties": {
                        "value": {"type": "string", "default": "hello"},
                    },
                }
            },
        }
        result = _make_strict_json_schema(schema)
        assert "default" not in result["properties"]["nested"]
        assert "default" not in result["properties"]["nested"]["properties"]["value"]

    async def test_handles_list_nodes(self):
        schema = {
            "type": "array",
            "items": {"type": "object", "default": "x", "properties": {}},
        }
        _strip_defaults_and_require_properties(schema)
        assert "default" not in schema["items"]

    async def test_returns_dict(self):
        from cv_ingestion.models.normalized_cv import NormalizedCVLLMOutput

        schema = NormalizedCVLLMOutput.model_json_schema()
        result = _make_strict_json_schema(schema)
        assert isinstance(result, dict)
