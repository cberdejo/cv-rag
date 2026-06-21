"""Tests for cv_factory.pipeline.tasks.enrich_cv.main."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cv_factory.exceptions import LLMEnrichmentShapeError, LLMStructuredOutputError
from cv_factory.models.cv import CV
from cv_factory.models.enrichment import CVEnrichmentOutput
from cv_factory.pipeline.tasks.enrich_cv.main import (
    _merge_enrichment,
    _validate_shape,
    enrich_cv,
)

pytestmark = pytest.mark.anyio


def _enrichment_for(cv: CV, *, about: str = "B" * 80) -> CVEnrichmentOutput:
    return CVEnrichmentOutput(
        about=about,
        experience=[
            {"index": i, "description": "E" * 80} for i in range(len(cv.experience))
        ],
        education=[
            {"index": i, "description": "D" * 80} for i in range(len(cv.education))
        ],
    )


class TestValidateShape:
    def test_passes_when_indexes_match(self, valid_cv: CV) -> None:
        _validate_shape(cv=valid_cv, enrichment=_enrichment_for(valid_cv))

    def test_raises_on_missing_experience_index(self, valid_cv: CV) -> None:
        enrichment = CVEnrichmentOutput(
            about="A" * 80,
            experience=[],
            education=[{"index": 0, "description": "D" * 80}],
        )

        with pytest.raises(LLMEnrichmentShapeError, match="experience"):
            _validate_shape(cv=valid_cv, enrichment=enrichment)

    def test_raises_on_missing_education_index(self, valid_cv: CV) -> None:
        enrichment = CVEnrichmentOutput(
            about="A" * 80,
            experience=[{"index": 0, "description": "E" * 80}],
            education=[],
        )

        with pytest.raises(LLMEnrichmentShapeError, match="education"):
            _validate_shape(cv=valid_cv, enrichment=enrichment)

    def test_raises_on_out_of_range_index(self, valid_cv: CV) -> None:
        enrichment = CVEnrichmentOutput(
            about="A" * 80,
            experience=[{"index": 5, "description": "E" * 80}],
            education=[{"index": 0, "description": "D" * 80}],
        )

        with pytest.raises(LLMEnrichmentShapeError):
            _validate_shape(cv=valid_cv, enrichment=enrichment)


class TestMergeEnrichment:
    def test_merges_about_and_descriptions(self, valid_cv: CV) -> None:
        enrichment = _enrichment_for(valid_cv, about="Merged summary " + "x" * 70)

        merged = _merge_enrichment(cv=valid_cv, enrichment=enrichment)

        assert merged.about == enrichment.about
        assert merged.experience[0].description == "E" * 80
        assert merged.education[0].description == "D" * 80

    def test_does_not_mutate_input_cv(self, valid_cv: CV) -> None:
        original_about = valid_cv.about
        _merge_enrichment(cv=valid_cv, enrichment=_enrichment_for(valid_cv))

        assert valid_cv.about == original_about

    def test_preserves_immutable_fields(self, valid_cv: CV) -> None:
        merged = _merge_enrichment(cv=valid_cv, enrichment=_enrichment_for(valid_cv))

        assert merged.name == valid_cv.name
        assert merged.experience[0].title == valid_cv.experience[0].title
        assert merged.experience[0].company == valid_cv.experience[0].company


class TestEnrichCV:
    async def test_uses_json_schema_result_when_valid(self, valid_cv: CV) -> None:
        enrichment = _enrichment_for(valid_cv)
        with patch(
            "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_schema_content",
            new=AsyncMock(return_value=enrichment.model_dump_json()),
        ):
            result = await enrich_cv(
                valid_cv,
                client=AsyncMock(),
                settings=SimpleNamespace(quality_model="quality-llm"),
            )

        assert result.about == enrichment.about

    async def test_falls_back_to_json_object_on_schema_failure(
        self, valid_cv: CV
    ) -> None:
        enrichment = _enrichment_for(valid_cv)
        with (
            patch(
                "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_schema_content",
                new=AsyncMock(side_effect=RuntimeError("schema mode unsupported")),
            ),
            patch(
                "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_object_content",
                new=AsyncMock(return_value=enrichment.model_dump_json()),
            ) as json_object_mock,
        ):
            result = await enrich_cv(
                valid_cv,
                client=AsyncMock(),
                settings=SimpleNamespace(quality_model="quality-llm"),
            )

        json_object_mock.assert_awaited_once()
        assert result.about == enrichment.about

    async def test_falls_back_when_schema_output_fails_validation(
        self, valid_cv: CV
    ) -> None:
        enrichment = _enrichment_for(valid_cv)
        with (
            patch(
                "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_schema_content",
                new=AsyncMock(return_value="{not valid json}"),
            ),
            patch(
                "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_object_content",
                new=AsyncMock(return_value=enrichment.model_dump_json()),
            ),
        ):
            result = await enrich_cv(
                valid_cv,
                client=AsyncMock(),
                settings=SimpleNamespace(quality_model="quality-llm"),
            )

        assert result.about == enrichment.about

    async def test_raises_shape_error_when_indexes_mismatch(self, valid_cv: CV) -> None:
        mismatched = CVEnrichmentOutput(
            about="A" * 80,
            experience=[],
            education=[{"index": 0, "description": "D" * 80}],
        )
        with patch(
            "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_schema_content",
            new=AsyncMock(return_value=mismatched.model_dump_json()),
        ):
            with pytest.raises(LLMEnrichmentShapeError):
                await enrich_cv(
                    valid_cv,
                    client=AsyncMock(),
                    settings=SimpleNamespace(quality_model="quality-llm"),
                )

    async def test_raises_structured_output_error_when_both_paths_empty(
        self, valid_cv: CV
    ) -> None:
        with (
            patch(
                "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_schema_content",
                new=AsyncMock(return_value=""),
            ),
            patch(
                "cv_factory.pipeline.tasks.enrich_cv.main.chat_json_object_content",
                new=AsyncMock(return_value=""),
            ),
        ):
            with pytest.raises(LLMStructuredOutputError):
                await enrich_cv(
                    valid_cv,
                    client=AsyncMock(),
                    settings=SimpleNamespace(quality_model="quality-llm"),
                )
