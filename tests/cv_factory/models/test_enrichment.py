"""Tests for cv_factory.models.enrichment — structured LLM output validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cv_factory.models.enrichment import CVEnrichmentOutput, DescriptionOutput


def _description(index: int = 0, length: int = 80) -> dict:
    return {"index": index, "description": "A" * length}


class TestDescriptionOutput:
    def test_accepts_valid_payload(self) -> None:
        output = DescriptionOutput(**_description())

        assert output.index == 0
        assert len(output.description) == 80

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValidationError):
            DescriptionOutput(**_description(index=-1))

    def test_rejects_description_below_min_length(self) -> None:
        with pytest.raises(ValidationError):
            DescriptionOutput(**_description(length=79))

    def test_rejects_description_above_max_length(self) -> None:
        with pytest.raises(ValidationError):
            DescriptionOutput(**_description(length=901))

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            DescriptionOutput(**_description(), unexpected="field")


class TestCVEnrichmentOutput:
    def _payload(self, **overrides) -> dict:
        payload = {
            "about": "A" * 80,
            "experience": [_description(index=0)],
            "education": [_description(index=0)],
        }
        payload.update(overrides)
        return payload

    def test_accepts_valid_payload(self) -> None:
        output = CVEnrichmentOutput(**self._payload())

        assert len(output.experience) == 1
        assert len(output.education) == 1

    def test_rejects_duplicate_experience_indexes(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate experience indexes"):
            CVEnrichmentOutput(
                **self._payload(
                    experience=[_description(index=0), _description(index=0)]
                )
            )

    def test_rejects_duplicate_education_indexes(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate education indexes"):
            CVEnrichmentOutput(
                **self._payload(
                    education=[_description(index=0), _description(index=0)]
                )
            )

    def test_accepts_multiple_unique_indexes(self) -> None:
        output = CVEnrichmentOutput(
            **self._payload(
                experience=[_description(index=0), _description(index=1)],
            )
        )

        assert [item.index for item in output.experience] == [0, 1]

    def test_rejects_about_below_min_length(self) -> None:
        with pytest.raises(ValidationError):
            CVEnrichmentOutput(**self._payload(about="too short"))

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            CVEnrichmentOutput(**self._payload(), unexpected="field")
