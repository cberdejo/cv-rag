"""Tests for cv_factory.pipeline.tasks.enrich_cv.prompts."""

from __future__ import annotations

from cv_factory.models.cv import CV
from cv_factory.pipeline.tasks.enrich_cv.prompts import (
    build_enrichment_messages,
    serialize_cv_for_prompt,
)


class TestSerializeCVForPrompt:
    def test_includes_zero_based_indexes(self, valid_cv: CV) -> None:
        payload = serialize_cv_for_prompt(valid_cv)

        assert [item["index"] for item in payload["experience"]] == [0]
        assert [item["index"] for item in payload["education"]] == [0]

    def test_does_not_leak_skills_outside_top_level(self, valid_cv: CV) -> None:
        payload = serialize_cv_for_prompt(valid_cv)

        assert payload["skills"] == valid_cv.skills
        assert "skills" not in payload["experience"][0]

    def test_open_ended_period_serializes_end_date_as_none(
        self, ongoing_cv: CV
    ) -> None:
        payload = serialize_cv_for_prompt(ongoing_cv)

        assert payload["experience"][0]["period"]["end_date"] is None


class TestBuildEnrichmentMessages:
    def test_returns_system_and_user_messages(self, valid_cv: CV) -> None:
        messages = build_enrichment_messages(valid_cv)

        assert [message["role"] for message in messages] == ["system", "user"]

    def test_user_message_embeds_serialized_cv(self, valid_cv: CV) -> None:
        messages = build_enrichment_messages(valid_cv)

        assert valid_cv.name in messages[1]["content"]
