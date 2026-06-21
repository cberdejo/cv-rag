"""Tests for cv_factory.models.catalog — CareerProfile and related models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cv_factory.models.catalog import (
    CareerProfile,
    EducationOption,
    JobTitleOption,
    Seniority,
    Skill,
    SkillCategory,
)


def _make_profile(**overrides) -> CareerProfile:
    """Return a valid CareerProfile, merging *overrides*."""
    defaults = dict(
        key="test_role",
        label="Test Role",
        family="software",
        titles=[
            JobTitleOption(
                title="Junior Test Role",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Senior Test Role",
                seniority=Seniority.SENIOR,
                min_years_experience=5,
            ),
        ],
        education=[
            EducationOption(
                title="Bachelor's Degree in Testing",
                institution_type="university",
                level="bachelor",
                relevance=5,
            )
        ],
        skills=[
            Skill(name="Python", category=SkillCategory.CORE, weight=5),
            Skill(name="pytest", category=SkillCategory.CORE, weight=4),
            Skill(name="Mocking", category=SkillCategory.CORE, weight=3),
        ],
    )
    defaults.update(overrides)
    return CareerProfile(**defaults)


class TestCareerProfileValidation:
    def test_valid_profile_is_accepted(self):
        profile = _make_profile()
        assert profile.key == "test_role"

    def test_key_must_be_snake_case(self):
        with pytest.raises(ValidationError):
            _make_profile(key="Test Role")  # spaces not allowed

    def test_key_with_uppercase_raises(self):
        with pytest.raises(ValidationError):
            _make_profile(key="TestRole")

    def test_empty_titles_raises(self):
        with pytest.raises(ValidationError, match="[Tt]itle"):
            _make_profile(titles=[])

    def test_empty_education_raises(self):
        with pytest.raises(ValidationError, match="[Ee]ducation"):
            _make_profile(education=[])

    def test_fewer_than_three_core_skills_raises(self):
        with pytest.raises(ValidationError, match="[Cc]ore"):
            _make_profile(
                skills=[
                    Skill(name="Python", category=SkillCategory.CORE, weight=5),
                    Skill(name="Git", category=SkillCategory.TOOL, weight=4),
                ]
            )

    def test_duplicate_skill_names_raises(self):
        with pytest.raises(ValidationError, match="[Dd]uplicate"):
            _make_profile(
                skills=[
                    Skill(name="Python", category=SkillCategory.CORE, weight=5),
                    Skill(
                        name="python", category=SkillCategory.CORE, weight=4
                    ),  # same, lowercase
                    Skill(name="pytest", category=SkillCategory.CORE, weight=3),
                ]
            )

    def test_exactly_three_core_skills_passes(self):
        profile = _make_profile()
        core = [s for s in profile.skills if s.category == SkillCategory.CORE]
        assert len(core) >= 3


class TestSkillModel:
    def test_weight_boundary_one(self):
        s = Skill(name="Python", category=SkillCategory.CORE, weight=1)
        assert s.weight == 1

    def test_weight_boundary_five(self):
        s = Skill(name="Python", category=SkillCategory.CORE, weight=5)
        assert s.weight == 5

    def test_weight_zero_raises(self):
        with pytest.raises(ValidationError):
            Skill(name="Python", category=SkillCategory.CORE, weight=0)

    def test_weight_six_raises(self):
        with pytest.raises(ValidationError):
            Skill(name="Python", category=SkillCategory.CORE, weight=6)

    def test_name_too_short_raises(self):
        with pytest.raises(ValidationError):
            Skill(name="X", category=SkillCategory.CORE, weight=3)
