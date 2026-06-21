"""Tests for cv_factory.models.cv — Period, Experience, Education, and CV."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from cv_factory.models.cv import CV, Education, Experience, Gender, Period


# ──────────────────────────────────────────────────────────────────────────────
# Period
# ──────────────────────────────────────────────────────────────────────────────


class TestPeriod:
    def test_valid_closed_period(self):
        p = Period(start_date=date(2020, 1, 1), end_date=date(2022, 12, 31))
        assert p.current is False

    def test_valid_open_period(self):
        p = Period(start_date=date(2023, 1, 1), end_date=None)
        assert p.current is True

    def test_same_day_period_is_valid(self):
        p = Period(start_date=date(2021, 6, 1), end_date=date(2021, 6, 1))
        assert p.end_date == p.start_date

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="end date"):
            Period(start_date=date(2022, 1, 1), end_date=date(2021, 12, 31))


# ──────────────────────────────────────────────────────────────────────────────
# Experience
# ──────────────────────────────────────────────────────────────────────────────


class TestExperience:
    def test_valid_experience(self):
        exp = Experience(
            title="Backend Developer",
            company="Acme",
            description="Built REST APIs for an e-commerce platform.",
            period=Period(start_date=date(2021, 1, 1), end_date=date(2023, 12, 31)),
        )
        assert exp.title == "Backend Developer"

    def test_description_too_short_raises(self):
        with pytest.raises(ValidationError):
            Experience(
                title="Dev",
                company="Co",
                description="Too short",
                period=Period(start_date=date(2021, 1, 1), end_date=date(2022, 1, 1)),
            )


# ──────────────────────────────────────────────────────────────────────────────
# CV — experience validator
# ──────────────────────────────────────────────────────────────────────────────


class TestCVExperienceValidation:
    def _base_education(self) -> list[Education]:
        return [
            Education(
                institution="MIT",
                title="Bachelor's Degree in Computer Science",
                period=Period(start_date=date(2010, 9, 1), end_date=date(2014, 6, 30)),
            )
        ]

    def _make_cv(self, experience: list[Experience]) -> CV:
        return CV(
            name="Test User One",
            email="test.user01@example.com",
            phone="+34 600 000 001",
            gender=Gender.MALE,
            about="A" * 60,
            experience=experience,
            education=self._base_education(),
            skills=["Python", "FastAPI"],
        )

    def test_no_overlap_passes(self):
        cv = self._make_cv(
            [
                Experience(
                    title="Dev",
                    company="A",
                    description="Description of responsibilities at company A for two years.",
                    period=Period(
                        start_date=date(2020, 1, 1), end_date=date(2021, 12, 31)
                    ),
                ),
                Experience(
                    title="Dev",
                    company="B",
                    description="Description of responsibilities at company B for two years.",
                    period=Period(
                        start_date=date(2018, 1, 1), end_date=date(2019, 12, 31)
                    ),
                ),
            ]
        )
        assert len(cv.experience) == 2

    def test_overlapping_jobs_raises(self):
        with pytest.raises(ValidationError, match="overlap"):
            self._make_cv(
                [
                    Experience(
                        title="Dev",
                        company="A",
                        description="Worked on backend services and API integrations.",
                        period=Period(
                            start_date=date(2020, 1, 1), end_date=date(2022, 6, 30)
                        ),
                    ),
                    Experience(
                        title="Dev",
                        company="B",
                        description="Led development of data pipelines and reporting tools.",
                        period=Period(
                            start_date=date(2021, 6, 1), end_date=date(2023, 12, 31)
                        ),
                    ),
                ]
            )

    def test_two_ongoing_jobs_raises(self):
        with pytest.raises(ValidationError, match="ongoing"):
            self._make_cv(
                [
                    Experience(
                        title="Dev",
                        company="A",
                        description="Building microservices and maintaining CI/CD pipelines.",
                        period=Period(start_date=date(2022, 1, 1), end_date=None),
                    ),
                    Experience(
                        title="Dev",
                        company="B",
                        description="Developing integrations between third-party services.",
                        period=Period(start_date=date(2020, 1, 1), end_date=None),
                    ),
                ]
            )

    def test_single_ongoing_job_passes(self):
        cv = self._make_cv(
            [
                Experience(
                    title="Senior Dev",
                    company="Current",
                    description="Leading the backend team and defining the architecture.",
                    period=Period(start_date=date(2023, 1, 1), end_date=None),
                ),
                Experience(
                    title="Junior Dev",
                    company="Previous",
                    description="Implemented features and fixed bugs across the product.",
                    period=Period(
                        start_date=date(2020, 1, 1), end_date=date(2022, 12, 31)
                    ),
                ),
            ]
        )
        assert cv.experience[0].period.current is True


# ──────────────────────────────────────────────────────────────────────────────
# CV — education validator
# ──────────────────────────────────────────────────────────────────────────────


class TestCVEducationValidation:
    def _base_experience(self) -> list[Experience]:
        return [
            Experience(
                title="Developer",
                company="Corp",
                description="Worked on full-stack web development across several projects.",
                period=Period(start_date=date(2022, 1, 1), end_date=date(2024, 12, 31)),
            )
        ]

    def _make_cv(self, education: list[Education]) -> CV:
        return CV(
            name="Test User Two",
            email="test.user02@example.com",
            phone="+34 600 000 002",
            gender=Gender.FEMALE,
            about="B" * 60,
            experience=self._base_experience(),
            education=education,
            skills=["SQL", "Python"],
        )

    def test_overlapping_education_raises(self):
        with pytest.raises(ValidationError, match="overlap"):
            self._make_cv(
                [
                    Education(
                        institution="Univ A",
                        title="Bachelor's Degree in Computer Engineering",
                        period=Period(
                            start_date=date(2015, 9, 1), end_date=date(2019, 6, 30)
                        ),
                    ),
                    Education(
                        institution="Univ B",
                        title="Higher Technician in Web Application Development",
                        period=Period(
                            start_date=date(2017, 9, 1), end_date=date(2019, 6, 30)
                        ),
                    ),
                ]
            )

    def test_master_without_prior_bachelor_raises(self):
        with pytest.raises(ValidationError, match="[Mm]aster"):
            self._make_cv(
                [
                    Education(
                        institution="Univ A",
                        title="Master's Degree in Data Science",
                        period=Period(
                            start_date=date(2020, 9, 1), end_date=date(2022, 6, 30)
                        ),
                    ),
                ]
            )

    def test_master_after_bachelor_passes(self):
        cv = self._make_cv(
            [
                Education(
                    institution="Univ B",
                    title="Master's Degree in Data Science",
                    period=Period(
                        start_date=date(2019, 9, 1), end_date=date(2021, 6, 30)
                    ),
                ),
                Education(
                    institution="Univ A",
                    title="Bachelor's Degree in Computer Engineering",
                    period=Period(
                        start_date=date(2015, 9, 1), end_date=date(2019, 6, 30)
                    ),
                ),
            ]
        )
        assert len(cv.education) == 2

    def test_two_ongoing_educations_raises(self):
        with pytest.raises(ValidationError, match="ongoing"):
            self._make_cv(
                [
                    Education(
                        institution="Univ A",
                        title="Bachelor's Degree in Computer Engineering",
                        period=Period(start_date=date(2020, 9, 1), end_date=None),
                    ),
                    Education(
                        institution="Bootcamp X",
                        title="Backend Development Bootcamp",
                        period=Period(start_date=date(2021, 1, 1), end_date=None),
                    ),
                ]
            )


# ──────────────────────────────────────────────────────────────────────────────
# CV — skills validator
# ──────────────────────────────────────────────────────────────────────────────


class TestCVSkillsValidation:
    def _base(self) -> dict:
        return dict(
            name="Test User Three",
            email="test.user03@example.com",
            phone="+34 600 000 003",
            gender=Gender.MALE,
            about="C" * 60,
            experience=[
                Experience(
                    title="Dev",
                    company="Corp",
                    description="Maintained backend services and RESTful APIs for clients.",
                    period=Period(
                        start_date=date(2020, 1, 1), end_date=date(2022, 12, 31)
                    ),
                )
            ],
            education=[
                Education(
                    institution="MIT",
                    title="Bachelor's Degree in Computer Science",
                    period=Period(
                        start_date=date(2016, 9, 1), end_date=date(2020, 6, 30)
                    ),
                )
            ],
        )

    def test_duplicate_skills_case_insensitive_raises(self):
        with pytest.raises(ValidationError, match="[Dd]uplicate"):
            CV(**self._base(), skills=["Python", "python"])

    def test_empty_skill_string_raises(self):
        with pytest.raises(ValidationError):
            CV(**self._base(), skills=["Python", "   "])

    def test_unique_skills_pass(self):
        cv = CV(**self._base(), skills=["Python", "FastAPI", "Docker"])
        assert len(cv.skills) == 3

    def test_skills_are_stripped(self):
        # Whitespace-only entries should be rejected, not silently dropped
        with pytest.raises(ValidationError):
            CV(**self._base(), skills=["Python", "  "])
