"""Shared pytest fixtures for the CV factory test suite."""

from __future__ import annotations

from datetime import date
from random import Random

import pytest
from faker import Faker

from cv_factory.models.catalog import (
    CareerProfile,
    EducationOption,
    JobTitleOption,
    Seniority,
    Skill,
    SkillCategory,
)
from cv_factory.models.cv import CV, Education, Experience, Gender, Period


# ──────────────────────────────────────────────────────────────────────────────
# Primitive helpers
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def rng() -> Random:
    return Random(42)


@pytest.fixture()
def fake() -> Faker:
    f = Faker("es_ES")
    f.seed_instance(42)
    return f


# ──────────────────────────────────────────────────────────────────────────────
# Catalog
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def minimal_career_profile() -> CareerProfile:
    """Smallest valid CareerProfile — useful for boundary tests."""
    return CareerProfile(
        key="test_profile",
        label="Test Profile",
        family="test",
        titles=[
            JobTitleOption(
                title="Junior Tester",
                seniority=Seniority.JUNIOR,
                min_years_experience=0,
            ),
            JobTitleOption(
                title="Senior Tester",
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
            Skill(name="Git", category=SkillCategory.TOOL, weight=4),
        ],
        possible_industries=["SaaS"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# CV building blocks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def single_experience() -> Experience:
    return Experience(
        title="Backend Developer",
        company="Acme Corp",
        description="Developed REST APIs and maintained PostgreSQL databases.",
        period=Period(start_date=date(2022, 1, 1), end_date=date(2024, 12, 31)),
    )


@pytest.fixture()
def single_education() -> Education:
    return Education(
        institution="Madrid Institute of Technology",
        title="Bachelor's Degree in Computer Engineering",
        period=Period(start_date=date(2018, 9, 1), end_date=date(2022, 6, 30)),
        gpa=7.5,
    )


@pytest.fixture()
def valid_cv(single_experience, single_education) -> CV:
    """Fully valid CV with one experience and one education entry."""
    return CV(
        name="Ana García López",
        email="ana.garcia42@example.com",
        phone="+34 612 345 678",
        gender=Gender.FEMALE,
        about="Experienced backend developer specialising in Python and REST APIs.",
        experience=[single_experience],
        education=[single_education],
        skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
    )


@pytest.fixture()
def ongoing_cv() -> CV:
    """CV where the most recent job has no end date (currently employed)."""
    return CV(
        name="Carlos Martín Ruiz",
        email="carlos.martin55@example.com",
        phone="+34 699 000 111",
        gender=Gender.MALE,
        about="Full-stack developer with five years of experience.",
        experience=[
            Experience(
                title="Senior Full Stack Engineer",
                company="StartupXYZ",
                description="Led frontend and backend development for a SaaS platform.",
                period=Period(start_date=date(2022, 1, 1), end_date=None),
            ),
            Experience(
                title="Full Stack Developer",
                company="OldCorp",
                description="Delivered multiple web projects for enterprise clients.",
                period=Period(start_date=date(2019, 3, 1), end_date=date(2021, 12, 31)),
            ),
        ],
        education=[
            Education(
                institution="Barcelona Institute of Technology",
                title="Bachelor's Degree in Software Engineering",
                period=Period(start_date=date(2015, 9, 1), end_date=date(2019, 6, 30)),
            )
        ],
        skills=["TypeScript", "React", "Node.js", "PostgreSQL"],
    )
