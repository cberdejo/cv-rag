"""Synthetic education generators."""

from __future__ import annotations

from datetime import date
from random import Random

from faker import Faker

from cv_factory.models.catalog import EducationOption
from cv_factory.models.cv import Education, Period


def _generate_gpa(rng: Random) -> float | None:
    """Randomly return a GPA on a 0-10 scale or omit it."""
    if not rng.choice([True, False]):
        return None

    return rng.randint(51, 100) / 10


def _education_duration_years(level: str, rng: Random) -> int:
    """Return a realistic duration in years for an education level."""
    durations = {
        "certificate": (1, 1),
        "vocational": (1, 2),
        "bachelor": (3, 4),
        "master": (1, 2),
        "phd": (3, 4),
        "bootcamp": (1, 1),
    }
    min_years, max_years = durations.get(level, (1, 4))
    return rng.randint(min_years, max_years)


def _normalize_title(title: str) -> str:
    """Return a stable key for comparing education titles."""
    return " ".join(title.lower().split())


def generate_education(
    *,
    fake: Faker,
    rng: Random,
    education_options: list[EducationOption],
    latest_end_year: int,
) -> list[Education]:
    """Generate non-overlapping education entries with valid degree progression.

    Args:
        fake: Faker instance used to synthesize institution names.
        rng: Random source for deterministic testable generation.
        education_options: Catalog education paths available for the profile.
        latest_end_year: Latest possible end year for the most recent entry.

    Returns:
        Education entries ordered from most recent to oldest.
    """
    education_count = rng.randint(1, 4)
    selected_options: list[EducationOption] = []
    education: list[Education] = []
    next_end_year = latest_end_year
    has_bachelor = False
    master_count = 0
    selected_titles: set[str] = set()

    for _ in range(education_count):
        eligible_options = [
            option
            for option in education_options
            if _normalize_title(option.title) not in selected_titles
            and (option.level != "master" or has_bachelor)
            and (option.level != "master" or master_count < 2)
            and (option.level != "bachelor" or not has_bachelor)
        ]
        if not eligible_options:
            if selected_options:
                break
            raise ValueError(
                "Education options must include a non-master path before masters."
            )
        education_option = rng.choice(eligible_options)
        selected_options.append(education_option)
        selected_titles.add(_normalize_title(education_option.title))
        if education_option.level == "bachelor":
            has_bachelor = True
        if education_option.level == "master":
            master_count += 1

    for education_option in reversed(selected_options):
        duration_years = _education_duration_years(education_option.level, rng)
        start_year = next_end_year - duration_years

        education.append(
            Education(
                institution=f"{fake.city()} Institute of Technology",
                title=education_option.title,
                period=Period(
                    start_date=date(start_year, 9, 1),
                    end_date=date(next_end_year, 6, 30),
                ),
                gpa=_generate_gpa(rng),
            )
        )

        next_end_year = start_year - 1

    return education


__all__ = ["generate_education"]
