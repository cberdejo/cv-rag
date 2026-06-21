"""Synthetic professional experience generators."""

from __future__ import annotations

from datetime import date
from random import Random

from faker import Faker

from cv_factory.models.catalog import JobTitleOption, Seniority
from cv_factory.models.cv import Experience, Period

# Ordered from most to least senior
_SENIORITY_ORDER: list[Seniority] = [
    Seniority.LEAD,
    Seniority.SENIOR,
    Seniority.MID,
    Seniority.JUNIOR,
]

_SENIORITY_RANK: dict[Seniority, int] = {s: i for i, s in enumerate(_SENIORITY_ORDER)}


def _titles_by_seniority(
    titles: list[JobTitleOption],
) -> dict[Seniority, list[JobTitleOption]]:
    """Group titles by seniority level."""
    grouped: dict[Seniority, list[JobTitleOption]] = {}
    for title in titles:
        grouped.setdefault(title.seniority, []).append(title)
    return grouped


def _pick_seniority_for_position(
    *,
    rng: Random,
    position_index: int,
    experience_count: int,
    available_seniorities: list[Seniority],
) -> Seniority:
    """
    Pick a seniority level for a given position in the career timeline.

    position_index=0 is the most recent role; higher indices are older roles.
    The most recent role gets a seniority drawn from the upper end of what's
    available, and each earlier role steps down one level (with a small chance
    of staying flat, to model real careers).
    """
    sorted_available = sorted(available_seniorities, key=lambda s: _SENIORITY_RANK[s])
    # For the most recent role, bias toward the top half of available seniorities
    top_half = sorted_available[: max(1, len(sorted_available) // 2 + 1)]
    latest_seniority = rng.choice(top_half)

    latest_rank = _SENIORITY_RANK[latest_seniority]
    # Each step back in time can go one level down, or stay flat (20% chance)
    target_rank = min(
        latest_rank + position_index
        if rng.random() > 0.2
        else latest_rank + position_index - 1,
        len(_SENIORITY_ORDER) - 1,
    )
    target_rank = max(target_rank, 0)

    # Find the closest available seniority at or below the target rank
    candidates = [s for s in sorted_available if _SENIORITY_RANK[s] >= target_rank]
    if not candidates:
        candidates = sorted_available  # fallback: use anything available
    return min(candidates, key=lambda s: abs(_SENIORITY_RANK[s] - target_rank))


def generate_experience(
    *,
    fake: Faker,
    rng: Random,
    profile_label: str,
    titles: list[JobTitleOption],
    current_year: int,
) -> list[Experience]:
    """Generate non-overlapping experience with coherent seniority progression.

    Args:
        fake: Faker instance used to synthesize company names.
        rng: Random source for deterministic testable generation.
        profile_label: Human-readable profile label used in draft descriptions.
        titles: Catalog job titles available for the profile.
        current_year: Reference year for the most recent role.

    Returns:
        Experience entries ordered from most recent to oldest.
    """
    experience_count = rng.randint(1, 4)
    has_current_experience = rng.choice([True, False])
    experiences: list[Experience] = []
    previous_start: date | None = None

    titles_by_seniority = _titles_by_seniority(titles)
    available_seniorities = list(titles_by_seniority.keys())

    for index in range(experience_count):
        seniority = _pick_seniority_for_position(
            rng=rng,
            position_index=index,
            experience_count=experience_count,
            available_seniorities=available_seniorities,
        )
        title = rng.choice(titles_by_seniority[seniority])
        duration_years = rng.randint(1, 3)

        if index == 0:
            if has_current_experience:
                end_date = None
                effective_end_year = current_year
            else:
                effective_end_year = current_year - rng.randint(0, 1)
                end_date = date(effective_end_year, 12, 31)
        else:
            assert previous_start is not None
            effective_end_year = previous_start.year - 1
            end_date = date(effective_end_year, 12, 31)

        start_date = date(effective_end_year - duration_years + 1, 1, 1)
        previous_start = start_date
        experiences.append(
            Experience(
                title=title.title,
                company=fake.company(),
                description=(
                    f"Draft {title.title} role focused on {profile_label} "
                    "responsibilities, delivery, collaboration, and measurable impact."
                ),
                period=Period(start_date=start_date, end_date=end_date),
            )
        )

    return experiences


__all__ = ["generate_experience"]
