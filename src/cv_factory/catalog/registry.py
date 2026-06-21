"""Query helpers for the built-in career profile catalog."""

from __future__ import annotations

from collections.abc import Iterable

from cv_factory.models.catalog import CareerProfile
from cv_factory.catalog.profiles import CAREER_PROFILES


class UnknownCareerProfileError(ValueError):
    """Raised when a career profile key is not present in the catalog."""


def list_career_profiles() -> tuple[CareerProfile, ...]:
    """Return all built-in career profiles."""
    return CAREER_PROFILES


def get_career_profile(key: str) -> CareerProfile:
    """Return one career profile by key."""
    try:
        return _PROFILE_BY_KEY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILE_BY_KEY))
        raise UnknownCareerProfileError(
            f"Unknown career profile '{key}'. Available profiles: {available}."
        ) from exc


def list_profile_families() -> tuple[str, ...]:
    """Return all profile family names present in the catalog."""
    return tuple(sorted({profile.family for profile in CAREER_PROFILES}))


def find_profiles_by_family(family: str) -> tuple[CareerProfile, ...]:
    """Return all profiles for a family name."""
    normalized_family = family.strip().lower()
    return tuple(
        profile
        for profile in CAREER_PROFILES
        if profile.family.lower() == normalized_family
    )


def _index_by_key(profiles: Iterable[CareerProfile]) -> dict[str, CareerProfile]:
    """Build a lookup table and fail fast on duplicate profile keys."""
    indexed: dict[str, CareerProfile] = {}
    for profile in profiles:
        if profile.key in indexed:
            raise ValueError(f"Duplicate career profile key: {profile.key}")
        indexed[profile.key] = profile
    return indexed


_PROFILE_BY_KEY = _index_by_key(CAREER_PROFILES)


__all__ = [
    "UnknownCareerProfileError",
    "find_profiles_by_family",
    "get_career_profile",
    "list_career_profiles",
    "list_profile_families",
]
