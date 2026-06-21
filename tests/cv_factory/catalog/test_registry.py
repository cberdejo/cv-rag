"""Tests for cv_factory.catalog.registry — query helpers."""

from __future__ import annotations

import pytest

from cv_factory.catalog.registry import (
    UnknownCareerProfileError,
    find_profiles_by_family,
    get_career_profile,
    list_career_profiles,
    list_profile_families,
)
from cv_factory.catalog.profiles import CAREER_PROFILES


class TestListCareerProfiles:
    def test_returns_all_built_in_profiles(self):
        profiles = list_career_profiles()
        assert profiles == CAREER_PROFILES

    def test_returns_non_empty_tuple(self):
        assert len(list_career_profiles()) > 0

    def test_all_keys_are_unique(self):
        keys = [p.key for p in list_career_profiles()]
        assert len(keys) == len(set(keys))


class TestGetCareerProfile:
    def test_known_key_returns_profile(self):
        profile = get_career_profile("backend_developer")
        assert profile.key == "backend_developer"
        assert profile.label == "Backend Developer"

    def test_unknown_key_raises(self):
        with pytest.raises(UnknownCareerProfileError, match="unknown_role"):
            get_career_profile("unknown_role")

    def test_error_message_lists_available_profiles(self):
        with pytest.raises(UnknownCareerProfileError, match="backend_developer"):
            get_career_profile("nonexistent")

    @pytest.mark.parametrize("key", [p.key for p in CAREER_PROFILES])
    def test_every_built_in_key_is_retrievable(self, key: str):
        profile = get_career_profile(key)
        assert profile.key == key


class TestListProfileFamilies:
    def test_returns_sorted_tuple(self):
        families = list_profile_families()
        assert list(families) == sorted(families)

    def test_known_families_present(self):
        families = list_profile_families()
        assert "software" in families
        assert "data" in families

    def test_no_duplicate_families(self):
        families = list_profile_families()
        assert len(families) == len(set(families))


class TestFindProfilesByFamily:
    def test_software_family_returns_profiles(self):
        profiles = find_profiles_by_family("software")
        assert len(profiles) > 0
        assert all(p.family == "software" for p in profiles)

    def test_case_insensitive_match(self):
        lower = find_profiles_by_family("software")
        upper = find_profiles_by_family("SOFTWARE")
        assert lower == upper

    def test_unknown_family_returns_empty_tuple(self):
        result = find_profiles_by_family("nonexistent_family")
        assert result == ()

    def test_family_with_leading_trailing_whitespace(self):
        result = find_profiles_by_family("  software  ")
        assert len(result) > 0
