"""Tests for cv_factory.pipeline.tasks.build_cv_base.contact."""

from __future__ import annotations

import re
from random import Random

from faker import Faker

from cv_factory.models.cv import Gender
from cv_factory.pipeline.tasks.build_cv_base.contact import (
    _ascii_token,
    generate_person,
    generate_phone,
)

# Phone pattern from the CV model
_PHONE_RE = re.compile(r"^\+?[0-9\s\-\(\)]{7,20}$")


class TestAsciiToken:
    def test_simple_ascii_name(self):
        assert _ascii_token("Ana") == "ana"

    def test_accented_characters_are_stripped(self):
        token = _ascii_token("García")
        assert token == "garcia"

    def test_spaces_are_removed(self):
        assert " " not in _ascii_token("Pedro Ruiz")

    def test_special_characters_removed(self):
        token = _ascii_token("Ñoño")
        assert token.isalnum()


class TestGeneratePerson:
    def test_returns_three_values(self, fake, rng):
        result = generate_person(fake, rng)
        assert len(result) == 3

    def test_gender_is_enum(self, fake, rng):
        _, _, gender = generate_person(fake, rng)
        assert isinstance(gender, Gender)

    def test_email_contains_at_symbol(self, fake, rng):
        _, email, _ = generate_person(fake, rng)
        assert "@" in email

    def test_email_ends_with_example_com(self, fake, rng):
        _, email, _ = generate_person(fake, rng)
        assert email.endswith("@example.com")

    def test_email_has_no_accents(self, fake, rng):
        for _ in range(20):
            _, email, _ = generate_person(fake, rng)
            assert email.isascii()

    def test_same_seed_produces_same_output(self):
        fake_a, rng_a = Faker("es_ES"), Random(99)
        fake_b, rng_b = Faker("es_ES"), Random(99)
        fake_a.seed_instance(99)
        fake_b.seed_instance(99)
        assert generate_person(fake_a, rng_a) == generate_person(fake_b, rng_b)

    def test_different_seeds_may_produce_different_genders(self):
        results = set()
        for seed in range(30):
            f, r = Faker("es_ES"), Random(seed)
            f.seed_instance(seed)
            _, _, gender = generate_person(f, r)
            results.add(gender)
        # Over 30 seeds we should see both genders at least once
        assert len(results) == 2


class TestGeneratePhone:
    def test_matches_phone_pattern(self, rng):
        for _ in range(50):
            phone = generate_phone(rng)
            assert _PHONE_RE.match(phone), f"Invalid phone: {phone!r}"

    def test_length_within_bounds(self, rng):
        for _ in range(50):
            phone = generate_phone(rng)
            assert 7 <= len(phone) <= 20, f"Phone length out of bounds: {phone!r}"

    def test_same_seed_is_reproducible(self):
        assert generate_phone(Random(7)) == generate_phone(Random(7))
