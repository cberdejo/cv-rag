"""Synthetic contact data generators."""

from __future__ import annotations

import random
from random import Random
from unicodedata import combining, normalize

from faker import Faker

from cv_factory.models.cv import Gender

PHONE_PREFIXES = ["+34"]  # In the first iteration we assume all phones are spanish


def _ascii_token(value: str) -> str:
    """Normalize a text value into a lowercase ASCII email token."""
    normalized = normalize("NFKD", value)
    ascii_value = "".join(char for char in normalized if not combining(char))
    return "".join(char.lower() for char in ascii_value if char.isalnum())


def generate_person(fake: Faker, rng: Random) -> tuple[str, str, Gender]:
    """Generate related name, email, and gender data."""
    gender = rng.choice([Gender.MALE, Gender.FEMALE])
    first_name = (
        fake.first_name_male() if gender == Gender.MALE else fake.first_name_female()
    )
    last_names = [fake.last_name(), fake.last_name()]
    name = " ".join([first_name, *last_names])
    email = (
        f"{_ascii_token(first_name)}.{_ascii_token(last_names[0])}"
        f"{rng.randint(10, 99)}@example.com"
    )

    return name, email, gender


def generate_phone(rng: random.Random) -> str:
    """Generate a valid mobile phone string."""
    prefix = rng.choice(PHONE_PREFIXES)
    digits = str(rng.choice([6, 7])) + "".join(
        str(rng.randint(0, 9)) for _ in range(8)
    )  # first number 6 or 7 to emulate spanish phone number
    formats = [
        digits,
        f"{digits[:3]} {digits[3:6]} {digits[6:]}",
        f"{digits[:3]}-{digits[3:6]}-{digits[6:]}",
        f"{prefix} {digits[:3]} {digits[3:6]} {digits[6:]}",
        f"{prefix}-{digits[:3]}-{digits[3:6]}-{digits[6:]}",
        f"{prefix}{digits}",
        f"{digits[:3]} {digits[3:5]} {digits[5:7]} {digits[7:]}",
    ]

    return rng.choice(formats)


__all__ = ["generate_person", "generate_phone"]
