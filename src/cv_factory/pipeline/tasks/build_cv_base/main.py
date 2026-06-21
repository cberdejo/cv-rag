"""Build the first-pass CV skeleton from catalog data."""

from __future__ import annotations

from datetime import date
from random import Random

from faker import Faker

from cv_factory.catalog.registry import list_career_profiles
from cv_factory.models.cv import CV
from cv_factory.pipeline.tasks.build_cv_base.contact import (
    generate_person,
    generate_phone,
)
from cv_factory.pipeline.tasks.build_cv_base.education import generate_education
from cv_factory.pipeline.tasks.build_cv_base.experience import generate_experience


def build_cv_base(*, seed: int | None = 52) -> CV:
    """Create a valid, non-enriched CV skeleton.

    The returned CV intentionally contains plain placeholder prose. LLM enrichment
    should happen in a later task over this validated structure.
    """
    rng = Random(seed)
    fake = Faker("es_ES")
    fake.seed_instance(seed)

    profile = rng.choice(list_career_profiles())
    industry = (
        rng.choice(profile.possible_industries)
        if profile.possible_industries
        else profile.family
    )

    current_year = date.today().year
    name, email, gender = generate_person(fake, rng)
    phone = generate_phone(rng)
    experience = generate_experience(
        fake=fake,
        rng=rng,
        profile_label=profile.label,
        titles=profile.titles,
        current_year=current_year,
    )
    oldest_experience_start = min(job.period.start_date for job in experience)
    education = generate_education(
        fake=fake,
        rng=rng,
        education_options=profile.education,
        latest_end_year=oldest_experience_start.year - 1,
    )

    selected_skills = sorted(
        profile.skills,
        key=lambda skill: (-skill.weight, skill.category.value, skill.name),
    )[:8]

    return CV(
        name=name,
        email=email,
        phone=phone,
        gender=gender,
        about=(
            f"{name} is a {experience[0].title} profile for the {industry} industry. "
            "This base CV contains validated catalog skills and draft placeholders "
            "for later AI enrichment."
        ),
        experience=experience,
        education=education,
        skills=[skill.name for skill in selected_skills],
    )
