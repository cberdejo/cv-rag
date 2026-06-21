"""Tests for cv_factory.pipeline.tasks.build_cv_base.education."""

from __future__ import annotations

from random import Random

from faker import Faker

from cv_factory.models.catalog import EducationOption
from cv_factory.pipeline.tasks.build_cv_base.education import generate_education


def _make_options() -> list[EducationOption]:
    return [
        EducationOption(
            title="Bachelor's Degree in Computer Engineering",
            institution_type="university",
            level="bachelor",
            relevance=5,
        ),
        EducationOption(
            title="Higher Technician in Web Application Development",
            institution_type="vocational_training",
            level="vocational",
            relevance=4,
        ),
        EducationOption(
            title="Master's Degree in Data Science",
            institution_type="university",
            level="master",
            relevance=5,
        ),
        EducationOption(
            title="Backend Development Bootcamp",
            institution_type="bootcamp",
            level="bootcamp",
            relevance=3,
        ),
    ]


class TestGenerateEducation:
    def _call(self, rng: Random, *, latest_end_year: int = 2020) -> list:
        fake = Faker("es_ES")
        fake.seed_instance(rng.randint(0, 9999))
        return generate_education(
            fake=fake,
            rng=rng,
            education_options=_make_options(),
            latest_end_year=latest_end_year,
        )

    def test_returns_at_least_one_entry(self):
        result = self._call(Random(1))
        assert len(result) >= 1

    def test_returns_at_most_four_entries(self):
        # Run many seeds to stress the upper bound
        for seed in range(50):
            result = self._call(Random(seed))
            assert len(result) <= 4, f"Too many education entries with seed={seed}"

    def test_periods_do_not_overlap(self):
        for seed in range(30):
            entries = self._call(Random(seed))
            dates = [(e.period.start_date, e.period.end_date) for e in entries]
            for i, (sa, ea) in enumerate(dates):
                for j, (sb, eb) in enumerate(dates):
                    if i >= j:
                        continue
                    # ea and eb are always set (all generated entries are closed)
                    assert not (max(sa, sb) < min(ea, eb)), (
                        f"Overlap detected with seed={seed}: {sa}–{ea} vs {sb}–{eb}"
                    )

    def test_master_only_after_bachelor(self):
        for seed in range(50):
            entries = self._call(Random(seed))
            master_entries = [
                e
                for e in entries
                if "master" in e.title.lower() or "mba" in e.title.lower()
            ]
            bachelor_entries = [e for e in entries if "bachelor" in e.title.lower()]

            for master in master_entries:
                completed_bachelors = [
                    b
                    for b in bachelor_entries
                    if b.period.end_date is not None
                    and b.period.end_date < master.period.start_date
                ]
                assert completed_bachelors, (
                    f"Master '{master.title}' found without a prior bachelor (seed={seed})"
                )

    def test_all_entries_end_before_latest_end_year(self):
        latest = 2018
        entries = self._call(Random(42), latest_end_year=latest)
        for e in entries:
            assert e.period.end_date is not None
            assert e.period.end_date.year <= latest

    def test_reproducible_with_same_seed(self):
        r1, r2 = Random(123), Random(123)
        f1, f2 = Faker("es_ES"), Faker("es_ES")
        f1.seed_instance(0)
        f2.seed_instance(0)
        e1 = generate_education(
            fake=f1, rng=r1, education_options=_make_options(), latest_end_year=2020
        )
        e2 = generate_education(
            fake=f2, rng=r2, education_options=_make_options(), latest_end_year=2020
        )
        assert [(e.title, e.period.start_date) for e in e1] == [
            (e.title, e.period.start_date) for e in e2
        ]
