"""Tests for cv_factory.pipeline.tasks.build_cv_base.experience."""

from __future__ import annotations

from datetime import date
from random import Random

from faker import Faker

from cv_factory.models.catalog import JobTitleOption, Seniority
from cv_factory.pipeline.tasks.build_cv_base.experience import generate_experience


def _titles() -> list[JobTitleOption]:
    return [
        JobTitleOption(
            title="Junior Developer", seniority=Seniority.JUNIOR, min_years_experience=0
        ),
        JobTitleOption(
            title="Developer", seniority=Seniority.MID, min_years_experience=2
        ),
        JobTitleOption(
            title="Senior Developer", seniority=Seniority.SENIOR, min_years_experience=5
        ),
        JobTitleOption(
            title="Lead Developer", seniority=Seniority.LEAD, min_years_experience=8
        ),
    ]


class TestGenerateExperience:
    def _call(self, rng: Random, current_year: int = 2024) -> list:
        fake = Faker("es_ES")
        fake.seed_instance(rng.randint(0, 9999))
        return generate_experience(
            fake=fake,
            rng=rng,
            profile_label="Backend Developer",
            titles=_titles(),
            current_year=current_year,
        )

    def test_returns_at_least_one_entry(self):
        assert len(self._call(Random(1))) >= 1

    def test_returns_at_most_four_entries(self):
        for seed in range(50):
            result = self._call(Random(seed))
            assert len(result) <= 4

    def test_at_most_one_ongoing_job(self):
        for seed in range(50):
            entries = self._call(Random(seed))
            ongoing = [e for e in entries if e.period.current]
            assert len(ongoing) <= 1, f"Multiple ongoing jobs with seed={seed}"

    def test_no_overlapping_periods(self):
        today = date.today()
        for seed in range(30):
            entries = self._call(Random(seed))
            ranges = [
                (e.period.start_date, e.period.end_date or today) for e in entries
            ]
            for i, (sa, ea) in enumerate(ranges):
                for j, (sb, eb) in enumerate(ranges):
                    if i >= j:
                        continue
                    assert not (max(sa, sb) < min(ea, eb)), (
                        f"Overlap with seed={seed}: {sa}–{ea} vs {sb}–{eb}"
                    )

    def test_titles_belong_to_provided_list(self):
        valid_titles = {t.title for t in _titles()}
        for seed in range(20):
            entries = self._call(Random(seed))
            for e in entries:
                assert e.title in valid_titles

    def test_reproducible_with_same_seed(self):
        r1, r2 = Random(77), Random(77)
        f1, f2 = Faker("es_ES"), Faker("es_ES")
        f1.seed_instance(0)
        f2.seed_instance(0)
        e1 = generate_experience(
            fake=f1, rng=r1, profile_label="Dev", titles=_titles(), current_year=2024
        )
        e2 = generate_experience(
            fake=f2, rng=r2, profile_label="Dev", titles=_titles(), current_year=2024
        )
        assert [(e.title, e.period.start_date) for e in e1] == [
            (e.title, e.period.start_date) for e in e2
        ]

    def test_seniority_progression_is_plausible(self):
        """Most recent role should not be more junior than the oldest one."""
        seniority_rank = {
            Seniority.LEAD: 0,
            Seniority.SENIOR: 1,
            Seniority.MID: 2,
            Seniority.JUNIOR: 3,
        }
        title_to_seniority = {t.title: t.seniority for t in _titles()}
        violations = 0

        for seed in range(50):
            entries = self._call(Random(seed))
            if len(entries) < 2:
                continue
            first_seniority = title_to_seniority.get(entries[0].title)
            last_seniority = title_to_seniority.get(entries[-1].title)
            if first_seniority and last_seniority:
                if seniority_rank[first_seniority] > seniority_rank[last_seniority]:
                    violations += 1

        # Allow a small minority of "unusual" careers, but not the majority
        assert violations < 25, f"Too many seniority regressions: {violations}/50"
