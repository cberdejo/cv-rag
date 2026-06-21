"""Tests for cv_factory.pipeline.tasks.build_cv_base.main — build_cv_base."""

from __future__ import annotations

from cv_factory.models.cv import CV
from cv_factory.pipeline.tasks.build_cv_base.main import build_cv_base


class TestBuildCvBase:
    def test_returns_cv_instance(self):
        cv = build_cv_base(seed=1)
        assert isinstance(cv, CV)

    def test_same_seed_produces_same_cv(self):
        cv1 = build_cv_base(seed=42)
        cv2 = build_cv_base(seed=42)
        assert cv1.name == cv2.name
        assert cv1.email == cv2.email
        assert cv1.skills == cv2.skills

    def test_different_seeds_produce_different_cvs(self):
        names = {build_cv_base(seed=i).name for i in range(20)}
        # Over 20 seeds we expect some variation
        assert len(names) > 1

    def test_cv_has_non_empty_experience(self):
        cv = build_cv_base(seed=10)
        assert len(cv.experience) >= 1

    def test_cv_has_non_empty_education(self):
        cv = build_cv_base(seed=10)
        assert len(cv.education) >= 1

    def test_cv_has_skills(self):
        cv = build_cv_base(seed=10)
        assert len(cv.skills) >= 1

    def test_skills_capped_at_eight(self):
        for seed in range(20):
            cv = build_cv_base(seed=seed)
            assert len(cv.skills) <= 8, f"Too many skills with seed={seed}"

    def test_about_references_candidate_name(self):
        cv = build_cv_base(seed=5)
        # The base about template includes the candidate's name
        assert cv.name in cv.about

    def test_cv_passes_own_pydantic_validation(self):
        """build_cv_base must return a model that re-validates without errors."""
        for seed in range(10):
            cv = build_cv_base(seed=seed)
            # model_validate re-runs all validators
            CV.model_validate(cv.model_dump())

    def test_none_seed_is_accepted(self):
        cv = build_cv_base(seed=None)
        assert isinstance(cv, CV)

    def test_education_precedes_experience(self):
        """All education end dates must precede the oldest experience start date."""
        for seed in range(20):
            cv = build_cv_base(seed=seed)
            oldest_exp_start = min(e.period.start_date for e in cv.experience)
            for edu in cv.education:
                assert edu.period.end_date is not None
                assert edu.period.end_date < oldest_exp_start, (
                    f"Education ends after experience starts (seed={seed})"
                )
