"""Tests for cv_factory.pipeline.tasks.render_cv.main."""

from __future__ import annotations

from pathlib import Path
from random import Random

from cv_factory.models.cv import CV
from cv_factory.pipeline.tasks.render_cv.main import (
    SECTION_TITLE_ALIASES,
    TEMPLATES,
    RenderedCVArtifacts,
    _random_section_titles,
    _safe_stem,
    render_cv,
)


class TestSafeStem:
    def test_simple_name(self) -> None:
        assert _safe_stem("Ana Garcia") == "ana_garcia"

    def test_special_characters_become_underscores(self) -> None:
        stem = _safe_stem("O'Brien & Co.!")
        assert " " not in stem
        assert "'" not in stem
        assert "!" not in stem

    def test_no_leading_or_trailing_underscores(self) -> None:
        stem = _safe_stem("  Ana  ")
        assert not stem.startswith("_")
        assert not stem.endswith("_")

    def test_empty_string_falls_back_to_cv(self) -> None:
        assert _safe_stem("") == "cv"

    def test_only_special_chars_falls_back_to_cv(self) -> None:
        assert _safe_stem("###") == "cv"


class TestRandomSectionTitles:
    def test_returns_one_title_per_section(self) -> None:
        titles = _random_section_titles(Random(1))

        assert set(titles) == set(SECTION_TITLE_ALIASES)

    def test_each_title_is_a_valid_alias_for_its_section(self) -> None:
        titles = _random_section_titles(Random(2))

        for section, title in titles.items():
            assert title in SECTION_TITLE_ALIASES[section]

    def test_same_seed_is_reproducible(self) -> None:
        assert _random_section_titles(Random(7)) == _random_section_titles(Random(7))


class TestRenderCV:
    def test_returns_rendered_artifacts_with_existing_pdf(
        self, valid_cv: CV, tmp_path: Path
    ) -> None:
        result = render_cv(valid_cv, output_dir=tmp_path, seed=1)

        assert isinstance(result, RenderedCVArtifacts)
        assert result.pdf_path.exists()
        assert result.pdf_path.suffix == ".pdf"

    def test_pdf_filename_uses_safe_stem_of_candidate_name(
        self, valid_cv: CV, tmp_path: Path
    ) -> None:
        result = render_cv(valid_cv, output_dir=tmp_path, seed=1)

        assert result.pdf_path.name == f"{_safe_stem(valid_cv.name)}.pdf"

    def test_creates_output_dir_if_missing(self, valid_cv: CV, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "outputs"

        result = render_cv(valid_cv, output_dir=nested, seed=1)

        assert result.pdf_path.exists()

    def test_same_seed_selects_same_template(
        self, valid_cv: CV, tmp_path: Path
    ) -> None:
        rng_a, rng_b = Random(42), Random(42)

        assert rng_a.choice(TEMPLATES) == rng_b.choice(TEMPLATES)

    def test_renders_with_profile_image(self, valid_cv: CV, tmp_path: Path) -> None:
        photo_url = "data:image/png;base64,iVBORw0KGgo="

        result = render_cv(valid_cv, output_dir=tmp_path, photo_url=photo_url, seed=1)

        assert result.pdf_path.exists()
