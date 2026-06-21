"""Tests for cv_factory.pipeline.pipeline — pure helper functions."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from cv_factory.pipeline.pipeline import (
    ArtifactStorage,
    _render_and_store,
    _safe_object_stem,
    minio_pdf_object_name,
)
from cv_factory.models.cv import CV, Education, Experience, Gender, Period


# ──────────────────────────────────────────────────────────────────────────────
# _safe_object_stem
# ──────────────────────────────────────────────────────────────────────────────


class TestSafeObjectStem:
    def test_ascii_name(self):
        assert _safe_object_stem("Ana Garcia") == "ana-garcia"

    def test_accented_characters_are_removed(self):
        stem = _safe_object_stem("García López")
        assert stem.isascii()
        assert stem == "garcia-lopez"

    def test_special_characters_replaced_with_hyphens(self):
        stem = _safe_object_stem("John  O'Brien!")
        assert "-" in stem
        assert "'" not in stem
        assert "!" not in stem

    def test_no_leading_or_trailing_hyphens(self):
        stem = _safe_object_stem("  Ana  ")
        assert not stem.startswith("-")
        assert not stem.endswith("-")

    def test_empty_string_returns_cv_fallback(self):
        assert _safe_object_stem("") == "cv"

    def test_only_special_chars_returns_cv_fallback(self):
        assert _safe_object_stem("!!!") == "cv"

    def test_ñ_handling(self):
        stem = _safe_object_stem("Ñoño")
        assert stem.isascii()


# ──────────────────────────────────────────────────────────────────────────────
# minio_pdf_object_name
# ──────────────────────────────────────────────────────────────────────────────


def _make_minimal_cv(name: str = "Ana Garcia Lopez") -> CV:
    return CV(
        name=name,
        email="ana.garcia01@example.com",
        phone="+34 600 111 222",
        gender=Gender.FEMALE,
        about="A" * 60,
        experience=[
            Experience(
                title="Backend Developer",
                company="Corp",
                description="Developed REST APIs for an e-commerce platform.",
                period=Period(start_date=date(2021, 1, 1), end_date=date(2023, 12, 31)),
            )
        ],
        education=[
            Education(
                institution="Madrid IT",
                title="Bachelor's Degree in Computer Engineering",
                period=Period(start_date=date(2017, 9, 1), end_date=date(2021, 6, 30)),
            )
        ],
        skills=["Python", "FastAPI"],
    )


class TestMinioPdfObjectName:
    def test_ends_with_pdf(self):
        cv = _make_minimal_cv()
        name = minio_pdf_object_name(cv=cv, today=date(2025, 1, 15))
        assert name.endswith(".pdf")

    def test_starts_with_iso_date(self):
        cv = _make_minimal_cv()
        name = minio_pdf_object_name(cv=cv, today=date(2025, 6, 1))
        assert name.startswith("2025-06-01-")

    def test_uses_today_when_date_not_provided(self):
        cv = _make_minimal_cv()
        name = minio_pdf_object_name(cv=cv)
        today_str = date.today().isoformat()
        assert name.startswith(today_str)

    def test_contains_sanitised_candidate_name(self):
        cv = _make_minimal_cv(name="Ana García López")
        name = minio_pdf_object_name(cv=cv, today=date(2025, 1, 1))
        assert "garcia" in name
        assert "lopez" in name

    def test_no_spaces_in_object_name(self):
        cv = _make_minimal_cv()
        name = minio_pdf_object_name(cv=cv, today=date(2025, 1, 1))
        assert " " not in name


# ──────────────────────────────────────────────────────────────────────────────
# ArtifactStorage enum
# ──────────────────────────────────────────────────────────────────────────────


class TestArtifactStorage:
    def test_all_variants_exist(self):
        assert ArtifactStorage.LOCAL == "local"
        assert ArtifactStorage.MINIO == "minio"
        assert ArtifactStorage.BOTH == "both"

    def test_str_coercion(self):
        assert ArtifactStorage("local") == ArtifactStorage.LOCAL


def test_render_and_store_passes_bucket_to_minio_upload(tmp_path):
    pdf_path = tmp_path / "cv.pdf"
    pdf_path.write_bytes(b"pdf")
    cv = _make_minimal_cv()

    with (
        patch(
            "cv_factory.pipeline.pipeline.render_cv",
            return_value=SimpleNamespace(pdf_path=pdf_path),
        ),
        patch(
            "cv_factory.pipeline.pipeline.upload_file",
            return_value="minio://cvs-test/cv.pdf",
        ) as upload_file,
    ):
        artifacts = _render_and_store(
            cv,
            output_dir=tmp_path,
            photo_url=None,
            seed=1,
            storage=ArtifactStorage.MINIO,
            minio_bucket="cvs-test",
            today=date(2025, 1, 1),
        )

    assert artifacts[0].minio_uri == "minio://cvs-test/cv.pdf"
    assert upload_file.call_args.kwargs["bucket"] == "cvs-test"
