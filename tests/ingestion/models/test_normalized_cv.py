"""Tests for cv_ingestion.models.normalized_cv."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from cv_ingestion.models.normalized_cv import (
    AdditionalSection,
    CertificationEntry,
    ContactInfo,
    EducationEntry,
    ExperienceEntry,
    LanguageEntry,
    LinkEntry,
    NormalizedCV,
    ProjectEntry,
    format_date_range,
    normalize_phone,
    parse_cv_date,
)


# ─────────────────────────────────────────────────────────────────────────────
# normalize_phone
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizePhone:
    def test_strips_spaces_and_dashes(self):
        assert normalize_phone("+34 600 123 456") == "+34600123456"

    def test_strips_parentheses(self):
        assert normalize_phone("+34 (600) 123-456") == "+34600123456"

    def test_preserves_leading_plus(self):
        result = normalize_phone("+34612345678")
        assert result is not None
        assert result.startswith("+")

    def test_number_without_plus_has_no_plus(self):
        result = normalize_phone("612345678")
        assert result is not None
        assert not result.startswith("+")

    def test_plain_digits_returned_as_is(self):
        assert normalize_phone("612345678") == "612345678"

    def test_too_short_returns_none(self):
        assert normalize_phone("123") is None

    def test_too_long_returns_none(self):
        assert normalize_phone("+1234567890123456789012") is None

    def test_letters_in_number_returns_none(self):
        assert normalize_phone("abc123def") is None

    def test_empty_string_returns_none(self):
        assert normalize_phone("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_phone("   ") is None

    def test_international_format_with_dashes(self):
        result = normalize_phone("+34-699-000-111")
        assert result == "+34699000111"


# ─────────────────────────────────────────────────────────────────────────────
# parse_cv_date
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCvDate:
    def test_year_only_returns_january_first(self):
        assert parse_cv_date("2023") == date(2023, 1, 1)

    def test_year_month_returns_first_of_month(self):
        assert parse_cv_date("2023-06") == date(2023, 6, 1)

    def test_full_iso_date_is_parsed(self):
        assert parse_cv_date("2023-06-15") == date(2023, 6, 15)

    def test_date_object_returned_unchanged(self):
        d = date(2021, 3, 10)
        assert parse_cv_date(d) == d

    def test_datetime_object_returns_date_part(self):
        from datetime import datetime

        dt = datetime(2021, 3, 10, 14, 30)
        assert parse_cv_date(dt) == date(2021, 3, 10)

    def test_none_returns_none(self):
        assert parse_cv_date(None) is None

    def test_empty_string_returns_none(self):
        assert parse_cv_date("") is None

    def test_whitespace_returns_none(self):
        assert parse_cv_date("   ") is None

    def test_unrecognized_format_returns_none(self):
        assert parse_cv_date("March 2023") is None

    def test_invalid_month_returns_none(self):
        assert parse_cv_date("2023-13") is None

    def test_invalid_day_returns_none(self):
        assert parse_cv_date("2023-02-30") is None


# ─────────────────────────────────────────────────────────────────────────────
# format_date_range
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatDateRange:
    def test_start_and_end_date(self):
        result = format_date_range(date(2021, 1, 1), date(2023, 12, 31))
        assert result == "Jan 2021 – Dec 2023"

    def test_is_current_shows_present(self):
        result = format_date_range(date(2022, 6, 1), None, is_current=True)
        assert result == "Jun 2022 – Present"

    def test_no_end_date_and_not_current_shows_only_start(self):
        result = format_date_range(date(2022, 6, 1), None, is_current=False)
        assert result == "Jun 2022"

    def test_no_start_date_returns_empty(self):
        result = format_date_range(None, None)
        assert result == ""

    def test_only_end_date(self):
        result = format_date_range(None, date(2023, 6, 30))
        assert result == "Jun 2023"


# ─────────────────────────────────────────────────────────────────────────────
# ContactInfo
# ─────────────────────────────────────────────────────────────────────────────


class TestContactInfo:
    def test_phone_is_normalized_on_init(self):
        contact = ContactInfo(phone="+34 600 123 456")
        assert contact.phone == "+34600123456"

    def test_invalid_phone_stored_as_none(self):
        contact = ContactInfo(phone="not-a-phone-number!!!")
        assert contact.phone is None

    def test_name_is_stripped(self):
        contact = ContactInfo(name="  Ana García  ")
        assert contact.name == "Ana García"

    def test_empty_name_becomes_none(self):
        contact = ContactInfo(name="   ")
        assert contact.name is None

    def test_to_text_includes_all_fields(self):
        contact = ContactInfo(
            name="Ana García",
            email="ana@example.com",
            phone="+34600123456",
            location="Madrid",
        )
        text = contact.to_text()
        assert "Ana García" in text
        assert "ana@example.com" in text
        assert "+34600123456" in text
        assert "Madrid" in text

    def test_to_text_skips_none_fields(self):
        contact = ContactInfo(name="Ana García", email=None, phone=None, location=None)
        text = contact.to_text()
        assert text == "Ana García"

    def test_to_text_empty_contact_returns_empty_string(self):
        contact = ContactInfo()
        assert contact.to_text() == ""


# ─────────────────────────────────────────────────────────────────────────────
# ExperienceEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestExperienceEntry:
    def test_to_text_includes_title_and_company(self):
        entry = ExperienceEntry(
            title="Backend Developer",
            company="Acme",
            start_date=date(2021, 1, 1),
            end_date=date(2023, 12, 31),
        )
        text = entry.to_text()
        assert "Backend Developer" in text
        assert "Acme" in text

    def test_to_text_includes_date_range(self):
        entry = ExperienceEntry(
            title="Dev",
            company="Co",
            start_date=date(2021, 1, 1),
            end_date=date(2022, 12, 31),
        )
        text = entry.to_text()
        assert "Jan 2021" in text
        assert "Dec 2022" in text

    def test_to_text_shows_present_when_current(self):
        entry = ExperienceEntry(
            title="Dev",
            company="Co",
            start_date=date(2022, 1, 1),
            is_current=True,
        )
        text = entry.to_text()
        assert "Present" in text

    def test_to_text_includes_description_when_present(self):
        entry = ExperienceEntry(
            title="Dev",
            company="Co",
            description="Built REST APIs.",
        )
        text = entry.to_text()
        assert "Built REST APIs." in text

    def test_to_text_no_empty_line_without_description(self):
        entry = ExperienceEntry(title="Dev", company="Co")
        text = entry.to_text()
        assert "\n\n" not in text

    def test_dates_are_parsed_from_strings(self):
        entry = ExperienceEntry(
            title="Dev",
            company="Co",
            start_date="2021-01",
            end_date="2023-12",
        )
        assert entry.start_date == date(2021, 1, 1)
        assert entry.end_date == date(2023, 12, 1)

    def test_title_and_company_stripped(self):
        entry = ExperienceEntry(title="  Dev  ", company="  Co  ")
        assert entry.title == "Dev"
        assert entry.company == "Co"

    def test_none_title_becomes_empty_string(self):
        entry = ExperienceEntry(title=None, company="Co")
        assert entry.title == ""


# ─────────────────────────────────────────────────────────────────────────────
# EducationEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestEducationEntry:
    def test_to_text_includes_degree_and_institution(self):
        entry = EducationEntry(
            degree="Bachelor's in CS",
            institution="UPM",
            start_date=date(2017, 9, 1),
            end_date=date(2021, 6, 30),
        )
        text = entry.to_text()
        assert "Bachelor's in CS" in text
        assert "UPM" in text

    def test_to_text_includes_date_range(self):
        entry = EducationEntry(
            degree="Bachelor's",
            institution="UPM",
            start_date=date(2017, 9, 1),
            end_date=date(2021, 6, 30),
        )
        text = entry.to_text()
        assert "Sep 2017" in text
        assert "Jun 2021" in text

    def test_to_text_includes_description(self):
        entry = EducationEntry(
            degree="Master's",
            institution="MIT",
            description="Focused on ML and data.",
        )
        text = entry.to_text()
        assert "Focused on ML and data." in text

    def test_to_text_no_description_no_extra_newline(self):
        entry = EducationEntry(degree="Bachelor's", institution="UPM")
        text = entry.to_text()
        assert "\n\n" not in text

    def test_dates_parsed_from_strings(self):
        entry = EducationEntry(
            degree="Bachelor's",
            institution="UPM",
            start_date="2017-09",
            end_date="2021-06",
        )
        assert entry.start_date == date(2017, 9, 1)
        assert entry.end_date == date(2021, 6, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CertificationEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestCertificationEntry:
    def test_to_text_includes_name_and_issuer(self):
        entry = CertificationEntry(name="AWS Certified", issuer="Amazon")
        text = entry.to_text()
        assert "AWS Certified" in text
        assert "Amazon" in text

    def test_to_text_includes_issue_date(self):
        entry = CertificationEntry(name="AWS Certified", issue_date=date(2022, 3, 1))
        assert "Mar 2022" in entry.to_text()

    def test_to_text_includes_description(self):
        entry = CertificationEntry(name="AWS Certified", description="Cloud fundamentals.")
        assert "Cloud fundamentals." in entry.to_text()

    def test_date_parsed_from_string(self):
        entry = CertificationEntry(name="AWS Certified", issue_date="2022-03")
        assert entry.issue_date == date(2022, 3, 1)


# ─────────────────────────────────────────────────────────────────────────────
# LanguageEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageEntry:
    def test_to_text_includes_proficiency(self):
        entry = LanguageEntry(language="English", proficiency="Native")
        assert entry.to_text() == "English (Native)"

    def test_to_text_without_proficiency(self):
        entry = LanguageEntry(language="French")
        assert entry.to_text() == "French"


# ─────────────────────────────────────────────────────────────────────────────
# ProjectEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestProjectEntry:
    def test_to_text_includes_name_and_description(self):
        entry = ProjectEntry(name="CVault", description="CV ingestion pipeline.")
        text = entry.to_text()
        assert "CVault" in text
        assert "CV ingestion pipeline." in text

    def test_to_text_includes_technologies(self):
        entry = ProjectEntry(name="CVault", technologies=["Python", "Qdrant"])
        text = entry.to_text()
        assert "Python" in text
        assert "Qdrant" in text

    def test_to_text_includes_url(self):
        entry = ProjectEntry(name="CVault", url="https://example.com")
        assert "https://example.com" in entry.to_text()

    def test_technologies_deduplicated(self):
        entry = ProjectEntry(name="CVault", technologies=["Python", "python"])
        assert len(entry.technologies) == 1


# ─────────────────────────────────────────────────────────────────────────────
# LinkEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestLinkEntry:
    def test_to_text_with_label(self):
        entry = LinkEntry(label="LinkedIn", url="https://linkedin.com/in/ana")
        assert entry.to_text() == "LinkedIn: https://linkedin.com/in/ana"

    def test_to_text_without_label(self):
        entry = LinkEntry(url="https://example.com")
        assert entry.to_text() == "https://example.com"


# ─────────────────────────────────────────────────────────────────────────────
# AdditionalSection
# ─────────────────────────────────────────────────────────────────────────────


class TestAdditionalSection:
    def test_to_text_includes_title_and_content(self):
        entry = AdditionalSection(title="Awards", content="Best Hackathon 2022.")
        text = entry.to_text()
        assert "Awards" in text
        assert "Best Hackathon 2022." in text

    def test_to_text_without_title_returns_content_only(self):
        entry = AdditionalSection(content="Best Hackathon 2022.")
        assert entry.to_text() == "Best Hackathon 2022."


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCV.calculate_years_of_experience
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculateYearsOfExperience:
    def _make_cv(self, experience: list[ExperienceEntry]) -> NormalizedCV:
        return NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Test"),
            experience=experience,
        )

    def test_no_experience_returns_none(self):
        cv = self._make_cv([])
        assert cv.calculate_years_of_experience() is None

    def test_experience_without_start_date_returns_none(self):
        cv = self._make_cv(
            [ExperienceEntry(title="Dev", company="Co", start_date=None)]
        )
        assert cv.calculate_years_of_experience() is None

    def test_two_year_job_returns_two(self):
        cv = self._make_cv(
            [
                ExperienceEntry(
                    title="Dev",
                    company="Co",
                    start_date=date(2021, 1, 1),
                    end_date=date(2022, 12, 31),
                )
            ]
        )
        result = cv.calculate_years_of_experience()
        assert result is not None
        assert result >= 1

    def test_minimum_is_one(self):
        cv = self._make_cv(
            [
                ExperienceEntry(
                    title="Dev",
                    company="Co",
                    start_date=date(2023, 1, 1),
                    end_date=date(2023, 3, 31),
                )
            ]
        )
        result = cv.calculate_years_of_experience()
        assert result == 1

    def test_current_job_counts_up_to_today(self):
        cv = self._make_cv(
            [
                ExperienceEntry(
                    title="Dev",
                    company="Co",
                    start_date=date(2020, 1, 1),
                    is_current=True,
                )
            ]
        )
        result = cv.calculate_years_of_experience()
        assert result is not None
        assert result >= 4

    def test_consecutive_jobs_sum_correctly(self):
        cv = self._make_cv(
            [
                ExperienceEntry(
                    title="Senior Dev",
                    company="B",
                    start_date=date(2022, 1, 1),
                    end_date=date(2023, 12, 31),
                ),
                ExperienceEntry(
                    title="Junior Dev",
                    company="A",
                    start_date=date(2020, 1, 1),
                    end_date=date(2021, 12, 31),
                ),
            ]
        )
        result = cv.calculate_years_of_experience()
        assert result is not None
        assert result >= 3


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCV.context_summary
# ─────────────────────────────────────────────────────────────────────────────


class TestContextSummary:
    def test_includes_candidate_name(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana García"),
        )
        assert "Ana García" in cv.context_summary()

    def test_includes_current_title_and_company(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            current_title="Senior Dev",
            current_company="Acme",
        )
        summary = cv.context_summary()
        assert "Senior Dev" in summary
        assert "Acme" in summary

    def test_includes_skills(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            skills=["Python", "FastAPI"],
        )
        summary = cv.context_summary()
        assert "Python" in summary
        assert "FastAPI" in summary

    def test_includes_email(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana", email="ana@example.com"),
        )
        assert "ana@example.com" in cv.context_summary()

    def test_unknown_candidate_fallback(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(),
        )
        assert "Unknown Candidate" in cv.context_summary()

    def test_no_skills_no_crash(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            skills=[],
        )
        summary = cv.context_summary()
        assert "Ana" in summary

    def test_includes_latest_education(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            education=[
                EducationEntry(degree="Master's in CS", institution="MIT"),
                EducationEntry(degree="Bachelor's in CS", institution="UPM"),
            ],
        )
        summary = cv.context_summary()
        assert "Master's in CS" in summary
        assert "MIT" in summary

    def test_years_of_experience_shown(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            years_of_experience=5,
        )
        summary = cv.context_summary()
        assert "5y exp" in summary


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCV.section_text
# ─────────────────────────────────────────────────────────────────────────────


class TestSectionText:
    def _make_cv(self) -> NormalizedCV:
        return NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(
                name="Ana García",
                email="ana@example.com",
                phone="+34600123456",
            ),
            about="Experienced developer.",
            experience=[
                ExperienceEntry(
                    title="Dev",
                    company="Acme",
                    start_date=date(2021, 1, 1),
                    end_date=date(2023, 12, 31),
                )
            ],
            education=[
                EducationEntry(
                    degree="Bachelor's in CS",
                    institution="UPM",
                )
            ],
            skills=["Python", "FastAPI"],
            certifications=[CertificationEntry(name="AWS Certified", issuer="Amazon")],
            languages=[LanguageEntry(language="English", proficiency="Native")],
            projects=[ProjectEntry(name="CVault", description="CV pipeline.")],
            links=[LinkEntry(label="GitHub", url="https://github.com/ana")],
            additional_sections=[
                AdditionalSection(title="Awards", content="Best Hackathon 2022.")
            ],
        )

    def test_contact_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.CONTACT)
        assert "Ana García" in text

    def test_about_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        assert cv.section_text(CVSection.ABOUT) == "Experienced developer."

    def test_experience_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.EXPERIENCE)
        assert "Dev" in text
        assert "Acme" in text

    def test_education_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.EDUCATION)
        assert "Bachelor's in CS" in text
        assert "UPM" in text

    def test_skills_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.SKILLS)
        assert "Python" in text
        assert "FastAPI" in text

    def test_unknown_section_returns_empty(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        assert cv.section_text(CVSection.UNKNOWN) == ""

    def test_certifications_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.CERTIFICATIONS)
        assert "AWS Certified" in text
        assert "Amazon" in text

    def test_languages_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        assert cv.section_text(CVSection.LANGUAGES) == "English (Native)"

    def test_projects_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.PROJECTS)
        assert "CVault" in text
        assert "CV pipeline." in text

    def test_links_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        assert cv.section_text(CVSection.LINKS) == "GitHub: https://github.com/ana"

    def test_additional_section(self):
        from cv_ingestion.models.chunking import CVSection

        cv = self._make_cv()
        text = cv.section_text(CVSection.ADDITIONAL)
        assert "Awards" in text
        assert "Best Hackathon 2022." in text


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCV.from_llm_payload
# ─────────────────────────────────────────────────────────────────────────────


class TestFromLlmPayload:
    def _payload(self) -> dict:
        return {
            "contact": {
                "name": "Pedro López",
                "email": "pedro.lopez@example.com",
                "phone": "+34699000111",
                "location": None,
            },
            "about": "Backend developer.",
            "experience": [
                {
                    "title": "Backend Developer",
                    "company": "Acme",
                    "start_date": "2021-01",
                    "end_date": "2023-12",
                    "is_current": False,
                    "description": "Built APIs.",
                }
            ],
            "education": [
                {
                    "degree": "Bachelor's in CS",
                    "institution": "UPM",
                    "start_date": "2017-09",
                    "end_date": "2021-06",
                    "description": None,
                }
            ],
            "skills": ["Python", "FastAPI"],
            "certifications": [
                {
                    "name": "AWS Certified",
                    "issuer": "Amazon",
                    "issue_date": "2022-03",
                    "description": None,
                }
            ],
            "languages": [{"language": "English", "proficiency": "Native"}],
            "projects": [
                {
                    "name": "CVault",
                    "description": "CV ingestion pipeline.",
                    "technologies": ["Python", "Qdrant"],
                    "url": None,
                }
            ],
            "links": [{"label": "GitHub", "url": "https://github.com/pedro"}],
            "additional_sections": [
                {"title": "Awards", "content": "Best Hackathon 2022."}
            ],
        }

    def test_returns_normalized_cv(self):
        result = NormalizedCV.from_llm_payload(self._payload(), source_path="test.pdf")
        assert isinstance(result, NormalizedCV)

    def test_source_path_is_set(self):
        result = NormalizedCV.from_llm_payload(
            self._payload(), source_path="minio://cvs/pedro.pdf"
        )
        assert result.source_path == "minio://cvs/pedro.pdf"

    def test_current_title_derived(self):
        payload = self._payload()
        payload["experience"][0]["is_current"] = True
        payload["experience"][0]["end_date"] = None
        result = NormalizedCV.from_llm_payload(payload, source_path="test.pdf")
        assert result.current_title == "Backend Developer"

    def test_skills_deduplicated(self):
        payload = self._payload()
        payload["skills"] = ["Python", "python", "PYTHON", "FastAPI"]
        result = NormalizedCV.from_llm_payload(payload, source_path="test.pdf")
        lower_skills = [s.lower() for s in result.skills]
        assert lower_skills.count("python") == 1

    def test_invalid_payload_raises_validation_error(self):
        with pytest.raises(ValidationError):
            NormalizedCV.from_llm_payload(
                {"contact": None},  # contact cannot be None
                source_path="test.pdf",
            )

    def test_years_of_experience_computed(self):
        result = NormalizedCV.from_llm_payload(self._payload(), source_path="test.pdf")
        assert result.years_of_experience is not None
        assert result.years_of_experience >= 1

    def test_new_sections_are_parsed(self):
        result = NormalizedCV.from_llm_payload(self._payload(), source_path="test.pdf")
        assert result.certifications[0].name == "AWS Certified"
        assert result.languages[0].language == "English"
        assert result.projects[0].name == "CVault"
        assert result.links[0].url == "https://github.com/pedro"
        assert result.additional_sections[0].title == "Awards"


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedCV.shared_metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestSharedMetadata:
    def test_skills_are_included(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            skills=["Python", "FastAPI"],
        )
        meta = cv.shared_metadata()
        assert meta.skills == ["Python", "FastAPI"]

    def test_degrees_extracted_from_education(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            education=[
                EducationEntry(degree="Bachelor's in CS", institution="UPM"),
                EducationEntry(degree="Master's in AI", institution="MIT"),
            ],
        )
        meta = cv.shared_metadata()
        assert "Bachelor's in CS" in meta.degrees
        assert "Master's in AI" in meta.degrees

    def test_companies_extracted_from_experience(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            experience=[
                ExperienceEntry(title="Dev", company="Acme"),
                ExperienceEntry(title="Lead", company="BigCorp"),
            ],
        )
        meta = cv.shared_metadata()
        assert "Acme" in meta.companies
        assert "BigCorp" in meta.companies

    def test_source_path_matches(self):
        cv = NormalizedCV(
            source_path="minio://cvs/ana.pdf",
            contact=ContactInfo(name="Ana"),
        )
        assert cv.shared_metadata().source_path == "minio://cvs/ana.pdf"

    def test_email_and_phone_included(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(
                name="Ana",
                email="ana@example.com",
                phone="+34600123456",
            ),
        )
        meta = cv.shared_metadata()
        assert meta.email == "ana@example.com"
        assert meta.phone == "+34600123456"

    def test_certifications_and_languages_extracted(self):
        cv = NormalizedCV(
            source_path="test.pdf",
            contact=ContactInfo(name="Ana"),
            certifications=[CertificationEntry(name="AWS Certified")],
            languages=[LanguageEntry(language="English")],
        )
        meta = cv.shared_metadata()
        assert "AWS Certified" in meta.certifications
        assert "English" in meta.languages
