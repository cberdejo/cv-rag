"""Prompt builders for CV narrative enrichment."""

from __future__ import annotations

import json
from typing import Any

from cv_factory.models.cv import CV


SYSTEM_PROMPT = """
You are a professional CV writer.

Your task is to enrich an existing synthetic CV.

Rules:
- Write in English.
- Do not invent new jobs, studies, institutions, companies, dates, skills, degrees or certifications.
- Do not modify names, emails, phones, titles, companies, institutions, dates, GPA or skills.
- Only generate:
  1. about
  2. experience descriptions
  3. education descriptions
- Keep the tone professional, realistic and concise.
- Avoid first person.
- Avoid using the candidate's name.
- Avoid third-person pronouns such as "he", "she", "his", "her", "they" or "their".
- Use an impersonal CV style with no explicit subject.
- Write as if the text belongs directly inside the candidate's CV.
- Prefer noun phrases and action-oriented sentence fragments commonly used in CVs.
- Avoid phrases such as "the candidate", "the applicant", "this person", or "the professional".
- Avoid exaggerated claims.
- Use the provided skills naturally when relevant.
- Return exactly one description for each input experience and education entry.
- Preserve the zero-based indexes.
""".strip()


USER_TEMPLATE = """
Enrich the following base CV.

Input CV:
{payload}
""".strip()


def build_enrichment_messages(cv: CV) -> list[dict[str, str]]:
    """Build chat messages used to ask the LLM for CV narrative enrichment."""
    payload = serialize_cv_for_prompt(cv)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                payload=json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        },
    ]


def serialize_cv_for_prompt(cv: CV) -> dict[str, Any]:
    """Serialize only the immutable CV facts needed by the enrichment prompt."""
    return {
        "name": cv.name,
        "gender": cv.gender.value,
        "about": cv.about,
        "skills": cv.skills,
        "experience": [
            {
                "index": index,
                "title": item.title,
                "company": item.company,
                "period": {
                    "start_date": item.period.start_date.isoformat(),
                    "end_date": (
                        item.period.end_date.isoformat()
                        if item.period.end_date is not None
                        else None
                    ),
                },
                "current_description": item.description,
            }
            for index, item in enumerate(cv.experience)
        ],
        "education": [
            {
                "index": index,
                "institution": item.institution,
                "title": item.title,
                "period": {
                    "start_date": item.period.start_date.isoformat(),
                    "end_date": (
                        item.period.end_date.isoformat()
                        if item.period.end_date is not None
                        else None
                    ),
                },
                "gpa": item.gpa,
                "current_description": item.description,
            }
            for index, item in enumerate(cv.education)
        ],
    }


__all__ = [
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "build_enrichment_messages",
    "serialize_cv_for_prompt",
]
