"""Render a CV to PDF using one of the bundled HTML templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from cv_factory.models.cv import CV

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATES = (
    "classic_ats.html",
    "compact.html",
    "modern_sidebar.html",
)
SECTION_TITLE_ALIASES = {
    "about": [
        "Profile",
        "Professional Profile",
        "About",
        "About Me",
        "Summary",
        "Professional Summary",
        "Career Summary",
        "Personal Statement",
        "Overview",
        "Introduction",
        "",
    ],
    "contact": [
        "Contact",
        "Contact Details",
        "Contact Information",
        "Personal Details",
        "Get in Touch",
        "Reach Me",
        "",
    ],
    "education": [
        "Education",
        "Academic Background",
        "Academic History",
        "Education & Training",
        "Qualifications",
        "Academic Qualifications",
        "Studies",
        "Learning Path",
        "",
    ],
    "experience": [
        "Experience",
        "Work Experience",
        "Professional Experience",
        "Employment History",
        "Career History",
        "Work History",
        "Professional Background",
        "Relevant Experience",
        "",
    ],
    "skills": [
        "Skills",
        "Key Skills",
        "Core Skills",
        "Technical Skills",
        "Competencies",
        "Core Competencies",
        "Areas of Expertise",
        "Expertise",
        "",
    ],
}


@dataclass(frozen=True)
class RenderedCVArtifacts:
    """Filesystem artifacts produced by the CV renderer."""

    pdf_path: Path


def _safe_stem(value: str) -> str:
    """Create a filesystem-friendly filename stem."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return normalized or "cv"


def _random_section_titles(rng: Random) -> dict[str, str]:
    """Select one display title per CV section."""
    return {
        section: rng.choice(aliases)
        for section, aliases in SECTION_TITLE_ALIASES.items()
    }


def render_cv(
    cv: CV,
    *,
    output_dir: Path,
    photo_url: str | None = None,
    seed: int | None = None,
) -> RenderedCVArtifacts:
    """Render ``cv`` to HTML in memory, convert it to PDF, and persist the PDF.

    A template is chosen randomly from the three bundled HTML templates. Passing
    ``seed`` makes the choice reproducible for pipeline runs that need it.

    ``photo_url`` is embedded directly as the profile photo's ``<img src>`` —
    typically a base64 ``data:`` URI — so no image file ever touches disk.
    """
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = Random(seed)
    template_name = rng.choice(TEMPLATES)
    section_titles = _random_section_titles(rng)

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(("html", "xml")),
    )
    template = environment.get_template(template_name)

    html = template.render(
        cv=cv,
        photo_url=photo_url,
        section_titles=section_titles,
    )

    stem = _safe_stem(cv.name)
    pdf_path = output_dir / f"{stem}.pdf"

    HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(pdf_path)

    return RenderedCVArtifacts(
        pdf_path=pdf_path.resolve(),
    )


__all__ = [
    "RenderedCVArtifacts",
    "SECTION_TITLE_ALIASES",
    "render_cv",
]
