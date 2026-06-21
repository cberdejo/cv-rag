"""Tests for RAG assistant Pydantic models and state types."""

import pytest
from pydantic import ValidationError

from rag.assistant.models import (
    AssistantResponse,
    CVSource,
    IntentDecision,
    IntentRoute,
    MetadataFilters,
    QueryRewrite,
    RetrievedChunk,
)


class TestIntentRoute:
    def test_values(self):
        assert IntentRoute.OUT_OF_SCOPE == "out_of_scope"
        assert IntentRoute.METADATA == "metadata"
        assert IntentRoute.HYBRID == "hybrid"
        assert IntentRoute.HYBRID_FILTERED == "hybrid_filtered"


class TestMetadataFilters:
    def test_default_has_no_filters(self):
        f = MetadataFilters()
        assert not f.has_any()

    def test_has_any_with_candidate_name(self):
        f = MetadataFilters(candidate_name="Ana García")
        assert f.has_any()

    def test_has_any_with_skills(self):
        f = MetadataFilters(skills=["Python", "Docker"])
        assert f.has_any()

    def test_has_any_with_current_title(self):
        f = MetadataFilters(current_title="Senior Engineer")
        assert f.has_any()

    def test_has_any_with_current_company(self):
        f = MetadataFilters(current_company="Acme Corp")
        assert f.has_any()

    def test_has_any_with_companies(self):
        f = MetadataFilters(companies=["Acme", "Initech"])
        assert f.has_any()

    def test_has_any_with_degrees(self):
        f = MetadataFilters(degrees=["Bachelor's in CS"])
        assert f.has_any()

    def test_has_any_with_institutions(self):
        f = MetadataFilters(institutions=["MIT"])
        assert f.has_any()

    def test_has_any_with_certifications(self):
        f = MetadataFilters(certifications=["AWS Certified Solutions Architect"])
        assert f.has_any()

    def test_has_any_with_languages(self):
        f = MetadataFilters(languages=["German"])
        assert f.has_any()

    def test_has_any_with_email(self):
        f = MetadataFilters(email="ana@example.com")
        assert f.has_any()

    def test_has_any_with_section(self):
        f = MetadataFilters(section="experience")
        assert f.has_any()

    def test_has_any_with_min_years(self):
        f = MetadataFilters(min_years_of_experience=3)
        assert f.has_any()

    def test_has_any_with_max_years(self):
        f = MetadataFilters(max_years_of_experience=10)
        assert f.has_any()

    def test_has_any_false_with_zero_min_years(self):
        # 0 is falsy in Python but None is the sentinel — 0 should count as set
        f = MetadataFilters(min_years_of_experience=0)
        assert f.has_any()  # `is not None` makes 0 valid

    def test_default_skills_is_empty_list(self):
        f = MetadataFilters()
        assert f.skills == []
        assert f.companies == []
        assert f.degrees == []
        assert f.institutions == []
        assert f.certifications == []
        assert f.languages == []


class TestQueryRewrite:
    def test_valid_rewrite(self):
        qr = QueryRewrite(standalone_query="Who knows Kubernetes?")
        assert qr.standalone_query == "Who knows Kubernetes?"

    def test_empty_query_allowed(self):
        qr = QueryRewrite(standalone_query="")
        assert qr.standalone_query == ""

    def test_model_validate_json(self):
        qr = QueryRewrite.model_validate_json(
            '{"standalone_query": "Find Python developers"}'
        )
        assert qr.standalone_query == "Find Python developers"


class TestIntentDecision:
    def test_default_metadata_filters(self):
        d = IntentDecision(route=IntentRoute.HYBRID)
        assert isinstance(d.metadata_filters, MetadataFilters)
        assert not d.metadata_filters.has_any()

    def test_with_filters(self):
        d = IntentDecision(
            route=IntentRoute.METADATA,
            metadata_filters=MetadataFilters(skills=["Python"]),
        )
        assert d.route == IntentRoute.METADATA
        assert d.metadata_filters.skills == ["Python"]

    def test_invalid_route_raises(self):
        with pytest.raises(ValidationError):
            IntentDecision(route="unknown_route")


class TestRetrievedChunk:
    def test_minimal_chunk(self):
        chunk = RetrievedChunk(id="abc", page_content="some text")
        assert chunk.id == "abc"
        assert chunk.page_content == "some text"
        assert chunk.metadata == {}
        assert chunk.score is None

    def test_chunk_with_score(self):
        chunk = RetrievedChunk(id="x", page_content="text", score=0.87)
        assert chunk.score == pytest.approx(0.87)

    def test_chunk_with_metadata(self):
        chunk = RetrievedChunk(
            id="1",
            page_content="content",
            metadata={"candidate_name": "Juan", "section": "experience"},
        )
        assert chunk.metadata["candidate_name"] == "Juan"


class TestCVSource:
    def test_minimal_source(self):
        src = CVSource(id="s1", source_path="/data/cv.pdf")
        assert src.id == "s1"
        assert src.source_path == "/data/cv.pdf"
        assert src.candidate_name is None
        assert src.snippet == ""
        assert src.score is None

    def test_full_source(self):
        src = CVSource(
            id="s2",
            candidate_name="María López",
            source_path="minio://cvs/maria.pdf",
            section="skills",
            score=0.95,
            snippet="Python, FastAPI, Docker",
        )
        assert src.candidate_name == "María López"
        assert src.score == pytest.approx(0.95)
        assert src.snippet == "Python, FastAPI, Docker"


class TestAssistantResponse:
    def test_minimal_response(self):
        r = AssistantResponse(answer="No results found.")
        assert r.answer == "No results found."
        assert r.sources == []

    def test_response_with_sources(self):
        src = CVSource(id="s1", source_path="/cv.pdf")
        r = AssistantResponse(answer="Found one candidate.", sources=[src])
        assert len(r.sources) == 1
        assert r.sources[0].id == "s1"
