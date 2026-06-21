"""
Integration tests for run_pipeline.

All external I/O is mocked:
  - LLM enrichment (OpenAI client)
  - Profile image generation (HTTP call to Hugging Face)
  - WeasyPrint PDF rendering

The goal is to verify the pipeline orchestration logic — task sequencing,
output structure, artifact storage routing — without hitting real services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cv_factory.models.cv import CV
from cv_factory.pipeline.pipeline import ArtifactStorage, PipelineResult, run_pipeline

pytestmark = pytest.mark.anyio


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def pipeline_patches(tmp_path):
    """
    Patch all external I/O used by run_pipeline.

    Yields a dict of the mock objects for assertion inside tests.
    """
    # We need the real build_cv_base to run to obtain a valid CV,
    # then we mock the LLM client used by LLMEnrichmentService.

    # Patch WeasyPrint so no PDF is actually rendered
    with patch("cv_factory.pipeline.tasks.render_cv.main.HTML") as mock_html:
        mock_html.return_value.write_pdf = MagicMock()

        async def enrich_without_llm(cv: CV) -> CV:
            return cv.model_copy(
                update={"about": "Enriched CV summary for orchestration tests."}
            )

        with patch(
            "cv_factory.pipeline.pipeline.enrich_cv",
            new=AsyncMock(side_effect=enrich_without_llm),
        ):
            yield {
                "mock_html": mock_html,
                "tmp_path": tmp_path,
            }


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRunPipelineLocal:
    async def test_returns_pipeline_result(self, pipeline_patches):
        result = await run_pipeline(
            seed=1,
            output_dir=pipeline_patches["tmp_path"],
            skip_image=True,
            artifact_storage=ArtifactStorage.LOCAL,
        )
        assert isinstance(result, PipelineResult)

    async def test_result_contains_enriched_cv(self, pipeline_patches):
        result = await run_pipeline(
            seed=2,
            output_dir=pipeline_patches["tmp_path"],
            skip_image=True,
            artifact_storage=ArtifactStorage.LOCAL,
        )
        assert isinstance(result.cv, CV)

    async def test_local_artifacts_have_pdf(self, pipeline_patches):
        result = await run_pipeline(
            seed=3,
            output_dir=pipeline_patches["tmp_path"],
            skip_image=True,
            artifact_storage=ArtifactStorage.LOCAL,
        )
        kinds = {a.kind for a in result.artifacts}
        assert "pdf" in kinds

    async def test_local_artifacts_have_no_minio_uri(self, pipeline_patches):
        result = await run_pipeline(
            seed=4,
            output_dir=pipeline_patches["tmp_path"],
            skip_image=True,
            artifact_storage=ArtifactStorage.LOCAL,
        )
        for artifact in result.artifacts:
            assert artifact.minio_uri is None

    async def test_pdf_renderer_is_called(self, pipeline_patches):
        tmp = pipeline_patches["tmp_path"]
        await run_pipeline(
            seed=5,
            output_dir=tmp,
            skip_image=True,
            artifact_storage=ArtifactStorage.LOCAL,
        )
        pipeline_patches["mock_html"].return_value.write_pdf.assert_called_once()

    async def test_different_seeds_produce_different_cv_names(self, pipeline_patches):
        tmp = pipeline_patches["tmp_path"]
        r1 = await run_pipeline(seed=10, output_dir=tmp, skip_image=True)
        r2 = await run_pipeline(seed=99, output_dir=tmp, skip_image=True)
        # Very unlikely to be equal across different seeds
        assert r1.cv.name != r2.cv.name or r1.cv.email != r2.cv.email

    async def test_same_seed_produces_same_cv_name(self, pipeline_patches):
        tmp = pipeline_patches["tmp_path"]
        r1 = await run_pipeline(seed=55, output_dir=tmp, skip_image=True)
        r2 = await run_pipeline(seed=55, output_dir=tmp, skip_image=True)
        assert r1.cv.name == r2.cv.name
