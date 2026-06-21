"""Tests for cv_factory.pipeline.batch_pipeline — async batch orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from cv_factory.pipeline.batch_pipeline import run_batch_pipeline
from cv_factory.pipeline.result import BatchResult

pytestmark = pytest.mark.anyio


class TestRunBatchPipelineSuccess:
    async def test_returns_all_succeeded_when_no_failures(self) -> None:
        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(return_value=None),
        ):
            result = await run_batch_pipeline(count=3)

        assert result == BatchResult(total=3, succeeded=3, failed=0, errors=[])

    async def test_passes_sequential_seeds_from_base_seed(self) -> None:
        seeds_used: list[int | None] = []

        async def fake_run_pipeline(*, seed, **kwargs):
            seeds_used.append(seed)

        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(side_effect=fake_run_pipeline),
        ):
            await run_batch_pipeline(count=3, base_seed=10)

        assert sorted(seeds_used) == [10, 11, 12]

    async def test_uses_none_seeds_when_base_seed_not_given(self) -> None:
        seeds_used: list[int | None] = []

        async def fake_run_pipeline(*, seed, **kwargs):
            seeds_used.append(seed)

        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(side_effect=fake_run_pipeline),
        ):
            await run_batch_pipeline(count=2)

        assert seeds_used == [None, None]

    async def test_forwards_storage_and_bucket_options(self) -> None:
        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(return_value=None),
        ) as run_pipeline_mock:
            from core.artifacts import ArtifactStorage

            await run_batch_pipeline(
                count=1,
                artifact_storage=ArtifactStorage.MINIO,
                minio_bucket="cvs-test",
            )

        run_pipeline_mock.assert_awaited_once_with(
            seed=None,
            output_dir=None,
            skip_image=False,
            artifact_storage=ArtifactStorage.MINIO,
            minio_bucket="cvs-test",
            minio_endpoint=None,
        )


class TestRunBatchPipelineFailures:
    async def test_collects_errors_without_failing_other_jobs(self) -> None:
        async def fake_run_pipeline(*, seed, **kwargs):
            if seed == 2:
                raise RuntimeError("boom")

        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(side_effect=fake_run_pipeline),
        ):
            result = await run_batch_pipeline(count=3, base_seed=1)

        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1
        assert len(result.errors) == 1
        assert isinstance(result.errors[0][1], RuntimeError)

    async def test_all_jobs_failing_reports_zero_succeeded(self) -> None:
        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await run_batch_pipeline(count=2)

        assert result.succeeded == 0
        assert result.failed == 2


class TestRunBatchPipelineConcurrency:
    async def test_never_exceeds_max_workers_concurrent_jobs(self) -> None:
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_run_pipeline(*, seed, **kwargs):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1

        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(side_effect=fake_run_pipeline),
        ):
            await run_batch_pipeline(count=10, max_workers=3)

        assert peak <= 3

    async def test_workers_bounded_by_count_when_smaller_than_max(self) -> None:
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_run_pipeline(*, seed, **kwargs):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1

        with patch(
            "cv_factory.pipeline.batch_pipeline.run_pipeline",
            new=AsyncMock(side_effect=fake_run_pipeline),
        ):
            await run_batch_pipeline(count=2, max_workers=10)

        assert peak <= 2
