"""Tests for core.settings — environment-backed configuration namespaces.

Every settings field that declares a ``validation_alias`` must be exercised
through its environment variable (not a constructor kwarg): pydantic-settings
only binds aliased fields by alias, and ``extra="ignore"`` silently drops an
unrecognized field-name kwarg instead of raising.
"""

from __future__ import annotations

import pytest

from core.settings import (
    AppSettings,
    CVFactorySettings,
    CVIngestionSettings,
    ImageGenerationSettings,
    LLMSettings,
    LoggingSettings,
    MinIOSettings,
    QdrantSettings,
    Settings,
    get_settings,
)


class TestAppSettings:
    def test_defaults(self) -> None:
        settings = AppSettings()

        assert settings.name == "RAG CV Platform"
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.api_prefix == "/api/v1"

    def test_cors_origins_splits_and_strips_comma_separated_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_CORS_ORIGINS", "http://a.test, http://b.test ,,")

        assert AppSettings().cors_origins == ["http://a.test", "http://b.test"]

    def test_port_rejects_out_of_range_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_PORT", "70000")

        with pytest.raises(ValueError):
            AppSettings()

    def test_host_accepts_legacy_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "0.0.0.0")

        assert AppSettings().host == "0.0.0.0"


class TestLoggingSettings:
    def test_defaults(self) -> None:
        settings = LoggingSettings()

        assert settings.level == "INFO"
        assert settings.to_file is False

    def test_expands_user_marker_in_log_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_DIR", "~/logs")

        assert "~" not in str(LoggingSettings().log_dir)


class TestMinIOSettings:
    def test_defaults(self) -> None:
        settings = MinIOSettings()

        assert settings.endpoint == "localhost:9100"
        assert settings.bucket == "cvs"

    def test_rejects_invalid_bucket_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIO_BUCKET", "ab")  # below the 3-character minimum

        with pytest.raises(ValueError):
            MinIOSettings()

    def test_accepts_valid_bucket_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIO_BUCKET", "my-valid-bucket")

        assert MinIOSettings().bucket == "my-valid-bucket"


class TestQdrantSettings:
    def test_default_url(self) -> None:
        assert QdrantSettings().url == "http://localhost:6333"

    def test_accepts_legacy_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CV_QDRANT_URL", "http://legacy:6333")

        assert QdrantSettings().url == "http://legacy:6333"


class TestCVIngestionSettings:
    def test_defaults(self) -> None:
        settings = CVIngestionSettings()

        assert settings.collection_name == "cvs_collection"
        assert settings.write_debug_chunks is True

    def test_expands_user_marker_in_chunks_debug_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CV_INGESTION_CHUNKS_DEBUG_DIR", "~/chunks")

        assert "~" not in str(CVIngestionSettings().chunks_debug_dir)


class TestLLMSettings:
    def test_defaults_match_litellm_config(self) -> None:
        settings = LLMSettings()

        assert settings.default_model == "default-llm"
        assert settings.quality_model == "quality-llm"
        assert settings.model == "default-llm"

    def test_accepts_legacy_model_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CV_LLM_MODEL", "legacy-default")

        settings = LLMSettings()

        assert settings.default_model == "legacy-default"
        assert settings.model == "legacy-default"

    def test_accepts_model_tier_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CV_LLM_DEFAULT_MODEL", "router-model")
        monkeypatch.setenv("CV_LLM_QUALITY_MODEL", "chat-model")

        settings = LLMSettings()

        assert settings.default_model == "router-model"
        assert settings.quality_model == "chat-model"


class TestImageGenerationSettings:
    def test_defaults(self) -> None:
        settings = ImageGenerationSettings()

        assert settings.width == 512
        assert settings.height == 512
        assert settings.steps == 28

    def test_accepts_token_via_legacy_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Importing chainlit elsewhere in the suite triggers its own
        # load_dotenv(), which can leak CV_HF_API_TOKEN into the real
        # environment. Clear it so the legacy alias is unambiguous here.
        monkeypatch.delenv("CV_HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HF_API_TOKEN", "hf_legacy_token")

        assert ImageGenerationSettings().hf_api_token == "hf_legacy_token"

    @pytest.mark.parametrize("value", ["64", "2048"])
    def test_rejects_out_of_bounds_width(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("CV_IMAGE_WIDTH", value)

        with pytest.raises(ValueError):
            ImageGenerationSettings()


class TestCVFactorySettings:
    def test_defaults(self) -> None:
        settings = CVFactorySettings()

        assert settings.count == 1
        assert settings.seed is None

    def test_rejects_non_positive_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CV_COUNT", "0")

        with pytest.raises(ValueError):
            CVFactorySettings()

    def test_expands_user_marker_in_output_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CV_OUTPUT_DIR", "~/outputs")

        assert "~" not in str(CVFactorySettings().output_dir)


class TestSettings:
    def test_instantiates_every_namespace(self) -> None:
        settings = Settings()

        assert isinstance(settings.app, AppSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.minio, MinIOSettings)
        assert isinstance(settings.qdrant, QdrantSettings)
        assert isinstance(settings.cv_ingestion, CVIngestionSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.image_generation, ImageGenerationSettings)
        assert isinstance(settings.cv_factory, CVFactorySettings)


class TestGetSettings:
    def test_returns_cached_instance(self) -> None:
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
