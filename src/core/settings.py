"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from minio.helpers import check_bucket_name
from pydantic import AliasChoices, Field, PositiveInt, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvSettings(BaseSettings):
    """Shared ``.env``-backed configuration base for all settings namespaces."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class AppSettings(_EnvSettings):
    """Runtime metadata and API-facing application settings."""

    name: str = Field(default="RAG CV Platform", validation_alias="RAG_APP_NAME")
    version: str = Field(default="0.1.0", validation_alias="RAG_APP_VERSION")
    debug: bool = Field(default=True, validation_alias="RAG_DEBUG")
    host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("RAG_HOST", "HOST"),
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("RAG_PORT", "PORT"),
    )
    api_prefix: str = Field(
        default="/api/v1",
        validation_alias=AliasChoices("RAG_API_PREFIX", "API_PREFIX"),
    )
    cors_origins_str: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("RAG_CORS_ORIGINS", "CORS_ORIGINS"),
    )

    @property
    def cors_origins(self) -> list[str]:
        """Return comma-separated CORS origins as a cleaned list."""
        return [
            origin.strip()
            for origin in self.cors_origins_str.split(",")
            if origin.strip()
        ]


class LoggingSettings(_EnvSettings):
    """Loguru logging configuration loaded from ``LOG_*`` variables."""

    level: Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(default="INFO", validation_alias="LOG_LEVEL")

    to_file: bool = Field(default=False, validation_alias="LOG_TO_FILE")
    log_dir: Path = Field(default=Path("logs"), validation_alias="LOG_DIR")

    @model_validator(mode="after")
    def normalize_paths(self) -> "LoggingSettings":
        """Expand user markers in configured filesystem paths."""
        self.log_dir = self.log_dir.expanduser()
        return self


class MinIOSettings(_EnvSettings):
    """MinIO connection and default artifact bucket settings."""

    endpoint: str = Field(default="localhost:9100", validation_alias="MINIO_ENDPOINT")
    access_key: str = Field(default="minioadmin", validation_alias="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="minioadmin", validation_alias="MINIO_SECRET_KEY")
    secure: bool = Field(default=False, validation_alias="MINIO_SECURE")
    bucket: str = Field(default="cvs", validation_alias="MINIO_BUCKET")

    @field_validator("bucket")
    @classmethod
    def validate_bucket_name(cls, value: str) -> str:
        """Validate the bucket using MinIO's own naming rules."""
        check_bucket_name(value)
        return value


class QdrantSettings(_EnvSettings):
    """Qdrant vector database endpoint configuration."""

    url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "CV_QDRANT_URL"),
    )


class CVIngestionSettings(_EnvSettings):
    """CV ingestion collection and debug artifact settings."""

    collection_name: str = Field(
        default="cvs_collection",
        validation_alias="CV_INGESTION_COLLECTION_NAME",
    )
    chunks_debug_dir: Path = Field(
        default=Path("outputs/ingestion_chunks"),
        validation_alias="CV_INGESTION_CHUNKS_DEBUG_DIR",
    )
    write_debug_chunks: bool = Field(
        default=True,
        validation_alias="CV_INGESTION_WRITE_DEBUG_CHUNKS",
    )

    @model_validator(mode="after")
    def normalize_paths(self) -> "CVIngestionSettings":
        """Expand the debug chunk output directory."""
        self.chunks_debug_dir = self.chunks_debug_dir.expanduser()
        return self


class LLMSettings(_EnvSettings):
    """LiteLLM-compatible chat model configuration."""

    default_model: str = Field(
        default="default-llm",
        validation_alias=AliasChoices("CV_LLM_DEFAULT_MODEL", "CV_LLM_MODEL"),
    )
    quality_model: str = Field(
        default="quality-llm",
        validation_alias="CV_LLM_QUALITY_MODEL",
    )
    litellm_base_url: str = Field(
        default="http://localhost:4000",
        validation_alias="CV_LITELLM_BASE_URL",
    )

    @property
    def model(self) -> str:
        """Backward-compatible default chat model name."""
        return self.default_model


class ImageGenerationSettings(_EnvSettings):
    """Hugging Face profile image generation settings and bounds."""

    hf_api_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CV_HF_API_TOKEN", "HF_API_TOKEN"),
    )
    width: int = Field(default=512, ge=128, le=1024, validation_alias="CV_IMAGE_WIDTH")
    height: int = Field(
        default=512, ge=128, le=1024, validation_alias="CV_IMAGE_HEIGHT"
    )
    guidance: float = Field(
        default=3.5, ge=1.0, le=20.0, validation_alias="CV_IMAGE_GUIDANCE"
    )
    steps: int = Field(default=28, ge=1, le=100, validation_alias="CV_IMAGE_STEPS")


class CVFactorySettings(_EnvSettings):
    """Synthetic CV generation defaults for CLI and pipeline runs."""

    count: PositiveInt = Field(default=1, validation_alias="CV_COUNT")
    output_dir: Path = Field(default=Path("outputs"), validation_alias="CV_OUTPUT_DIR")
    seed: int | None = Field(default=None, validation_alias="CV_SEED")

    @model_validator(mode="after")
    def normalize_paths(self) -> "CVFactorySettings":
        """Expand the generated CV artifact output directory."""
        self.output_dir = self.output_dir.expanduser()
        return self


class Settings:
    """Root settings container.

    Keeps a single application entry point while separating configuration domains.
    """

    def __init__(self) -> None:
        """Instantiate every configuration namespace from environment variables."""
        self.app = AppSettings()
        self.logging = LoggingSettings()
        self.minio = MinIOSettings()
        self.qdrant = QdrantSettings()
        self.cv_ingestion = CVIngestionSettings()
        self.llm = LLMSettings()
        self.image_generation = ImageGenerationSettings()
        self.cv_factory = CVFactorySettings()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


@lru_cache
def get_minio_settings() -> MinIOSettings:
    """Return cached MinIO settings."""
    return get_settings().minio


@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    """Return cached Qdrant settings."""
    return get_settings().qdrant


@lru_cache
def get_cv_ingestion_settings() -> CVIngestionSettings:
    """Return cached CV ingestion settings."""
    return get_settings().cv_ingestion


@lru_cache
def get_cv_factory_settings() -> CVFactorySettings:
    """Return cached CV factory settings."""
    return get_settings().cv_factory


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Return cached LLM settings."""
    return get_settings().llm


@lru_cache
def get_image_generation_settings() -> ImageGenerationSettings:
    """Return cached image generation settings."""
    return get_settings().image_generation


__all__ = [
    "AppSettings",
    "LoggingSettings",
    "MinIOSettings",
    "QdrantSettings",
    "CVIngestionSettings",
    "LLMSettings",
    "ImageGenerationSettings",
    "CVFactorySettings",
    "Settings",
    "get_settings",
    "get_minio_settings",
    "get_qdrant_settings",
    "get_cv_ingestion_settings",
    "get_cv_factory_settings",
    "get_llm_settings",
    "get_image_generation_settings",
]
