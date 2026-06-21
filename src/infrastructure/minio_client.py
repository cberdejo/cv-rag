"""Shared MinIO client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from posixpath import basename

from core.settings import MinIOSettings, get_minio_settings
from minio import Minio

SUPPORTED_CV_SUFFIXES = frozenset({".pdf"})


@dataclass(frozen=True)
class DownloadedArtifact:
    """A MinIO object staged locally with its original object identity."""

    local_path: Path
    object_name: str
    uri: str


@lru_cache
def get_minio_client(endpoint: str | None = None) -> Minio:
    """Return a cached MinIO client, optionally overriding the configured endpoint."""
    settings = get_minio_settings()
    return Minio(
        endpoint=endpoint or settings.endpoint,
        access_key=settings.access_key,
        secret_key=settings.secret_key,
        secure=settings.secure,
    )


def ensure_bucket(
    *,
    client: Minio | None = None,
    bucket: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Create the configured bucket when it does not exist and return its name."""
    settings = get_minio_settings()
    bucket_name = bucket or settings.bucket
    minio_client = client or get_minio_client(endpoint)

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    return bucket_name


def upload_file(
    path: Path,
    *,
    object_name: str,
    content_type: str | None = None,
    client: Minio | None = None,
    bucket: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Upload a file to MinIO and return its object URI."""
    minio_client = client or get_minio_client(endpoint)
    bucket_name = ensure_bucket(client=minio_client, bucket=bucket)

    kwargs = {
        "bucket_name": bucket_name,
        "object_name": object_name,
        "file_path": str(path),
    }
    if content_type is not None:
        kwargs["content_type"] = content_type

    minio_client.fput_object(**kwargs)
    return f"minio://{bucket_name}/{object_name}"


def list_artifact_objects(
    *,
    bucket: str,
    prefix: str = "",
    client: Minio | None = None,
    supported_suffixes: set[str] | frozenset[str] | None = None,
    endpoint: str | None = None,
) -> list[str]:
    """Return supported artifact object names from a MinIO bucket."""
    minio_client = client or get_minio_client(endpoint)

    if not minio_client.bucket_exists(bucket):
        raise FileNotFoundError(f"MinIO bucket does not exist: {bucket}")

    suffixes = supported_suffixes or SUPPORTED_CV_SUFFIXES
    objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)

    return sorted(
        object_name
        for obj in objects
        if (object_name := obj.object_name) is not None
        and Path(object_name).suffix.lower() in suffixes
    )


def download_artifact_sources(
    *,
    bucket: str,
    target_dir: Path,
    prefix: str = "",
    client: Minio | None = None,
    supported_suffixes: set[str] | frozenset[str] | None = None,
    filename_prefix: str = "",
    endpoint: str | None = None,
) -> list[DownloadedArtifact]:
    """
    Download supported MinIO artifacts into ``target_dir``.

    Args:
        bucket: MinIO bucket name.
        target_dir: Local directory where objects will be downloaded.
        prefix: Optional MinIO object prefix to filter objects.
        client: Optional MinIO client, useful for tests.
        supported_suffixes: File suffixes to include. Defaults to PDF.
        filename_prefix: Prefix added to downloaded local filenames. Useful when
            combining local and MinIO artifacts in the same staging directory.
        endpoint: Optional MinIO endpoint override (``host:port``).

    Returns:
        Downloaded artifacts with local path and durable ``minio://`` URI.
    """
    minio_client = client or get_minio_client(endpoint)
    object_names = list_artifact_objects(
        bucket=bucket,
        prefix=prefix,
        client=minio_client,
        supported_suffixes=supported_suffixes,
    )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Could not create MinIO download directory: {target_dir}"
        ) from exc

    downloaded: list[DownloadedArtifact] = []
    used_names: set[str] = set()

    for object_name in object_names:
        filename = _download_filename(
            object_name=object_name,
            used_names=used_names,
            filename_prefix=filename_prefix,
        )
        target_path = target_dir / filename

        minio_client.fget_object(
            bucket,
            object_name,
            str(target_path),
        )

        downloaded.append(
            DownloadedArtifact(
                local_path=target_path,
                object_name=object_name,
                uri=f"minio://{bucket}/{object_name}",
            )
        )

    return downloaded


def _download_filename(
    *,
    object_name: str,
    used_names: set[str],
    filename_prefix: str = "",
) -> str:
    """
    Build a unique flat filename for a downloaded MinIO object.

    Keeps the basename when possible:
        nested/a.pdf -> a.pdf

    If there is a collision, it falls back to a path-based flat name:
        nested/a.pdf -> nested__a.pdf

    If that still collides, appends a numeric suffix.
    """
    base_name = basename(object_name.rstrip("/")) or Path(object_name).name
    base_name = f"{filename_prefix}{base_name}"

    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    flattened = object_name.strip("/").replace("/", "__")
    candidate = f"{filename_prefix}{flattened}"

    suffix = Path(candidate).suffix
    stem = Path(candidate).stem
    counter = 1

    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate


__all__ = [
    "DownloadedArtifact",
    "MinIOSettings",
    "SUPPORTED_CV_SUFFIXES",
    "download_artifact_sources",
    "ensure_bucket",
    "get_minio_client",
    "get_minio_settings",
    "list_artifact_objects",
    "upload_file",
]
