"""Chainlit frontend for the CV RAG assistant."""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from core.logging import get_logger
from cv_rag.api.v1.services import get_assistant
from infrastructure.minio_client import get_minio_client
from rag.assistant import AssistantResponse, CVSource

import chainlit as cl  # noqa: E402

logger = get_logger(__name__)

CV_PREVIEW_CACHE_DIR = Path(gettempdir()) / "cv-rag-chainlit-cv-previews"


class CVPreview:
    """Local PDF artifact prepared for Chainlit preview/download elements."""

    def __init__(self, *, path: Path, filename: str) -> None:
        """Store the local path and display filename for a CV preview."""
        self.path = path
        self.filename = filename


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize a Chainlit chat session."""
    cl.user_session.set("thread_id", str(uuid4()))
    await cl.Message(
        content=(
            "Ask about candidates, experience, skills, or companies from the indexed CVs."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer user messages through the shared CV assistant service."""
    thread_id = cl.user_session.get("thread_id") or str(uuid4())
    cl.user_session.set("thread_id", thread_id)

    assistant = get_assistant()
    response_message = cl.Message(content="")
    streamed_token = False

    async def stream_token(token: str) -> None:
        """Forward one streamed assistant token to the Chainlit response."""
        nonlocal streamed_token
        streamed_token = True
        await response_message.stream_token(token)

    try:
        response = await assistant.ask_stream(
            message.content,
            thread_id=thread_id,
            on_token=stream_token,
        )
    except Exception:
        error_content = "I couldn't access the assistant at this time. Please make sure Qdrant and LiteLLM are running and that CVs are indexed."
        if streamed_token:
            response_message.content = error_content
            await response_message.update()
        else:
            await cl.Message(content=error_content).send()
        return

    elements = await build_cv_source_elements(response)
    answer_content = append_cv_source_links(response.answer, elements)

    if streamed_token:
        response_message.content = answer_content
        response_message.elements = elements
        await response_message.update()
    else:
        await cl.Message(content=answer_content, elements=elements).send()


async def build_cv_source_elements(
    response: AssistantResponse,
    *,
    thread_id: str | None = None,
) -> list[Any]:
    """Build Chainlit elements that open retrieved CV PDFs as single-document pages."""
    elements: list[Any] = []
    unresolved_sources: list[CVSource] = []
    element_kwargs = {"thread_id": thread_id} if thread_id else {}
    used_names: set[str] = set()

    for source in response.sources:
        preview = await resolve_cv_preview(source.source_path)
        if preview is None:
            unresolved_sources.append(source)
            continue

        name = _unique_element_name(
            f"CV - {_source_label(source, preview.filename)}",
            used_names,
        )
        elements.append(
            cl.Pdf(
                name=name,
                path=str(preview.path),
                display="page",
                page=1,
                **element_kwargs,
            )
        )

    if unresolved_sources:
        elements.append(
            cl.Text(
                name="CVs without preview",
                content=_format_unresolved_sources(unresolved_sources),
                display="inline",
                **element_kwargs,
            )
        )

    return elements


def append_cv_source_links(answer: str, elements: list[Any]) -> str:
    """Append page-preview links for PDF source elements to a Chainlit answer."""
    pdf_names = [element.name for element in elements if isinstance(element, cl.Pdf)]
    if not pdf_names:
        return answer

    document_links = "\n".join(f"- {name}" for name in pdf_names)
    return f"{answer.rstrip()}\n\nDocuments:\n{document_links}"


async def resolve_cv_preview(source_path: str) -> CVPreview | None:
    """Resolve PDF source path to a local file for use in Chainlit."""
    normalized_source = source_path.strip()
    if not normalized_source:
        return None

    minio_ref = parse_minio_uri(normalized_source)
    if minio_ref is not None:
        bucket, object_name = minio_ref
        if Path(object_name).suffix.lower() != ".pdf":
            return None
        cache_path = _cache_path_for_minio_object(bucket, object_name)
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                await asyncio.to_thread(
                    get_minio_client().fget_object,
                    bucket,
                    object_name,
                    str(cache_path),
                )
            except Exception as exc:
                logger.warning(
                    "Could not download CV preview artifact '{}' from MinIO: {}",
                    normalized_source,
                    exc,
                )
                return None
        return CVPreview(path=cache_path, filename=Path(object_name).name)

    local_path = Path(normalized_source).expanduser()
    if local_path.is_file() and local_path.suffix.lower() == ".pdf":
        return CVPreview(path=local_path.resolve(), filename=local_path.name)

    return None


def parse_minio_uri(uri: str) -> tuple[str, str] | None:
    """Parse 'minio://bucket/object' style URIs for MinIO references."""
    parsed = urlparse(uri)
    if parsed.scheme != "minio" or not parsed.netloc:
        return None

    object_name = unquote(parsed.path.lstrip("/"))
    if not object_name:
        return None

    return parsed.netloc, object_name


def _cache_path_for_minio_object(bucket: str, object_name: str) -> Path:
    """Return the deterministic local cache path for a MinIO PDF object."""
    source_key = f"{bucket}/{object_name}"
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
    filename = Path(object_name).name or "cv.pdf"
    stem = _safe_filename(Path(filename).stem) or "cv"
    return CV_PREVIEW_CACHE_DIR / f"{stem}-{digest}.pdf"


def _source_label(source: CVSource, fallback_filename: str) -> str:
    """Choose a readable label for a retrieved CV source."""
    candidate = (source.candidate_name or "").strip()
    if candidate:
        return candidate
    return Path(fallback_filename).stem or source.id


def _format_unresolved_sources(sources: list[CVSource]) -> str:
    """Format sources that could not be resolved into local PDF previews."""
    blocks: list[str] = []
    for source in sources:
        blocks.append(
            "\n".join(
                [
                    f"{source.id}: {source.candidate_name or 'Unknown candidate'}",
                    f"CV not available for preview: {source.source_path}",
                    f"Section: {source.section or 'N/A'}",
                    f"Score: {source.score if source.score is not None else 'N/A'}",
                    source.snippet,
                ]
            )
        )
    return "\n\n".join(blocks)


def _safe_filename(value: str) -> str:
    """Replace unsafe filename characters with hyphens."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")


def _unique_element_name(name: str, used_names: set[str]) -> str:
    """Return a Chainlit element name that has not been used in this response."""
    candidate = name
    suffix = 2
    while candidate in used_names:
        candidate = f"{name} ({suffix})"
        suffix += 1
    used_names.add(candidate)
    return candidate
