# cv-rag

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Chainlit](https://img.shields.io/badge/UI-Chainlit-6E56CF)](https://chainlit.io/)
[![Qdrant](https://img.shields.io/badge/vectors-Qdrant-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1C3C3C)](https://www.langchain.com/langgraph)
[![Docker Compose](https://img.shields.io/badge/infra-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

cv-rag is a local RAG platform for synthetic CV generation, ingestion, search,
and assistant-style questioning. It can generate fake CV PDFs, store artifacts
locally or in MinIO, normalize and chunk CV content, index it in Qdrant, and
serve a FastAPI plus Chainlit interface for asking questions over the indexed
candidate database.

## Table of Contents

- [Prerequisites](#prerequisites)
- [High-Level Architecture](#high-level-architecture)
- [Modules](#modules)
- [Quick Start](#quick-start)
- [Changing LLM Models](#changing-llm-models)
- [Storage and Indexing](#storage-and-indexing)
- [Retrieval Model and Reranking Decisions](#retrieval-model-and-reranking-decisions)
- [Further Reading](#further-reading)
- [Profile Image Generation](#profile-image-generation)
- [License](#license)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose (for Ollama, LiteLLM, MinIO, and Qdrant)

## High-Level Architecture

```text
CV Factory -> Artifact Storage -> CV Ingestion -> Qdrant Index -> RAG Assistant
     |              |                  |              |              |
  Faker/LLM     local/MinIO       PDF parsing      hybrid        FastAPI
  rendering                     normalization     retrieval      Chainlit
```

## Modules

### `cv_factory`

Builds synthetic CVs from catalog profiles, enriches narrative fields with an
LLM, optionally generates profile images, renders HTML templates to PDF, and
writes artifacts to local storage, MinIO, or both.

### `cv_ingestion`

Reads generated CV artifacts, extracts PDF text, asks the LLM to normalize the
CV into structured data, chunks the normalized CV, and indexes the chunks into
Qdrant.

### `rag`

Contains retrieval and assistant logic. Retrieval uses Qdrant dense and sparse
vectors, metadata filters, and optional reranking. The assistant is a LangGraph
workflow that rewrites queries, routes intent, retrieves candidate chunks, and
generates answers with source references.

### `cv_rag.api`

Creates the FastAPI application, mounts API routes under `/api/v1`, and mounts
the Chainlit chat UI at `/chainlit`.

### `cv_rag.cli`

Provides the main CLI entry point:

```bash
cv-rag generate-cv
cv-rag ingest-cv
cv-rag serve
```

### `core`

Shared configuration, logging, and artifact storage enums. Settings are loaded
from environment variables and `.env`.

### `infrastructure`

Shared service clients for LiteLLM/OpenAI-compatible chat completions and MinIO
artifact access.

## Quick Start

Install dependencies:

```bash
uv sync
```

Create local environment settings:

```bash
cp .env.example .env
```

Start local services (Ollama, LiteLLM, MinIO, Qdrant):

```bash
docker compose up -d
```

| Service | URL |
| --- | --- |
| LiteLLM proxy | `http://localhost:4000` |
| MinIO API / console | `http://localhost:9100` / `http://localhost:9101` |
| Qdrant | `http://localhost:6333` |
| Ollama | `http://localhost:11434` |

Generate sample CVs:

```bash
uv run cv-rag generate-cv --count 5 --output-dir outputs --skip-image
```

Ingest generated CVs into Qdrant:

```bash
uv run cv-rag ingest-cv --input-dir outputs --recreate
```

Run the API and Chainlit UI:

```bash
uv run cv-rag serve --reload
```

Open:

- API docs: `http://127.0.0.1:8000/api/v1/docs`
- Chat UI: `http://127.0.0.1:8000/chainlit`

Run tests:

```bash
uv run pytest
```

## Changing LLM Models

cv-rag calls an OpenAI-compatible LiteLLM proxy. The default local setup uses
Ollama through `litellm_config.yaml`.

The application-level model names are configured with environment variables:

```env
CV_LLM_DEFAULT_MODEL=default-llm
CV_LLM_QUALITY_MODEL=quality-llm
CV_LITELLM_BASE_URL=http://localhost:4000
```

- `CV_LLM_DEFAULT_MODEL` (alias: `CV_LLM_MODEL`) is the default model used for
  routine assistant and extraction calls.
- `CV_LLM_QUALITY_MODEL` is used for higher-quality CV enrichment.
- `CV_LITELLM_BASE_URL` points the application to the LiteLLM proxy.

To change the underlying provider or model, edit `litellm_config.yaml`:

```yaml
model_list:
  - model_name: default-llm
    litellm_params:
      model: ollama/llama3:8b
      api_base: http://ollama:11434

  - model_name: quality-llm
    litellm_params:
      model: ollama/llama3:8b
      api_base: http://ollama:11434
```

Keep `model_name` aligned with `CV_LLM_DEFAULT_MODEL` and
`CV_LLM_QUALITY_MODEL`, then restart LiteLLM:

```bash
docker compose restart litellm
```

When using a different Ollama model, also pull it in the Ollama container or
adjust the `ollama-pull` service in `docker-compose.yaml`.

## Storage and Indexing

Generated PDFs can stay on disk, be uploaded to MinIO, or be written to both:

```bash
uv run cv-rag generate-cv --count 10 --storage both --bucket cvs
uv run cv-rag ingest-cv --storage minio --bucket cvs --recreate
```

The default Qdrant collection is configured with:

```env
CV_INGESTION_COLLECTION_NAME=cvs_collection
QDRANT_URL=http://localhost:6333
```

MinIO connection settings (used when `--storage minio` or `--storage both`):

```env
MINIO_ENDPOINT=localhost:9100
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET=cvs
```

## Retrieval Model and Reranking Decisions

`rag/retrieval/indexer.py` indexes each CV as one chunk per section (contact,
about, experience, education, skills, ...), so a single CV typically becomes
8-10 chunks, each one already containing a compact cross-section context
header. This shape — few, coarse, context-enriched chunks per candidate —
drove the choices below.

### Dense vector model: `intfloat/multilingual-e5-large`

[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) is trained specifically for retrieval and covers ~100 languages,
seeing [MTEB](#further-reading) `BAAI/bge-m3` would be an equally strong (or
stronger) alternative, but it is not in the supported-model list of the
`fastembed` version this project pins — revisit it if that changes.

E5 models are trained with literal `"query: "` / `"passage: "` text prefixes
that tell the model which side of the asymmetry it's embedding; skipping them
measurably hurts retrieval quality. `indexer.py` adds `E5_QUERY_PREFIX` to the
search query and `E5_PASSAGE_PREFIX` to each chunk's `page_content` before
embedding — but only for the **dense** vector. The sparse (BM25) vector keeps
the raw text, since prefixing would only add noise to lexical matching.

### Sparse vector model: `Qdrant/bm25`

[Qdrant/bm25](https://huggingface.co/Qdrant/bm25) is classic lexical search (term frequency + IDF): not
neural, but for that same reason **language-agnostic** — it does not rely on
a vocabulary learned in one language, so mixed-language CVs and queries don't
break it. If the corpus and queries were expected to be purely English,
SPLADE-style models are worth revisiting since they outperform BM25 on
English-only retrieval.

### Reranker: `jinaai/jina-reranker-v2-base-multilingual`

It's a modern, multilingual cross-encoder (278M params) that scores
well on multilingual reranking benchmarks and is fast enough to rerank dozens
of candidates per query. No need to swap it out unless a concrete quality
problem shows up in production.

### How many chunks to fetch before/after reranking

Because each CV is only ~8-10 chunks, a small prefetch pool spans very few
distinct candidates, the cross-encoder can only promote chunks that already
made it into that pool, so a narrow first-stage pool caps recall regardless of
how good the reranker is. `RERANK_PREFETCH_LIMIT` (in `indexer.py`) is set to
**50**: each of the dense/sparse branches fetches up to 50 candidates before
RRF fusion and reranking, following the common 3-6x oversampling guidance for
two-stage retrieval (see [Cohere reranking guide](#further-reading)). The
final `limit` returned to callers stays small (5 by default) since that is
what is actually shown to the user or passed to the LLM as context.

When reranking is not used (plain RRF fusion), the candidate pool equals
`limit` directly — there is no second stage to recover from a narrow pool, so
oversampling would only add latency without improving results.

### Deduplication by `source_path` — only where it makes sense

`search_hybrid` and `search_metadata_then_rerank` return the top `limit`
**chunks** by score, with no deduplication by candidate. This was tried the
other way first (collapsing to one chunk per `source_path`) and reverted: since
a CV is chunked one-per-section, the strongest match can legitimately
contribute more than one chunk (e.g. its "experience" and "skills" sections
both ranking high). Deduplicating there would drop a top candidate's second-
and third-best sections to make room for weaker, single-section matches from
other candidates — actively hurting the context passed to the LLM for no
benefit, since the chunk's `[CONTEXT]` header does not carry every section's
detail (certifications, languages, full work history, etc. are not
summarized there — see `context_summary()` in `normalized_cv.py`).

`search_metadata` is the one exception: it deduplicates by `source_path`
because its job is literally "list distinct candidates matching a filter"
(no free-text query, no ranking), so one row per CV is the correct output
shape. The citation list shown to the user (`chunks_to_sources` in
`rag/assistant/nodes.py`) is deduplicated separately, downstream — so the UI
never shows the same CV twice even though `retrieved_chunks` (the context
fed to the LLM) can.

## Further Reading

Resources used to make the model and retrieval-sizing decisions above, useful
if you want to re-evaluate them later:

| Resource | Use it to... |
| --- | --- |
| [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Compare dense embedding models (e.g. `bge-m3` vs `multilingual-e5-large` vs MiniLM) on real multilingual retrieval benchmarks before swapping `DENSE_MODEL`. |
| [Qdrant — Hybrid Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) | Understand RRF fusion and how to size `prefetch limit` relative to the final `limit`. |
| [Qdrant — Hybrid Search Revamped](https://qdrant.tech/articles/hybrid-search/) | See why the right sparse model (SPLADE vs BM25) depends on the corpus language. |
| [BGE-M3 paper](https://arxiv.org/abs/2402.03216) | Understand why `bge-m3` is multi-functional (dense + sparse + multi-vector) and multilingual by design. |
| [Cohere — Reranking Best Practices](https://docs.cohere.com/docs/reranking-best-practices) | Reference oversampling ratios (candidates fetched vs. final top-k) for any cross-encoder reranker, not just Cohere's. |
| [FastEmbed Supported Models](https://github.com/qdrant/fastembed/blob/main/docs/source/Supported_Models.md) | Look up exact, currently-supported model identifiers before changing `DENSE_MODEL` / `SPARSE_MODEL` / `RERANKER_MODEL`. |

## Profile Image Generation

Profile images are generated via a Hugging Face inference endpoint and are
skipped entirely with `--skip-image`. To enable them, set a token in `.env`:

```env
CV_HF_API_TOKEN=hf_your_token_here
CV_IMAGE_OUTPUT_DIR=outputs/images
CV_IMAGE_WIDTH=512
CV_IMAGE_HEIGHT=512
CV_IMAGE_GUIDANCE=3.5
CV_IMAGE_STEPS=28
```

See `.env.example` for the full list of available settings.

## License

Licensed under the [MIT License](LICENSE).
