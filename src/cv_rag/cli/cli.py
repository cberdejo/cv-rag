"""Global CLI entry point for cv-rag."""

from __future__ import annotations

import typer

from cv_rag.api.application import serve_command
from cv_factory.cli import generate_cv_command
from cv_ingestion.cli import ingest_command


app = typer.Typer(
    name="cv-rag",
    help="Generate synthetic CVs and ingest them into the RAG index.",
    add_completion=False,
    no_args_is_help=True,
)


app.command("generate-cv")(generate_cv_command)
app.command("ingest-cv")(ingest_command)
app.command("serve")(serve_command)


if __name__ == "__main__":
    app()
