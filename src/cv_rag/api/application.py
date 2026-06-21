"""ASGI application and server runner for cv-rag."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import typer
import uvicorn

from core.logging import setup_logger
from core.settings import get_settings
from cv_rag.api.v1.api import api_router


APP_IMPORT_PATH = "cv_rag.api.application:app"
CHAINLIT_PATH = "/chainlit"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure process-level services on startup."""
    settings = get_settings()
    setup_logger(
        level=settings.logging.level,
        log_to_file=settings.logging.to_file,
        log_dir=settings.logging.log_dir,
    )
    yield


def create_app(*, mount_frontend: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
        docs_url=f"{settings.app.api_prefix}/docs",
        redoc_url=f"{settings.app.api_prefix}/redoc",
        openapi_url=f"{settings.app.api_prefix}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Return service entry points for API docs and the mounted chat UI."""
        return {
            "api": settings.app.api_prefix,
            "docs": f"{settings.app.api_prefix}/docs",
            "chainlit": CHAINLIT_PATH,
        }

    app.include_router(api_router, prefix=settings.app.api_prefix)

    if mount_frontend:
        from chainlit.utils import mount_chainlit
        from rag.frontend import chainlit_app as chainlit_frontend

        chainlit_target = Path(chainlit_frontend.__file__)
        mount_chainlit(app=app, target=str(chainlit_target), path=CHAINLIT_PATH)
        _mount_chainlit_assets_at_root(app)

    return app


def _mount_chainlit_assets_at_root(app: FastAPI) -> None:
    """Expose Chainlit frontend assets at /assets for hardcoded worker URLs."""
    from chainlit import server as chainlit_server

    assets_dir = Path(chainlit_server.build_dir) / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="chainlit-assets",
        )


def run_application(
    *,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """Run cv-rag with uvicorn."""
    settings = get_settings()
    uvicorn.run(
        APP_IMPORT_PATH,
        host=host if host is not None else settings.app.host,
        port=port if port is not None else settings.app.port,
        reload=reload,
    )


def serve_command(
    host: Annotated[
        str | None,
        typer.Option(help="Host to bind. Defaults to settings."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(min=1, max=65535, help="Port to bind. Defaults to settings."),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(help="Reload on source changes."),
    ] = False,
) -> None:
    """Run the FastAPI API with the Chainlit frontend mounted at /chainlit."""
    run_application(host=host, port=port, reload=reload)


app = create_app()


if __name__ == "__main__":
    run_application()


__all__ = [
    "APP_IMPORT_PATH",
    "CHAINLIT_PATH",
    "app",
    "create_app",
    "run_application",
    "serve_command",
]
