"""
Logging setup for query-cvs-with-ai.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_LEVEL = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

# Colour palette aligned with each severity
_LEVEL_COLOURS: dict[str, str] = {
    "TRACE": "<dim>",
    "DEBUG": "<cyan>",
    "INFO": "<blue>",
    "SUCCESS": "<green>",
    "WARNING": "<yellow>",
    "ERROR": "<red>",
    "CRITICAL": "<bold><red>",
}

# ──────────────────────────────────────────────────────────────────────────────
# Format strings
# ──────────────────────────────────────────────────────────────────────────────

_CONSOLE_FORMAT = (
    "<dim>{time:HH:mm:ss}</dim> "
    "│ <level>{level: <8}</level> "
    "│ <dim>{name}</dim> "
    "▸ {message}"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} "
    "| {level: <8} "
    "| {name}:{function}:{line} "
    "- {message}"
)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────


def setup_logger(
    level: str = "DEBUG",
    log_to_file: bool = False,
    log_dir: Path = LOG_DIR,
) -> None:
    """Configure the global loguru logger.

    Call this once at startup (Singleton pattern).

    """
    logger.remove()  # Drop loguru's default handler

    logger.add(
        sys.stdout,
        format=_CONSOLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_dir / "cv_screener.log",
            format=_FILE_FORMAT,
            level="DEBUG",
            rotation="00:00",
            retention="7 days",
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=False,
        )

        logger.add(
            log_dir / "errors.log",
            format=_FILE_FORMAT,
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=False,
        )

    logger.debug("Logger initialised | level={} | files={}", level, log_to_file)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_logger(name: str):
    """Return a logger bound to *name* (typically ``__name__``)."""
    return logger.bind(name=name)
