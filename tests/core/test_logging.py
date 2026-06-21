"""Tests for core.logging — loguru setup and logger access."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from core.logging import get_logger, setup_logger


class TestSetupLogger:
    def test_configures_console_sink_only_by_default(self) -> None:
        setup_logger(level="INFO")

        captured: list[str] = []
        handler_id = logger.add(captured.append, level="INFO")
        try:
            logger.bind(name=__name__).info("hello")
        finally:
            logger.remove(handler_id)

        assert any("hello" in message for message in captured)

    def test_writes_log_files_when_enabled(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"

        setup_logger(level="DEBUG", log_to_file=True, log_dir=log_dir)
        logger.bind(name=__name__).debug("file sink check")

        assert (log_dir / "cv_screener.log").exists()
        assert (log_dir / "errors.log").exists()


class TestGetLogger:
    def test_binds_name_to_logger(self) -> None:
        setup_logger(level="INFO")
        bound = get_logger("my.module")

        captured: list[str] = []
        handler_id = logger.add(captured.append, format="{extra[name]} | {message}")
        try:
            bound.info("bound message")
        finally:
            logger.remove(handler_id)

        assert any("my.module | bound message" in message for message in captured)
