"""Tests for shared/logging_utils.py – Structured logging utilities."""

import json
import logging
import sys
import pytest

from shared.logging_utils import JSONFormatter, setup_logging, log_with_context, get_agent_logger


# ── JSONFormatter ──────────────────────────────────────────────────────────


class TestJSONFormatter:
    def test_basic_record(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Hello world",
            args=(),
            exc_info=None,
        )
        data = json.loads(fmt.format(record))

        assert data["level"] == "INFO"
        assert data["message"] == "Hello world"
        assert data["logger"] == "test_logger"
        assert "timestamp" in data

    def test_exception_included(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        data = json.loads(fmt.format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_extra_fields_merged(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="task done",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"task_id": "t001", "duration": 5.2}

        data = json.loads(fmt.format(record))
        assert data["task_id"] == "t001"
        assert data["duration"] == 5.2


# ── setup_logging ──────────────────────────────────────────────────────────


class TestSetupLogging:
    def test_returns_logger(self, tmp_path, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "project_root", tmp_path)

        logger = setup_logging("test_agent")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_agent"

    def test_log_level_override(self, tmp_path, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "project_root", tmp_path)

        logger = setup_logging("test_debug", log_level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_creates_agent_log_directory(self, tmp_path, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "project_root", tmp_path)

        setup_logging("test_logdir")
        assert (tmp_path / "logs" / "test_logdir").is_dir()

    def test_non_json_formatting(self, tmp_path, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "project_root", tmp_path)

        logger = setup_logging("test_plain", use_json=False)
        assert len(logger.handlers) >= 1
        # At least one handler should use a standard Formatter (not JSON)
        has_standard = any(
            not isinstance(h.formatter, JSONFormatter)
            for h in logger.handlers
            if h.formatter is not None
        )
        assert has_standard


# ── log_with_context ───────────────────────────────────────────────────────


class TestLogWithContext:
    def test_does_not_raise(self, tmp_path, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "project_root", tmp_path)

        logger = setup_logging("ctx_test")
        log_with_context(logger, "info", "Task started", task_id="t001", agent="dq")


# ── get_agent_logger ───────────────────────────────────────────────────────


class TestGetAgentLogger:
    def test_returns_named_logger(self, tmp_path, monkeypatch):
        from shared.config import settings

        monkeypatch.setattr(settings, "project_root", tmp_path)

        logger = get_agent_logger("my_agent")
        assert logger.name == "my_agent"
