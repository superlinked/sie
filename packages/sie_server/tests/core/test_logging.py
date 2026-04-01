"""Tests for structured JSON logging."""

from __future__ import annotations

import json
import logging

import pytest
from sie_server.core.logging import JSONFormatter, TextFormatter, configure_logging


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_message(self) -> None:
        """Test basic log message is formatted as JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_extra_fields(self) -> None:
        """Test extra fields are included in JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.model = "bge-m3"
        record.request_id = "abc123"
        record.latency_ms = 45.2

        result = formatter.format(record)
        data = json.loads(result)

        assert data["model"] == "bge-m3"
        assert data["request_id"] == "abc123"
        assert data["latency_ms"] == 45.2

    def test_exception_info(self) -> None:
        """Test exception info is included in JSON."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_format(self) -> None:
        """Test text formatter produces expected format."""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test.module" in result
        assert "Test message" in result


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_json_format_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit JSON format configuration."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)

        configure_logging(json_format=True)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_text_format_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit text format configuration."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)

        configure_logging(json_format=False)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    def test_json_format_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test JSON format from environment variable."""
        monkeypatch.setenv("SIE_LOG_JSON", "true")

        configure_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_verbose_sets_debug_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test verbose flag sets DEBUG level."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)

        configure_logging(verbose=True, json_format=False)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
