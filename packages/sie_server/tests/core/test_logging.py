"""Tests for structured JSON logging."""

from __future__ import annotations

import json
import logging

import pytest
from sie_server.core.logging import (
    JSONFormatter,
    TextFormatter,
    configure_logging,
    is_valid_log_level,
    valid_log_levels,
)


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
        monkeypatch.delenv("SIE_LOG_LEVEL", raising=False)

        configure_logging(json_format=True)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_text_format_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test explicit text format configuration."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.delenv("SIE_LOG_LEVEL", raising=False)

        configure_logging(json_format=False)

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    def test_json_format_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test JSON format from environment variable."""
        monkeypatch.setenv("SIE_LOG_JSON", "true")
        monkeypatch.delenv("SIE_LOG_LEVEL", raising=False)

        configure_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_verbose_sets_debug_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test verbose flag sets DEBUG level."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "INFO")

        configure_logging(verbose=True, json_format=False)

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_sie_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_LOG_LEVEL controls root level when not verbose."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "DEBUG")

        configure_logging(json_format=False)

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_level_name_param_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit level_name wins over SIE_LOG_LEVEL."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "DEBUG")

        configure_logging(json_format=False, level_name="WARNING")

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_invalid_log_level_falls_back_to_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "not-a-real-level")

        configure_logging(json_format=False)

        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_invalid_log_level_says_so(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The fallback must be announced, not silent.

        An operator who sets DEBUG and mistypes it otherwise gets a
        normal-looking server with none of the output they asked for and
        nothing anywhere explaining why.

        Asserted against real stdout rather than ``caplog``: ``configure_logging``
        replaces every root handler (including the one caplog installs), and
        stdout is what the operator actually reads.
        """
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "DEBIG")

        configure_logging(json_format=False)
        logging.getLogger().handlers[0].flush()

        out = capsys.readouterr().out
        assert "DEBIG" in out
        assert "falling back to INFO" in out
        # The message must list what would have worked.
        assert "DEBUG" in out

    def test_warning_outranks_the_fallback_level(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """WARNING is above the INFO fallback, so the notice survives it."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "nope")

        configure_logging(json_format=False)
        logging.getLogger().handlers[0].flush()

        assert "WARNING" in capsys.readouterr().out
        assert logging.getLogger().level == logging.INFO

    def test_valid_level_warns_about_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "warning")  # lowercase still resolves

        configure_logging(json_format=False)
        logging.getLogger().handlers[0].flush()

        assert "falling back" not in capsys.readouterr().out
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_is_never_reported_as_invalid(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--verbose short-circuits resolution, so a stale env value is moot."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "garbage")

        configure_logging(verbose=True, json_format=False)
        logging.getLogger().handlers[0].flush()

        assert "falling back" not in capsys.readouterr().out
        assert logging.getLogger().level == logging.DEBUG


class TestLogLevelHelpers:
    """The validity check backing the CLI's eager `--log-level` rejection."""

    @pytest.mark.parametrize("name", ["DEBUG", "info", "  Warning  ", "ERROR", "critical"])
    def test_accepts_real_levels_case_and_space_insensitively(self, name: str) -> None:
        assert is_valid_log_level(name)

    @pytest.mark.parametrize("name", ["DEBIG", "verbose", "", "   ", "trace"])
    def test_rejects_unknown_levels(self, name: str) -> None:
        assert not is_valid_log_level(name)

    @pytest.mark.parametrize("name", ["NOTSET", "WARN", "FATAL"])
    def test_rejects_names_python_knows_but_we_do_not_advertise(self, name: str) -> None:
        """Validation and the advertised list must be the same set.

        `logging.getLevelNamesMapping()` also carries NOTSET and the deprecated
        WARN/FATAL aliases. Accepting them while `--help` documents five names
        would let a value through that the documented contract does not cover,
        and NOTSET on the root logger means "inherit", not a severity.
        """
        assert name in logging.getLevelNamesMapping()
        assert not is_valid_log_level(name)
        assert name not in valid_log_levels()

    def test_advertised_levels_are_all_real(self) -> None:
        """The list printed in the error must not itself contain a bad name."""
        assert all(is_valid_log_level(name) for name in valid_log_levels())


class TestRejectedValueIsNotEchoedRaw:
    """A rejected level is arbitrary input and must not reach the sink verbatim."""

    def test_long_value_is_truncated(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        secret = "A" * 200
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", secret)

        configure_logging(json_format=False)
        logging.getLogger().handlers[0].flush()

        out = capsys.readouterr().out
        assert secret not in out
        assert "..." in out

    def test_punctuation_is_masked(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Anything that could carry a token shape is replaced, not echoed."""
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "tok=abc.def/ghi")

        configure_logging(json_format=False)
        logging.getLogger().handlers[0].flush()

        out = capsys.readouterr().out
        assert "tok=abc.def/ghi" not in out
        assert "falling back to INFO" in out

    def test_empty_value_is_labelled(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.delenv("SIE_LOG_JSON", raising=False)
        monkeypatch.setenv("SIE_LOG_LEVEL", "!!!")

        configure_logging(json_format=False)
        logging.getLogger().handlers[0].flush()

        assert "falling back to INFO" in capsys.readouterr().out
