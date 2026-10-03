"""`sie-server serve --log-level` rejects a typo instead of ignoring it.

The parameter is bound to both a flag and `SIE_LOG_LEVEL`, and the two get
different treatment on purpose: a typed flag is an immediate, cheap-to-fix
mistake, while a bad environment value comes from a rendered Helm value and
must not crash-loop a pod.
"""

from __future__ import annotations

from typing import Any

import pytest
from sie_server import cli
from typer.testing import CliRunner


@pytest.fixture
def no_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the server so `serve` returns instead of binding a port."""
    started: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> None:
        started["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    monkeypatch.delenv("SIE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("SIE_PINNED_MODELS", raising=False)
    monkeypatch.delenv("SIE_EXTRA_MODELS", raising=False)
    return started


def test_invalid_flag_is_rejected_before_startup(no_server: dict[str, Any]) -> None:
    """A mistyped flag must not start a server that silently ignores it."""
    result = CliRunner().invoke(cli.app, ["serve", "--log-level", "DEBIG"])

    assert result.exit_code == 1
    assert "Invalid --log-level 'DEBIG'" in result.output
    # The error must name the levels that would have worked.
    assert "DEBUG" in result.output
    assert "kwargs" not in no_server, "server started despite an invalid log level"


def test_undocumented_level_name_is_rejected(no_server: dict[str, Any]) -> None:
    """NOTSET resolves in Python's mapping but is not an advertised level."""
    result = CliRunner().invoke(cli.app, ["serve", "--log-level", "NOTSET"])

    assert result.exit_code == 1
    assert "Invalid --log-level 'NOTSET'" in result.output


def test_invalid_environment_value_does_not_stop_the_server(
    no_server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad SIE_LOG_LEVEL degrades loudly; it must never crash-loop a pod."""
    monkeypatch.setenv("SIE_LOG_LEVEL", "DEBIG")

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    assert "kwargs" in no_server, "an environment typo prevented startup"
    # configure_logging reports it and falls back.
    assert no_server["kwargs"]["uvicorn_log_level"] == "info"


def test_verbose_wins_over_an_invalid_environment_value(
    no_server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--verbose must still reach DEBUG regardless of a stale env value."""
    monkeypatch.setenv("SIE_LOG_LEVEL", "garbage")

    result = CliRunner().invoke(cli.app, ["serve", "--verbose"])

    assert result.exit_code == 0, result.output
    assert no_server["kwargs"]["uvicorn_log_level"] == "debug"


def test_valid_flag_starts_normally(no_server: dict[str, Any]) -> None:
    result = CliRunner().invoke(cli.app, ["serve", "--log-level", "warning"])

    assert result.exit_code == 0, result.output
    assert no_server["kwargs"]["uvicorn_log_level"] == "warning"


def test_valid_environment_value_starts_normally(no_server: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_LOG_LEVEL", "error")

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    assert no_server["kwargs"]["uvicorn_log_level"] == "error"
