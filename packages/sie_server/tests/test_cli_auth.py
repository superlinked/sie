"""`sie-server serve`: bind address, token auth resolution and exposure warnings."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from sie_server import cli
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.server_auth import token_path
from typer.testing import CliRunner

TOKEN = "s3cret-token-value"  # noqa: S105 - test credential
EXPOSED_HOST = "0.0.0.0"  # noqa: S104 - no server is started


@pytest.fixture(autouse=True)
def _restore_environment() -> Iterator[None]:
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    for name in (
        "SIE_AUTH_MODE",
        "SIE_AUTH_TOKEN",
        "SIE_AUTH_TOKENS",
        "SIE_SERVER_HOST",
        "XDG_DATA_HOME",
        "SIE_UPSTREAMS_FILE",
        "SIE_REMOTE_SERVING",
        "SIE_LOG_LEVEL",
        "SIE_PRELOAD_MODELS",
        "SIE_PINNED_MODELS",
        "SIE_EXTRA_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path / "data"))
    return calls


def serve(*args: str) -> Any:
    return CliRunner().invoke(cli.app, ["serve", *args])


def resolved_mode(started: dict[str, Any]) -> str:
    config: AppStateConfig = started["config"]
    return config.auth_mode


def test_default_bind_is_unchanged_and_warns(started: dict[str, Any]) -> None:
    result = serve()

    assert result.exit_code == 0, result.output
    assert started["host"] == "0.0.0.0"  # noqa: S104
    assert resolved_mode(started) == "none"
    assert "WARNING: sie-server listens on 0.0.0.0:8080 without authentication." in result.output
    assert "From 0.13.0, an exposed sie-server requires a token." in result.output
    assert "From 0.13.0 the default is 127.0.0.1." in result.output


def test_loopback_bind_does_not_warn(started: dict[str, Any]) -> None:
    result = serve("--host", "127.0.0.1")

    assert result.exit_code == 0, result.output
    assert started["host"] == "127.0.0.1"
    assert "WARNING" not in result.output
    assert "Notice" not in result.output


def test_bind_address_from_the_environment(started: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_SERVER_HOST", "127.0.0.1")

    result = serve()

    assert result.exit_code == 0, result.output
    assert started["host"] == "127.0.0.1"
    assert "WARNING" not in result.output


def test_an_explicit_exposed_bind_skips_the_default_notice(started: dict[str, Any]) -> None:
    result = serve("--host", "0.0.0.0")  # noqa: S104

    assert result.exit_code == 0, result.output
    assert "WARNING: sie-server listens on 0.0.0.0:8080 without authentication." in result.output
    assert "Notice" not in result.output


def test_explicitly_open_server_warns_without_the_deprecation(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "none")

    result = serve("--host", "0.0.0.0")  # noqa: S104

    assert result.exit_code == 0, result.output
    assert resolved_mode(started) == "none"
    assert "authentication is off (SIE_AUTH_MODE=none)" in result.output
    assert "From 0.13.0" not in result.output


def test_a_configured_token_turns_auth_on(started: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)

    result = serve("--host", "0.0.0.0")  # noqa: S104

    assert result.exit_code == 0, result.output
    assert resolved_mode(started) == "token"
    assert "WARNING" not in result.output
    assert TOKEN not in result.output


def test_token_mode_generates_and_prints_the_token_once(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "token")
    monkeypatch.setattr(cli, "_stderr_is_terminal", lambda: True)

    first = serve("--port", "9090")

    assert first.exit_code == 0, first.output
    assert resolved_mode(started) == "token"
    token = token_path(tmp_path / "data").read_text().strip()
    assert f"export SIE_API_KEY={token}" in first.output
    assert "http://localhost:9090/v1/models" in first.output

    second = serve()

    assert second.exit_code == 0, second.output
    assert token not in second.output
    assert str(token_path(tmp_path / "data")) in second.output


def test_a_generated_token_is_not_printed_into_logs(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data $HOME"
    monkeypatch.setenv("SIE_DATA_DIR", str(data))
    monkeypatch.setenv("SIE_AUTH_MODE", "token")
    monkeypatch.setattr(cli, "_stderr_is_terminal", lambda: False)

    result = serve()

    assert result.exit_code == 0, result.output
    path = token_path(data)
    assert path.read_text().strip() not in result.output
    assert f"export SIE_API_KEY=$(cat '{path}')" in result.output
    assert "set SIE_AUTH_TOKEN yourself" in result.output


@pytest.mark.parametrize("value", [" ", ",", ""])
def test_a_blank_token_stops_startup(started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SIE_AUTH_TOKEN", value)

    result = serve("--host", "0.0.0.0")  # noqa: S104

    assert result.exit_code == 1
    assert "holds no token" in result.output
    assert not started


def test_the_bind_address_reaches_the_app(started: dict[str, Any]) -> None:
    result = serve("--host", "127.0.0.1")

    assert result.exit_code == 0, result.output
    config: AppStateConfig = started["config"]
    assert config.bind_host == "127.0.0.1"


def test_auto_leaves_a_loopback_server_open(started: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "auto")

    result = serve("--host", "127.0.0.1")

    assert result.exit_code == 0, result.output
    assert resolved_mode(started) == "none"


def test_auto_protects_an_exposed_server(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "auto")

    result = serve("--host", "0.0.0.0")  # noqa: S104

    assert result.exit_code == 0, result.output
    assert resolved_mode(started) == "token"
    assert token_path(tmp_path / "data").exists()
    assert "WARNING" not in result.output


def test_a_token_with_auth_off_stops_startup(started: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "none")
    monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)

    result = serve()

    assert result.exit_code == 1
    assert "SIE_AUTH_MODE=none" in result.output
    assert TOKEN not in result.output
    assert not started


def test_an_unknown_mode_stops_startup(started: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "open")

    result = serve()

    assert result.exit_code == 1
    assert "none, token, static or auto" in result.output
    assert not started


def test_an_unsafe_token_file_stops_startup(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "token")
    directory = tmp_path / "data"
    directory.mkdir()
    token_path(directory).write_text(f"{TOKEN}\n")
    token_path(directory).chmod(0o644)

    result = serve()

    assert result.exit_code == 1
    assert "chmod 600" in result.output
    assert TOKEN not in result.output
    assert not started


@pytest.mark.parametrize("value", ["TOKEN", "Token", "None"])
def test_auth_modes_are_case_sensitive_as_on_the_gateway(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", value)

    result = serve()

    assert result.exit_code == 1
    assert "lowercase" in result.output
    assert not started


@pytest.mark.parametrize(
    ("mode", "token", "host", "required", "generated"),
    [
        (None, None, EXPOSED_HOST, "none", False),
        ("none", None, EXPOSED_HOST, "none", False),
        (None, TOKEN, EXPOSED_HOST, "token", False),
        ("token", TOKEN, EXPOSED_HOST, "token", False),
        ("static", TOKEN, EXPOSED_HOST, "token", False),
        ("auto", TOKEN, "127.0.0.1", "token", False),
        ("auto", None, "127.0.0.1", "none", False),
        ("auto", None, EXPOSED_HOST, "token", True),
        ("token", None, "127.0.0.1", "token", True),
        ("static", None, "127.0.0.1", "token", True),
    ],
)
def test_required_auth_mode_matches_existing_resolution(
    started: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str | None,
    token: str | None,
    host: str,
    required: str,
    generated: bool,
) -> None:
    if mode is not None:
        monkeypatch.setenv("SIE_AUTH_MODE", mode)
    if token is not None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", token)
    monkeypatch.setattr(cli, "_stderr_is_terminal", lambda: False)

    result = serve("--host", host, "--require-auth-mode", required)

    assert result.exit_code == 0, result.output
    assert resolved_mode(started) == required
    assert token_path(tmp_path / "data").exists() is generated
    assert os.environ.get("SIE_AUTH_MODE") == mode
    assert os.environ.get("SIE_AUTH_TOKEN") == token
    assert TOKEN not in result.output


@pytest.fixture
def no_token_provisioning(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    directory = Mock(side_effect=AssertionError("unexpected token directory resolution"))
    token = Mock(side_effect=AssertionError("unexpected token provisioning"))
    monkeypatch.setattr(cli, "resolve_data_dir", directory)
    monkeypatch.setattr(cli, "load_or_create_token", token)
    yield
    directory.assert_not_called()
    token.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "token", "host", "required"),
    [
        (None, None, EXPOSED_HOST, "token"),
        ("none", None, EXPOSED_HOST, "token"),
        ("token", None, "127.0.0.1", "none"),
        ("static", None, "127.0.0.1", "none"),
        (None, TOKEN, "127.0.0.1", "none"),
        ("auto", None, EXPOSED_HOST, "none"),
        ("auto", None, "127.0.0.1", "token"),
    ],
)
@pytest.mark.usefixtures("no_token_provisioning")
def test_required_auth_mode_mismatch_stops_before_provisioning_or_serving(
    started: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str | None,
    token: str | None,
    host: str,
    required: str,
) -> None:
    if mode is not None:
        monkeypatch.setenv("SIE_AUTH_MODE", mode)
    if token is not None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", token)

    result = serve("--host", host, "--require-auth-mode", required)

    assert result.exit_code == 1
    assert "resolved authentication mode does not match --require-auth-mode" in result.output
    assert not started
    assert not (tmp_path / "data").exists()
    assert os.environ.get("SIE_AUTH_MODE") == mode
    assert os.environ.get("SIE_AUTH_TOKEN") == token
    assert TOKEN not in result.output


@pytest.mark.parametrize("required", ["", "auto", "static", "TOKEN", "None", " token", "unknown"])
@pytest.mark.usefixtures("no_token_provisioning")
def test_required_auth_mode_rejects_invalid_option_before_auth_resolution(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, required: str
) -> None:
    prepare = Mock(side_effect=AssertionError("unexpected auth resolution"))
    monkeypatch.setattr(cli, "_prepare_auth", prepare)

    result = serve("--require-auth-mode", required)

    assert result.exit_code == 2
    assert "Invalid value" in result.output
    prepare.assert_not_called()
    assert not started


@pytest.mark.parametrize("name", ["SIE_AUTH_TOKEN", "SIE_AUTH_TOKENS"])
@pytest.mark.parametrize("value", ["", " ", ","])
@pytest.mark.parametrize("required", ["token", "none"])
@pytest.mark.usefixtures("no_token_provisioning")
def test_required_auth_mode_does_not_bypass_blank_token_refusal(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, name: str, value: str, required: str
) -> None:
    monkeypatch.setenv(name, value)

    result = serve("--require-auth-mode", required)

    assert result.exit_code == 1
    assert "holds no token" in result.output
    assert not started


@pytest.mark.parametrize("required", ["token", "none"])
@pytest.mark.usefixtures("no_token_provisioning")
def test_required_auth_mode_does_not_override_auth_off_with_token_refusal(
    started: dict[str, Any], monkeypatch: pytest.MonkeyPatch, required: str
) -> None:
    monkeypatch.setenv("SIE_AUTH_MODE", "none")
    monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)

    result = serve("--require-auth-mode", required)

    assert result.exit_code == 1
    assert "SIE_AUTH_MODE=none turns authentication off" in result.output
    assert TOKEN not in result.output
    assert not started
