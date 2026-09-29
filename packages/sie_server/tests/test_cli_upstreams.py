"""`sie-server serve --upstreams-file`: startup-only upstream definitions.

A typed flag with an invalid file stops startup. The same file from
`SIE_UPSTREAMS_FILE` (a Helm value) warns and loads no upstream, so nothing is
sent outside the deployment and the pod does not crash-loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sie_server import cli
from sie_server.app.app_state_config import AppStateConfig
from typer.testing import CliRunner

VALID = (
    "upstreams:\n"
    "  team-sie:\n"
    "    kind: sie\n"
    "    base_url: https://sie.example.internal\n"
    "    api_key_secret: TEAM_SIE_KEY\n"
    "    rate_cap: {requests_per_minute: 600, max_concurrency: 32}\n"
)
INVALID = VALID.replace("https://", "http://")


@pytest.fixture
def no_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    started: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> None:
        started["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_server", fake_run_server)
    for name in ("SIE_UPSTREAMS_FILE", "SIE_LOG_LEVEL", "SIE_PRELOAD_MODELS", "SIE_PINNED_MODELS", "SIE_EXTRA_MODELS"):
        monkeypatch.delenv(name, raising=False)
    return started


def upstreams_file(tmp_path: Path, text: str) -> str:
    path = tmp_path / "upstreams.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_valid_file_reaches_the_server_process(no_server: dict[str, Any], tmp_path: Path) -> None:
    path = upstreams_file(tmp_path, VALID)

    result = CliRunner().invoke(cli.app, ["serve", "--upstreams-file", path])

    assert result.exit_code == 0, result.output
    assert "Upstreams: team-sie" in result.output
    config: AppStateConfig = no_server["kwargs"]["config"]
    assert config.upstreams_file == path


def test_an_invalid_typed_file_stops_startup(no_server: dict[str, Any], tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["serve", "--upstreams-file", upstreams_file(tmp_path, INVALID)])

    assert result.exit_code == 1
    assert "must use https outside loopback" in result.output
    assert "kwargs" not in no_server


def test_an_invalid_environment_file_loads_no_upstream(
    no_server: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIE_UPSTREAMS_FILE", upstreams_file(tmp_path, INVALID))

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    assert "No upstream is loaded" in result.output
    assert no_server["kwargs"]["config"].upstreams_file is None


def test_the_file_survives_the_uvicorn_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_UPSTREAMS_FILE", raising=False)
    path = upstreams_file(tmp_path, VALID)

    AppStateConfig(upstreams_file=path).save_to_env_vars()
    assert AppStateConfig.from_env_vars().upstreams_file == path

    AppStateConfig().save_to_env_vars()
    assert AppStateConfig.from_env_vars().upstreams_file is None
