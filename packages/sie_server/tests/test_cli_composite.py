"""``sie-server serve`` with a composite bundle."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import typer
from sie_server import cli
from sie_server.app.app_state_config import AppStateConfig
from sie_server.composite_bundles import COMPONENT_ENV

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # ``serve`` exports SIE_BUNDLE and friends straight into os.environ, which
    # monkeypatch cannot undo for variables that were unset; restore it all.
    saved = dict(os.environ)
    for name in (
        "SIE_BUNDLE",
        "SIE_DEVICE",
        "SIE_DEVICES",
        "SIE_EXTRA_MODELS",
        "SIE_POOL",
        "SIE_PRELOAD_MODELS",
        "SIE_PINNED_MODELS",
        COMPONENT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _serve(**overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "port": 8080,
        "host": "127.0.0.1",
        "device": "cpu",
        "models_dir": str(MODELS_DIR),
        "bundle": "default-sglang",
        "models": None,
        "local_cache": None,
        "cluster_cache": None,
        "hf_fallback": True,
        "reload": False,
        "tracing": False,
        "instrumentation": False,
        "verbose": False,
        "log_level": "info",
        "preload": None,
        "json_logs": False,
    }
    arguments.update(overrides)
    cli.serve(**arguments)


def test_serve_supervises_the_components_of_a_composite_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_supervisor(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_composite_supervisor", fake_supervisor)
    monkeypatch.setattr(cli, "run_server", lambda **_: pytest.fail("the supervisor must not serve models itself"))

    with pytest.raises(typer.Exit) as exit_info:
        _serve(host="0.0.0.0", port=8081)  # noqa: S104 - the container port

    assert exit_info.value.exit_code == 0
    assert captured == {
        "bundle": "default-sglang",
        "components": ("default", "sglang"),
        "host": "0.0.0.0",  # noqa: S104
        "port": 8081,
    }


def test_component_process_preloads_only_its_own_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, AppStateConfig] = {}

    def fake_run_server(*, config: AppStateConfig, **_: Any) -> None:
        captured["config"] = config

    monkeypatch.setenv(COMPONENT_ENV, "default")
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    _serve(preload="BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507", uds=str(tmp_path / "default.http.sock"))

    config = captured["config"]
    assert config.preload_models == ["BAAI/bge-m3"]
    # The catalog stays the composite's, so every component hashes the same scope.
    assert config.model_filter is not None
    assert "Qwen/Qwen3-4B-Instruct-2507" in config.model_filter
    assert "BAAI/bge-m3" in config.model_filter


def test_a_component_outside_the_bundle_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPONENT_ENV, "transformers5")
    monkeypatch.setattr(cli, "run_server", lambda **_: pytest.fail("must not start"))

    with pytest.raises(typer.Exit) as exit_info:
        _serve()

    assert exit_info.value.exit_code == 1
