"""Real SIE apps on loopback for the remote adapter tests.

An upstream is a real SIE app serving the built-in fake models, so a request
and its answer cross the actual wire format in both directions. A local app
serves the remote-backed model ``acme/remote-fake`` through it.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.upstreams import install_upstreams

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_CREDENTIAL = "sk-canary-3d9f1a7b52c6e804"
_CREDENTIAL_ENV = "REMOTE_SIE_TEST_KEY"
_REMOTE_MODEL = """\
sie_id: acme/remote-fake
remote_backed: true
inputs:
  text: true
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: {upstream}
        upstream_model: {upstream_model}
"""


@dataclass
class LoopbackUpstream:
    """A running upstream SIE app, the ``Authorization`` header of each encode and the path of each POST it received."""

    url: str
    seen_authorization: list[str | None] = field(default_factory=list)
    posted_paths: list[str] = field(default_factory=list)


@contextmanager
def _serve(app: FastAPI) -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "the app on loopback did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def _offline_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apps built by these fixtures send no usage telemetry off the machine."""
    monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "1")


@pytest.fixture
def upstream_credential_env() -> str:
    """The environment variable that holds the credential of the local app's upstream."""
    return _CREDENTIAL_ENV


@pytest.fixture
def upstream_credential(monkeypatch: pytest.MonkeyPatch, upstream_credential_env: str) -> Iterator[str]:
    """The credential a local app sends its upstream. It must never reach a response or a log."""
    monkeypatch.setenv(upstream_credential_env, _CREDENTIAL)
    yield _CREDENTIAL
    install_upstreams({})


@pytest.fixture
def serve_on_loopback() -> Callable[[FastAPI], AbstractContextManager[str]]:
    """Run an app on 127.0.0.1 for the length of a ``with`` block, which receives its base URL."""
    return _serve


@pytest.fixture
def sie_upstream(_offline_apps: None) -> Callable[..., AbstractContextManager[LoopbackUpstream]]:
    """Start a real SIE app serving the fake models named, ``sie-fake`` when none is named.

    With ``models_dir`` the app serves every model in that directory instead.
    """

    @contextmanager
    def start(*models: str, models_dir: Path | None = None) -> Iterator[LoopbackUpstream]:
        model_filter = list(models) if models else (None if models_dir is not None else ["sie-fake"])
        app = AppFactory.create_app(
            AppStateConfig(models_dir=str(models_dir or MODELS_DIR), model_filter=model_filter, device="cpu")
        )
        upstream = LoopbackUpstream(url="")

        @app.middleware("http")
        async def record_requests(request: Request, call_next: Any) -> Any:
            if request.url.path.startswith("/v1/encode/"):
                upstream.seen_authorization.append(request.headers.get("authorization"))
            if request.method == "POST":
                upstream.posted_paths.append(request.url.path)
            return await call_next(request)

        with _serve(app) as url:
            upstream.url = url
            yield upstream

    return start


@pytest.fixture
def remote_app(
    tmp_path: Path, upstream_credential: str, upstream_credential_env: str, _offline_apps: None
) -> Callable[..., FastAPI]:
    """Build a local app that serves ``acme/remote-fake`` through the upstream at a URL."""
    _ = upstream_credential

    def build(
        upstream_url: str,
        *,
        remote_serving: bool = True,
        upstream: str = "fake-sie",
        upstream_model: str = "sie-fake",
        extra_models: dict[str, str] | None = None,
        preload: bool = False,
    ) -> FastAPI:
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        (models / "remote-fake.yaml").write_text(
            _REMOTE_MODEL.format(upstream=upstream, upstream_model=upstream_model), encoding="utf-8"
        )
        for file_name, text in (extra_models or {}).items():
            (models / file_name).write_text(text, encoding="utf-8")
        upstreams = tmp_path / "upstreams.yaml"
        upstreams.write_text(
            "upstreams:\n"
            "  fake-sie:\n"
            "    kind: sie\n"
            f"    base_url: {upstream_url}\n"
            f"    api_key_secret: {upstream_credential_env}\n"
            "    rate_cap: {requests_per_minute: 600, max_concurrency: 8}\n",
            encoding="utf-8",
        )
        return AppFactory.create_app(
            AppStateConfig(
                models_dir=str(models),
                device="cpu",
                upstreams_file=str(upstreams),
                remote_serving=remote_serving,
                preload_models=["acme/remote-fake"] if preload else None,
            )
        )

    return build


@pytest.fixture
def encode_when_loaded() -> Callable[..., httpx.Response]:
    """POST one text item to ``/v1/encode`` until the answer is not a 503, for at most 30 s."""

    def post(client: TestClient, model: str, text: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        body: dict[str, Any] = {"items": [{"text": text}]}
        if params is not None:
            body["params"] = params
        deadline = time.monotonic() + 30
        while True:
            response = client.post(f"/v1/encode/{model}", json=body, headers={"Accept": "application/json"})
            if response.status_code != 503 or time.monotonic() > deadline:
                return response
            time.sleep(0.1)

    return post
