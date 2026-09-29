"""A remote-backed model served on a single node through an SIE upstream.

The upstream is a real SIE app serving the built-in fake model on loopback, so
the request and the vectors cross the actual wire format in both directions.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_server.adapters.remote import sie as remote_sie
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.types.inputs import Item

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CANARY = "sk-canary-3d9f1a7b52c6e804"
KEY_ENV = "REMOTE_SIE_TEST_KEY"
REMOTE_MODEL = """\
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
        upstream_model: sie-fake
"""


@pytest.fixture(autouse=True)
def _reset_upstreams(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(KEY_ENV, CANARY)
    yield
    install_upstreams({})


@contextmanager
def fake_sie_upstream() -> Iterator[tuple[str, list[str | None]]]:
    app = AppFactory.create_app(AppStateConfig(models_dir=str(MODELS_DIR), model_filter=["sie-fake"], device="cpu"))
    seen_authorization: list[str | None] = []

    @app.middleware("http")
    async def record_authorization(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/v1/encode/"):
            seen_authorization.append(request.headers.get("authorization"))
        return await call_next(request)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "the upstream SIE app did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", seen_authorization
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def local_app(tmp_path: Path, upstream_url: str, *, remote_serving: bool = True, upstream: str = "fake-sie") -> FastAPI:
    models = tmp_path / "models"
    models.mkdir()
    (models / "remote-fake.yaml").write_text(REMOTE_MODEL.format(upstream=upstream), encoding="utf-8")
    upstreams = tmp_path / "upstreams.yaml"
    upstreams.write_text(
        "upstreams:\n"
        "  fake-sie:\n"
        "    kind: sie\n"
        f"    base_url: {upstream_url}\n"
        f"    api_key_secret: {KEY_ENV}\n"
        "    rate_cap: {requests_per_minute: 600, max_concurrency: 8}\n",
        encoding="utf-8",
    )
    return AppFactory.create_app(
        AppStateConfig(
            models_dir=str(models),
            device="cpu",
            upstreams_file=str(upstreams),
            remote_serving=remote_serving,
        )
    )


def encode_when_loaded(client: TestClient, model: str, text: str) -> httpx.Response:
    deadline = time.monotonic() + 30
    while True:
        response = client.post(
            f"/v1/encode/{model}", json={"items": [{"text": text}]}, headers={"Accept": "application/json"}
        )
        if response.status_code != 503 or time.monotonic() > deadline:
            return response
        time.sleep(0.1)


def test_a_remote_backed_model_returns_the_upstreams_vectors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with fake_sie_upstream() as (upstream_url, seen_authorization):
        with SIEClient(upstream_url) as upstream:
            expected = upstream.encode("sie-fake", {"text": "remote backends"})["dense"]
        seen_authorization.clear()

        with TestClient(local_app(tmp_path, upstream_url)) as client:
            response = encode_when_loaded(client, "acme/remote-fake", "remote backends")

    assert response.status_code == 200, response.text
    served = np.asarray(response.json()["items"][0]["dense"]["values"], dtype=np.float32)
    np.testing.assert_allclose(served, expected, rtol=1e-6)
    assert seen_authorization == [f"Bearer {CANARY}"]
    assert CANARY not in response.text
    assert CANARY not in str(response.headers)
    assert CANARY not in caplog.text


def test_the_global_switch_refuses_remote_profiles_and_sends_nothing(tmp_path: Path) -> None:
    with fake_sie_upstream() as (upstream_url, seen_authorization):
        with TestClient(local_app(tmp_path, upstream_url, remote_serving=False)) as client:
            response = encode_when_loaded(client, "acme/remote-fake", "remote backends")

    assert response.status_code == 502, response.text
    assert "remote serving is switched off" in response.text
    assert seen_authorization == []


def test_a_model_naming_an_undefined_upstream_stops_startup(tmp_path: Path) -> None:
    app = local_app(tmp_path, "http://127.0.0.1:9", upstream="nobody")

    with pytest.raises(ValueError, match="names an undefined upstream 'nobody'"), TestClient(app):
        pass


def adapter_over(monkeypatch: pytest.MonkeyPatch, respond: httpx.Response) -> remote_sie.SieUpstreamAdapter:
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "https://sie.example.internal",
            "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
        }
    )
    install_upstreams({"team-sie": upstream})
    monkeypatch.setattr(
        remote_sie,
        "upstream_sync_client",
        lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(lambda _request: respond)),
    )
    adapter = remote_sie.SieUpstreamAdapter(upstream="team-sie", upstream_model="sie-fake", dense_dim=4)
    adapter.load("cpu")
    return adapter


def msgpack_response(items: list[dict[str, Any]]) -> httpx.Response:
    from sie_sdk._msgpack import packb

    return httpx.Response(200, content=packb({"items": items}), headers={"Content-Type": "application/msgpack"})


def test_a_dimension_mismatch_is_an_error_not_a_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = np.ones(8, dtype=np.float32)
    adapter = adapter_over(monkeypatch, msgpack_response([{"dense": {"dims": 8, "dtype": "float32", "values": wrong}}]))

    with pytest.raises(remote_sie.RemoteUpstreamError, match="8-dimensional vectors, the model declares 4"):
        adapter.encode([Item(text="a")], ["dense"])


def test_a_short_answer_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    vector = np.ones(4, dtype=np.float32)
    adapter = adapter_over(
        monkeypatch, msgpack_response([{"dense": {"dims": 4, "dtype": "float32", "values": vector}}])
    )

    with pytest.raises(remote_sie.RemoteUpstreamError, match="different number of results"):
        adapter.encode([Item(text="a"), Item(text="b")], ["dense"])


def test_an_upstream_error_is_raised_to_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = adapter_over(
        monkeypatch, httpx.Response(404, json={"detail": {"code": "MODEL_NOT_FOUND", "message": "no such model"}})
    )

    with pytest.raises(Exception, match="no such model"):
        adapter.encode([Item(text="a")], ["dense"])
