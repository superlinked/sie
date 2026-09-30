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


def test_a_malformed_credential_never_reaches_the_caller_or_the_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(KEY_ENV, f"{CANARY} trailing-garbage")
    caplog.set_level(logging.DEBUG)
    with fake_sie_upstream() as (upstream_url, seen_authorization):
        with TestClient(local_app(tmp_path, upstream_url)) as client:
            response = encode_when_loaded(client, "acme/remote-fake", "remote backends")

    assert response.status_code != 200
    assert seen_authorization == []
    assert CANARY not in response.text
    assert CANARY not in caplog.text


def test_with_the_switch_off_a_stale_upstream_name_does_not_stop_startup(tmp_path: Path) -> None:
    app = local_app(tmp_path, "http://127.0.0.1:9", upstream="nobody", remote_serving=False)

    with TestClient(app) as client:
        response = encode_when_loaded(client, "acme/remote-fake", "remote backends")

    assert response.status_code == 502, response.text
    assert "remote serving is switched off" in response.text


class Recorder:
    def __init__(self, respond: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond


def adapter_over(
    monkeypatch: pytest.MonkeyPatch, recorder: Recorder, *, upstream_model: str = "sie-fake"
) -> remote_sie.SieUpstreamAdapter:
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "https://sie.example.internal/tenant-a",
            "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
        }
    )
    install_upstreams({"team-sie": upstream})
    monkeypatch.setattr(
        remote_sie,
        "upstream_sync_client",
        lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(recorder)),
    )
    adapter = remote_sie.SieUpstreamAdapter(upstream="team-sie", upstream_model=upstream_model, dense_dim=4)
    adapter.load("cpu")
    return adapter


def encode_payload(items: list[dict[str, Any]]) -> bytes:
    from sie_sdk._msgpack import packb

    return packb({"items": items})


def dense_item(values: np.ndarray) -> dict[str, Any]:
    return {"dense": {"dims": int(values.shape[0]), "dtype": "float32", "values": values}}


class _Body(httpx.SyncByteStream):
    """A body that streams like a network response, rather than one read eagerly from bytes."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    def __iter__(self) -> Iterator[bytes]:
        yield self._content


def streamed(status: int, content: bytes, **headers: str) -> httpx.Response:
    return httpx.Response(status, stream=_Body(content), headers=headers)


def ok(content: bytes, **headers: str) -> httpx.Response:
    return streamed(200, content, **{"Content-Type": "application/msgpack", **headers})


def test_the_request_goes_to_the_encode_path_under_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(ok(encode_payload([dense_item(np.ones(4, dtype=np.float32))])))
    adapter = adapter_over(monkeypatch, recorder, upstream_model="org/name:profile")

    output = adapter.encode([Item(text="a")], ["dense"])

    assert output.dense is not None
    assert output.dense.shape == (1, 4)
    assert recorder.requests[0].url.raw_path == b"/tenant-a/v1/encode/org/name:profile"
    assert recorder.requests[0].headers["accept-encoding"] == "identity"
    assert recorder.requests[0].extensions["timeout"]["read"] == remote_sie.READ_TIMEOUT_S


@pytest.mark.parametrize("upstream_model", ["../../admin", "a/../../../v1/configs/models"])
def test_a_model_id_that_escapes_the_encode_path_is_never_sent(
    monkeypatch: pytest.MonkeyPatch, upstream_model: str
) -> None:
    recorder = Recorder(ok(b""))
    adapter = adapter_over(monkeypatch, recorder, upstream_model=upstream_model)

    with pytest.raises(remote_sie.RemoteUpstreamError, match="does not form an encode path"):
        adapter.encode([Item(text="a")], ["dense"])

    assert recorder.requests == []


def test_upstream_error_text_is_never_relayed(monkeypatch: pytest.MonkeyPatch) -> None:
    echoed = f"Authorization: Bearer {CANARY}"
    recorder = Recorder(streamed(404, echoed.encode(), **{"X-SIE-Error-Code": "MODEL_NOT_FOUND"}))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert str(raised.value) == "upstream answered 404 MODEL_NOT_FOUND"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_an_unexpected_error_code_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(streamed(500, b"x", **{"X-SIE-Error-Code": f"leak {CANARY}"}))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert str(raised.value) == "upstream answered 500"


def test_a_compressed_body_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(ok(b"\x1f\x8b" + b"\0" * 32, **{"Content-Encoding": "gzip"}))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError, match="compressed body"):
        adapter.encode([Item(text="a")], ["dense"])


def test_an_oversized_body_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(ok(b"\0" * (2 << 20)))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError, match="exceeds the size limit"):
        adapter.encode([Item(text="a")], ["dense"])


class _SlowBody(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        for _ in range(5):
            time.sleep(0.1)
            yield b"\0"


def test_a_slow_body_is_cut_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote_sie, "REQUEST_DEADLINE_S", 0.2)
    recorder = Recorder(httpx.Response(200, stream=_SlowBody()))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError, match="exceeded the deadline"):
        adapter.encode([Item(text="a")], ["dense"])


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"\xc1", "not an encode response"),
        (encode_payload([dense_item(np.ones(8, dtype=np.float32))]), "8-dimensional vectors, the model declares 4"),
        (encode_payload([dense_item(np.array([1.0, np.nan, 0.0, 0.0], dtype=np.float32))]), "non-finite"),
        (encode_payload([{"dense": None}]), "without a dense vector"),
    ],
    ids=["undecodable", "wrong-dimension", "non-finite", "no-vector"],
)
def test_a_malformed_answer_is_an_error_not_a_vector(
    monkeypatch: pytest.MonkeyPatch, content: bytes, reason: str
) -> None:
    adapter = adapter_over(monkeypatch, Recorder(ok(content)))

    with pytest.raises(remote_sie.RemoteUpstreamError, match=reason):
        adapter.encode([Item(text="a")], ["dense"])


def test_an_item_without_text_is_refused_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(ok(b""))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(ValueError, match="text items only"):
        adapter.encode([Item(text="a"), Item()], ["dense"])

    assert recorder.requests == []


def test_a_short_answer_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = adapter_over(monkeypatch, Recorder(ok(encode_payload([dense_item(np.ones(4, dtype=np.float32))]))))

    with pytest.raises(remote_sie.RemoteUpstreamError, match="different number of results"):
        adapter.encode([Item(text="a"), Item(text="b")], ["dense"])
