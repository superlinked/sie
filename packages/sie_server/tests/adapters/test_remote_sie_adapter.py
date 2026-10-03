"""A remote-backed model served on a single node through an SIE upstream.

The upstream is a real SIE app serving the built-in fake model on loopback, so
the request and the vectors cross the actual wire format in both directions.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_sdk._msgpack import packb
from sie_server.adapters.errors import InputTooLongError, UpstreamUnavailableError
from sie_server.adapters.remote import _http as remote_http
from sie_server.adapters.remote import sie as remote_sie
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.oom import is_oom_error
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.types.inputs import InvalidInputError, Item

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CANARY = "sk-canary-77c2e41b09d8f356"


MIXED_MODEL = """\
sie_id: acme/mixed
package_backed: true
inputs:
  text: true
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.fake.adapter:FakeAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        memory_footprint_bytes: 67108864
        fault_key: acme/mixed
  remote:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: fake-sie
        upstream_model: sie-fake
"""


@pytest.fixture(autouse=True)
def _credential(upstream_credential: str) -> str:
    return upstream_credential


def test_a_remote_backed_model_returns_the_upstreams_vectors(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    encode_when_loaded: Callable[..., httpx.Response],
    upstream_credential: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    with sie_upstream() as upstream:
        with SIEClient(upstream.url) as direct:
            expected = direct.encode("sie-fake", {"text": "remote backends"})["dense"]
        upstream.seen_authorization.clear()

        with TestClient(remote_app(upstream.url)) as client:
            response = encode_when_loaded(client, "acme/remote-fake", "remote backends")
            embeddings = client.post("/v1/embeddings", json={"model": "acme/remote-fake", "input": "remote backends"})
            listed = client.get("/v1/models/acme/remote-fake")

    assert response.status_code == 200, response.text
    served = np.asarray(response.json()["items"][0]["dense"]["values"], dtype=np.float32)
    np.testing.assert_allclose(served, expected, rtol=1e-6)
    assert response.headers["x-sie-served-by"] == "remote"
    assert response.headers["x-sie-upstream"] == "fake-sie"
    assert embeddings.status_code == 200, embeddings.text
    assert embeddings.headers["x-sie-served-by"] == "remote"
    assert embeddings.headers["x-sie-upstream"] == "fake-sie"
    assert listed.json()["routing"] == {"policy": "remote_only", "upstream_kind": "sie"}
    assert upstream.seen_authorization == [f"Bearer {upstream_credential}", f"Bearer {upstream_credential}"]
    assert upstream_credential not in response.text
    assert upstream_credential not in str(response.headers)
    assert upstream_credential not in caplog.text


def test_a_local_model_discloses_local_serving(encode_when_loaded: Callable[..., httpx.Response]) -> None:
    app = AppFactory.create_app(AppStateConfig(models_dir=str(MODELS_DIR), model_filter=["sie-fake"], device="cpu"))

    with TestClient(app) as client:
        response = encode_when_loaded(client, "sie-fake", "local")
        listed = client.get("/v1/models/sie-fake")

    assert response.status_code == 200, response.text
    assert response.headers["x-sie-served-by"] == "local"
    assert "x-sie-upstream" not in response.headers
    assert listed.json()["routing"] == {"policy": None, "upstream_kind": None}


def test_the_served_side_follows_the_model_id_not_a_request_option(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    encode_when_loaded: Callable[..., httpx.Response],
) -> None:
    with sie_upstream() as upstream:
        with SIEClient(upstream.url) as direct:
            direct.encode("sie-fake", {"text": "warm"})
        app = remote_app(upstream.url, extra_models={"mixed.yaml": MIXED_MODEL})
        with TestClient(app) as client:
            selected = encode_when_loaded(client, "acme/mixed", "mixed", params={"options": {"profile": "remote"}})
            variant = encode_when_loaded(client, "acme/mixed:remote", "mixed")

    assert selected.status_code == 200, selected.text
    assert selected.headers["x-sie-served-by"] == "local"
    assert "x-sie-upstream" not in selected.headers
    assert variant.status_code == 200, variant.text
    assert variant.headers["x-sie-served-by"] == "remote"
    assert variant.headers["x-sie-upstream"] == "fake-sie"


def test_the_global_switch_refuses_remote_profiles_and_sends_nothing(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    encode_when_loaded: Callable[..., httpx.Response],
) -> None:
    with sie_upstream() as upstream:
        with TestClient(remote_app(upstream.url, remote_serving=False)) as client:
            response = encode_when_loaded(client, "acme/remote-fake", "remote backends")

    assert response.status_code == 502, response.text
    assert "remote serving is switched off" in response.text
    assert upstream.seen_authorization == []


def test_a_model_naming_an_undefined_upstream_stops_startup(remote_app: Callable[..., Any]) -> None:
    app = remote_app("http://127.0.0.1:9", upstream="nobody")

    with pytest.raises(ValueError, match="names an undefined upstream 'nobody'"), TestClient(app):
        pass


def test_a_malformed_credential_never_reaches_the_caller_or_the_upstream(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    encode_when_loaded: Callable[..., httpx.Response],
    upstream_credential: str,
    upstream_credential_env: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(upstream_credential_env, f"{upstream_credential} trailing-garbage")
    caplog.set_level(logging.DEBUG)
    with sie_upstream() as upstream:
        with TestClient(remote_app(upstream.url)) as client:
            response = encode_when_loaded(client, "acme/remote-fake", "remote backends")

    assert response.status_code == 500, response.text
    assert upstream.seen_authorization == []
    assert upstream_credential not in response.text
    assert upstream_credential not in caplog.text


def test_with_the_switch_off_a_stale_upstream_name_does_not_stop_startup(
    remote_app: Callable[..., Any], encode_when_loaded: Callable[..., httpx.Response]
) -> None:
    app = remote_app("http://127.0.0.1:9", upstream="nobody", remote_serving=False)

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
    monkeypatch: pytest.MonkeyPatch,
    recorder: Callable[[httpx.Request], httpx.Response],
    *,
    upstream_model: str = "sie-fake",
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
    recorder = Recorder(streamed(404, b"x", **{"X-SIE-Error-Code": f"leak {CANARY}"}))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert str(raised.value) == "upstream answered 404"


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

    with pytest.raises(UpstreamUnavailableError, match="exceeded the deadline") as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert raised.value.kind == "unavailable"
    assert raised.value.retry_after_s == remote_http.DEFAULT_RETRY_AFTER_S


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


def test_an_input_other_than_text_or_images_is_refused_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(ok(b""))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(InvalidInputError, match="text and image inputs only"):
        adapter.encode([Item(text="a"), Item(document={"data": b"%PDF", "format": "pdf"})], ["dense"])

    assert recorder.requests == []


def test_an_answer_with_more_results_than_items_sent_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = encode_payload([dense_item(np.ones(4, dtype=np.float32))] * 2)
    adapter = adapter_over(monkeypatch, Recorder(ok(answer)))

    with pytest.raises(remote_sie.RemoteUpstreamError, match="different number of results"):
        adapter.encode([Item(text="a")], ["dense"])


def error_answer(
    status: int,
    code: str | None = None,
    *,
    envelope: str = "detail",
    retry_after: str | None = None,
    header_code: bool = False,
    msgpack: bool = False,
    padding: int = 0,
) -> httpx.Response:
    """An upstream error answer. Its message carries the canary, which must never be relayed."""
    headers = {"Content-Type": "application/msgpack" if msgpack else "application/json"}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    detail: dict[str, Any] = {"message": f"upstream says {CANARY}" + " " * padding}
    if code is not None:
        if header_code:
            headers["X-SIE-Error-Code"] = code
        else:
            detail["code"] = code
    content = packb({envelope: detail}) if msgpack else json.dumps({envelope: detail}).encode()
    return streamed(status, content, **headers)


RETRYABLE_ANSWERS = [
    pytest.param(lambda: error_answer(503, "MODEL_LOADING", retry_after="5"), "not_ready", 5, id="503-model-loading"),
    pytest.param(
        lambda: error_answer(503, "PROVISIONING", header_code=True, retry_after="60"),
        "not_ready",
        60,
        id="503-provisioning-in-header",
    ),
    pytest.param(
        lambda: error_answer(503, "provisioning", envelope="error"),
        "not_ready",
        5,
        id="503-provisioning-openai-envelope",
    ),
    pytest.param(lambda: error_answer(503, "LORA_LOADING", retry_after="1"), "not_ready", 1, id="503-lora-loading"),
    pytest.param(
        lambda: error_answer(429, "RATE_LIMIT", envelope="error", retry_after="2"), "busy", 2, id="429-rate-limit"
    ),
    pytest.param(lambda: error_answer(429), "busy", 5, id="429-without-code-or-wait"),
    pytest.param(lambda: error_answer(503, "QUEUE_FULL", retry_after="1"), "busy", 1, id="503-queue-full"),
    pytest.param(
        lambda: error_answer(503, "RESOURCE_EXHAUSTED", msgpack=True, retry_after="7"),
        "busy",
        7,
        id="503-resource-exhausted-msgpack",
    ),
    pytest.param(
        lambda: error_answer(503, "BILLING_CAPACITY_UNAVAILABLE", envelope="error", retry_after="1"),
        "busy",
        1,
        id="503-billing-capacity-unavailable",
    ),
    pytest.param(
        lambda: error_answer(503, "QUEUE_UNAVAILABLE", retry_after="5"), "busy", 5, id="503-queue-unavailable-with-wait"
    ),
    pytest.param(
        lambda: error_answer(503, "transport_failure", envelope="error", retry_after="3"),
        "busy",
        3,
        id="503-transport-failure-with-wait",
    ),
    pytest.param(
        lambda: error_answer(503, "QUEUE_UNAVAILABLE"), "unavailable", 5, id="503-queue-unavailable-without-wait"
    ),
    pytest.param(lambda: error_answer(503), "unavailable", 5, id="503-without-code"),
    pytest.param(
        lambda: error_answer(503, "SOMETHING_NEW", retry_after="9"), "unavailable", 9, id="503-with-unknown-code"
    ),
    pytest.param(lambda: error_answer(500, "INFERENCE_ERROR"), "unavailable", 5, id="500"),
    pytest.param(lambda: error_answer(502), "unavailable", 5, id="502"),
    pytest.param(
        lambda: error_answer(504, "GATEWAY_TIMEOUT", header_code=True, retry_after="5"), "unavailable", 5, id="504"
    ),
    pytest.param(
        lambda: error_answer(503, "MODEL_LOADING", padding=128 << 10), "unavailable", 5, id="oversized-error-body"
    ),
    pytest.param(lambda: error_answer(503, "MODEL_LOADING", retry_after="0"), "not_ready", 1, id="wait-raised-to-1-s"),
    pytest.param(lambda: error_answer(503, "QUEUE_FULL", retry_after="86400"), "busy", 60, id="wait-capped-at-60-s"),
    pytest.param(lambda: error_answer(503, "QUEUE_FULL", retry_after="soon"), "busy", 5, id="unusable-wait"),
]


@pytest.mark.parametrize(("answer", "kind", "retry_after_s"), RETRYABLE_ANSWERS)
def test_an_answer_the_same_request_may_survive_is_retryable(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[], httpx.Response], kind: str, retry_after_s: int
) -> None:
    adapter = adapter_over(monkeypatch, Recorder(answer()))

    with pytest.raises(UpstreamUnavailableError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert (raised.value.upstream, raised.value.kind, raised.value.retry_after_s) == ("team-sie", kind, retry_after_s)
    assert CANARY not in str(raised.value)
    assert "SOMETHING_NEW" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not is_oom_error(raised.value)


FINAL_ANSWERS = [
    pytest.param(
        lambda: error_answer(429, "COLD_START_RATE_LIMITED", envelope="error"),
        remote_http.RemoteUpstreamError,
        "upstream answered 429 COLD_START_RATE_LIMITED",
        id="429-cold-start-rate-limited",
    ),
    pytest.param(
        lambda: error_answer(503, "ACCOUNT_STATE_UNAVAILABLE", envelope="error", retry_after="1"),
        remote_http.RemoteUpstreamError,
        "upstream answered 503 ACCOUNT_STATE_UNAVAILABLE",
        id="503-account-state-unavailable",
    ),
    pytest.param(
        lambda: error_answer(502, "MODEL_LOAD_FAILED"),
        remote_http.RemoteUpstreamError,
        "upstream answered 502 MODEL_LOAD_FAILED",
        id="502-model-load-failed",
    ),
    pytest.param(lambda: error_answer(501), remote_http.RemoteUpstreamError, "upstream answered 501", id="501"),
    pytest.param(lambda: error_answer(505), remote_http.RemoteUpstreamError, "upstream answered 505", id="505"),
    pytest.param(lambda: error_answer(401), remote_http.RemoteUpstreamError, "upstream answered 401", id="401"),
    pytest.param(
        lambda: error_answer(402, "INSUFFICIENT_CREDITS", envelope="error"),
        remote_http.RemoteUpstreamError,
        "upstream answered 402 INSUFFICIENT_CREDITS",
        id="402",
    ),
    pytest.param(
        lambda: error_answer(403, "ACCOUNT_SUSPENDED", envelope="error"),
        remote_http.RemoteUpstreamError,
        "upstream answered 403 ACCOUNT_SUSPENDED",
        id="403",
    ),
    pytest.param(
        lambda: error_answer(404, "MODEL_NOT_FOUND"),
        remote_http.RemoteUpstreamError,
        "upstream answered 404 MODEL_NOT_FOUND",
        id="404",
    ),
    pytest.param(lambda: error_answer(400), remote_http.RemoteUpstreamError, "upstream answered 400", id="400"),
    pytest.param(lambda: error_answer(413), remote_http.RemoteUpstreamError, "upstream answered 413", id="413"),
    pytest.param(
        lambda: error_answer(400, "INVALID_INPUT"),
        InvalidInputError,
        "the upstream refused the input (400 INVALID_INPUT)",
        id="400-invalid-input",
    ),
    pytest.param(
        lambda: error_answer(422, "INVALID_INPUT"),
        InvalidInputError,
        "the upstream refused the input (422 INVALID_INPUT)",
        id="422-invalid-input",
    ),
    pytest.param(
        lambda: error_answer(400, "INPUT_TOO_LONG"),
        InputTooLongError,
        "the upstream refused the input as too long (400 INPUT_TOO_LONG)",
        id="400-input-too-long",
    ),
    pytest.param(
        lambda: error_answer(413, "INPUT_TOO_LONG", header_code=True),
        InputTooLongError,
        "the upstream refused the input as too long (413 INPUT_TOO_LONG)",
        id="413-input-too-long",
    ),
]


@pytest.mark.parametrize(("answer", "error_type", "message"), FINAL_ANSWERS)
def test_an_answer_no_retry_would_fix_is_final(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[], httpx.Response], error_type: type[Exception], message: str
) -> None:
    adapter = adapter_over(monkeypatch, Recorder(answer()))

    with pytest.raises(error_type) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert type(raised.value) is error_type
    assert str(raised.value) == message


class Raiser:
    def __init__(self, error: Exception) -> None:
        self.requests: list[httpx.Request] = []
        self._error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect"),
        httpx.PoolTimeout("pool"),
        httpx.ProxyError("proxy"),
        httpx.WriteError("write"),
        httpx.WriteTimeout("write"),
        httpx.ReadError("read"),
        httpx.ReadTimeout("read"),
        httpx.RemoteProtocolError("dropped"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_failed_connection_is_retryable(monkeypatch: pytest.MonkeyPatch, error: httpx.TransportError) -> None:
    adapter = adapter_over(monkeypatch, Raiser(error))

    with pytest.raises(UpstreamUnavailableError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert str(raised.value) == f"upstream 'team-sie' is unavailable: request failed ({type(error).__name__})"
    assert raised.value.retry_after_s == remote_http.DEFAULT_RETRY_AFTER_S
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize(
    "error", [httpx.UnsupportedProtocol("scheme"), httpx.LocalProtocolError("header")], ids=lambda e: type(e).__name__
)
def test_a_request_the_client_cannot_send_is_final(monkeypatch: pytest.MonkeyPatch, error: httpx.HTTPError) -> None:
    adapter = adapter_over(monkeypatch, Raiser(error))

    with pytest.raises(remote_sie.RemoteUpstreamError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert str(raised.value) == f"upstream request failed ({type(error).__name__})"


def test_a_redirect_is_final_and_never_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(httpx.Response(307, headers={"Location": "https://collector.example/steal"}))
    adapter = adapter_over(monkeypatch, recorder)

    with pytest.raises(remote_sie.RemoteUpstreamError, match="redirect, which is refused"):
        adapter.encode([Item(text="a")], ["dense"])

    assert len(recorder.requests) == 1


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("5", 5),
        (" 12 ", 12),
        ("0", 1),
        ("1.5", 2),
        ("86400", 60),
        ("1e400", None),
        ("-1", None),
        ("nan", None),
        ("soon", None),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 1),
    ],
)
def test_retry_after_is_whole_seconds_within_the_bounded_range(value: str | None, seconds: int | None) -> None:
    assert remote_http.parse_retry_after(value) == seconds


def test_a_retry_after_date_is_the_wait_until_that_date() -> None:
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)

    seconds = remote_http.parse_retry_after(when)

    assert seconds is not None
    assert 29 <= seconds <= 30


@pytest.mark.parametrize(
    ("headers", "body", "code"),
    [
        ({"X-SIE-Error-Code": "QUEUE_FULL"}, b"", "QUEUE_FULL"),
        ({"X-SIE-Error-Code": f"leak {CANARY}"}, b'{"detail": {"code": "MODEL_LOADING"}}', "MODEL_LOADING"),
        ({}, b'{"error": {"code": "provisioning"}}', "PROVISIONING"),
        (
            {"Content-Type": "application/msgpack"},
            packb({"detail": {"code": "RESOURCE_EXHAUSTED"}}),
            "RESOURCE_EXHAUSTED",
        ),
        ({}, b'{"detail": {"code": "NOT_A_CODE_THIS_SERVER_KNOWS"}}', None),
        ({}, b'{"detail": "Service unavailable"}', None),
        ({}, b'{"detail": {"code": ["MODEL_LOADING"]}}', None),
        ({}, b"[1, 2]", None),
        ({}, b"\xff not json", None),
    ],
    ids=[
        "header",
        "unknown-header-then-body",
        "openai-envelope",
        "msgpack",
        "unknown-code",
        "detail-without-object",
        "code-not-a-string",
        "not-an-object",
        "undecodable",
    ],
)
def test_only_an_error_code_this_server_knows_is_read(headers: dict[str, str], body: bytes, code: str | None) -> None:
    assert remote_http.upstream_error_code(httpx.Headers(headers), body) == code


def mock_upstream_app(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any], answer: Callable[[], httpx.Response]
) -> Any:
    monkeypatch.setattr(
        remote_sie,
        "upstream_sync_client",
        lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(lambda _request: answer())),
    )
    return remote_app("https://sie.example.internal", preload=True)


def test_a_busy_upstream_is_a_retryable_queue_full_on_both_routes(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any]
) -> None:
    app = mock_upstream_app(monkeypatch, remote_app, lambda: error_answer(429, "RATE_LIMIT", retry_after="7"))

    with TestClient(app) as client:
        native = client.post(
            "/v1/encode/acme/remote-fake", json={"items": [{"text": "a"}]}, headers={"Accept": "application/json"}
        )
        openai = client.post("/v1/embeddings", json={"model": "acme/remote-fake", "input": "a"})

    assert native.status_code == 503, native.text
    assert native.json() == {
        "detail": {
            "code": "QUEUE_FULL",
            "message": "The upstream serving model 'acme/remote-fake' is busy, please retry",
        }
    }
    assert openai.status_code == 503, openai.text
    assert openai.json()["error"]["code"] == "QUEUE_FULL"
    assert openai.json()["error"]["type"] == "server_error"
    for response in (native, openai):
        assert response.headers["retry-after"] == "7"
        assert response.headers["x-sie-served-by"] == "remote"
        assert response.headers["x-sie-upstream"] == "fake-sie"
        assert CANARY not in response.text


@pytest.mark.parametrize(
    ("answer", "native_code", "openai_code", "message"),
    [
        pytest.param(
            lambda: error_answer(400, "INPUT_TOO_LONG"),
            "INPUT_TOO_LONG",
            "INPUT_TOO_LONG",
            "the upstream refused the input as too long (400 INPUT_TOO_LONG)",
            id="too-long",
        ),
        pytest.param(
            lambda: error_answer(422, "INVALID_INPUT"),
            "INVALID_INPUT",
            "invalid_request",
            "the upstream refused the input (422 INVALID_INPUT)",
            id="invalid",
        ),
    ],
)
def test_an_input_the_upstream_refuses_is_the_callers_error_on_both_routes(
    monkeypatch: pytest.MonkeyPatch,
    remote_app: Callable[..., Any],
    answer: Callable[[], httpx.Response],
    native_code: str,
    openai_code: str,
    message: str,
) -> None:
    app = mock_upstream_app(monkeypatch, remote_app, answer)

    with TestClient(app) as client:
        native = client.post(
            "/v1/encode/acme/remote-fake", json={"items": [{"text": "a"}]}, headers={"Accept": "application/json"}
        )
        openai = client.post("/v1/embeddings", json={"model": "acme/remote-fake", "input": "a"})

    assert native.status_code == 400, native.text
    assert native.json()["detail"] == {"code": native_code, "message": message}
    assert openai.status_code == 400, openai.text
    assert openai.json()["error"] == {
        "code": openai_code,
        "message": message,
        "type": "invalid_request_error",
        "param": "input",
    }


def test_an_upstream_that_refuses_the_credential_is_a_server_error(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any]
) -> None:
    app = mock_upstream_app(monkeypatch, remote_app, lambda: error_answer(401))

    with TestClient(app) as client:
        response = client.post(
            "/v1/encode/acme/remote-fake", json={"items": [{"text": "a"}]}, headers={"Accept": "application/json"}
        )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == {"code": "INFERENCE_ERROR", "message": "internal error during encoding"}
    assert "retry-after" not in response.headers
