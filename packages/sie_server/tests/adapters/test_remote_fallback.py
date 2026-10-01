"""Routing a request between a model's local profile and its remote profile on a single node.

The local profile is the fake adapter. Its load is held open with the fake's
load latch, so a model stays cold or loading for as long as a test needs. The
remote profile calls a real SIE app on loopback that serves the fake model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config import routing as routing_config

HYBRID_MODEL = """\
sie_id: acme/hybrid
package_backed: true
inputs:
  text: true
tasks:
  encode:
    dense:
      dim: 384
routing:
  policy: fallback
  fallback_profile: remote
profiles:
  default:
    adapter_path: sie_server.adapters.fake.adapter:FakeAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        memory_footprint_bytes: 67108864
        fault_key: acme/hybrid
        faults:
          load_latch_file: {latch}
          latch_timeout_s: 60
  query:
    extends: default
    adapter_options:
      runtime:
        is_query: true
  remote:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: fake-sie
        upstream_model: sie-fake
"""

REMOTE_EVERYTHING_MODEL = """\
sie_id: acme/remote-all
remote_backed: true
inputs:
  text: true
  audio: true
tasks:
  encode:
    dense:
      dim: 384
  score: {}
  extract: {}
  generate:
    context_length: 4096
    max_output_tokens: 64
profiles:
  default:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    kv_budget_tokens: 8192
    adapter_options:
      loadtime:
        upstream: fake-sie
        upstream_model: sie-fake
"""

FORBID = {"X-SIE-Remote": "forbid"}
UNREACHABLE = "http://127.0.0.1:9"


@pytest.fixture(autouse=True)
def _hybrid_encode_allowed(monkeypatch: pytest.MonkeyPatch, upstream_credential: str) -> str:
    """Lift the configuration gate that refuses hybrid encode until equivalence is shown.

    Encode is the primitive an SIE upstream serves, so the bridge is exercised
    through it; equivalence is a separate rule with its own tests.
    """
    monkeypatch.setattr(routing_config, "hybrid_equivalence_refusal", lambda config: None)
    return upstream_credential


@pytest.fixture
def latch(tmp_path: Path) -> Path:
    """The file whose creation releases every held local load of ``acme/hybrid``."""
    return tmp_path / "release-local-load"


def hybrid(latch: Path) -> dict[str, str]:
    return {"hybrid.yaml": HYBRID_MODEL.format(latch=latch)}


@contextmanager
def serving(app: FastAPI, latch: Path) -> Iterator[TestClient]:
    """Serve ``app``; on the way out, release the held local loads and let them finish."""
    with TestClient(app) as client:
        try:
            yield client
        finally:
            latch.touch()
            registry = app.state.registry
            wait_for(lambda: not any(registry.is_loading(name) for name in registry.model_names))


def encode(client: TestClient, model: str, headers: dict[str, str] | None = None, **params: Any) -> httpx.Response:
    body: dict[str, Any] = {"items": [{"text": "remote backends"}]}
    if params:
        body["params"] = params
    return client.post(f"/v1/encode/{model}", json=body, headers={"Accept": "application/json", **(headers or {})})


def warm(upstream: Any) -> None:
    with SIEClient(upstream.url) as direct:
        direct.encode("sie-fake", {"text": "warm"})
    upstream.seen_authorization.clear()


def wait_for(predicate: Callable[[], bool], timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.02)


def fallback_headers(response: httpx.Response) -> dict[str, str]:
    return {name: value for name, value in response.headers.items() if name.startswith("x-sie-fallback-")}


def test_a_cold_model_is_bridged_while_its_local_load_runs_and_then_served_locally(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    latch: Path,
    upstream_credential: str,
) -> None:
    with sie_upstream() as upstream:
        warm(upstream)
        app = remote_app(upstream.url, extra_models=hybrid(latch))
        with serving(app, latch) as client:
            registry = app.state.registry
            bridged = encode(client, "acme/hybrid")
            local_load_started = registry.is_loading("acme/hybrid")
            bridged_while_loading = encode(client, "acme/hybrid")
            listed = client.get("/v1/models/acme/hybrid")
            upstream_calls = list(upstream.seen_authorization)

            latch.touch()
            wait_for(lambda: registry.is_loaded("acme/hybrid"))
            local = encode(client, "acme/hybrid")

    assert bridged.status_code == 200, bridged.text
    assert bridged.json()["model"] == "acme/hybrid"
    assert bridged.headers["x-sie-served-by"] == "remote"
    assert bridged.headers["x-sie-upstream"] == "fake-sie"
    assert bridged.headers["x-sie-fallback-reason"] == "model_loading"
    assert "x-sie-fallback-error" not in bridged.headers
    assert local_load_started
    assert bridged_while_loading.status_code == 200, bridged_while_loading.text
    assert bridged_while_loading.headers["x-sie-fallback-reason"] == "model_loading"
    assert listed.json()["routing"] == {"policy": "fallback", "upstream_kind": "sie"}
    assert upstream_calls == [f"Bearer {upstream_credential}"] * 2

    assert local.status_code == 200, local.text
    assert local.headers["x-sie-served-by"] == "local"
    assert "x-sie-upstream" not in local.headers
    assert fallback_headers(local) == {}
    assert len(upstream.seen_authorization) == 2


def test_forbid_answers_with_the_local_refusal_and_sends_nothing_upstream(
    sie_upstream: Callable[..., Any], remote_app: Callable[..., Any], latch: Path
) -> None:
    with sie_upstream() as upstream:
        warm(upstream)
        app = remote_app(upstream.url, extra_models=hybrid(latch))
        with serving(app, latch) as client:
            cold = encode(client, "acme/hybrid", FORBID)
            local_load_started = app.state.registry.is_loading("acme/hybrid")
            loading = encode(client, "acme/hybrid", FORBID)

    for response in (cold, loading):
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "MODEL_LOADING"
        assert response.headers["retry-after"] == "5"
        assert "x-sie-served-by" not in response.headers
        assert fallback_headers(response) == {}
    assert local_load_started
    assert upstream.seen_authorization == []


def test_a_failed_remote_attempt_answers_the_local_refusal_and_names_both_outcomes(
    remote_app: Callable[..., Any], latch: Path
) -> None:
    app = remote_app(UNREACHABLE, extra_models=hybrid(latch))
    with serving(app, latch) as client:
        native = encode(client, "acme/hybrid")
        openai = client.post("/v1/embeddings", json={"model": "acme/hybrid", "input": "remote backends"})

    assert native.status_code == 503, native.text
    assert native.json()["detail"]["code"] == "MODEL_LOADING"
    assert openai.status_code == 503, openai.text
    assert openai.json()["error"]["code"] == "MODEL_LOADING"
    for response in (native, openai):
        assert response.headers["retry-after"] == "5"
        assert response.headers["x-sie-served-by"] == "local"
        assert "x-sie-upstream" not in response.headers
        assert fallback_headers(response) == {
            "x-sie-fallback-reason": "model_loading",
            "x-sie-fallback-error": "QUEUE_FULL",
        }


def test_a_request_that_names_a_profile_is_served_as_written(
    sie_upstream: Callable[..., Any], remote_app: Callable[..., Any], latch: Path
) -> None:
    with sie_upstream() as upstream:
        warm(upstream)
        app = remote_app(upstream.url, extra_models=hybrid(latch))
        with serving(app, latch) as client:
            by_option = encode(client, "acme/hybrid", options={"profile": "query"})
            by_variant = encode(client, "acme/hybrid:query")
            local_calls = list(upstream.seen_authorization)
            remote_variant = encode(client, "acme/hybrid:remote")

    for response in (by_option, by_variant):
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "MODEL_LOADING"
        assert fallback_headers(response) == {}
    assert local_calls == []
    assert remote_variant.status_code == 200, remote_variant.text
    assert remote_variant.headers["x-sie-served-by"] == "remote"
    assert remote_variant.headers["x-sie-upstream"] == "fake-sie"
    assert fallback_headers(remote_variant) == {}


def test_with_remote_serving_switched_off_the_policy_is_reported_and_never_bridges(
    sie_upstream: Callable[..., Any], remote_app: Callable[..., Any], latch: Path
) -> None:
    with sie_upstream() as upstream:
        warm(upstream)
        app = remote_app(upstream.url, remote_serving=False, extra_models=hybrid(latch))
        with serving(app, latch) as client:
            response = encode(client, "acme/hybrid")
            listed = client.get("/v1/models/acme/hybrid")

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "MODEL_LOADING"
    assert fallback_headers(response) == {}
    assert listed.json()["routing"] == {"policy": "fallback", "upstream_kind": "sie"}
    assert upstream.seen_authorization == []


def test_a_remote_backed_model_answers_its_first_request(
    sie_upstream: Callable[..., Any], remote_app: Callable[..., Any]
) -> None:
    with sie_upstream() as upstream:
        warm(upstream)
        with TestClient(remote_app(upstream.url)) as client:
            response = encode(client, "acme/remote-fake")

    assert response.status_code == 200, response.text
    assert response.headers["x-sie-served-by"] == "remote"
    assert fallback_headers(response) == {}


ROUTE_REQUESTS: dict[str, Callable[[TestClient, dict[str, str]], httpx.Response]] = {
    "encode": lambda client, headers: client.post(
        "/v1/encode/acme/remote-all", json={"items": [{"text": "x"}]}, headers=headers
    ),
    "score": lambda client, headers: client.post(
        "/v1/score/acme/remote-all", json={"query": {"text": "q"}, "items": [{"text": "d"}]}, headers=headers
    ),
    "extract": lambda client, headers: client.post(
        "/v1/extract/acme/remote-all", json={"items": [{"text": "x"}]}, headers=headers
    ),
    "generate": lambda client, headers: client.post(
        "/v1/generate/acme__remote-all", json={"prompt": "hi", "max_new_tokens": 4}, headers=headers
    ),
    "embeddings": lambda client, headers: client.post(
        "/v1/embeddings", json={"model": "acme/remote-all", "input": "x"}, headers=headers
    ),
    "completions": lambda client, headers: client.post(
        "/v1/completions", json={"model": "acme/remote-all", "prompt": "hi", "max_tokens": 4}, headers=headers
    ),
    "responses": lambda client, headers: client.post(
        "/v1/responses", json={"model": "acme/remote-all", "input": "hi", "max_output_tokens": 4}, headers=headers
    ),
    "chat": lambda client, headers: client.post(
        "/v1/chat/completions",
        json={"model": "acme/remote-all", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    ),
    "rerank": lambda client, headers: client.post(
        "/v1/rerank", json={"model": "acme/remote-all", "query": "q", "documents": ["d"]}, headers=headers
    ),
    "transcription": lambda client, headers: client.post(
        "/v1/audio/transcriptions",
        data={"model": "acme/remote-all"},
        files={"file": ("clip.wav", b"RIFF\x00\x00\x00\x00WAVEtest", "application/octet-stream")},
        headers=headers,
    ),
}


def error_message(response: httpx.Response) -> str:
    body = response.json()
    error = body.get("detail") or body.get("error") or {}
    return str(error.get("message", ""))


@pytest.mark.parametrize("route", sorted(ROUTE_REQUESTS))
def test_forbid_refuses_a_model_served_only_by_an_upstream_on_every_route(
    remote_app: Callable[..., Any], route: str
) -> None:
    app = remote_app(UNREACHABLE, extra_models={"remote-all.yaml": REMOTE_EVERYTHING_MODEL})
    with TestClient(app) as client:
        response = ROUTE_REQUESTS[route](client, FORBID)
        loaded = app.state.registry.is_loaded("acme/remote-all")

    assert response.status_code == 400, response.text
    assert "X-SIE-Remote: forbid" in error_message(response)
    assert not loaded


def test_forbid_refuses_an_explicit_remote_profile(remote_app: Callable[..., Any], latch: Path) -> None:
    app = remote_app(UNREACHABLE, extra_models=hybrid(latch))
    with serving(app, latch) as client:
        response = encode(client, "acme/hybrid:remote", FORBID)

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("value", ["allow", "Forbid", "forbid,forbid"])
def test_a_remote_header_other_than_forbid_is_refused(remote_app: Callable[..., Any], latch: Path, value: str) -> None:
    app = remote_app(UNREACHABLE, extra_models=hybrid(latch))
    with serving(app, latch) as client:
        native = encode(client, "acme/hybrid", {"X-SIE-Remote": value})
        openai = client.post(
            "/v1/embeddings", json={"model": "acme/hybrid", "input": "x"}, headers={"X-SIE-Remote": value}
        )
        load_started = app.state.registry.is_loading("acme/hybrid")

    assert native.status_code == 400, native.text
    assert native.json()["detail"]["code"] == "INVALID_INPUT"
    assert openai.status_code == 400, openai.text
    assert openai.json()["error"]["code"] == "INVALID_INPUT"
    assert openai.json()["error"]["type"] == "invalid_request_error"
    assert not load_started


def test_every_route_that_serves_a_model_declares_the_remote_header() -> None:
    spec = AppFactory.create_app(AppStateConfig()).openapi()
    operations = [
        ("/v1/encode/{model}", "post"),
        ("/v1/score/{model}", "post"),
        ("/v1/extract/{model}", "post"),
        ("/v1/generate/{model}", "post"),
        ("/v1/embeddings", "post"),
        ("/v1/completions", "post"),
        ("/v1/responses", "post"),
        ("/v1/audio/transcriptions", "post"),
        ("/v1/chat/completions", "post"),
        ("/v1/rerank", "post"),
    ]

    for path, method in operations:
        parameters = spec["paths"][path][method].get("parameters", [])
        assert any(p["name"] == "X-SIE-Remote" and p["in"] == "header" for p in parameters), path
    assert not any(p["name"] == "X-SIE-Remote" for p in spec["paths"]["/v1/models"]["get"].get("parameters", []))
