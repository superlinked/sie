"""What the caller sees when an SIE upstream cannot serve a request yet.

Every upstream here is a real SIE app on loopback. Its fake model is held in its
load, fails to load, or is never reached, or the local rate cap or circuit
breaker refuses the call, so the local server answers through the real error
path on both ingress paths: single-node HTTP and the queue worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_server.adapters.errors import UpstreamRefusedError, UpstreamUnavailableError
from sie_server.api.helpers import upstream_unavailable_exception
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_types import EncodeBatchItem, ItemOutcome, ProcessEncodeBatchRequest
from sie_server.queue_executor import QueueExecutor, _inference_exception_outcome

REMOTE_MODEL = "acme/remote-fake"
NOT_READY_MESSAGE = "Model 'acme/remote-fake' is loading on its upstream, please retry"


@pytest.fixture(autouse=True)
def _credential(upstream_credential: str) -> str:
    return upstream_credential


@pytest.fixture
def held_upstream_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Hold the upstream's fake model in its load until the returned file exists."""
    release = tmp_path / "release-upstream-load"
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS", json.dumps({"sie-fake": {"load_latch_file": str(release), "latch_timeout_s": 60}})
    )
    yield release
    release.touch()


def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def encode(client: TestClient, text: str = "cold") -> httpx.Response:
    return client.post(
        f"/v1/encode/{REMOTE_MODEL}", json={"items": [{"text": text}]}, headers={"Accept": "application/json"}
    )


def embeddings(client: TestClient, text: str = "cold") -> httpx.Response:
    return client.post("/v1/embeddings", json={"model": REMOTE_MODEL, "input": text})


def test_a_cold_upstream_is_a_retryable_model_loading_with_its_wait(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    encode_when_loaded: Callable[..., httpx.Response],
    held_upstream_load: Path,
    upstream_credential: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    with sie_upstream() as upstream, TestClient(remote_app(upstream.url, preload=True)) as client:
        native = encode(client)
        openai = embeddings(client)
        held_upstream_load.touch()
        served = encode_when_loaded(client, REMOTE_MODEL, "cold")

    assert native.status_code == 503, native.text
    assert native.json() == {"detail": {"code": "MODEL_LOADING", "message": NOT_READY_MESSAGE}}
    assert openai.status_code == 503, openai.text
    assert openai.json()["error"] == {
        "code": "MODEL_LOADING",
        "message": NOT_READY_MESSAGE,
        "type": "server_error",
        "param": None,
    }
    for response in (native, openai):
        assert response.headers["retry-after"] == "5"
        assert response.headers["x-sie-served-by"] == "remote"
        assert response.headers["x-sie-upstream"] == "fake-sie"
        assert upstream_credential not in response.text
    assert served.status_code == 200, served.text
    assert upstream_credential not in caplog.text


def test_the_sdk_retries_a_cold_upstream_and_returns_its_vectors(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    serve_on_loopback: Callable[..., Any],
    held_upstream_load: Path,
) -> None:
    with sie_upstream() as upstream:

        def release_after_the_first_refusal() -> None:
            deadline = time.monotonic() + 30
            while not upstream.seen_authorization and time.monotonic() < deadline:
                time.sleep(0.05)
            held_upstream_load.touch()

        with serve_on_loopback(remote_app(upstream.url, preload=True)) as local_url:
            releaser = threading.Thread(target=release_after_the_first_refusal, daemon=True)
            releaser.start()
            with SIEClient(local_url) as client:
                served = client.encode(REMOTE_MODEL, {"text": "cold"})["dense"]
            releaser.join(timeout=30)
        with SIEClient(upstream.url) as direct:
            expected = direct.encode("sie-fake", {"text": "cold"})["dense"]

    np.testing.assert_allclose(served, expected, rtol=1e-6)
    assert len(upstream.seen_authorization) >= 3, "the refused request, the retried one, and the direct one"


def test_an_unreachable_upstream_is_a_retryable_queue_full(remote_app: Callable[..., Any]) -> None:
    with TestClient(remote_app(f"http://127.0.0.1:{closed_port()}", preload=True)) as client:
        native = encode(client)
        openai = embeddings(client)

    message = "The upstream serving model 'acme/remote-fake' is unavailable, please retry"
    assert native.status_code == 503, native.text
    assert native.json() == {"detail": {"code": "QUEUE_FULL", "message": message}}
    assert openai.status_code == 503, openai.text
    assert openai.json()["error"] == {"code": "QUEUE_FULL", "message": message, "type": "server_error", "param": None}
    for response in (native, openai):
        assert response.headers["retry-after"] == "5"
        assert response.headers["x-sie-served-by"] == "remote"
        assert response.headers["x-sie-upstream"] == "fake-sie"


def test_an_upstream_that_cannot_load_the_model_is_a_final_error(
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    encode_when_loaded: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIE_FAKE_FAULTS", json.dumps({"sie-fake": {"fail_load": True}}))
    with sie_upstream() as upstream, TestClient(remote_app(upstream.url, preload=True)) as client:
        response = encode_when_loaded(client, REMOTE_MODEL, "never")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": {"code": "INFERENCE_ERROR", "message": "internal error during encoding"}}
    assert "retry-after" not in response.headers


RETRYABLE_CODES = ("QUEUE_FULL", "MODEL_LOADING")


async def queue_outcome(
    registry: ModelRegistry, text: str = "cold", *, attempts: int = 1, interval_s: float = 0.0
) -> ItemOutcome:
    """Process one encode work item, again while it is refused as retryable, up to ``attempts`` times."""
    executor = QueueExecutor(registry)
    for attempt in range(attempts):
        batch = await executor.process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id=REMOTE_MODEL,
                items=[
                    EncodeBatchItem(
                        work_item_id=f"req-{attempt}.0",
                        request_id=f"req-{attempt}",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": text},
                    )
                ],
            )
        )
        outcome = batch.outcomes[0]
        if outcome.error_code not in RETRYABLE_CODES:
            return outcome
        await asyncio.sleep(interval_s)
    return outcome


async def queue_worker(
    tmp_path: Path, upstream_url: str, credential_env: str, *, requests_per_minute: int = 600
) -> ModelRegistry:
    models = tmp_path / "queue-models"
    models.mkdir()
    (models / "remote-fake.yaml").write_text(
        "sie_id: acme/remote-fake\n"
        "remote_backed: true\n"
        "inputs: {text: true}\n"
        "tasks: {encode: {dense: {dim: 384}}}\n"
        "profiles:\n"
        "  default:\n"
        "    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter\n"
        "    max_batch_tokens: 8192\n"
        "    adapter_options: {loadtime: {upstream: fake-sie, upstream_model: sie-fake}}\n",
        encoding="utf-8",
    )
    install_upstreams(
        {
            "fake-sie": Upstream.model_validate(
                {
                    "kind": "sie",
                    "base_url": upstream_url,
                    "api_key_secret": credential_env,
                    "rate_cap": {"requests_per_minute": requests_per_minute, "max_concurrency": 8},
                }
            )
        }
    )
    registry = ModelRegistry(models_dir=str(models), device="cpu", enable_hot_reload=False)
    await registry.load_async(REMOTE_MODEL, "cpu")
    return registry


def single_server_answer(error: UpstreamUnavailableError) -> tuple[int, str, str]:
    """The status, code and ``Retry-After`` the single server answers ``error`` with."""
    answer = upstream_unavailable_exception(error, REMOTE_MODEL)
    assert isinstance(answer.detail, dict)
    assert answer.headers is not None
    return answer.status_code, answer.detail["code"], answer.headers["Retry-After"]


def assert_answered_at_once(outcome: ItemOutcome, error: UpstreamUnavailableError) -> None:
    """The queue worker publishes the single server's code and wait instead of redelivering."""
    assert outcome.disposition == "publish_error_and_ack", outcome
    assert outcome.nak_delay_ms is None
    assert single_server_answer(error) == (503, outcome.error_code, str(outcome.retry_after_s))
    assert outcome.error is not None
    assert error.upstream not in outcome.error


async def test_the_queue_worker_answers_an_item_its_cold_upstream_refused_at_once(
    sie_upstream: Callable[..., Any], held_upstream_load: Path, tmp_path: Path, upstream_credential_env: str
) -> None:
    with sie_upstream() as upstream:
        registry = await queue_worker(tmp_path, upstream.url, upstream_credential_env)
        try:
            refused = await queue_outcome(registry)
            await asyncio.to_thread(held_upstream_load.touch)
            served = await queue_outcome(registry, attempts=100, interval_s=0.1)
        finally:
            await registry.unload_all_async()

    assert_answered_at_once(refused, UpstreamUnavailableError("fake-sie", "not_ready", retry_after_s=5, reason=""))
    assert (refused.error_code, refused.retry_after_s) == ("MODEL_LOADING", 5)
    assert served.disposition == "publish_and_ack", served.error


async def test_the_queue_worker_answers_an_item_its_unreachable_upstream_never_saw_at_once(
    tmp_path: Path, upstream_credential_env: str
) -> None:
    registry = await queue_worker(tmp_path, f"http://127.0.0.1:{closed_port()}", upstream_credential_env)
    try:
        outcome = await queue_outcome(registry)
    finally:
        await registry.unload_all_async()

    assert_answered_at_once(outcome, UpstreamUnavailableError("fake-sie", "unavailable", retry_after_s=5, reason=""))
    assert (outcome.error_code, outcome.retry_after_s) == ("QUEUE_FULL", 5)


async def test_the_queue_worker_answers_an_item_over_the_rate_cap_at_once(
    sie_upstream: Callable[..., Any], tmp_path: Path, upstream_credential_env: str
) -> None:
    def warm(url: str) -> None:
        with SIEClient(url) as direct:
            direct.encode("sie-fake", {"text": "warm"})

    with sie_upstream() as upstream:
        await asyncio.to_thread(warm, upstream.url)
        registry = await queue_worker(tmp_path, upstream.url, upstream_credential_env, requests_per_minute=1)
        try:
            served = await queue_outcome(registry, "first")
            capped = await queue_outcome(registry, "second")
        finally:
            await registry.unload_all_async()

    assert served.disposition == "publish_and_ack", served.error
    assert capped.retry_after_s is not None
    assert_answered_at_once(capped, UpstreamRefusedError("fake-sie", "rate_cap", retry_after_s=capped.retry_after_s))
    assert capped.error_code == "QUEUE_FULL"
    assert 1 <= capped.retry_after_s <= 60


async def test_the_queue_worker_answers_an_item_its_open_breaker_refused_at_once(
    tmp_path: Path, upstream_credential_env: str
) -> None:
    registry = await queue_worker(tmp_path, f"http://127.0.0.1:{closed_port()}", upstream_credential_env)
    try:
        failures = [await queue_outcome(registry) for _ in range(5)]
        refused = await queue_outcome(registry)
    finally:
        await registry.unload_all_async()

    assert {(outcome.error_code, outcome.retry_after_s) for outcome in failures} == {("QUEUE_FULL", 5)}
    assert refused.retry_after_s is not None
    assert_answered_at_once(
        refused, UpstreamRefusedError("fake-sie", "breaker_open", retry_after_s=refused.retry_after_s)
    )
    assert (refused.error_code, refused.retry_after_s) == ("QUEUE_FULL", 60)


async def test_the_queue_worker_publishes_a_final_upstream_error(
    sie_upstream: Callable[..., Any], tmp_path: Path, upstream_credential_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIE_FAKE_FAULTS", json.dumps({"sie-fake": {"fail_load": True}}))
    with sie_upstream() as upstream:
        registry = await queue_worker(tmp_path, upstream.url, upstream_credential_env)
        try:
            outcome = await queue_outcome(registry, attempts=100, interval_s=0.1)
        finally:
            await registry.unload_all_async()

    assert outcome.disposition == "publish_error_and_ack"
    assert outcome.error_code == "inference_error"
    assert outcome.error == "upstream answered 502 MODEL_LOAD_FAILED"


def work_item() -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id="req-1.0", request_id="req-1", item_index=0, total_items=1, timestamp=0.0, item={"text": "a"}
    )


@pytest.mark.parametrize("base_delay", ["2.0", "5.0", "120.0"])
@pytest.mark.parametrize("retry_after_s", [1, 30, 60])
@pytest.mark.parametrize("kind", ["busy", "unavailable", "not_ready"])
def test_a_refused_item_carries_the_upstreams_own_wait_whatever_the_redelivery_delay(
    monkeypatch: pytest.MonkeyPatch, base_delay: str, retry_after_s: int, kind: Any
) -> None:
    monkeypatch.setenv("SIE_NAK_DELAY_S", base_delay)
    error = UpstreamUnavailableError("fake-sie", kind, retry_after_s=retry_after_s, reason="answered 503")

    outcome = _inference_exception_outcome(work_item(), error)

    assert_answered_at_once(outcome, error)
    assert outcome.retry_after_s == retry_after_s
