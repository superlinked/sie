"""What the caller sees when an SIE upstream cannot serve a request yet.

Every upstream here is a real SIE app on loopback. Its fake model is held in its
load, fails to load, or is never reached, so the local server answers through
the real error path on both ingress paths: single-node HTTP and the queue worker.
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
from sie_server.adapters.errors import UpstreamUnavailableError
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


async def queue_outcome(
    registry: ModelRegistry, text: str = "cold", *, attempts: int = 1, interval_s: float = 0.0
) -> ItemOutcome:
    """Process one encode work item, again while it is NAKed, up to ``attempts`` times."""
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
        if outcome.disposition != "nak_retry":
            return outcome
        await asyncio.sleep(interval_s)
    return outcome


async def queue_worker(tmp_path: Path, upstream_url: str, credential_env: str) -> ModelRegistry:
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
                    "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
                }
            )
        }
    )
    registry = ModelRegistry(models_dir=str(models), device="cpu", enable_hot_reload=False)
    await registry.load_async(REMOTE_MODEL, "cpu")
    return registry


async def test_the_queue_worker_redelivers_an_item_its_cold_upstream_refused(
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

    assert refused.disposition == "nak_retry"
    assert refused.nak_delay_ms == 5_000
    assert refused.error is None
    assert refused.error_code is None
    assert served.disposition == "publish_and_ack", served.error


async def test_the_queue_worker_redelivers_an_item_its_unreachable_upstream_never_saw(
    tmp_path: Path, upstream_credential_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIE_NAK_DELAY_S", "2.0")
    registry = await queue_worker(tmp_path, f"http://127.0.0.1:{closed_port()}", upstream_credential_env)
    try:
        outcome = await queue_outcome(registry)
    finally:
        await registry.unload_all_async()

    assert outcome.disposition == "nak_retry"
    assert outcome.nak_delay_ms == 5_000, "the upstream's wait, which is longer than the base delay"
    assert outcome.error_code is None


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


@pytest.mark.parametrize(
    ("base_delay", "retry_after_s", "delay_ms"),
    [("5.0", 1, 5_000), ("5.0", 30, 30_000), ("2.0", 60, 60_000), ("120.0", 5, 60_000)],
    ids=["base-delay-is-the-floor", "upstream-wait-above-the-floor", "upstream-wait-at-the-cap", "capped-at-60-s"],
)
def test_the_redelivery_delay_is_the_upstream_wait_between_the_base_delay_and_a_minute(
    monkeypatch: pytest.MonkeyPatch, base_delay: str, retry_after_s: int, delay_ms: int
) -> None:
    monkeypatch.setenv("SIE_NAK_DELAY_S", base_delay)
    error = UpstreamUnavailableError("fake-sie", "busy", retry_after_s=retry_after_s, reason="answered 429")

    outcome = _inference_exception_outcome(work_item(), error)

    assert outcome.disposition == "nak_retry"
    assert outcome.nak_delay_ms == delay_ms
