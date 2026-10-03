"""A remote-backed model served by a remote-lane worker through the queue path.

The upstream is a real SIE app serving the built-in fake model on loopback. The
worker receives the model the way a cluster's remote lane does, as an export
for the ``remote`` bundle, and serves an encode batch through the executor the
worker sidecar drives over IPC.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest
import uvicorn
from fastapi import Request
from sie_sdk import SIEClient
from sie_sdk._msgpack import unpackb
from sie_server.adapters.errors import UpstreamUnavailableError
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_types import (
    EncodeBatchItem,
    ItemOutcome,
    ProcessEncodeBatchRequest,
    ReplaceModelConfigEntry,
    ReplaceModelConfigsRequest,
)
from sie_server.queue_executor import QueueExecutor, _inference_exception_outcome

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
KEY_ENV = "REMOTE_QUEUE_PATH_TEST_KEY"
CANARY = "sk-canary-5a0e3c9d71b2f486"
MODEL_ID = "acme/remote-fake"
REMOTE_MODEL = f"""\
sie_id: {MODEL_ID}
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
        upstream: fake-sie
        upstream_model: sie-fake
"""


@contextmanager
def sie_upstream() -> Iterator[tuple[str, list[str | None]]]:
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


async def wait_until_ready(executor: QueueExecutor, model_id: str) -> None:
    deadline = time.monotonic() + 30
    while (state := await executor.ensure_model_ready(model_id)) != "ready":
        assert state in {"loading_started", "loading_in_progress"}, state
        assert time.monotonic() < deadline, "the remote profile did not load"
        await asyncio.sleep(0.05)


def dense_values(outcome: ItemOutcome) -> np.ndarray:
    if outcome.raw_output is not None and outcome.raw_output.dense is not None:
        return np.asarray(outcome.raw_output.dense.values, dtype=np.float32)
    assert outcome.result_msgpack is not None
    return np.asarray(unpackb(outcome.result_msgpack, numeric_arrays=True)["dense"]["values"], dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_upstreams(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(KEY_ENV, CANARY)
    yield
    install_upstreams({})


async def test_a_remote_lane_worker_serves_a_remote_backed_model_through_the_queue_path() -> None:
    with sie_upstream() as (upstream_url, seen_authorization):
        with SIEClient(upstream_url) as upstream:
            expected = upstream.encode("sie-fake", {"text": "remote lane"})["dense"]
        seen_authorization.clear()
        install_upstreams(
            {
                "fake-sie": Upstream.model_validate(
                    {
                        "kind": "sie",
                        "base_url": upstream_url,
                        "api_key_secret": KEY_ENV,
                        "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
                    }
                )
            }
        )
        executor = QueueExecutor(ModelRegistry(models_dir=None))

        applied = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="remote",
                epoch=1,
                bundle_config_hash="",
                models=[ReplaceModelConfigEntry(model_id=MODEL_ID, model_config=REMOTE_MODEL)],
            )
        )
        await wait_until_ready(executor, MODEL_ID)
        batch = await executor.process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id=MODEL_ID,
                items=[
                    EncodeBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": "remote lane"},
                        output_types=["dense"],
                        bundle_config_hash=applied.bundle_config_hash,
                    )
                ],
            )
        )

    assert applied.applied_models == [MODEL_ID]
    assert applied.unsupported_models == []
    assert applied.bundle_config_hash
    (outcome,) = batch.outcomes
    assert outcome.disposition == "publish_and_ack", outcome.error
    np.testing.assert_allclose(dense_values(outcome), expected, rtol=1e-6)
    assert seen_authorization == [f"Bearer {CANARY}"]


@pytest.mark.parametrize("kind", ["busy", "unavailable", "not_ready"])
def test_remote_refusal_preserves_retry_hint_on_the_ipc_wire(kind: Any) -> None:
    item = EncodeBatchItem(
        work_item_id="req.0",
        request_id="req",
        item_index=0,
        total_items=1,
        timestamp=time.time(),
        item={"text": "remote lane"},
    )
    outcome = _inference_exception_outcome(
        item, UpstreamUnavailableError("fake-sie", kind, retry_after_s=9, reason="unavailable")
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(outcome), type=ItemOutcome)
    assert decoded.disposition == "nak_retry"
    assert decoded.error_code == "QUEUE_FULL"
    assert decoded.retry_after_s == 9
    assert decoded.nak_delay_ms is not None
    assert decoded.nak_delay_ms >= 9_000
    assert decoded.error is None
