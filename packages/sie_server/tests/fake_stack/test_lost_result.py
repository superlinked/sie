"""Fake Engine regression test (#1850, final scenario): the lost-result
timeout, characterized on the first full queue topology in any test —
gateway → NATS(JetStream) → sidecar → server, all real containers.

The fake worker registers normally (real sidecar heartbeats on
``sie.health.*``), consumes the work item off JetStream, and delivers it over
IPC to the Python server — where the dispatch-latch fault (#1849) holds it
forever. The gateway published successfully but no result ever arrives, so
after ``SIE_GATEWAY_REQUEST_TIMEOUT`` (shrunk from the 120 s default to 5 s
here; same code path) it returns the canonical lost-result response:
HTTP 504, ``x-sie-error-code: GATEWAY_TIMEOUT``, ``retry-after``, and the
"request was published, but no worker result reached the gateway" message
(``handlers/proxy.rs``). This also exercises the sidecar end to end —
closing the remaining #1850 acceptance items.

Notes pinned by this harness:
- First touch of a model over the queue path costs ~5 s (cold load + first
  batch + scheduler warmup), so the control request retries until warm.
- The latched model's stalls do NOT starve the sibling model — per-model
  inference executors isolate them (asserted at the end).
- The single ``sie-fake`` model declares ``tasks.generate``, so every fake
  (base + variants) classifies as a generation model for pool isolation —
  they can share a queue pool. The worker serves the base plus the
  ``small-a`` variant (``-m sie-fake,sie-fake:small-a``) so the latched and
  control models are separate adapter instances.
"""

from __future__ import annotations

import contextlib
import os
import socket
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

pytestmark = [pytest.mark.fake_stack, pytest.mark.docker]

GATEWAY_TIMEOUT_S = 5
_NATS_IMAGE = "nats:2.10-alpine"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _image(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


@dataclass
class QueueStack:
    gateway_url: str
    containers: list[Any]


@pytest.fixture(scope="module")
def queue_stack() -> Iterator[QueueStack]:
    """Boot the full queue topology with fake models, zero downloads."""
    docker = pytest.importorskip("docker")
    try:
        docker_client = docker.from_env(timeout=600)
        docker_client.ping()
    except Exception:  # noqa: BLE001 — no daemon = skip, matching sibling docker tests
        pytest.skip("Docker daemon not available")
    run_id = uuid.uuid4().hex[:8]
    network = docker_client.networks.create(f"sie-fake-queue-{run_id}")
    volume = docker_client.volumes.create(f"sie-fake-ipc-{run_id}")
    containers: list[Any] = []
    gateway_port = _free_port()

    def _run(name: str, image: str, **kwargs: Any) -> Any:
        container = docker_client.containers.run(
            image,
            name=f"{name}-{run_id}",
            network=network.name,
            detach=True,
            **kwargs,
        )
        containers.append(container)
        return container

    try:
        docker_client.images.pull(*_NATS_IMAGE.split(":"))
        _run("sie-fake-nats", _NATS_IMAGE, command=["-js"])
        # The IPC socket volume must be writable by both containers; align
        # them on uid 0 and open the mount point (volumes are root-owned).
        docker_client.containers.run(
            "alpine",
            command=["chmod", "0777", "/var/run/sie"],
            volumes={volume.name: {"bind": "/var/run/sie", "mode": "rw"}},
            remove=True,
        )
        common_volumes = {volume.name: {"bind": "/var/run/sie", "mode": "rw"}}
        worker_env = {
            "SIE_POOL": "default",
            "SIE_BUNDLE": "default",
            "SIE_MACHINE_PROFILE": "cpu",
            "SIE_IPC_SOCKET_PATH": "/var/run/sie/ipc.sock",
        }
        _run(
            "sie-fake-server",
            _image("SIE_DOCKER_IMAGE", "sie-server:test"),
            user="0:0",
            volumes=common_volumes,
            environment={
                **worker_env,
                "SIE_FAKE_MEMORY_BUDGET": "4GiB",
                # The lost-result fault: rerank dispatches block on a latch
                # that is never released (timeout far above the gateway's).
                "SIE_FAKE_FAULTS": (
                    '{"sie-fake:small-a": {"dispatch_latch_file": "/tmp/never-released", "latch_timeout_s": 600}}'
                ),
            },
            command=[
                "serve",
                "--host",
                "0.0.0.0",  # noqa: S104 — intentional bind to all interfaces in container
                "--port",
                "8080",
                "--device",
                "cpu",
                "-m",
                "sie-fake,sie-fake:small-a",
            ],
        )
        _run(
            "sie-fake-sidecar",
            _image("SIE_SIDECAR_IMAGE", "sie-server-sidecar:test"),
            user="0:0",
            volumes=common_volumes,
            environment={
                **worker_env,
                "SIE_NATS_URL": f"nats://sie-fake-nats-{run_id}:4222",
                "SIE_WORKER_ID": f"fake-worker-{run_id}",
                "SIE_GATEWAY_URL": f"http://sie-fake-gateway-{run_id}:8080",
            },
        )
        _run(
            "sie-fake-gateway",
            _image("SIE_GATEWAY_IMAGE", "sie-gateway:test"),
            ports={"8080/tcp": gateway_port},
            environment={
                "SIE_GATEWAY_HEALTH_MODE": "nats",
                "SIE_NATS_URL": f"nats://sie-fake-nats-{run_id}:4222",
                "SIE_GATEWAY_REQUEST_TIMEOUT": str(GATEWAY_TIMEOUT_S),
                "SIE_GATEWAY_ENABLE_POOLS": "1",
                "SIE_GATEWAY_CONFIGURED_GPUS": "cpu",
                "SIE_GATEWAY_CONFIGURED_PHYSICAL_LANES": (
                    '[{"pool":"default","machineProfile":"cpu","bundle":"default"}]'
                ),
            },
            command=["--port", "8080", "--host", "0.0.0.0"],  # noqa: S104 — container-internal bind
        )

        gateway_url = f"http://127.0.0.1:{gateway_port}"
        deadline = time.monotonic() + 120
        while True:
            try:
                pools = httpx.get(f"{gateway_url}/v1/pools", timeout=2.0).json()["pools"]
                if any(pool["status"].get("assigned_workers") for pool in pools):
                    break
            except (httpx.HTTPError, KeyError, ValueError):
                pass
            if time.monotonic() >= deadline:
                for container in containers:
                    container.reload()
                msg = "queue stack did not become ready (no worker assigned to any pool)"
                raise RuntimeError(msg)
            time.sleep(1.0)
        yield QueueStack(gateway_url=gateway_url, containers=containers)
    finally:
        for container in containers:
            with contextlib.suppress(Exception):  # best-effort teardown
                container.remove(force=True)
        try:
            volume.remove(force=True)
        finally:
            network.remove()


def _warm_embed(gateway_url: str) -> None:
    """First queue-path touch costs ~5 s (cold load + scheduler warmup) and
    may 504 against the shrunk gateway timeout; retry until warm.
    """
    deadline = time.monotonic() + 90
    while True:
        response = httpx.post(
            f"{gateway_url}/v1/embeddings",
            json={"model": "sie-fake", "input": ["warmup"]},
            timeout=GATEWAY_TIMEOUT_S + 10,
        )
        if response.status_code == 200:
            return
        assert response.status_code in (503, 504), response.text
        if time.monotonic() >= deadline:
            pytest.fail(f"queue path never warmed: {response.status_code} {response.text[:200]}")
        time.sleep(1.0)


def test_lost_result_times_out_at_gateway(queue_stack: QueueStack) -> None:
    gateway_url = queue_stack.gateway_url

    # Control: the topology genuinely works end to end (gateway → NATS →
    # sidecar → IPC → fake adapter → result), so the 504 below is a lost
    # result, not mis-wiring.
    _warm_embed(gateway_url)
    warm = httpx.post(
        f"{gateway_url}/v1/embeddings",
        json={"model": "sie-fake", "input": ["hello queue"]},
        timeout=30.0,
    )
    assert warm.status_code == 200
    assert len(warm.json()["data"][0]["embedding"]) == 384

    # The lost result: work is published and consumed, the dispatch latches
    # forever inside the adapter, and the gateway gives up at its deadline.
    start = time.monotonic()
    lost = httpx.post(
        f"{gateway_url}/v1/score/sie-fake:small-a",
        json={"query": {"text": "q"}, "items": [{"text": "a"}, {"text": "b"}]},
        timeout=GATEWAY_TIMEOUT_S * 6,
    )
    elapsed = time.monotonic() - start

    assert lost.status_code == 504, lost.text
    assert lost.headers.get("x-sie-error-code") == "GATEWAY_TIMEOUT"
    assert "retry-after" in {k.lower() for k in lost.headers}
    body = lost.text
    assert "was published" in body, body
    assert f"{GATEWAY_TIMEOUT_S}" in body
    assert elapsed >= GATEWAY_TIMEOUT_S * 0.9
    assert elapsed < GATEWAY_TIMEOUT_S * 5

    # Per-model executor isolation: the latched model's stalls must not
    # starve the sibling — embed still answers instantly.
    start = time.monotonic()
    sibling = httpx.post(
        f"{gateway_url}/v1/embeddings",
        json={"model": "sie-fake", "input": ["still alive"]},
        timeout=30.0,
    )
    assert sibling.status_code == 200
    assert time.monotonic() - start < GATEWAY_TIMEOUT_S
