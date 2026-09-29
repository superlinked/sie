"""The NATS users and permissions of the chart, on a real nats-server.

The server runs the configuration the chart renders by default
(tools/ci/fixtures/sie-cluster-nats.conf, which test_helm_render.py keeps equal
to the chart).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import nats
import pytest
from nats.aio.client import Client
from sie_config.nats_publisher import NatsPublisher

ROOT = Path(__file__).resolve().parents[3]
SERVER_CONFIG = ROOT / "tools/ci/fixtures/sie-cluster-nats.conf"
PASSWORDS = {
    "config": "ConfigPassword0123456789abcdefghij",
    "gateway": "GatewayPassword0123456789abcdefghi",
    "worker": "WorkerPassword0123456789abcdefghij",
}
WORKER_INBOX_PREFIX = "_INBOX_WORKER"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def nats_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server is not on PATH")
    work = tmp_path_factory.mktemp("nats")
    port = _free_port()
    env = {
        **os.environ,
        "SERVER_NAME": "sie-auth-matrix",
        **{f"SIE_NATS_AUTH_{name.upper()}_PASSWORD": value for name, value in PASSWORDS.items()},
    }
    server = subprocess.Popen(  # noqa: S603
        [
            binary,
            "-c",
            str(SERVER_CONFIG),
            "-a",
            "127.0.0.1",
            "-p",
            str(port),
            "-m",
            str(_free_port()),
            "-sd",
            str(work / "jetstream"),
            "-P",
            str(work / "nats.pid"),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if time.monotonic() > deadline or server.poll() is not None:
                    raise RuntimeError("nats-server did not start") from None
                time.sleep(0.05)
        yield f"nats://127.0.0.1:{port}"
    finally:
        server.terminate()
        server.wait(timeout=10)


class Session:
    """A NATS connection that records the server's asynchronous errors."""

    def __init__(self, client: Client, errors: list[str]) -> None:
        self.client = client
        self.errors = errors

    def violations(self) -> list[str]:
        return [error for error in self.errors if "permissions violation" in error.lower()]


async def _connect(url: str, component: str) -> Session:
    errors: list[str] = []

    async def record(error: Exception) -> None:
        errors.append(str(error))

    client = await nats.connect(
        url,
        user=f"sie-{component}",
        password=PASSWORDS[component],
        inbox_prefix=WORKER_INBOX_PREFIX if component == "worker" else "_INBOX",
        error_cb=record,
        allow_reconnect=False,
    )
    return Session(client, errors)


async def _received(subscription: nats.aio.subscription.Subscription) -> list[bytes]:
    messages = []
    while True:
        try:
            message = await subscription.next_msg(timeout=0.5)
        except nats.errors.TimeoutError:
            return messages
        messages.append(message.data)


async def _api(session: Session, subject: str, body: dict | None = None) -> dict:
    reply = await session.client.request(subject, json.dumps(body or {}).encode(), timeout=2)
    return json.loads(reply.data)


async def _refused(session: Session, subject: str) -> bool:
    try:
        await session.client.request(subject, b"{}", timeout=1)
    except nats.errors.TimeoutError:
        return True
    return False


def _run(coroutine):
    return asyncio.run(coroutine)


@pytest.mark.parametrize(
    ("user", "password"),
    [(None, None), ("sie-gateway", PASSWORDS["worker"]), ("sie-unknown", PASSWORDS["worker"])],
)
def test_unauthenticated_connections_are_refused(nats_url: str, user: str | None, password: str | None) -> None:
    async def scenario() -> None:
        with pytest.raises(Exception, match=r"(?i)authorization"):
            await nats.connect(nats_url, user=user, password=password, allow_reconnect=False, max_reconnect_attempts=0)

    _run(scenario())


def test_sie_config_publishes_config_deltas_and_nothing_else(nats_url: str) -> None:
    async def scenario() -> None:
        gateway = await _connect(nats_url, "gateway")
        worker = await _connect(nats_url, "worker")
        config = await _connect(nats_url, "config")
        at_gateway = await gateway.client.subscribe("sie.config.models._all")
        at_worker = await worker.client.subscribe("sie.config.models.default")
        cancels = await worker.client.subscribe("cancel.>")
        await gateway.client.flush()
        await worker.client.flush()

        await config.client.publish("sie.config.models._all", b"all")
        await config.client.publish("sie.config.models.default", b"bundle")
        await config.client.publish("cancel.router.request", b"")
        await config.client.subscribe("sie.health.>")
        await config.client.flush()

        assert await _received(at_gateway) == [b"all"]
        assert await _received(at_worker) == [b"bundle"]
        assert await _received(cancels) == []
        assert len(config.violations()) == 2, config.errors
        for session in (gateway, worker, config):
            await session.client.close()

    _run(scenario())


def test_sie_config_client_authenticates_with_its_environment(nats_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_NATS_USER", "sie-config")
    monkeypatch.setenv("SIE_NATS_PASSWORD", PASSWORDS["config"])

    async def scenario() -> None:
        gateway = await _connect(nats_url, "gateway")
        deltas = await gateway.client.subscribe("sie.config.models._all")
        await gateway.client.flush()
        publisher = NatsPublisher(nats_url=nats_url)
        await publisher.connect()
        assert publisher.connected
        await publisher.publish_config_notification(
            model_id="org/model",
            profiles_added=["default"],
            affected_bundles=["default"],
            bundle_config_hashes={"default": "hash"},
            epoch=3,
            model_config_yaml="name: org/model\n",
        )
        (delta,) = await _received(deltas)
        assert json.loads(delta)["epoch"] == 3
        await publisher.disconnect()
        await gateway.client.close()

    _run(scenario())


def test_sie_config_client_ignores_url_credentials(nats_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_NATS_USER", raising=False)
    monkeypatch.delenv("SIE_NATS_PASSWORD", raising=False)
    monkeypatch.setenv("SIE_NATS_STARTUP_CONNECT_TIMEOUT_SEC", "2")
    url_with_credentials = nats_url.replace("nats://", f"nats://sie-config:{PASSWORDS['config']}@")

    async def scenario() -> None:
        publisher = NatsPublisher(nats_url=url_with_credentials)
        await publisher.connect()
        assert not publisher.connected
        await publisher.disconnect()

    _run(scenario())


def test_gateway_permissions(nats_url: str) -> None:
    async def scenario() -> None:
        gateway = await _connect(nats_url, "gateway")
        worker = await _connect(nats_url, "worker")
        at_worker = await worker.client.subscribe("work_cancel.>")
        await worker.client.flush()

        stream = {
            "name": "WORK_POOL_matrix",
            "subjects": ["sie.work.matrix.*.*.*"],
            "retention": "workqueue",
            "storage": "memory",
            "num_replicas": 1,
        }
        created = await _api(gateway, "$JS.API.STREAM.CREATE.WORK_POOL_matrix", stream)
        assert "error" not in created, created
        info = await _api(gateway, "$JS.API.STREAM.INFO.WORK_POOL_matrix")
        assert info["config"]["name"] == "WORK_POOL_matrix"
        ack = await gateway.client.request("sie.work.matrix.cpu.default.model", b"work", timeout=2)
        assert json.loads(ack.data)["seq"] == 1
        dlq = {"name": "DEAD_LETTERS", "subjects": ["sie.dlq.>"], "storage": "memory", "num_replicas": 1}
        assert "error" not in await _api(gateway, "$JS.API.STREAM.CREATE.DEAD_LETTERS", dlq)
        await gateway.client.publish("work_cancel.router.request", b"")
        await gateway.client.subscribe("sie.health.>")
        await gateway.client.subscribe("$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>")
        await gateway.client.flush()
        assert await _received(at_worker) == [b""]
        assert gateway.violations() == []

        await gateway.client.publish("sie.config.models._all", b"{}")
        await gateway.client.subscribe(f"{WORKER_INBOX_PREFIX}.>")
        await gateway.client.flush()
        assert await _refused(gateway, "$JS.API.STREAM.DELETE.WORK_POOL_matrix")
        assert await _refused(gateway, "$JS.API.STREAM.PURGE.WORK_POOL_matrix")
        assert len(gateway.violations()) >= 2, gateway.errors
        for session in (gateway, worker):
            await session.client.close()

    _run(scenario())


def test_worker_permissions(nats_url: str) -> None:
    async def scenario() -> None:
        gateway = await _connect(nats_url, "gateway")
        worker = await _connect(nats_url, "worker")
        results = await gateway.client.subscribe("_INBOX.router.>")
        heartbeats = await gateway.client.subscribe("sie.health.>")
        await gateway.client.flush()

        stream = {
            "name": "WORK_WORKER_matrix-0",
            "subjects": ["sie.work.wmatrix.cpu.default.*.matrix-0"],
            "retention": "workqueue",
            "storage": "memory",
            "num_replicas": 1,
        }
        assert "error" not in await _api(worker, "$JS.API.STREAM.CREATE.WORK_WORKER_matrix-0", stream)
        consumer = {
            "stream_name": "WORK_WORKER_matrix-0",
            "config": {
                "durable_name": "gen-matrix-0",
                "filter_subject": "sie.work.wmatrix.cpu.default.*.matrix-0",
                "ack_policy": "explicit",
            },
            "action": "create",
        }
        created = await _api(
            worker,
            "$JS.API.CONSUMER.CREATE.WORK_WORKER_matrix-0.gen-matrix-0.sie.work.wmatrix.cpu.default.*.matrix-0",
            consumer,
        )
        assert "error" not in created, created
        assert "error" not in await _api(worker, "$JS.API.CONSUMER.LIST.WORK_WORKER_matrix-0")
        ack = await gateway.client.request("sie.work.wmatrix.cpu.default.model.matrix-0", b"work", timeout=2)
        assert json.loads(ack.data)["seq"] == 1
        pull_inbox = worker.client.new_inbox()
        pulled = await worker.client.subscribe(pull_inbox)
        await worker.client.publish(
            "$JS.API.CONSUMER.MSG.NEXT.WORK_WORKER_matrix-0.gen-matrix-0", b'{"batch":1}', reply=pull_inbox
        )
        work = await pulled.next_msg(timeout=2)
        assert work.data == b"work"
        assert pull_inbox.startswith(f"{WORKER_INBOX_PREFIX}.")
        await work.ack()
        await worker.client.publish("_INBOX.router.request", b"result")
        await worker.client.publish("sie.health.matrix-0", b"heartbeat")
        for subject in ("cancel.>", "work_cancel.>", "batch_cancel.>", "sie.config.models.default"):
            await worker.client.subscribe(subject)
        await worker.client.flush()
        assert await _received(results) == [b"result"]
        assert await _received(heartbeats) == [b"heartbeat"]
        assert worker.violations() == [], worker.errors

        for subject in ("sie.config.models._all", "sie.work.wmatrix.cpu.default.model.matrix-0", "cancel.r.q"):
            await worker.client.publish(subject, b"forged")
        await worker.client.subscribe("_INBOX.>")
        await worker.client.subscribe("sie.health.>")
        await worker.client.flush()
        assert await _refused(worker, "$JS.API.STREAM.DELETE.WORK_WORKER_matrix-0")
        assert await _refused(worker, "$JS.API.STREAM.PURGE.WORK_WORKER_matrix-0")
        assert await _refused(worker, "$JS.API.STREAM.MSG.GET.WORK_WORKER_matrix-0")
        assert await _refused(worker, "$JS.API.DIRECT.GET.WORK_WORKER_matrix-0")
        assert len(worker.violations()) >= 5, worker.errors
        info = await _api(gateway, "$JS.API.STREAM.INFO.WORK_WORKER_matrix-0")
        assert info["state"]["messages"] == 0
        for session in (gateway, worker):
            await session.client.close()

    _run(scenario())
