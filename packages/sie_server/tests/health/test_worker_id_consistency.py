"""Regression test: WS-status ``name`` must match queue-runtime ``worker_id``.

C1 from the routing-rollout review: when ``SIE_WORKER_ID`` is set (or the
queue runtime falls through to ``uuid4``), the gateway used to register
the worker under a different name than the worker subscribed on,
silently breaking HRW direct dispatch.

The fix threads the queue runtime's resolved ``worker_id`` into
``build_status_message`` and uses it for the WS payload's ``name``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


async def _gpu_healthy() -> bool:
    return True


class _FakeQueueRuntime:
    """Minimal stand-in exposing only the ``worker_id`` attribute the
    status builder reads.
    """

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def update_saturation(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_build_status_message_uses_queue_runtime_worker_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the queue runtime is wired in, its ``worker_id`` wins over env vars.

    The scenario this prevents: ``SIE_WORKER_ID="w-prod-3"`` is set and
    ``HOSTNAME`` is the K8s pod name. Pre-fix, the queue runtime subscribed
    on ``sie.work.{pool}.{machine_profile}.{bundle}.*.w-prod-3`` but the
    WS payload reported
    ``name = {pod-name}`` — the gateway's HRW pick used ``{pod-name}``
    for the dispatch subject and no worker was listening on it.
    """
    monkeypatch.setenv("SIE_WORKER_ID", "w-prod-3")
    monkeypatch.setenv("HOSTNAME", "pod-abc-123")

    from sie_server.api.ws import build_status_message

    # `build_status_message` reaches into the registry for memory
    # thresholds, model status, etc.; we only care about the `name`
    # field here, so stub out the registry with the minimum surface.
    registry = MagicMock()
    registry.memory_manager.pressure_threshold_pct = 0.0
    registry._loaded = {}

    # Stub the helpers the builder calls so we don't drag in GPU or model
    # registry plumbing.
    monkeypatch.setattr("sie_server.api.ws.get_gpu_metrics", list)
    monkeypatch.setattr("sie_server.api.ws.get_model_status", lambda r: [])
    monkeypatch.setattr("sie_server.api.ws.compute_bundle_config_hash_cached", lambda r, b: "")
    monkeypatch.setattr("sie_server.api.ws.is_ready", lambda: True)
    monkeypatch.setattr("sie_server.api.ws.gpu_is_healthy_async", _gpu_healthy)

    queue_runtime = _FakeQueueRuntime(worker_id="w-prod-3")
    status: dict[str, Any] = await build_status_message(registry, queue_runtime=queue_runtime)

    # The direct-dispatch routing contract: the registered name must equal the
    # subscription's worker_id.
    assert status["name"] == "w-prod-3"


@pytest.mark.asyncio
async def test_build_status_message_falls_back_to_env_without_queue_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-queue deployments (no queue runtime) keep the legacy behaviour
    but now honour ``SIE_WORKER_ID`` too.
    """
    monkeypatch.setenv("SIE_WORKER_ID", "w-env-1")
    monkeypatch.setenv("HOSTNAME", "pod-other")

    from sie_server.api.ws import build_status_message

    registry = MagicMock()
    registry.memory_manager.pressure_threshold_pct = 0.0
    registry._loaded = {}
    monkeypatch.setattr("sie_server.api.ws.get_gpu_metrics", list)
    monkeypatch.setattr("sie_server.api.ws.get_model_status", lambda r: [])
    monkeypatch.setattr("sie_server.api.ws.compute_bundle_config_hash_cached", lambda r, b: "")
    monkeypatch.setattr("sie_server.api.ws.is_ready", lambda: True)
    monkeypatch.setattr("sie_server.api.ws.gpu_is_healthy_async", _gpu_healthy)

    status: dict[str, Any] = await build_status_message(registry, queue_runtime=None)
    assert status["name"] == "w-env-1"
