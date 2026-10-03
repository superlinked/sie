"""The single-node routing decision for a model with the ``fallback`` policy.

These tests call :func:`route_request` against a real registry. The local
profile is the fake adapter, held unloading, failed or with a full queue by
the fake's faults and the worker's own queue. The remote profile is never
called here: the decision and the local warm-up are what is under test.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi import HTTPException, Request
from opentelemetry import trace
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.api.routing import ServingRoute, route_request
from sie_server.config import routing as routing_config
from sie_server.config.engine import EngineConfig
from sie_server.config.upstreams import Upstream, UpstreamConfigError, install_upstreams
from sie_server.core.prepared import make_text_item
from sie_server.core.registry import ModelRegistry
from sie_server.types.inputs import Item

SPAN = trace.INVALID_SPAN
TIMEOUT_S = 10.0


def model_config(sie_id: str, *, faults: dict[str, Any] | None = None, triggers: list[str] | None = None) -> dict:
    loadtime: dict[str, Any] = {"memory_footprint_bytes": 64 << 20, "fault_key": sie_id}
    if faults is not None:
        loadtime["faults"] = faults
    routing: dict[str, Any] = {"policy": "fallback", "fallback_profile": "remote"}
    if triggers is not None:
        routing["triggers"] = triggers
    return {
        "sie_id": sie_id,
        "package_backed": True,
        "inputs": {"text": True},
        "tasks": {"encode": {"dense": {"dim": 384}}},
        "routing": routing,
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": loadtime},
            },
            "remote": {
                "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "sie-fake"}},
            },
        },
    }


def registry_for(tmp_path: Path, *configs: dict, engine_config: EngineConfig | None = None) -> ModelRegistry:
    models = tmp_path / "models"
    models.mkdir()
    for config in configs:
        (models / f"{config['sie_id'].replace('/', '__')}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return ModelRegistry(models_dir=models, device="cpu", enable_hot_reload=False, engine_config=engine_config)


def request_for(registry: ModelRegistry) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(registry=registry)),
        }
    )


async def wait_until(predicate: Callable[[], bool], timeout_s: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            msg = "condition not reached within timeout"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def _remote_profiles(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Define the upstream the remote profiles name, and allow hybrid encode.

    Encode is the primitive an SIE upstream serves, so the decision is
    exercised through it; equivalence is a separate rule with its own tests.
    """
    monkeypatch.setattr(routing_config, "hybrid_equivalence_refusal", lambda config: None)
    install_upstreams(
        {
            "team-sie": Upstream.model_validate(
                {
                    "kind": "sie",
                    "base_url": "http://127.0.0.1:9",
                    "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
                }
            )
        }
    )
    yield
    install_upstreams({})


async def test_an_unloading_model_is_bridged_and_reloaded_behind_its_unload(tmp_path: Path) -> None:
    registry = registry_for(tmp_path, model_config("acme/hybrid", faults={"teardown_hang_s": 2}))
    try:
        await registry.load_async("acme/hybrid", device="cpu")
        unloading = asyncio.create_task(registry.unload_async("acme/hybrid"))
        await wait_until(lambda: registry.is_unloading("acme/hybrid"))

        route = await route_request(request_for(registry), "acme/hybrid", SPAN)
        await unloading
        await wait_until(lambda: registry.is_loaded("acme/hybrid"))

        assert route.key == "acme/hybrid:remote"
        assert route.upstream == "team-sie"
        assert route.fallback_reason == "model_loading"
        assert route.local_refusal is not None
        assert route.local_refusal.detail["code"] == "MODEL_NOT_LOADED"
    finally:
        await registry.unload_all_async()


async def test_a_failed_local_load_is_bridged_only_when_unhealthy_is_a_trigger(tmp_path: Path) -> None:
    registry = registry_for(
        tmp_path,
        model_config("acme/opted-in", faults={"fail_load": True}, triggers=["model_loading", "unhealthy"]),
        model_config("acme/default", faults={"fail_load": True}),
    )
    try:
        for name in ("acme/opted-in", "acme/default"):
            await registry.start_load_async(name, device="cpu")
        await wait_until(lambda: registry.is_failed("acme/opted-in") and registry.is_failed("acme/default"))

        route = await route_request(request_for(registry), "acme/opted-in", SPAN)
        with pytest.raises(HTTPException) as refused:
            await route_request(request_for(registry), "acme/default", SPAN)

        assert route.key == "acme/opted-in:remote"
        assert route.fallback_reason == "unhealthy"
        assert refused.value.status_code == 502
        assert refused.value.detail["code"] == "MODEL_LOAD_FAILED"
    finally:
        await registry.unload_all_async()


async def test_a_full_queue_is_bridged_before_submission_only_when_saturated_is_a_trigger(tmp_path: Path) -> None:
    registry = registry_for(
        tmp_path,
        model_config("acme/opted-in", triggers=["model_loading", "saturated"]),
        model_config("acme/default"),
        engine_config=EngineConfig(max_concurrent_requests=1),
    )
    try:
        decisions: dict[str, ServingRoute] = {}
        held = []
        for name in ("acme/opted-in", "acme/default"):
            await registry.load_async(name, device="cpu")
            worker = await registry.start_worker(name)
            held.append(await worker.submit([make_text_item([1, 2], 0)], [Item(text="held")], ["dense"]))
            decisions[name] = await route_request(request_for(registry), name, SPAN, queued_items=1)
        await asyncio.wait_for(asyncio.gather(*held), TIMEOUT_S)

        bridged = decisions["acme/opted-in"]
        assert bridged.key == "acme/opted-in:remote"
        assert bridged.fallback_reason == "saturated"
        assert bridged.local_refusal is not None
        assert bridged.local_refusal.detail["code"] == "QUEUE_FULL"
        assert decisions["acme/default"] == ServingRoute(key="acme/default")
    finally:
        await registry.unload_all_async()


async def test_a_remote_profile_that_cannot_load_answers_the_local_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(adapter: SieUpstreamAdapter, device: str) -> None:
        raise UpstreamConfigError("upstream 'team-sie' is not of kind 'sie'")

    monkeypatch.setattr(SieUpstreamAdapter, "load", refuse)
    latch = tmp_path / "release-local-load"
    registry = registry_for(tmp_path, model_config("acme/hybrid", faults={"load_latch_file": str(latch)}))
    try:
        with pytest.raises(HTTPException) as refused:
            await route_request(request_for(registry), "acme/hybrid", SPAN)

        assert refused.value.status_code == 503
        assert refused.value.detail["code"] == "MODEL_LOADING"
        assert refused.value.headers == {
            "Retry-After": "5",
            "X-SIE-Served-By": "local",
            "X-SIE-Fallback-Reason": "model_loading",
            "X-SIE-Fallback-Error": "MODEL_LOAD_FAILED",
        }
        assert registry.is_loading("acme/hybrid")
        assert registry.is_failed("acme/hybrid:remote")
    finally:
        latch.touch()
        await wait_until(lambda: not registry.is_loading("acme/hybrid"))
        await registry.unload_all_async()


LOCAL_REFUSAL = HTTPException(
    status_code=503,
    detail={"code": "MODEL_LOADING", "message": "Model 'acme/hybrid' is loading, please retry"},
    headers={"Retry-After": "5"},
)


@pytest.mark.parametrize(
    ("status_code", "code", "fallback_error"),
    [
        (503, "QUEUE_FULL", "QUEUE_FULL"),
        (503, "MODEL_LOADING", "MODEL_LOADING"),
        (503, "server_overloaded", "QUEUE_FULL"),
        (500, "inference_error", "INFERENCE_ERROR"),
        (400, "INPUT_TOO_LONG", "INPUT_TOO_LONG"),
        (400, "invalid_request", "INVALID_INPUT"),
        (502, "transport_failure", "INFERENCE_ERROR"),
        (422, None, "INVALID_INPUT"),
    ],
)
def test_a_failed_bridged_attempt_is_answered_with_the_local_refusal(
    status_code: int, code: str | None, fallback_error: str
) -> None:
    route = ServingRoute(
        key="acme/hybrid:remote", upstream="team-sie", fallback_reason="model_loading", local_refusal=LOCAL_REFUSAL
    )

    refusal = route.refusal_after(status_code, code)

    assert refusal is not None
    assert refusal.status_code == 503
    assert refusal.detail == LOCAL_REFUSAL.detail
    assert refusal.headers == {
        "Retry-After": "5",
        "X-SIE-Served-By": "local",
        "X-SIE-Fallback-Reason": "model_loading",
        "X-SIE-Fallback-Error": fallback_error,
    }


def test_a_route_that_was_not_bridged_replaces_no_error() -> None:
    assert ServingRoute(key="acme/hybrid").refusal_after(500, "INFERENCE_ERROR") is None
    assert ServingRoute(key="acme/remote-only", upstream="team-sie").refusal_after(503, "QUEUE_FULL") is None


@pytest.mark.parametrize("profile", ["default", "fast"])
async def test_an_explicit_profile_keeps_the_local_loading_refusal(tmp_path: Path, profile: str) -> None:
    latch = tmp_path / "release-local-load"
    registry = registry_for(tmp_path, model_config("acme/hybrid", faults={"load_latch_file": str(latch)}))
    try:
        await registry.start_load_async("acme/hybrid", device="cpu")
        await wait_until(lambda: registry.is_loading("acme/hybrid"))
        request = request_for(registry)
        with pytest.raises(HTTPException) as refused:
            await route_request(request, "acme/hybrid", SPAN, profile=profile)
        assert refused.value.status_code == 503
        assert refused.value.detail["code"] == "MODEL_LOADING"
        assert refused.value.headers == {"Retry-After": "5"}
        assert not registry.is_loaded("acme/hybrid:remote")
        assert not registry.is_loading("acme/hybrid:remote")
        assert not hasattr(request.state, "serving_route")
    finally:
        latch.touch()
        await wait_until(lambda: not registry.is_loading("acme/hybrid"))
        await registry.unload_all_async()
