"""The single-node routing decision for a model with the ``fallback`` policy.

These tests call :func:`route_request` against a real registry. The local
profile is the fake adapter, held unloading, failed or with a full queue by
the fake's faults and the worker's own queue. The remote profile is never
called here: the decision and the local warm-up are what is under test.

The vectors in ``wire-fixtures/remote_routing.json``, which the gateway checks
too, drive the decision, ``X-SIE-Remote`` and the refusal that answers a failed
remote attempt.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
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
VECTORS = json.loads(
    (Path(__file__).resolve().parents[3] / "wire-fixtures" / "remote_routing.json").read_text(encoding="utf-8")
)


def single_node(vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The vectors that apply to a single server."""
    return [vector for vector in vectors if "single_node" in vector.get("topologies", ["single_node"])]


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


def request_for(registry: ModelRegistry, remote: list[str] | None = None) -> Request:
    """A request whose ``X-SIE-Remote`` fields carry ``remote``, one field per value."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(b"x-sie-remote", value.encode()) for value in remote or []],
            "app": SimpleNamespace(state=SimpleNamespace(registry=registry)),
        }
    )


REMOTE_ONLY = {
    "sie_id": "acme/remote-only",
    "remote_backed": True,
    "inputs": {"text": True},
    "tasks": {"encode": {"dense": {"dim": 384}}},
    "profiles": {
        "default": {
            "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
            "max_batch_tokens": 8192,
            "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "sie-fake"}},
        }
    },
}
MODEL_NAMES = {
    "bare": "acme/hybrid",
    "local_profile": "acme/hybrid",
    "remote_profile": "acme/hybrid:remote",
    "remote_only": "acme/remote-only",
}


@asynccontextmanager
async def in_local_state(tmp_path: Path, state: str, triggers: list[str] | None = None) -> AsyncIterator[ModelRegistry]:
    """A registry whose ``acme/hybrid`` local profile is in the abstract local ``state`` of the vectors.

    ``saturated`` holds the one slot of the local worker, so a request that
    adds one item finds the queue full.
    """
    latch = tmp_path / "release-local-load"
    faults = {"fail_load": True} if state == "unhealthy" else {"load_latch_file": str(latch)}
    registry = registry_for(
        tmp_path,
        model_config("acme/hybrid", faults=faults, triggers=triggers),
        REMOTE_ONLY,
        engine_config=EngineConfig(max_concurrent_requests=1) if state == "saturated" else None,
    )
    held = []
    try:
        if state in {"ready", "saturated"}:
            latch.touch()
            await registry.load_async("acme/hybrid", device="cpu")
        if state == "saturated":
            worker = await registry.start_worker("acme/hybrid")
            held.append(await worker.submit([make_text_item([1, 2], 0)], [Item(text="held")], ["dense"]))
        if state == "loading":
            await registry.start_load_async("acme/hybrid", device="cpu")
            await wait_until(lambda: registry.is_loading("acme/hybrid"))
        if state == "unhealthy":
            await registry.start_load_async("acme/hybrid", device="cpu")
            await wait_until(lambda: registry.is_failed("acme/hybrid"))
        yield registry
    finally:
        latch.touch()
        await asyncio.wait_for(asyncio.gather(*held), TIMEOUT_S)
        await wait_until(lambda: not registry.is_loading("acme/hybrid"))
        await registry.unload_all_async()


def fallback_headers(error: HTTPException) -> dict[str, str]:
    return {name: value for name, value in (error.headers or {}).items() if name.startswith("X-SIE-Fallback")}


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


@pytest.mark.parametrize("vector", single_node(VECTORS["fallback_error"]), ids=lambda vector: str(vector["code"]))
def test_a_failed_bridged_attempt_is_answered_with_the_local_refusal(vector: dict[str, Any]) -> None:
    route = ServingRoute(
        key="acme/hybrid:remote", upstream="team-sie", fallback_reason="model_loading", local_refusal=LOCAL_REFUSAL
    )

    refusal = route.refusal_after(vector["status"], vector["code"])

    assert refusal is not None
    assert refusal.status_code == 503
    assert refusal.detail == LOCAL_REFUSAL.detail
    assert refusal.headers == {
        "Retry-After": "5",
        "X-SIE-Served-By": "local",
        "X-SIE-Fallback-Reason": "model_loading",
        "X-SIE-Fallback-Error": vector["fallback_error"],
    }


@pytest.mark.parametrize(
    "vector", single_node(VECTORS["restored_refusal"]), ids=lambda vector: vector["fallback_reason"]
)
def test_the_restored_refusal_keeps_the_local_answer_and_names_both_outcomes(vector: dict[str, Any]) -> None:
    local = vector["local"]
    retry = {"Retry-After": local["retry_after"]} if local["retry_after"] is not None else {}
    route = ServingRoute(
        key="acme/hybrid:remote",
        upstream="team-sie",
        fallback_reason=vector["fallback_reason"],
        local_refusal=HTTPException(
            status_code=local["status"],
            detail={"code": local["code"], "message": "local refusal"},
            headers=retry or None,
        ),
    )

    refusal = route.refusal_after(vector["attempt"]["status"], vector["attempt"]["code"])

    assert refusal is not None
    assert refusal.status_code == local["status"]
    assert refusal.detail == {"code": local["code"], "message": "local refusal"}
    assert refusal.headers == {
        **retry,
        "X-SIE-Served-By": "local",
        "X-SIE-Fallback-Reason": vector["fallback_reason"],
        "X-SIE-Fallback-Error": vector["fallback_error"],
    }


@pytest.mark.parametrize(
    "vector",
    single_node(VECTORS["bridge"]),
    ids=lambda vector: f"{vector['local']}-{'+'.join(vector['triggers'] or ['default'])}",
)
async def test_the_local_state_decides_the_bridge_as_the_shared_vectors_say(
    tmp_path: Path, vector: dict[str, Any]
) -> None:
    refusal = None
    async with in_local_state(tmp_path, vector["local"], vector["triggers"]) as registry:
        try:
            route = await route_request(request_for(registry), "acme/hybrid", SPAN, queued_items=1)
        except HTTPException as error:
            route, refusal = None, error

    if refusal is not None:
        assert fallback_headers(refusal) == {}
    if vector["remote"]:
        assert route is not None
        assert route.key == "acme/hybrid:remote"
        assert route.upstream == "team-sie"
        assert route.fallback_reason == vector["fallback_reason"]
    else:
        assert route in (None, ServingRoute(key="acme/hybrid"))


@pytest.mark.parametrize("remote", VECTORS["remote_header"]["refused"], ids=repr)
async def test_a_remote_header_other_than_one_exact_forbid_is_refused(tmp_path: Path, remote: list[str]) -> None:
    async with in_local_state(tmp_path, "ready") as registry:
        with pytest.raises(HTTPException) as refused:
            await route_request(request_for(registry, remote), "acme/hybrid", SPAN)

    assert refused.value.status_code == VECTORS["remote_header"]["refusal"]["status"]
    assert refused.value.detail["code"] == VECTORS["remote_header"]["refusal"]["code"]


@pytest.mark.parametrize("remote", VECTORS["remote_header"]["accepted"], ids=repr)
async def test_one_exact_forbid_keeps_a_ready_model_local(tmp_path: Path, remote: list[str]) -> None:
    async with in_local_state(tmp_path, "ready") as registry:
        route = await route_request(request_for(registry, remote), "acme/hybrid", SPAN)

    assert route == ServingRoute(key="acme/hybrid")


@pytest.mark.parametrize("vector", single_node(VECTORS["forbid"]), ids=lambda vector: vector["model"])
async def test_forbid_follows_the_shared_vectors(tmp_path: Path, vector: dict[str, Any]) -> None:
    refusal = None
    async with in_local_state(tmp_path, vector.get("local", "ready")) as registry:
        try:
            route = await route_request(request_for(registry, ["forbid"]), MODEL_NAMES[vector["model"]], SPAN)
        except HTTPException as error:
            route, refusal = None, error
        remote_loaded = registry.is_loaded("acme/hybrid:remote") or registry.is_loaded("acme/remote-only")

    assert not remote_loaded
    if refusal is not None:
        assert fallback_headers(refusal) == {}
    if "refusal" in vector:
        assert refusal is not None
        assert refusal.status_code == vector["refusal"]["status"]
        assert refusal.detail["code"] == vector["refusal"]["code"]
    else:
        assert not vector["remote"]
        assert route in (None, ServingRoute(key="acme/hybrid"))


@pytest.mark.parametrize("vector", single_node(VECTORS["named_profile"]), ids=lambda vector: vector["model"])
async def test_a_named_profile_is_served_as_written(tmp_path: Path, vector: dict[str, Any]) -> None:
    profile = "default" if vector["model"] == "local_profile" else None
    refusal = None
    async with in_local_state(tmp_path, vector["local"]) as registry:
        try:
            route = await route_request(request_for(registry), MODEL_NAMES[vector["model"]], SPAN, profile=profile)
        except HTTPException as error:
            route, refusal = None, error

    if refusal is not None:
        assert fallback_headers(refusal) == {}
    if vector["remote"]:
        assert route == ServingRoute(key="acme/hybrid:remote", upstream="team-sie")
    else:
        assert route is None


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
