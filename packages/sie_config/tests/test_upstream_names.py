"""sie-config refuses a write whose remote profile names an upstream the remote workers do not define."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_config.config_api import router as config_router
from sie_config.model_registry import REMOTE_ADAPTER_MODULE_PREFIX, ModelRegistry
from sie_server.config.model import REMOTE_ADAPTER_MODULE_PREFIX as WORKER_REMOTE_ADAPTER_MODULE_PREFIX
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.registry import ModelRegistry as WorkerModelRegistry
from sie_server.ipc_types import ReplaceModelConfigEntry, ReplaceModelConfigsRequest
from sie_server.queue_executor import QueueExecutor

BUNDLES_DIR = Path(__file__).resolve().parents[2] / "sie_server" / "bundles"
REMOTE_ADAPTER = "sie_server.adapters.remote.sie:SieUpstreamAdapter"
LOCAL_ADAPTER = "sie_server.adapters.bert_flash:BertFlashAdapter"
TASKS = {"encode": {"dense": {"dim": 384}}}


def remote_profile(upstream: object) -> dict[str, Any]:
    return {
        "adapter_path": REMOTE_ADAPTER,
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": {"upstream": upstream, "upstream_model": "org/name"}},
    }


def remote_backed(upstream: object) -> dict[str, Any]:
    return {
        "sie_id": "acme/remote",
        "remote_backed": True,
        "tasks": TASKS,
        "profiles": {"default": remote_profile(upstream)},
    }


def hybrid(upstream: object, **profiles: dict[str, Any]) -> dict[str, Any]:
    return {
        "sie_id": "acme/hybrid",
        "hf_id": "acme/hybrid",
        "tasks": TASKS,
        "profiles": {
            "default": {"adapter_path": LOCAL_ADAPTER, "max_batch_tokens": 4096},
            "remote": remote_profile(upstream),
            **profiles,
        },
    }


def override_upstream(upstream: str) -> dict[str, Any]:
    return {"extends": "remote", "adapter_options": {"loadtime": {"upstream": upstream, "upstream_model": "org/name"}}}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    app = FastAPI()
    app.include_router(config_router)
    app.state.model_registry = ModelRegistry(BUNDLES_DIR, models_dir)
    app.state.nats_publisher = None
    app.state.config_store = None
    return TestClient(app)


@pytest.fixture
def declare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_UPSTREAM_NAMES", "team-sie")


def post(client: TestClient, config: dict[str, Any]) -> httpx.Response:
    return client.post("/v1/configs/models", content=yaml.safe_dump(config))


def refusal(profile: str, model: str, upstream: object) -> dict[str, Any]:
    return {
        "error": "validation_error",
        "details": [{"message": f"Profile '{profile}' of '{model}' names an undefined upstream {upstream!r}"}],
    }


@pytest.mark.usefixtures("declare")
def test_a_remote_profile_naming_an_undefined_upstream_is_refused_and_not_stored(client: TestClient) -> None:
    response = post(client, remote_backed("nobody"))

    assert response.status_code == 422
    assert response.json()["detail"] == refusal("default", "acme/remote", "nobody")
    assert client.get("/v1/configs/models/acme/remote").status_code == 404
    assert post(client, remote_backed("team-sie")).status_code == 201


@pytest.mark.parametrize("names", ["team-sie", " other , team-sie ,"])
def test_the_list_is_comma_separated(client: TestClient, monkeypatch: pytest.MonkeyPatch, names: str) -> None:
    monkeypatch.setenv("SIE_UPSTREAM_NAMES", names)

    assert post(client, remote_backed("team-sie")).status_code == 201


def test_without_the_list_upstream_names_are_left_to_the_worker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SIE_UPSTREAM_NAMES", raising=False)

    assert post(client, remote_backed("nobody")).status_code == 201


def test_an_empty_list_refuses_every_remote_profile_and_no_local_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIE_UPSTREAM_NAMES", "")
    local = {"sie_id": "acme/local", "hf_id": "acme/local", "tasks": TASKS, "profiles": hybrid("x")["profiles"]}
    del local["profiles"]["remote"]

    assert post(client, remote_backed("team-sie")).status_code == 422
    assert post(client, local).status_code == 201


@pytest.mark.usefixtures("declare")
@pytest.mark.parametrize("upstream", [["team-sie"], 7, None])
def test_an_upstream_that_is_not_a_name_is_refused(client: TestClient, upstream: object) -> None:
    assert post(client, remote_backed(upstream)).status_code == 422


@pytest.mark.usefixtures("declare")
def test_an_appended_profile_is_checked_after_extends(client: TestClient) -> None:
    assert post(client, hybrid("team-sie")).status_code == 201

    inherits = post(client, {"sie_id": "acme/hybrid", "profiles": {"alias": {"extends": "remote"}}})
    overrides = post(client, {"sie_id": "acme/hybrid", "profiles": {"elsewhere": override_upstream("nobody")}})

    assert inherits.status_code == 201
    assert overrides.status_code == 422
    assert overrides.json()["detail"] == refusal("elsewhere", "acme/hybrid", "nobody")


def test_only_the_written_profiles_are_checked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert post(client, hybrid("retired")).status_code == 201
    monkeypatch.setenv("SIE_UPSTREAM_NAMES", "team-sie")

    appended = {"sie_id": "acme/hybrid", "profiles": {"small": {"extends": "default", "max_batch_tokens": 1024}}}

    assert post(client, appended).status_code == 201


@pytest.mark.usefixtures("declare")
def test_a_replacement_is_checked_and_keeps_the_stored_config(client: TestClient) -> None:
    assert post(client, remote_backed("team-sie")).status_code == 201

    response = client.put("/v1/configs/models/acme/remote", content=yaml.safe_dump(remote_backed("nobody")))

    assert response.status_code == 422
    assert response.json()["detail"] == refusal("default", "acme/remote", "nobody")
    stored = client.app.state.model_registry.get_full_config("acme/remote")
    assert stored["profiles"]["default"]["adapter_options"]["loadtime"]["upstream"] == "team-sie"


@pytest.fixture
def remote_worker_upstreams() -> Iterator[None]:
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


async def accepted_by_a_remote_worker(config: dict[str, Any]) -> bool:
    executor = QueueExecutor(WorkerModelRegistry(models_dir=None))
    response = await executor.replace_model_configs(
        ReplaceModelConfigsRequest(
            bundle_id="remote",
            epoch=1,
            bundle_config_hash="",
            models=[ReplaceModelConfigEntry(model_id=config["sie_id"], model_config=yaml.safe_dump(config))],
        )
    )
    return config["sie_id"] in response.applied_models


@pytest.mark.usefixtures("declare", "remote_worker_upstreams")
@pytest.mark.parametrize(
    ("config", "accepted"),
    [
        pytest.param(remote_backed("team-sie"), True, id="remote-backed-defined"),
        pytest.param(remote_backed("nobody"), False, id="remote-backed-undefined"),
        pytest.param(hybrid("nobody"), False, id="hybrid-undefined"),
        pytest.param(hybrid("team-sie", alias={"extends": "remote"}), True, id="alias-inherits-defined"),
        pytest.param(hybrid("team-sie", alias=override_upstream("nobody")), False, id="alias-overrides-undefined"),
        pytest.param(hybrid("team-sie", small={"extends": "default", "max_batch_tokens": 1024}), True, id="local"),
    ],
)
def test_sie_config_refuses_exactly_what_a_remote_worker_refuses(
    client: TestClient, config: dict[str, Any], accepted: bool
) -> None:
    assert (post(client, config).status_code == 201) is accepted
    assert asyncio.run(accepted_by_a_remote_worker(config)) is accepted


def test_the_remote_adapter_prefix_is_the_workers() -> None:
    assert REMOTE_ADAPTER_MODULE_PREFIX == WORKER_REMOTE_ADAPTER_MODULE_PREFIX
