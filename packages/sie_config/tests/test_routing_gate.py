"""sie-config refuses a routing block that the workers would refuse when they load the model."""

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
from sie_config.model_registry import ModelRegistry
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.registry import ModelRegistry as WorkerModelRegistry
from sie_server.ipc_types import ReplaceModelConfigEntry, ReplaceModelConfigsRequest
from sie_server.queue_executor import QueueExecutor

BUNDLES_DIR = Path(__file__).resolve().parents[2] / "sie_server" / "bundles"
LOCAL_PROFILE = {"adapter_path": "sie_server.adapters.bert_flash:BertFlashAdapter", "max_batch_tokens": 4096}
REMOTE_PROFILE = {
    "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
    "max_batch_tokens": 8192,
    "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "org/name"}},
}
FALLBACK = {"policy": "fallback", "fallback_profile": "remote"}
THRESHOLD = {
    "policy": "threshold",
    "fallback_profile": "remote",
    "wake_above": 2.0,
    "sleep_below": 0.5,
    "window_s": 60,
    "cooldown_s": 300,
}
ENCODE = {"encode": {"dense": {"dim": 384}}}


def hybrid(tasks: dict[str, Any], routing: dict[str, Any] | None = FALLBACK) -> dict[str, Any]:
    config: dict[str, Any] = {
        "sie_id": "acme/hybrid",
        "hf_id": "acme/hybrid",
        "tasks": tasks,
        "profiles": {"default": LOCAL_PROFILE, "remote": REMOTE_PROFILE},
    }
    if routing is not None:
        config["routing"] = routing
    return config


def remote_backed() -> dict[str, Any]:
    return {
        "sie_id": "acme/remote",
        "remote_backed": True,
        "tasks": ENCODE,
        "routing": {"policy": "remote_only"},
        "profiles": {"default": REMOTE_PROFILE},
    }


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


@pytest.fixture(autouse=True)
def remote_worker_upstreams(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SIE_UPSTREAM_NAMES", "team-sie")
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


def post(client: TestClient, config: dict[str, Any]) -> httpx.Response:
    return client.post("/v1/configs/models", content=yaml.safe_dump(config))


ENCODE_REFUSAL = (
    "Model 'acme/hybrid' would serve encode from both its local profile and remote profile 'remote' under "
    "routing policy 'fallback'; that is refused until the remote profile is shown to be equivalent to the local one"
)


def test_a_hybrid_encode_model_under_fallback_is_refused_and_not_stored(client: TestClient) -> None:
    response = post(client, hybrid(ENCODE))

    assert response.status_code == 422
    assert response.json()["detail"] == {"error": "validation_error", "details": [{"message": ENCODE_REFUSAL}]}
    assert client.get("/v1/configs/models/acme/hybrid").status_code == 404


def test_a_routing_block_appended_to_a_stored_encode_model_is_refused(client: TestClient) -> None:
    assert post(client, hybrid(ENCODE, routing=None)).status_code == 201

    appended = {"sie_id": "acme/hybrid", "routing": FALLBACK, "profiles": {"small": {"extends": "default"}}}

    response = post(client, appended)

    assert response.status_code == 422
    assert response.json()["detail"]["details"] == [{"message": ENCODE_REFUSAL}]
    assert "routing" not in client.app.state.model_registry.get_full_config("acme/hybrid")


def test_a_replacement_is_checked_and_keeps_the_stored_config(client: TestClient) -> None:
    assert post(client, hybrid(ENCODE, routing=None)).status_code == 201

    response = client.put("/v1/configs/models/acme/hybrid", content=yaml.safe_dump(hybrid(ENCODE)))

    assert response.status_code == 422
    assert response.json()["detail"]["details"] == [{"message": ENCODE_REFUSAL}]
    assert "routing" not in client.app.state.model_registry.get_full_config("acme/hybrid")


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


@pytest.mark.parametrize(
    ("config", "accepted"),
    [
        pytest.param(hybrid(ENCODE), False, id="fallback-encode"),
        pytest.param(hybrid({"score": {}}), False, id="fallback-score"),
        pytest.param(hybrid({**ENCODE, "extract": {}}), False, id="fallback-encode-and-extract"),
        pytest.param(hybrid({"extract": {}}), True, id="fallback-extract"),
        pytest.param(hybrid({"extract": {}}, routing=THRESHOLD), False, id="threshold"),
        pytest.param(hybrid(ENCODE, routing=None), True, id="no-routing"),
        pytest.param(remote_backed(), True, id="remote-only"),
    ],
)
def test_sie_config_refuses_exactly_what_a_worker_refuses(
    client: TestClient, config: dict[str, Any], accepted: bool
) -> None:
    assert (post(client, config).status_code == 201) is accepted
    assert asyncio.run(accepted_by_a_remote_worker(config)) is accepted
