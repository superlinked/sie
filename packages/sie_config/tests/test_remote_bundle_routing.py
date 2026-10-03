"""sie-config routes remote profiles to the remote bundle, and a remote-lane worker agrees on its hash."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sie_config.model_registry import ModelRegistry
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.registry import ModelRegistry as WorkerModelRegistry
from sie_server.ipc_types import ReplaceModelConfigEntry, ReplaceModelConfigsRequest
from sie_server.queue_executor import QueueExecutor

BUNDLES_DIR = Path(__file__).resolve().parents[2] / "sie_server" / "bundles"
REMOTE_PROFILE: dict[str, Any] = {
    "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
    "max_batch_tokens": 8192,
    "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "org/name"}},
}
REMOTE_BACKED: dict[str, Any] = {
    "sie_id": "acme/remote",
    "remote_backed": True,
    "tasks": {"encode": {"dense": {"dim": 384}}},
    "profiles": {"default": REMOTE_PROFILE},
}
HYBRID: dict[str, Any] = {
    "sie_id": "acme/hybrid",
    "hf_id": "acme/hybrid",
    "tasks": {"encode": {"dense": {"dim": 384}}},
    "profiles": {
        "default": {"adapter_path": "sie_server.adapters.bert_flash:BertFlashAdapter", "max_batch_tokens": 4096},
        "remote": REMOTE_PROFILE,
    },
}


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return ModelRegistry(BUNDLES_DIR, models_dir)


@pytest.fixture
def team_sie_upstream() -> Iterator[None]:
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


def test_a_remote_backed_model_routes_to_the_remote_bundle(registry: ModelRegistry) -> None:
    _, _, affected_bundles = registry.validate_model_config(copy.deepcopy(REMOTE_BACKED))
    registry.add_model_config(copy.deepcopy(REMOTE_BACKED))

    assert affected_bundles == ["remote", "default"]
    assert registry.resolve_bundle("acme/remote") == "remote"


def test_only_the_remote_profile_of_a_local_model_routes_to_the_remote_bundle(registry: ModelRegistry) -> None:
    registry.add_model_config(copy.deepcopy(HYBRID))

    assert registry.get_model_profile_bundles("acme/hybrid") == {
        "default": ["default"],
        "remote": ["remote", "default"],
    }
    assert registry.resolve_bundle("acme/hybrid") == "default"
    assert registry.resolve_bundle("acme/hybrid:remote") == "remote"


@pytest.mark.usefixtures("team_sie_upstream")
async def test_a_remote_lane_worker_reports_the_control_plane_hash(registry: ModelRegistry) -> None:
    registry.add_model_config(copy.deepcopy(REMOTE_BACKED))
    executor = QueueExecutor(WorkerModelRegistry(models_dir=None))

    response = await executor.replace_model_configs(
        ReplaceModelConfigsRequest(
            bundle_id="remote",
            epoch=1,
            bundle_config_hash="",
            models=[ReplaceModelConfigEntry(model_id="acme/remote", model_config=yaml.safe_dump(REMOTE_BACKED))],
        )
    )

    assert response.unsupported_models == []
    assert response.bundle_config_hash
    assert response.bundle_config_hash == registry.compute_bundle_config_hash("remote")
