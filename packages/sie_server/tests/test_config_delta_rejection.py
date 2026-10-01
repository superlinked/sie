"""A config delta the worker rejects is reported per model and does not stall its bundle.

The sidecar advertises a delta's result only when the worker's hash equals the
control-plane hash. A rejected delta must therefore still produce that hash, the
way a rejected entry in a full export does, or the worker drops out of routing
for every model of the bundle until the next export replay.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import yaml
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_types import ApplyModelConfigRequest, ReplaceModelConfigEntry, ReplaceModelConfigsRequest
from sie_server.queue_executor import QueueExecutor

LOCAL_MODEL = """\
sie_id: acme/local
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""
UNKNOWN_FIELD_MODEL = LOCAL_MODEL.replace("acme/local", "acme/unknown-field") + "not_a_model_field: 1\n"


def remote_model(model_id: str, upstream: str) -> str:
    return f"""\
sie_id: {model_id}
remote_backed: true
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
        upstream: {upstream}
        upstream_model: sie-fake
"""


@pytest.fixture(autouse=True)
def _no_upstreams() -> Iterator[None]:
    install_upstreams({})
    yield
    install_upstreams({})


def delta(model_id: str, model_config: str, epoch: int) -> ApplyModelConfigRequest:
    return ApplyModelConfigRequest(
        bundle_id="default",
        model_id=model_id,
        epoch=epoch,
        bundle_config_hash="",
        model_config=model_config,
    )


async def export_view(*entries: tuple[str, str]) -> tuple[str, list[str]]:
    """The hash and unsupported ids a fresh worker reports after an export of ``entries``."""
    executor = QueueExecutor(ModelRegistry(models_dir=None))
    response = await executor.replace_model_configs(
        ReplaceModelConfigsRequest(
            bundle_id="default",
            epoch=1,
            bundle_config_hash="",
            models=[ReplaceModelConfigEntry(model_id=model_id, model_config=text) for model_id, text in entries],
        )
    )
    return response.bundle_config_hash, response.unsupported_models


@pytest.mark.parametrize(
    ("model_id", "rejected_config"),
    [
        ("acme/remote", remote_model("acme/remote", "team-sie")),
        ("acme/unknown-field", UNKNOWN_FIELD_MODEL),
    ],
    ids=["undefined-upstream", "schema-violation"],
)
async def test_a_rejected_delta_reports_the_export_hash_and_keeps_the_other_models(
    model_id: str, rejected_config: str
) -> None:
    registry = ModelRegistry(models_dir=None)
    executor = QueueExecutor(registry)
    await executor.apply_model_config(delta("acme/local", LOCAL_MODEL, 1))

    response = await executor.apply_model_config(delta(model_id, rejected_config, 2))

    assert response.applied is True
    assert response.unsupported_models == [model_id]
    assert (response.bundle_config_hash, response.unsupported_models) == await export_view(
        ("acme/local", LOCAL_MODEL), (model_id, rejected_config)
    )
    assert registry.has_model("acme/local")
    assert not registry.has_model(model_id)


async def test_a_rejected_delta_keeps_the_models_current_config() -> None:
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
    registry = ModelRegistry(models_dir=None)
    executor = QueueExecutor(registry)
    defined = remote_model("acme/remote", "team-sie")
    await executor.apply_model_config(delta("acme/remote", defined, 1))

    response = await executor.apply_model_config(delta("acme/remote", remote_model("acme/remote", "nobody"), 2))

    assert response.unsupported_models == ["acme/remote"]
    config = registry.get_config("acme/remote")
    assert config.resolve_profile("default").loadtime["upstream"] == "team-sie"


async def test_a_later_valid_delta_clears_the_rejection() -> None:
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
    registry = ModelRegistry(models_dir=None)
    executor = QueueExecutor(registry)
    await executor.apply_model_config(delta("acme/remote", remote_model("acme/remote", "nobody"), 1))

    corrected = remote_model("acme/remote", "team-sie")
    response = await executor.apply_model_config(delta("acme/remote", corrected, 2))

    assert response.unsupported_models == []
    assert registry.has_model("acme/remote")
    assert (response.bundle_config_hash, response.unsupported_models) == await export_view(("acme/remote", corrected))


@pytest.mark.parametrize(
    "model_config",
    [
        "- not a mapping\n",
        "sie_id: [unclosed\n",
        LOCAL_MODEL.replace("acme/local", "acme/other"),
    ],
    ids=["not-a-mapping", "invalid-yaml", "another-model"],
)
async def test_a_delta_that_does_not_name_its_model_still_fails(model_config: str) -> None:
    registry = ModelRegistry(models_dir=None)
    executor = QueueExecutor(registry)

    with pytest.raises((ValueError, yaml.YAMLError)):
        await executor.apply_model_config(delta("acme/local", model_config, 1))

    assert not registry.has_model("acme/local")
    assert not registry.has_model("acme/other")
