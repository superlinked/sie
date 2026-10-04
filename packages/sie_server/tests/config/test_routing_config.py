"""The routing block of a model config and the gates that refuse it at load."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.config.model import ModelConfig
from sie_server.config.routing import hybrid_equivalence_refusal, remote_output_refusal, validate_model_routing
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.loader import expand_profile_variants
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_types import ReplaceModelConfigEntry, ReplaceModelConfigsRequest
from sie_server.queue_executor import QueueExecutor

LOCAL_ADAPTER = "sie_server.adapters.fake.adapter:FakeAdapter"
REMOTE_ADAPTER = "sie_server.adapters.remote.sie:SieUpstreamAdapter"
DECLARED_OUTPUTS_MODULE = "sie_server.adapters.remote.declared_outputs"
ENCODE_REMOTE = f"{DECLARED_OUTPUTS_MODULE}:EncodeRemoteAdapter"
EXTRACT_REMOTE = f"{DECLARED_OUTPUTS_MODULE}:ExtractRemoteAdapter"
GENERATE_REMOTE = f"{DECLARED_OUTPUTS_MODULE}:GenerateRemoteAdapter"
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
EXTRACT: dict[str, Any] = {"extract": {}}
GENERATE = {"generate": {"context_length": 4096, "max_output_tokens": 256}}


def local_profile(**extra: Any) -> dict[str, Any]:
    return {"adapter_path": LOCAL_ADAPTER, "max_batch_tokens": 8192, **extra}


def remote_profile(adapter_path: str = REMOTE_ADAPTER, **extra: Any) -> dict[str, Any]:
    return {
        "adapter_path": adapter_path,
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "acme/upstream-model"}},
        **extra,
    }


def hybrid(
    *, sie_id: str = "acme/hybrid", tasks: dict[str, Any] | None = None, routing: Any = FALLBACK, **overrides: Any
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "sie_id": sie_id,
        "package_backed": True,
        "tasks": EXTRACT if tasks is None else tasks,
        "profiles": {"default": local_profile(), "remote": remote_profile()},
        "routing": routing,
    }
    spec.update(overrides)
    return spec


def remote_backed(*, routing: Any = None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "sie_id": "acme/remote",
        "remote_backed": True,
        "tasks": ENCODE,
        "profiles": {"default": remote_profile()},
    }
    if routing is not None:
        spec["routing"] = routing
    return spec


class EncodeRemoteAdapter(BaseAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("dense",), unload_fields=())

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ExtractRemoteAdapter(BaseAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("json",), unload_fields=())

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class GenerateRemoteAdapter(BaseAdapter, GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    async def generate(self, *args: Any, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop")


@pytest.fixture(autouse=True)
def _declared_outputs_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType(DECLARED_OUTPUTS_MODULE)
    module.EncodeRemoteAdapter = EncodeRemoteAdapter  # type: ignore[attr-defined]
    module.ExtractRemoteAdapter = ExtractRemoteAdapter  # type: ignore[attr-defined]
    module.GenerateRemoteAdapter = GenerateRemoteAdapter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, DECLARED_OUTPUTS_MODULE, module)


def extract_hybrid(*, sie_id: str = "acme/hybrid", **overrides: Any) -> dict[str, Any]:
    profiles = {"default": local_profile(), "remote": remote_profile(EXTRACT_REMOTE)}
    return hybrid(sie_id=sie_id, tasks=EXTRACT, profiles=profiles, **overrides)


@pytest.fixture(autouse=True)
def _upstreams() -> Iterator[None]:
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


def test_a_fallback_model_routes_to_its_remote_profile_on_the_default_triggers() -> None:
    config = ModelConfig.model_validate(hybrid())

    assert config.routing is not None
    assert config.routing.policy == "fallback"
    assert config.routing.fallback_profile == "remote"
    assert config.routing.effective_triggers == {"provisioning", "model_loading"}


def test_declared_triggers_replace_the_default() -> None:
    config = ModelConfig.model_validate(hybrid(routing={**FALLBACK, "triggers": ["model_loading", "unhealthy"]}))

    assert config.routing is not None
    assert config.routing.effective_triggers == {"model_loading", "unhealthy"}


def test_a_model_without_a_routing_block_is_served_locally() -> None:
    config = ModelConfig.model_validate(hybrid(routing=None))

    assert config.routing is None
    assert hybrid_equivalence_refusal(config) is None
    validate_model_routing(config)


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ({"policy": "fallback"}, "must name the remote profile in 'fallback_profile'"),
        ({**FALLBACK, "triggers": []}, "must name at least one trigger"),
        ({**FALLBACK, "triggers": ["model_loading", "model_loading"]}, "must not repeat a trigger"),
        ({**FALLBACK, "triggers": ["sometimes"]}, "triggers"),
        ({**FALLBACK, "wake_above": 2.0}, "'fallback' does not use: wake_above"),
        ({**FALLBACK, "upstream": "team-sie"}, "Extra inputs are not permitted"),
        ({"policy": "sometimes", "fallback_profile": "remote"}, "policy"),
        ({**THRESHOLD, "cooldown_s": None}, "'threshold' must set: cooldown_s"),
        ({**THRESHOLD, "sleep_below": 2.0}, "sleep_below must be lower than routing.wake_above"),
        ({**THRESHOLD, "window_s": 0}, "greater than 0"),
        ({**THRESHOLD, "triggers": ["model_loading"]}, "'threshold' does not use 'triggers'"),
        ({**FALLBACK, "fallback_profile": "default"}, "must name a non-default profile"),
        ({**FALLBACK, "fallback_profile": "missing"}, "must name a non-default profile"),
    ],
    ids=[
        "no-fallback-profile",
        "empty-triggers",
        "repeated-trigger",
        "unknown-trigger",
        "threshold-field-under-fallback",
        "unknown-field",
        "unknown-policy",
        "threshold-field-missing",
        "sleep-not-below-wake",
        "non-positive-window",
        "triggers-under-threshold",
        "default-as-fallback",
        "unknown-fallback-profile",
    ],
)
def test_an_invalid_routing_block_is_rejected(routing: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        ModelConfig.model_validate(hybrid(routing=routing))


def test_the_fallback_profile_must_be_a_remote_profile() -> None:
    profiles = {"default": local_profile(), "remote": remote_profile(), "fast": {"extends": "default"}}

    with pytest.raises(ValidationError, match=r"routing\.fallback_profile 'fast' must be a remote profile"):
        ModelConfig.model_validate(hybrid(profiles=profiles, routing={**FALLBACK, "fallback_profile": "fast"}))


def test_a_fallback_profile_may_inherit_its_remote_adapter() -> None:
    profiles = {
        "default": local_profile(),
        "remote": remote_profile(),
        "remote-large": {
            "extends": "remote",
            "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "acme/upstream-large"}},
        },
    }

    config = ModelConfig.model_validate(
        hybrid(profiles=profiles, routing={**FALLBACK, "fallback_profile": "remote-large"})
    )

    assert config.routing is not None
    assert config.routing.fallback_profile == "remote-large"


def test_a_routed_model_needs_a_local_default_profile() -> None:
    profiles = {"default": remote_profile(), "remote": remote_profile()}

    with pytest.raises(ValidationError, match="needs a local 'default' profile"):
        ModelConfig.model_validate(hybrid(profiles=profiles))


def test_remote_only_is_the_policy_of_a_remote_backed_model() -> None:
    config = ModelConfig.model_validate(remote_backed(routing={"policy": "remote_only"}))

    assert config.routing is not None
    assert config.routing.effective_triggers == frozenset()
    validate_model_routing(config)


def test_remote_only_is_refused_for_a_model_with_local_weights() -> None:
    with pytest.raises(ValidationError, match="set 'remote_backed: true'"):
        ModelConfig.model_validate(hybrid(routing={"policy": "remote_only"}))


@pytest.mark.parametrize(
    "routing",
    [
        {"policy": "remote_only", "fallback_profile": "default"},
        {"policy": "remote_only", "triggers": ["model_loading"]},
        {"policy": "remote_only", "wake_above": 2.0},
    ],
    ids=["fallback-profile", "triggers", "threshold-field"],
)
def test_remote_only_takes_no_other_field(routing: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="'remote_only' takes no other field"):
        ModelConfig.model_validate(remote_backed(routing=routing))


@pytest.mark.parametrize("routing", [FALLBACK, THRESHOLD], ids=["fallback", "threshold"])
def test_a_remote_backed_model_has_no_local_profile_to_route_to(routing: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="has no local profile"):
        ModelConfig.model_validate(remote_backed(routing=routing))


def test_profile_variants_carry_no_routing_block() -> None:
    expanded = expand_profile_variants([ModelConfig.model_validate(hybrid())])

    assert expanded["acme/hybrid"].routing is not None
    assert expanded["acme/hybrid:remote"].routing is None


@pytest.mark.parametrize(
    ("tasks", "refused"),
    [
        (ENCODE, "encode"),
        ({"score": {}}, "score"),
        ({**ENCODE, "score": {}}, "encode and score"),
    ],
    ids=["encode", "score", "encode-and-score"],
)
def test_hybrid_encode_and_score_are_refused(tasks: dict[str, Any], refused: str) -> None:
    config = ModelConfig.model_validate(hybrid(tasks=tasks))

    refusal = hybrid_equivalence_refusal(config)

    assert refusal is not None
    assert f"would serve {refused} from both its local profile and remote profile 'remote'" in refusal
    with pytest.raises(ValueError, match="shown to be equivalent"):
        validate_model_routing(config)


def test_hybrid_extract_is_allowed() -> None:
    validate_model_routing(ModelConfig.model_validate(extract_hybrid()))


def test_hybrid_generation_is_supported_with_pre_output_fallback_handling() -> None:
    profiles = {
        "default": local_profile(kv_budget_tokens=4096),
        "remote": remote_profile(GENERATE_REMOTE, kv_budget_tokens=4096),
    }

    validate_model_routing(ModelConfig.model_validate(hybrid(tasks=GENERATE, profiles=profiles)))


def test_declaring_tokens_does_not_make_an_adapter_a_generation_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class TokenOnlyAdapter(BaseAdapter):
        spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    monkeypatch.setattr(sys.modules[DECLARED_OUTPUTS_MODULE], "GenerateRemoteAdapter", TokenOnlyAdapter)
    profiles = {
        "default": local_profile(kv_budget_tokens=4096),
        "remote": remote_profile(GENERATE_REMOTE, kv_budget_tokens=4096),
    }
    with pytest.raises(ValueError, match="requires a GenerationAdapter"):
        validate_model_routing(ModelConfig.model_validate(hybrid(tasks=GENERATE, profiles=profiles)))


@pytest.mark.parametrize(
    ("tasks", "uncovered"),
    [
        (EXTRACT, "json"),
        ({"encode": {"dense": {"dim": 384}, "sparse": {"dim": 30000}}}, "sparse"),
    ],
    ids=["extract-through-an-encode-adapter", "sparse-through-a-dense-adapter"],
)
def test_a_remote_profile_must_produce_every_declared_output(tasks: dict[str, Any], uncovered: str) -> None:
    profiles = {"default": local_profile(), "remote": remote_profile(ENCODE_REMOTE)}
    config = ModelConfig.model_validate(hybrid(tasks=tasks, profiles=profiles))

    refusal = remote_output_refusal(config)

    assert refusal is not None
    assert f"declares {uncovered}, which remote profile 'remote' does not produce" in refusal


def test_an_extract_model_with_an_encode_only_remote_profile_is_refused() -> None:
    profiles = {"default": local_profile(), "remote": remote_profile(ENCODE_REMOTE)}
    with pytest.raises(ValueError, match="which remote profile 'remote' does not produce"):
        validate_model_routing(ModelConfig.model_validate(hybrid(tasks=EXTRACT, profiles=profiles)))


@pytest.mark.parametrize(
    "config",
    [
        hybrid(tasks=ENCODE),
        extract_hybrid(),
        remote_backed(routing={"policy": "remote_only"}),
        hybrid(tasks=EXTRACT, routing=None),
    ],
    ids=["dense-through-a-dense-adapter", "json-through-an-extract-adapter", "remote-only", "local-only"],
)
def test_a_remote_profile_covering_every_declared_output_is_accepted(config: dict[str, Any]) -> None:
    assert remote_output_refusal(ModelConfig.model_validate(config)) is None


def test_a_remote_profile_whose_adapter_cannot_be_imported_is_refused() -> None:
    profiles = {"default": local_profile(), "remote": remote_profile(f"{DECLARED_OUTPUTS_MODULE}:MissingAdapter")}
    config = ModelConfig.model_validate(hybrid(profiles=profiles))

    assert remote_output_refusal(config) == (
        "Model 'acme/hybrid': the adapter of remote profile 'remote' cannot be imported"
    )


def test_a_remote_only_model_serves_from_one_backend() -> None:
    config = ModelConfig.model_validate(remote_backed(routing={"policy": "remote_only"}))

    assert hybrid_equivalence_refusal(config) is None


def test_threshold_is_refused_until_it_is_available() -> None:
    config = ModelConfig.model_validate(hybrid(routing=THRESHOLD))

    with pytest.raises(ValueError, match="routing policy 'threshold' is not available yet"):
        validate_model_routing(config)


def test_a_refused_model_in_the_models_directory_stops_startup(tmp_path: Path) -> None:
    (tmp_path / "hybrid.yaml").write_text(yaml.safe_dump(hybrid(tasks=ENCODE)), encoding="utf-8")

    with pytest.raises(ValueError, match="shown to be equivalent"):
        ModelRegistry(models_dir=tmp_path, enable_hot_reload=False)


def test_a_refused_model_is_not_added_at_runtime() -> None:
    registry = ModelRegistry(models_dir=None)

    with pytest.raises(ValueError, match="routing policy 'threshold' is not available yet"):
        registry.add_config(ModelConfig.model_validate(hybrid(routing=THRESHOLD)))

    assert not registry.has_model("acme/hybrid")
    assert not registry.has_model("acme/hybrid:remote")


async def test_a_config_snapshot_rejects_only_the_refused_entry(caplog: pytest.LogCaptureFixture) -> None:
    registry = ModelRegistry(models_dir=None)
    executor = QueueExecutor(registry)
    caplog.set_level(logging.WARNING, logger="sie_server.queue_executor")

    response = await executor.replace_model_configs(
        ReplaceModelConfigsRequest(
            bundle_id="default",
            epoch=1,
            bundle_config_hash="",
            models=[
                ReplaceModelConfigEntry(
                    model_id="acme/extract", model_config=yaml.safe_dump(extract_hybrid(sie_id="acme/extract"))
                ),
                ReplaceModelConfigEntry(
                    model_id="acme/encode", model_config=yaml.safe_dump(hybrid(sie_id="acme/encode", tasks=ENCODE))
                ),
            ],
        )
    )

    assert registry.has_model("acme/extract")
    assert registry.has_model("acme/extract:remote")
    assert not registry.has_model("acme/encode")
    assert "acme/encode" not in response.applied_models
    assert "Model 'acme/encode' would serve encode from both" in caplog.text


async def test_authoritative_replacement_refuses_unsupported_remote_outputs_without_changing_registry() -> None:
    registry = ModelRegistry(models_dir=None)
    existing = ModelConfig.model_validate(hybrid(sie_id="acme/kept", routing=None))
    registry.add_config(existing)
    profiles = {"default": local_profile(), "remote": remote_profile(ENCODE_REMOTE)}
    refused = ModelConfig.model_validate(hybrid(tasks=EXTRACT, profiles=profiles))

    with pytest.raises(ValueError, match="which remote profile 'remote' does not produce"):
        await registry.replace_configs_async([refused])

    assert registry.has_model("acme/kept")
    assert not registry.has_model("acme/hybrid")
    assert not registry.has_model("acme/hybrid:remote")


def test_threshold_requires_flag_and_queue_worker_and_keeps_numerical_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = ModelConfig.model_validate(hybrid(routing=THRESHOLD))
    monkeypatch.setenv("SIE_THRESHOLD_ROUTING_ENABLED", "true")
    monkeypatch.delenv("SIE_IPC_SOCKET_PATH", raising=False)
    with pytest.raises(ValueError, match="queue worker"):
        validate_model_routing(config)
    monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(tmp_path / "ipc.sock"))
    validate_model_routing(config)
    with pytest.raises(ValueError, match="fleet equivalence"):
        validate_model_routing(ModelConfig.model_validate(hybrid(tasks=ENCODE, routing=THRESHOLD)))
    with pytest.raises(ValueError, match="86400"):
        validate_model_routing(ModelConfig.model_validate(hybrid(routing={**THRESHOLD, "window_s": 86401})))
    monkeypatch.setenv("SIE_THRESHOLD_ROUTING_ENABLED", "false")
    with pytest.raises(ValueError, match="not available yet"):
        validate_model_routing(config)
