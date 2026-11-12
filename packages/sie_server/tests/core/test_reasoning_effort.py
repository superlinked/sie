"""Chat ``reasoning_effort``: the per-profile declaration and its resolution at every ingress."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sie_server.adapters._generation_base import GenerationInvalidRequestError, GenerationUnsupportedFieldError
from sie_server.api.models import _resolve_capabilities
from sie_server.config.equivalence import canonical_digest
from sie_server.config.model import ModelConfig
from sie_server.core.loader import expand_profile_variants
from sie_server.core.reasoning_effort import MALFORMED_MESSAGE, apply_reasoning_effort, parse_reasoning_effort
from sie_server.processors.streaming import (
    _ChatMessage,
    _GenerateRequestParams,
    _MessagesInput,
    _remote_chat_parameters,
)

_VECTORS = json.loads(
    (Path(__file__).parents[4] / "conformance" / "openai_generation" / "reasoning_effort_vectors.json").read_text(
        encoding="utf-8"
    )
)
_LOCAL = "sie_server.adapters.fake.adapter:FakeAdapter"
_REMOTE = "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter"


def _config(name: str) -> ModelConfig:
    return ModelConfig.model_validate(copy.deepcopy(_VECTORS["models"][name]["config"]))


def _hybrid(local: dict[str, Any] | None, remote: dict[str, Any] | None) -> dict[str, Any]:
    default: dict[str, Any] = {"adapter_path": _LOCAL, "max_batch_tokens": 8192, "kv_budget_tokens": 8192}
    relay: dict[str, Any] = {
        "adapter_path": _REMOTE,
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": {"upstream": "acme-openai", "upstream_model": "acme/upstream-model"}},
    }
    if local is not None:
        default["reasoning_effort"] = local
    if remote is not None:
        relay["reasoning_effort"] = remote
    return {
        "sie_id": "acme/hybrid",
        "package_backed": True,
        "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 256}},
        "profiles": {"default": default, "relay": relay},
        "routing": {"policy": "fallback", "fallback_profile": "relay"},
    }


@pytest.mark.parametrize("case", _VECTORS["cases"], ids=lambda case: case["name"])
def test_shared_vectors_resolve_or_refuse(case: dict[str, Any]) -> None:
    config = _config(case["model"])
    if "message" in case:
        with pytest.raises(GenerationUnsupportedFieldError) as refused:
            apply_reasoning_effort(config, case["value"], model=case["model"])
        assert refused.value.param == "reasoning_effort"
        assert str(refused.value) == case["message"]
        return
    applied = apply_reasoning_effort(config, case["value"], model=case["model"])
    if case["applied"] is None:
        assert applied is None
    else:
        assert applied is not None
        assert dict(applied.chat_template_kwargs) == case["applied"]["chat_template_kwargs"]
        assert applied.upstream == case["applied"]["upstream"]


@pytest.mark.parametrize("name", sorted(_VECTORS["models"]))
def test_shared_vectors_capability(name: str) -> None:
    capabilities = _resolve_capabilities(_config(name))
    assert capabilities is not None
    reasoning = capabilities.reasoning
    expected = _VECTORS["models"][name]["capability"]
    assert (reasoning.model_dump() if reasoning is not None else None) == expected


@pytest.mark.parametrize("raw", _VECTORS["malformed"], ids=repr)
def test_a_value_outside_openais_set_is_malformed(raw: Any) -> None:
    with pytest.raises(GenerationInvalidRequestError) as refused:
        parse_reasoning_effort(raw)
    assert refused.value.param == "reasoning_effort"
    assert str(refused.value) == MALFORMED_MESSAGE == _VECTORS["malformed_message"]


def test_absent_and_null_change_nothing() -> None:
    assert parse_reasoning_effort(None) is None
    assert apply_reasoning_effort(_config("acme/thinking"), None, model="acme/thinking") is None


@pytest.mark.parametrize(
    ("declaration", "match"),
    [
        ({"values": {}}, "at least one value"),
        ({"values": {"extreme": {}}}, "values"),
        ({"default": "low", "values": {"high": {}}}, "is not one of its values"),
        ({"values": {"low": {"chat_template_kwargs": {"enable_thinking": "false"}}}}, "unsupported"),
        ({"values": {"low": {"chat_template_kwargs": {"reasoning_effort": "low; drop"}}}}, "letters, digits"),
        ({"values": {"low": {"budget": 512}}}, "budget"),
    ],
)
def test_malformed_declarations_fail_at_load(declaration: dict[str, Any], match: str) -> None:
    spec = copy.deepcopy(_VECTORS["models"]["acme/silent"]["config"])
    spec["tasks"]["generate"]["reasoning_effort"] = declaration
    with pytest.raises(ValidationError, match=match):
        ModelConfig.model_validate(spec)


def test_a_remote_profile_cannot_name_template_kwargs() -> None:
    spec = copy.deepcopy(_VECTORS["models"]["acme/remote"]["config"])
    spec["tasks"]["generate"]["reasoning_effort"] = {
        "values": {"low": {"chat_template_kwargs": {"reasoning_effort": "low"}}}
    }
    with pytest.raises(ValidationError, match="is remote"):
        ModelConfig.model_validate(spec)


def test_a_local_profile_cannot_name_an_upstream_value() -> None:
    spec = copy.deepcopy(_VECTORS["models"]["acme/silent"]["config"])
    spec["tasks"]["generate"]["reasoning_effort"] = {"values": {"low": {"upstream": "low"}}}
    with pytest.raises(ValidationError, match="is local"):
        ModelConfig.model_validate(spec)


def test_a_declaration_needs_a_generate_task() -> None:
    spec = copy.deepcopy(_VECTORS["models"]["acme/silent"]["config"])
    spec["tasks"] = {"encode": {"dense": {"dim": 8}}}
    spec["profiles"]["default"].pop("kv_budget_tokens")
    spec["profiles"]["default"]["reasoning_effort"] = {"values": {"low": {}}}
    with pytest.raises(ValidationError, match="no generate task"):
        ModelConfig.model_validate(spec)


def test_both_profiles_of_a_routed_model_accept_the_same_values() -> None:
    low_local = {"values": {"low": {"chat_template_kwargs": {"reasoning_effort": "low"}}}}
    low_remote = {"values": {"low": {"upstream": "low"}}}
    config = ModelConfig.model_validate(_hybrid(low_local, low_remote))
    assert apply_reasoning_effort(config, "low", model="acme/hybrid") is not None
    with pytest.raises(ValidationError, match="the same reasoning_effort values"):
        ModelConfig.model_validate(_hybrid(low_local, None))
    with pytest.raises(ValidationError, match="the same reasoning_effort values"):
        ModelConfig.model_validate(_hybrid(low_local, {"values": {"low": {}, "high": {}}}))
    with pytest.raises(ValidationError, match="the same reasoning_effort default"):
        ModelConfig.model_validate(_hybrid(low_local, {"default": "low", "values": {"low": {"upstream": "low"}}}))


def test_a_variant_carries_its_profiles_declaration() -> None:
    low_local = {"values": {"low": {"chat_template_kwargs": {"reasoning_effort": "low"}}}}
    low_remote = {"values": {"low": {"upstream": "low"}}}
    configs = expand_profile_variants([ModelConfig.model_validate(_hybrid(low_local, low_remote))])
    applied = apply_reasoning_effort(configs["acme/hybrid:relay"], "low", model="acme/hybrid:relay")
    assert applied is not None
    assert applied.upstream == "low"
    assert dict(applied.chat_template_kwargs) == {}
    local = apply_reasoning_effort(configs["acme/hybrid"], "low", model="acme/hybrid")
    assert local is not None
    assert dict(local.chat_template_kwargs) == {"reasoning_effort": "low"}


def test_a_profile_inherits_its_parents_declaration() -> None:
    spec = copy.deepcopy(_VECTORS["models"]["acme/profile-declared"]["config"])
    spec["profiles"]["long"] = {"extends": "default"}
    config = ModelConfig.model_validate(spec)
    assert config.reasoning_effort_for("long") == config.reasoning_effort_for("default")
    variant = expand_profile_variants([config])["acme/profile-declared:long"]
    assert apply_reasoning_effort(variant, "low", model="acme/profile-declared:long") is not None


def test_an_undeclared_config_serializes_and_digests_as_before() -> None:
    config = _config("acme/silent")
    dumped = config.model_dump(mode="json")
    assert "reasoning_effort" not in dumped["tasks"]["generate"]
    assert "reasoning_effort" not in dumped["profiles"]["default"]
    spec = copy.deepcopy(_VECTORS["models"]["acme/silent"]["config"])
    spec["tasks"]["generate"]["reasoning_effort"] = None
    spec["profiles"]["default"]["reasoning_effort"] = None
    explicit_null = ModelConfig.model_validate(spec)
    assert canonical_digest(explicit_null.model_dump(mode="json")) == canonical_digest(dumped)
    declared = _config("acme/declared").model_dump(mode="json")
    assert declared["tasks"]["generate"]["reasoning_effort"]["default"] == "max"


@pytest.mark.parametrize(
    ("name", "reasons"),
    [("acme/silent", False), ("acme/thinking", True), ("acme/declared", True), ("acme/remote", True)],
)
def test_declares_no_reasoning(name: str, reasons: bool) -> None:
    assert _config(name).declares_no_reasoning is not reasons


def test_a_reasoning_parser_counts_as_reasoning() -> None:
    spec = copy.deepcopy(_VECTORS["models"]["acme/silent"]["config"])
    spec["profiles"]["default"]["adapter_options"] = {"loadtime": {"reasoning_parser": "qwen3"}}
    assert ModelConfig.model_validate(spec).declares_no_reasoning is False


def _chat_params(**overrides: Any) -> _GenerateRequestParams:
    return _GenerateRequestParams(
        input=_MessagesInput(messages=(_ChatMessage(role="user", content="Classify this."),)),
        max_new_tokens=64,
        **overrides,
    )


def test_the_relay_body_carries_only_the_declared_upstream_value() -> None:
    assert "reasoning_effort" not in _remote_chat_parameters(_chat_params())
    body = _remote_chat_parameters(_chat_params(reasoning_effort="max"), reasoning_effort="high")
    assert body["reasoning_effort"] == "high"
