"""Upstreams of kind openai: the endpoints they offer and the request fields an operator sets or strips."""

from __future__ import annotations

import copy
import datetime
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import (
    ENDPOINT_OUTPUTS,
    RESERVED_UPSTREAM_PARAMS,
    Upstream,
    UpstreamConfigError,
    UpstreamEndpoint,
    install_upstreams,
    load_upstreams,
    validate_profile_upstreams,
)
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_types import ReplaceModelConfigEntry, ReplaceModelConfigsRequest
from sie_server.queue_executor import QueueExecutor

OPENAI_ADAPTER = "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter"
SIE_ADAPTER = "sie_server.adapters.remote.sie:SieUpstreamAdapter"
RATE_CAP = {"requests_per_minute": 60, "max_concurrency": 4}
CANARY = "canary-7d41c2e9"
DENSE = {"encode": {"dense": {"dim": 8}}}
GENERATE = {"generate": {"context_length": 8192, "max_output_tokens": 1024}}


def openai_upstream(**fields: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "kind": "openai",
        "base_url": "https://host.example.com/v1",
        "endpoints": ["chat"],
        "rate_cap": RATE_CAP,
    }
    spec.update(fields)
    return spec


def load_one(tmp_path: Path, spec: dict[str, Any]) -> Upstream:
    path = tmp_path / "upstreams.yaml"
    path.write_text(yaml.safe_dump({"upstreams": {"open-host": spec}}), encoding="utf-8")
    return load_upstreams(path)["open-host"]


def install(*endpoints: str) -> None:
    install_upstreams({"open-host": Upstream.model_validate(openai_upstream(endpoints=list(endpoints)))})


def remote_model(tasks: dict[str, Any], sie_id: str = "acme/remote", adapter: str = OPENAI_ADAPTER) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "adapter_path": adapter,
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": {"upstream": "open-host", "upstream_model": "org/model"}},
    }
    if "generate" in tasks:
        profile["kv_budget_tokens"] = 65536
    return {"sie_id": sie_id, "remote_backed": True, "tasks": tasks, "profiles": {"default": profile}}


def nested(levels: int) -> Any:
    value: Any = "leaf"
    for _ in range(levels):
        value = {"next": value}
    return value


@pytest.fixture(autouse=True)
def _no_upstreams_after() -> Iterator[None]:
    yield
    install_upstreams({})


def test_the_documented_openai_upstream_parses(tmp_path: Path) -> None:
    path = tmp_path / "upstreams.yaml"
    path.write_text(
        "upstreams:\n"
        "  open-host:\n"
        "    kind: openai\n"
        "    base_url: https://host.example.com/v1\n"
        "    api_key_secret: OPEN_HOST_KEY\n"
        "    endpoints: [completions, chat, embeddings, rerank]\n"
        "    set_params:\n"
        "      provider: {data_collection: deny, zdr: true}\n"
        "    strip_params: [user]\n"
        "    rate_cap: {requests_per_minute: 300, max_concurrency: 16}\n",
        encoding="utf-8",
    )

    upstream = load_upstreams(path)["open-host"]

    assert upstream.endpoints == frozenset(UpstreamEndpoint)
    assert upstream.set_params == {"provider": {"data_collection": "deny", "zdr": True}}
    assert upstream.strip_params == frozenset({"user"})
    assert upstream.outputs == frozenset({"tokens", "dense", "score"})


@pytest.mark.parametrize("endpoints", [None, []], ids=["absent", "empty"])
def test_an_openai_upstream_must_declare_its_endpoints(tmp_path: Path, endpoints: list[str] | None) -> None:
    spec = openai_upstream(endpoints=endpoints)
    if endpoints is None:
        del spec["endpoints"]

    with pytest.raises(UpstreamConfigError, match="kind openai must declare the endpoints it offers"):
        load_one(tmp_path, spec)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"endpoints": ["chat", "chat"]}, "endpoints lists the same entry more than once"),
        ({"endpoints": ["responses"]}, "Input should be 'completions', 'chat', 'embeddings' or 'rerank'"),
        ({"strip_params": ["user", "user"]}, "strip_params lists the same entry more than once"),
    ],
    ids=["repeated-endpoint", "unknown-endpoint", "repeated-strip"],
)
def test_a_list_is_never_silently_shortened_or_widened(tmp_path: Path, fields: dict[str, Any], reason: str) -> None:
    with pytest.raises(UpstreamConfigError, match=re.escape(reason)):
        load_one(tmp_path, openai_upstream(**fields))


@pytest.mark.parametrize(
    "field",
    [{"endpoints": ["chat"]}, {"set_params": {"provider": {"zdr": True}}}, {"strip_params": ["user"]}],
    ids=["endpoints", "set_params", "strip_params"],
)
def test_kind_sie_takes_no_openai_field(field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=f"kind sie takes no {next(iter(field))}"):
        Upstream.model_validate(
            {"kind": "sie", "base_url": "https://sie.example.internal", "rate_cap": RATE_CAP, **field}
        )


def test_the_model_and_the_caller_input_are_reserved() -> None:
    assert {"model", "input", "messages", "prompt", "query", "documents", "stream"} <= RESERVED_UPSTREAM_PARAMS


@pytest.mark.parametrize("name", sorted(RESERVED_UPSTREAM_PARAMS))
def test_a_field_the_server_sends_itself_can_be_neither_set_nor_stripped(name: str) -> None:
    with pytest.raises(ValidationError, match=f"set_params cannot set '{name}', which the server sends itself"):
        Upstream.model_validate(openai_upstream(set_params={name: "x"}))
    with pytest.raises(ValidationError, match=f"strip_params cannot strip '{name}', which the server sends itself"):
        Upstream.model_validate(openai_upstream(strip_params=[name]))


@pytest.mark.parametrize("name", ["", "1st", "a-b", "a.b", "$ref", "x" * 65, "naïve"])
def test_a_field_name_is_an_identifier(name: str) -> None:
    with pytest.raises(ValidationError, match="set_params names are letters, digits and underscores"):
        Upstream.model_validate(openai_upstream(set_params={name: 1}))
    with pytest.raises(ValidationError, match="strip_params names are letters, digits and underscores"):
        Upstream.model_validate(openai_upstream(strip_params=[name]))


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (float("inf"), "holds a number that is not finite"),
        (float("nan"), "holds a number that is not finite"),
        (datetime.date(2026, 10, 1), "holds a value that is not JSON"),
        ({1: "one"}, "holds an object key that is not a string"),
        (nested(8), "nests deeper than 8 levels"),
    ],
    ids=["infinity", "nan", "date", "integer-key", "too-deep"],
)
def test_set_params_hold_bounded_json_and_errors_never_repeat_a_value(
    tmp_path: Path, value: Any, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    with pytest.raises(UpstreamConfigError, match=re.escape(f"set_params['provider'] {reason}")) as raised:
        load_one(tmp_path, openai_upstream(set_params={"provider": [CANARY, value]}))

    assert CANARY not in str(raised.value)
    assert CANARY not in caplog.text


def test_eight_levels_of_nesting_are_accepted() -> None:
    upstream = Upstream.model_validate(openai_upstream(set_params={"provider": nested(8)}))

    assert upstream.set_params["provider"] == nested(8)


def test_set_params_and_strip_params_are_bounded() -> None:
    too_many = {f"field_{index}": index for index in range(33)}

    with pytest.raises(ValidationError, match="set_params sets at most 32 fields"):
        Upstream.model_validate(openai_upstream(set_params=too_many))
    with pytest.raises(ValidationError, match="strip_params strips at most 32 fields"):
        Upstream.model_validate(openai_upstream(strip_params=list(too_many)))
    with pytest.raises(ValidationError, match="set_params is larger than 16384 bytes as JSON"):
        Upstream.model_validate(openai_upstream(set_params={"a": "x" * 8192, "b": "y" * 8192}))


def test_a_value_that_repeats_one_subtree_is_measured_without_expanding_it() -> None:
    level: list[Any] = ["x" * 16] * 10
    for _ in range(7):
        level = [level] * 10

    with pytest.raises(ValidationError, match=re.escape("set_params['provider'] is larger than 16384 bytes as JSON")):
        Upstream.model_validate(openai_upstream(set_params={"provider": level}))


def test_a_field_cannot_be_both_set_and_stripped() -> None:
    with pytest.raises(ValidationError, match="set_params and strip_params both name 'temperature'"):
        Upstream.model_validate(openai_upstream(set_params={"temperature": 0}, strip_params=["temperature", "user"]))


def test_apply_params_sets_then_strips_and_changes_neither_the_body_nor_the_upstream() -> None:
    upstream = Upstream.model_validate(
        openai_upstream(set_params={"provider": {"zdr": True}, "temperature": 0}, strip_params=["user"])
    )
    body = {"model": "org/model", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7, "user": "u-7"}
    sent_before = copy.deepcopy(body)

    sent = upstream.apply_params(body)

    assert sent == {
        "model": "org/model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "provider": {"zdr": True},
    }
    assert body == sent_before
    sent["provider"]["zdr"] = False
    assert upstream.apply_params(body)["provider"] == {"zdr": True}


def test_set_params_stay_out_of_the_upstream_repr() -> None:
    upstream = Upstream.model_validate(openai_upstream(set_params={"provider": {"order": [CANARY]}}))

    assert CANARY not in repr(upstream)
    assert CANARY not in str(upstream)


def test_each_endpoint_produces_an_output_the_model_config_names() -> None:
    every_task = {
        "encode": {"dense": {"dim": 8}, "sparse": {"dim": 8}, "multivector": {"dim": 8}},
        "score": {},
        "extract": {},
        **GENERATE,
    }
    config = ModelConfig.model_validate(remote_model(every_task))

    assert set(ENDPOINT_OUTPUTS) == set(UpstreamEndpoint)
    assert set(ENDPOINT_OUTPUTS.values()) <= set(config.outputs)


@pytest.mark.parametrize(
    ("tasks", "endpoints"),
    [
        (DENSE, ["embeddings"]),
        ({"encode": {"dense": {"dim": 8}, "sparse": {"dim": 8}}}, ["embeddings"]),
        ({"score": {}}, ["rerank"]),
        (GENERATE, ["chat"]),
        (GENERATE, ["completions"]),
        ({**DENSE, "score": {}}, ["rerank"]),
    ],
    ids=["dense", "dense-and-sparse", "score", "chat", "completions", "one-of-two-tasks"],
)
def test_a_remote_profile_is_accepted_when_an_endpoint_produces_one_of_the_model_outputs(
    tasks: dict[str, Any], endpoints: list[str]
) -> None:
    install(*endpoints)

    validate_profile_upstreams(ModelConfig.model_validate(remote_model(tasks)))


@pytest.mark.parametrize(
    ("tasks", "endpoints", "reason"),
    [
        (DENSE, ["chat", "rerank"], "none of the model's outputs (dense); declare embeddings"),
        ({"score": {}}, ["embeddings"], "none of the model's outputs (score); declare rerank"),
        (GENERATE, ["embeddings", "rerank"], "none of the model's outputs (tokens); declare chat or completions"),
        (
            {"encode": {"sparse": {"dim": 8}}, "extract": {}},
            ["completions", "chat", "embeddings", "rerank"],
            "none of the model's outputs (sparse, json); an upstream of kind openai has no endpoint for them",
        ),
    ],
    ids=["dense", "score", "generation", "sparse-and-extract"],
)
def test_a_remote_profile_is_refused_when_no_endpoint_produces_a_model_output(
    tasks: dict[str, Any], endpoints: list[str], reason: str
) -> None:
    install(*endpoints)
    config = ModelConfig.model_validate(remote_model(tasks))

    expected = f"Profile 'default' of 'acme/remote' names upstream 'open-host', whose endpoints produce {reason}"
    with pytest.raises(ValueError, match=re.escape(expected)):
        validate_profile_upstreams(config)


def test_an_sie_upstream_is_not_limited_by_endpoints() -> None:
    sie = Upstream.model_validate({"kind": "sie", "base_url": "https://sie.example.internal", "rate_cap": RATE_CAP})
    install_upstreams({"open-host": sie})

    assert sie.outputs is None
    validate_profile_upstreams(ModelConfig.model_validate(remote_model({"extract": {}}, adapter=SIE_ADAPTER)))


def test_with_remote_serving_off_the_endpoints_are_not_checked() -> None:
    install_upstreams({"open-host": Upstream.model_validate(openai_upstream(endpoints=["chat"]))}, remote_serving=False)

    validate_profile_upstreams(ModelConfig.model_validate(remote_model({"score": {}})))


def test_a_model_added_at_runtime_is_refused_when_its_upstream_cannot_serve_it() -> None:
    install("embeddings")
    registry = ModelRegistry(models_dir=None)

    with pytest.raises(ValueError, match="declare rerank"):
        registry.add_config(ModelConfig.model_validate(remote_model({"score": {}})))

    assert not registry.has_model("acme/remote")


async def test_a_config_snapshot_rejects_only_the_model_its_upstream_cannot_serve() -> None:
    install("embeddings")
    registry = ModelRegistry(models_dir=None)
    executor = QueueExecutor(registry)

    response = await executor.replace_model_configs(
        ReplaceModelConfigsRequest(
            bundle_id="default",
            epoch=1,
            bundle_config_hash="",
            models=[
                ReplaceModelConfigEntry(
                    model_id="acme/embed", model_config=yaml.safe_dump(remote_model(DENSE, "acme/embed"))
                ),
                ReplaceModelConfigEntry(
                    model_id="acme/rank", model_config=yaml.safe_dump(remote_model({"score": {}}, "acme/rank"))
                ),
            ],
        )
    )

    assert registry.has_model("acme/embed")
    assert not registry.has_model("acme/rank")
    assert response.applied_models == ["acme/embed"]
