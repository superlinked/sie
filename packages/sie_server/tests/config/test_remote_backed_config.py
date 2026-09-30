"""Remote-backed models and remote profiles in the model config."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams, validate_profile_upstreams

REMOTE_ADAPTER = "sie_server.adapters.remote.sie:SieUpstreamAdapter"
LOCAL_ADAPTER = "sie_server.adapters.fake.adapter:FakeAdapter"


def remote_profile(**loadtime: Any) -> dict[str, Any]:
    return {
        "adapter_path": REMOTE_ADAPTER,
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": loadtime or {"upstream": "team-sie", "upstream_model": "sie-fake"}},
    }


def remote_backed(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "sie_id": "acme/remote-fake",
        "remote_backed": True,
        "tasks": {"encode": {"dense": {"dim": 384}}},
        "profiles": {"default": remote_profile()},
    }
    spec.update(overrides)
    return spec


@pytest.fixture(autouse=True)
def _upstreams() -> Any:
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


def test_a_remote_backed_model_needs_no_weight_source() -> None:
    config = ModelConfig.model_validate(remote_backed())

    assert config.remote_backed is True
    assert config.hf_id is None
    validate_profile_upstreams(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [("hf_id", "BAAI/bge-m3"), ("weights_path", "/weights"), ("package_backed", True), ("hf_revision", "a" * 40)],
)
def test_a_remote_backed_model_sets_no_weight_source(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match="'remote_backed' models must not set"):
        ModelConfig.model_validate(remote_backed(**{field: value}))


def test_every_profile_of_a_remote_backed_model_is_remote() -> None:
    profiles = {"default": remote_profile(), "local": {"adapter_path": LOCAL_ADAPTER, "max_batch_tokens": 8192}}

    with pytest.raises(ValidationError, match="Profile 'local' of a 'remote_backed' model must use a remote adapter"):
        ModelConfig.model_validate(remote_backed(profiles=profiles))


@pytest.mark.parametrize(
    "loadtime",
    [
        {"upstream": "team-sie", "upstream_model": "sie-fake", "base_url": "https://attacker.example"},
        {"upstream": "team-sie", "upstream_model": "sie-fake", "api_key": "sk-literal"},
        {"upstream": "team-sie", "upstream_model": "sie-fake", "api_key_secret": "SOME_ENV"},
    ],
    ids=["base_url", "api_key", "api_key_secret"],
)
def test_a_model_config_cannot_define_an_upstream(loadtime: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="may only name an upstream and its model"):
        ModelConfig.model_validate(remote_backed(profiles={"default": remote_profile(**loadtime)}))


def test_a_remote_profile_names_both_the_upstream_and_its_model() -> None:
    with pytest.raises(ValidationError, match="must set load-time options: upstream_model"):
        ModelConfig.model_validate(remote_backed(profiles={"default": remote_profile(upstream="team-sie")}))


@pytest.mark.parametrize(
    "upstream_model",
    [
        "../admin",
        "a/../../v1/configs/models",
        "a/./b",
        "/abs",
        "a/",
        "a//b",
        "a?b",
        "a#b",
        "a b",
        "a%2e",
        "a\\b",
        5,
        ["a"],
    ],
)
def test_upstream_model_is_a_plain_model_id(upstream_model: Any) -> None:
    with pytest.raises(ValidationError, match="'upstream_model' must be a model id"):
        ModelConfig.model_validate(
            remote_backed(profiles={"default": remote_profile(upstream="team-sie", upstream_model=upstream_model)})
        )


@pytest.mark.parametrize("upstream", ["Team_SIE", "../x", ["team-sie"], 7])
def test_upstream_is_an_upstream_name(upstream: Any) -> None:
    with pytest.raises(ValidationError, match="'upstream' must be an upstream name"):
        ModelConfig.model_validate(
            remote_backed(profiles={"default": remote_profile(upstream=upstream, upstream_model="sie-fake")})
        )


@pytest.mark.parametrize("upstream_model", ["sie-fake", "BAAI/bge-m3", "org/name:profile", "org/v1.5_x-y"])
def test_ordinary_model_ids_are_accepted(upstream_model: str) -> None:
    ModelConfig.model_validate(
        remote_backed(profiles={"default": remote_profile(upstream="team-sie", upstream_model=upstream_model)})
    )


def test_an_undefined_upstream_name_is_rejected() -> None:
    config = ModelConfig.model_validate(
        remote_backed(profiles={"default": remote_profile(upstream="nobody", upstream_model="sie-fake")})
    )

    with pytest.raises(ValueError, match="names an undefined upstream 'nobody'"):
        validate_profile_upstreams(config)

    install_upstreams({}, remote_serving=False)
    validate_profile_upstreams(config)


def test_a_local_model_may_add_a_remote_profile() -> None:
    config = ModelConfig.model_validate(
        {
            "sie_id": "sie-fake-local",
            "package_backed": True,
            "tasks": {"encode": {"dense": {"dim": 384}}},
            "profiles": {
                "default": {"adapter_path": LOCAL_ADAPTER, "max_batch_tokens": 8192},
                "remote": remote_profile(),
            },
        }
    )

    assert config.resolve_profile("remote").loadtime["upstream"] == "team-sie"
    validate_profile_upstreams(config)
