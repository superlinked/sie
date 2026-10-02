import pytest
from sie_server.config.model import ModelConfig


def generation_config(remote: bool) -> dict:
    return {
        "sie_id": "acme/generation",
        "remote_backed": remote,
        **({} if remote else {"package_backed": True}),
        "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64}},
        "profiles": {
            "default": {
                "adapter_path": (
                    "sie_server.adapters.remote.sie:SieUpstreamAdapter"
                    if remote
                    else "sie_server.adapters.fake.adapter:FakeAdapter"
                ),
                "max_batch_tokens": 8192,
                **(
                    {"adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "acme/model"}}}
                    if remote
                    else {}
                ),
            }
        },
    }


def test_remote_generation_has_no_local_kv_budget() -> None:
    config = ModelConfig.model_validate(generation_config(True))
    assert config.profiles["default"].kv_budget_tokens is None


def test_remote_generation_inheritance_has_no_local_kv_budget() -> None:
    data = generation_config(True)
    data["profiles"]["child"] = {"extends": "default"}
    config = ModelConfig.model_validate(data)
    assert config.resolve_profile("child").kv_budget_tokens is None


def test_local_generation_still_requires_a_kv_budget() -> None:
    with pytest.raises(ValueError, match="missing 'kv_budget_tokens'"):
        ModelConfig.model_validate(generation_config(False))
