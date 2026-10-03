from copy import deepcopy

import pytest
from sie_server.config.engine import EngineConfig
from sie_server.config.model import ModelConfig
from sie_server.core import profile_identity

PIN = "a" * 40


def config(**changes: object) -> ModelConfig:
    data = {
        "sie_id": "catalog/model",
        "hf_id": "weights/model",
        "hf_revision": PIN,
        "inputs": {"text": True},
        "tasks": {"encode": {"dense": {"dim": 384}}},
        "max_sequence_length": 512,
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.bge_m3:BGEM3Adapter",
                "max_batch_tokens": 8192,
                "compute_precision": "float32",
                "adapter_options": {
                    "loadtime": {"pooling": "mean", "trust_remote_code": False},
                    "runtime": {"normalize": True},
                },
            }
        },
        **changes,
    }
    return ModelConfig.model_validate(data)


@pytest.fixture(autouse=True)
def execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_identity, "_execution_code", lambda: {"sources": "f" * 64, "dependencies": {"torch": "2.9"}}
    )


def identity(model: ModelConfig, profile: str = "default", *, device: str = "cpu") -> str | None:
    return profile_identity.local_profile_identity(model, profile, device=device)


def test_alias_and_inheritance_preserve_resolved_identity() -> None:
    original = config()
    renamed = config(sie_id="alias/model")
    inherited = config(profiles={**original.model_dump()["profiles"], "other": {"extends": "default"}})
    assert identity(original) == identity(renamed)
    assert identity(original) == identity(inherited, "other")
    result = identity(original)
    assert result is not None
    assert result.startswith("v1:sha256:")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hf_id", "different/weights"),
        ("hf_revision", "b" * 40),
        ("max_sequence_length", 1024),
        ("hf_tokenizer_dependencies", {"tokenizer/model": "b" * 40}),
    ],
)
def test_model_semantics_change_identity(field: str, value: object) -> None:
    assert identity(config()) != identity(config(**{field: value}))


@pytest.mark.parametrize(("section", "key", "value"), [("loadtime", "pooling", "cls"), ("runtime", "normalize", False)])
def test_profile_semantics_change_identity(section: str, key: str, value: object) -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"][section][key] = value
    assert identity(config()) != identity(ModelConfig.model_validate(data))


def test_precision_and_device_change_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["compute_precision"] = "float16"
    assert identity(config()) != identity(ModelConfig.model_validate(data))
    assert identity(config(), device="cpu") != identity(config(), device="cuda:0")


def test_execution_stack_change_changes_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    first = identity(config())
    monkeypatch.setattr(
        profile_identity, "_execution_code", lambda: {"sources": "e" * 64, "dependencies": {"torch": "2.9"}}
    )
    assert identity(config()) != first
    monkeypatch.setattr(
        profile_identity, "_execution_code", lambda: {"sources": "f" * 64, "dependencies": {"torch": "2.10"}}
    )
    assert identity(config()) != first


def test_runtime_settings_are_observed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_identity, "_execution_runtime", lambda: {"matmul_precision": "highest"})
    first = identity(config())
    assert first is not None
    monkeypatch.setattr(profile_identity, "_execution_runtime", lambda: {"matmul_precision": "high"})
    assert identity(config()) != first


def test_runtime_version_failure_does_not_break_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def incompatible_cudnn() -> None:
        raise RuntimeError("incompatible cuDNN")

    monkeypatch.setattr(profile_identity.torch.backends.cudnn, "version", incompatible_cudnn)
    assert identity(config()) is None


@pytest.mark.parametrize("revision", [None, "main", "short"])
def test_mutable_weights_have_no_identity(revision: str | None) -> None:
    assert identity(config(hf_revision=revision)) is None


def test_unpinned_tokenizer_has_no_identity() -> None:
    assert identity(config(hf_tokenizer_dependencies={"tokenizer/model": "main"})) is None


def test_remote_profile_cannot_claim_local_identity() -> None:
    data = config().model_dump()
    data["profiles"]["remote"] = {
        "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
        "max_batch_tokens": 8192,
        "adapter_options": {"loadtime": {"upstream": "private", "upstream_model": "served/model"}},
    }
    assert identity(ModelConfig.model_validate(data), "remote") is None


def test_unidentified_custom_adapter_has_no_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_path"] = "custom.vendor:Adapter"
    assert identity(ModelConfig.model_validate(data)) is None


def test_unserializable_or_nonfinite_options_fail_closed() -> None:
    data = config().model_dump()
    for value in (object(), float("nan")):
        changed = deepcopy(data)
        changed["profiles"]["default"]["adapter_options"]["loadtime"]["custom"] = value
        assert identity(ModelConfig.model_validate(changed)) is None


def test_settings_order_does_not_change_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"] = {
        "runtime": {"normalize": True},
        "loadtime": {"pooling": "mean", "trust_remote_code": False},
    }
    assert identity(config()) == identity(ModelConfig.model_validate(data))


@pytest.mark.parametrize("source", ["/operator/local/model", "./local-model", "../local-model"])
def test_path_like_hf_source_has_no_identity(source: str) -> None:
    assert identity(config(hf_id=source)) is None


def test_local_weights_override_has_no_identity() -> None:
    assert identity(config(weights_path="/operator/local/model")) is None


def test_repo_shaped_local_directory_has_no_identity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights" / "model").mkdir(parents=True)
    assert identity(config()) is None


@pytest.mark.parametrize(
    "adapter",
    [
        "sie_server.adapters.sglang:SGLangGenerationAdapter",
        "sie_server.adapters.mlx:MLXGenerationAdapter",
        "sie_server.adapters.bge_m3_flag:BGEM3FlagAdapter",
        "sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter",
        "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
        "sie_server.adapters.sentence_transformer:SentenceTransformerSparseAdapter",
        "sie_server.adapters.cross_encoder:CrossEncoderAdapter",
    ],
)
def test_child_or_alternative_weights_engine_has_no_identity(adapter: str) -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_path"] = adapter
    data["profiles"]["default"]["adapter_options"]["loadtime"]["mlx_repo"] = "mutable/alternate-repo"
    assert identity(ModelConfig.model_validate(data)) is None


def test_lora_weights_have_no_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["runtime"]["lora_id"] = "mutable/adapter"
    assert identity(ModelConfig.model_validate(data)) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("revision", "main"),
        ("model_name_or_path", "mutable/alternate"),
        ("config_kwargs", {"revision": "main"}),
        ("config_kwargs", {"trust_remote_code": True}),
    ],
)
def test_loadtime_weight_or_remote_code_overrides_have_no_identity(key: str, value: object) -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["loadtime"][key] = value
    assert identity(ModelConfig.model_validate(data)) is None


def test_engine_defaults_must_be_identified_and_change_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["compute_precision"] = None
    model = ModelConfig.model_validate(data)
    assert identity(model, device="cuda:0") is None
    first = profile_identity.local_profile_identity(
        model, "default", device="cuda:0", engine_config=EngineConfig(default_compute_precision="float16")
    )
    second = profile_identity.local_profile_identity(
        model, "default", device="cuda:0", engine_config=EngineConfig(default_compute_precision="float32")
    )
    assert first is not None
    assert second is not None
    assert first != second


def test_remote_code_execution_has_no_identity() -> None:
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["loadtime"]["trust_remote_code"] = True
    assert identity(ModelConfig.model_validate(data)) is None


def test_deep_options_fail_closed() -> None:
    value: object = "nested"
    for _ in range(1200):
        value = [value]
    data = config().model_dump()
    data["profiles"]["default"]["adapter_options"]["loadtime"]["custom"] = value
    assert identity(ModelConfig.model_validate(data)) is None
