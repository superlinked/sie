import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError
from sie_config.model_schema import model_config_schema_errors
from sie_server.config.equivalence import model_contract_digest
from sie_server.config.model import ModelConfig, ProfileConfig
from sie_server.core.batcher import BatchFormer
from sie_server.core.loader import load_model_configs
from sie_server.core.model_loader import ModelLoader
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.preprocessor_registry import PreprocessorRegistry
from sie_server.core.progress_batcher import ProgressBatchFormer


def _payload(parent: object = None, child: object = None) -> dict:
    return {
        "sie_id": "example/model",
        "hf_id": "example/model",
        "hf_revision": "a" * 40,
        "tasks": {},
        "profiles": {
            "default": {
                "adapter_path": "example:Adapter",
                "max_batch_tokens": 100,
                "bounded_batch_progress": parent,
            },
            "child": {"extends": "default", "bounded_batch_progress": child},
        },
    }


@pytest.mark.parametrize("parent", [None, False, True])
@pytest.mark.parametrize("child", [None, False, True])
def test_inheritance_and_yaml_variants_reach_real_worker(
    parent: bool | None, child: bool | None, tmp_path: Path
) -> None:
    payload = _payload(parent, child)
    assert model_config_schema_errors(payload) == []
    config = ModelConfig.model_validate(payload)
    expected = child if child is not None else parent is True
    assert config.resolve_profile("default").bounded_batch_progress is (parent is True)
    assert config.resolve_profile("child").bounded_batch_progress is expected
    # Explicit False in a child of True must not disappear during serialization.
    serialized = config.model_dump(mode="json")
    if child is False:
        assert serialized["profiles"]["child"]["bounded_batch_progress"] is False
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(serialized))
    configs = load_model_configs(tmp_path)
    variant = configs["example/model:child"]
    assert variant.resolve_profile("default").bounded_batch_progress is expected

    pool = ThreadPoolExecutor(max_workers=1)
    loader = ModelLoader(PreprocessorRegistry(), PostprocessorRegistry(pool), configs)
    adapter = MagicMock()
    adapter.max_concurrent_dispatch = 1
    adapter.get_preprocessor.return_value = None
    adapter.get_postprocessors.return_value = {}
    adapter.memory_footprint.return_value = 0
    adapter.lora_capability.return_value = None
    adapter.lora_runtime_config.return_value = None
    loaded = None
    try:
        # Actual loader worker construction, without checkpoint or inference.
        loaded = loader._finish_load(variant.sie_id, "cpu", adapter, variant)
        assert loaded.worker.config.bounded_batch_progress is expected
        expected_type = ProgressBatchFormer if expected else BatchFormer
        for selection in (None, "first", "second"):
            batcher = loaded.worker._get_batcher(selection)
            assert type(batcher) is expected_type
            assert hasattr(batcher, "_arrivals") is expected
            assert hasattr(batcher, "_reserved_request") is expected
    finally:
        if loaded is not None:
            loaded.worker._inference_executor.shutdown(wait=True)
        loader._load_executor.shutdown(wait=True)
        loader._download_executor.shutdown(wait=True)
        pool.shutdown(wait=True)


@pytest.mark.parametrize("invalid", [0, 1, -1, 0.0, 1.0, "true", "false", "yes", "", [], {}])
def test_invalid_option_is_rejected_by_worker_and_config_service(invalid: object) -> None:
    payload = _payload(invalid)
    with pytest.raises(ValidationError, match="bounded_batch_progress"):
        ModelConfig.model_validate(payload)
    errors = model_config_schema_errors(payload)
    assert any(error["loc"] == ["profiles", "default", "bounded_batch_progress"] for error in errors)


def test_omitted_and_null_defaults_preserve_serialized_contract() -> None:
    omitted = _payload()
    for profile in omitted["profiles"].values():
        profile.pop("bounded_batch_progress")
    implicit = ModelConfig.model_validate(omitted)
    explicit_null = ModelConfig.model_validate(_payload())
    assert implicit.model_dump() == explicit_null.model_dump()
    assert model_contract_digest(implicit) == model_contract_digest(explicit_null)
    for profile in implicit.model_dump()["profiles"].values():
        assert "bounded_batch_progress" not in profile
    for name in implicit.profiles:
        assert "bounded_batch_progress" not in implicit.resolve_profile(name).model_dump()
    enabled = ModelConfig.model_validate(_payload(True))
    assert model_contract_digest(enabled) != model_contract_digest(implicit)
    assert enabled.resolve_profile("default").model_dump()["bounded_batch_progress"] is True


@pytest.mark.parametrize("value", [None, False, True])
@pytest.mark.parametrize("mode", ["python", "json"])
def test_profile_serialization_respects_selection(value: bool | None, mode: str) -> None:
    profile = ProfileConfig(bounded_batch_progress=value)
    assert "bounded_batch_progress" not in profile.model_dump(mode=mode, exclude={"bounded_batch_progress"})
    assert "bounded_batch_progress" not in profile.model_dump(mode=mode, include={"max_batch_tokens"})
    expected = {} if value is None else {"bounded_batch_progress": value}
    for kwargs in ({"exclude_unset": True}, {"exclude_defaults": True}, {"include": {"bounded_batch_progress"}}):
        assert profile.model_dump(mode=mode, **kwargs) == expected
    assert json.loads(profile.model_dump_json(include={"bounded_batch_progress"})) == expected


def test_materialized_default_variants_do_not_gain_explicit_false(tmp_path: Path) -> None:
    payload = _payload()
    for profile in payload["profiles"].values():
        profile.pop("bounded_batch_progress")
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(payload))
    for config in load_model_configs(tmp_path).values():
        for profile in config.model_dump()["profiles"].values():
            assert "bounded_batch_progress" not in profile


def test_wrap_serializer_preserves_field_level_json_schemas() -> None:
    validation = ProfileConfig.model_json_schema(mode="validation")
    serialization = ProfileConfig.model_json_schema(mode="serialization")
    assert serialization == validation
    assert "bounded_batch_progress" in serialization["properties"]
    assert "max_batch_tokens" in serialization["properties"]
    assert serialization["additionalProperties"] is False
