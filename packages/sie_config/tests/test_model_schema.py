import copy
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from sie_config.config_api import router as config_router
from sie_config.config_store import ConfigStore
from sie_config.model_registry import ModelRegistry, validate_routing_config
from sie_config.model_schema import SCHEMA_PATH, _without_required, model_config_schema_errors
from sie_server.config.model import ModelConfig
from sie_server.config.routing import validate_model_routing

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGENERATE = (
    "mise exec -- uv run --frozen --project . --no-sync python -c "
    "'import json; from sie_server.config.model import ModelConfig; "
    "print(json.dumps(ModelConfig.model_json_schema(), indent=2))' "
    "> packages/sie_config/src/sie_config/model_config.schema.json"
)
_ADAPTER = "sie_server.adapters.bert_flash:BertFlashAdapter"
_UNKNOWN = "Unknown field: the worker model config schema does not define it."


def _model(**top_level: object) -> dict:
    profile = {"adapter_path": _ADAPTER, "max_batch_tokens": 4096}
    return {"sie_id": "acme/bert", "profiles": {"default": profile}, **top_level}


def test_checked_in_schema_matches_the_worker_model_config() -> None:
    expected = json.dumps(ModelConfig.model_json_schema(), indent=2) + "\n"
    assert SCHEMA_PATH.read_text(encoding="utf-8") == expected, f"Regenerate the schema: {_REGENERATE}"


def test_every_shipped_model_config_passes() -> None:
    models_dir = _REPO_ROOT / "packages" / "sie_server" / "models"
    paths = sorted(models_dir.glob("*.yaml"))
    assert paths
    rejected = {
        path.name: errors for path in paths if (errors := model_config_schema_errors(yaml.safe_load(path.read_text())))
    }
    assert rejected == {}


def test_partial_schema_preserves_required_named_properties_and_literal_data() -> None:
    literal = {"required": ["literal"]}
    schema = {
        "type": "object",
        "required": ["required", "payload"],
        "properties": {
            "required": {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
            },
            "payload": {"const": literal, "default": literal, "examples": [literal]},
        },
    }

    partial = _without_required(schema)
    validator = Draft202012Validator(partial)

    assert validator.is_valid({})
    assert validator.is_valid({"required": {}, "payload": literal})
    assert not validator.is_valid({"required": {"count": "wrong"}})
    assert not validator.is_valid({"payload": {}})
    assert partial["properties"]["payload"] == schema["properties"]["payload"]
    assert schema["required"] == ["required", "payload"]
    assert schema["properties"]["required"]["required"] == ["count"]


def test_partial_append_body_passes() -> None:
    assert model_config_schema_errors(_model()) == []


@pytest.mark.parametrize(
    "case",
    [
        "valid_fallback",
        "valid_triggers",
        "inherited_fallback",
        "remote_only",
        "unknown_policy",
        "missing_fallback",
        "default_fallback",
        "local_fallback",
        "missing_default",
        "remote_default",
        "empty_triggers",
        "duplicate_triggers",
        "threshold_field",
        "threshold",
        "encode",
        "score",
        "generate",
        "remote_backed_fallback",
        "remote_only_fields",
        "remote_only_local",
        "remote_only_weights",
        "remote_only_hf",
        "remote_only_revision",
        "remote_only_package",
        "remote_only_not_backed",
        "dangling_parent",
        "chained_parent",
        "cyclic_parent",
    ],
)
def test_routing_guard_agrees_with_worker(case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(tmp_path / "ipc.sock"))
    remote = {
        "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
        "max_batch_tokens": 8192,
        "kv_budget_tokens": 4096,
        "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "org/name"}},
    }
    config = _model(
        hf_id="acme/bert",
        tasks={"extract": {}},
        profiles={
            "default": {"adapter_path": _ADAPTER, "max_batch_tokens": 4096, "kv_budget_tokens": 4096},
            "remote": remote,
        },
        routing={"policy": "fallback", "fallback_profile": "remote"},
    )
    routing = config["routing"]
    profiles = config["profiles"]
    if case == "valid_triggers":
        routing["triggers"] = ["unhealthy", "saturated"]
    elif case == "inherited_fallback":
        profiles["parent"] = copy.deepcopy(remote)
        profiles["remote"] = {"extends": "parent"}
    elif case.startswith("remote_only") or case == "remote_backed_fallback":
        config.pop("hf_id")
        config["remote_backed"] = True
        profiles["default"] = copy.deepcopy(remote)
        if case != "remote_backed_fallback":
            config["routing"] = {"policy": "remote_only"}
        if case == "remote_only_fields":
            config["routing"]["triggers"] = []
        elif case == "remote_only_local":
            profiles["remote"]["adapter_path"] = _ADAPTER
        elif case == "remote_only_weights":
            config["weights_path"] = "/synthetic/weights"
        elif case == "remote_only_hf":
            config["hf_id"] = "acme/bert"
        elif case == "remote_only_revision":
            config["hf_revision"] = "0" * 40
        elif case == "remote_only_package":
            config["package_backed"] = True
        elif case == "remote_only_not_backed":
            config.pop("remote_backed")
            config["hf_id"] = "acme/bert"
    elif case == "unknown_policy":
        routing["policy"] = "unknown"
    elif case == "missing_fallback":
        routing.pop("fallback_profile")
    elif case == "default_fallback":
        routing["fallback_profile"] = "default"
    elif case == "local_fallback":
        profiles["remote"]["adapter_path"] = _ADAPTER
    elif case == "missing_default":
        profiles.pop("default")
    elif case == "remote_default":
        profiles["default"] = copy.deepcopy(remote)
    elif case == "empty_triggers":
        routing["triggers"] = []
    elif case == "duplicate_triggers":
        routing["triggers"] = ["model_loading", "model_loading"]
    elif case == "threshold_field":
        routing["wake_above"] = 1
    elif case == "threshold":
        routing.update(policy="threshold", wake_above=2, sleep_below=1, window_s=1, cooldown_s=1)
    elif case == "encode":
        config["tasks"]["encode"] = {"dense": {"dim": 384}}
    elif case == "score":
        config["tasks"]["score"] = {}
    elif case == "generate":
        config["tasks"]["generate"] = {"context_length": 8192, "max_output_tokens": 64}
    elif case in {"dangling_parent", "chained_parent", "cyclic_parent"}:
        profiles["remote"] = {"extends": "parent"}
        if case != "dangling_parent":
            profiles["parent"] = {"extends": "remote" if case == "cyclic_parent" else "default"}

    try:
        worker = ModelConfig.model_validate(config)
        validate_model_routing(worker)
    except ValueError:
        with pytest.raises(ValueError, match=r"routing|remote_backed"):
            validate_routing_config(config)
    else:
        validate_routing_config(config)


def test_unknown_keys_are_reported_per_field() -> None:
    config = _model(bogus=1, tasks={"generate": {"context_length": 8, "max_output_tokens": 8, "wat": True}})
    config["profiles"]["default"]["max_output_token"] = 5

    assert model_config_schema_errors(config) == [
        {"loc": ["bogus"], "message": _UNKNOWN},
        {"loc": ["profiles", "default", "max_output_token"], "message": _UNKNOWN},
        {"loc": ["tasks", "generate", "wat"], "message": _UNKNOWN},
    ]


def test_a_routing_block_passes_and_its_unknown_keys_are_reported() -> None:
    routing = {"policy": "fallback", "fallback_profile": "remote", "triggers": ["model_loading", "unhealthy"]}

    assert model_config_schema_errors(_model(routing=routing)) == []
    assert model_config_schema_errors(_model(routing={**routing, "upstream": "team-sie"})) == [
        {"loc": ["routing", "upstream"], "message": _UNKNOWN}
    ]
    assert model_config_schema_errors(_model(routing={**routing, "triggers": ["sometimes"]})) == [
        {
            "loc": ["routing", "triggers", 0],
            "message": "'sometimes' is not one of ['provisioning', 'model_loading', 'saturated', 'unhealthy']",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "loc", "message"),
    [
        ("max_batch_tokens", "8192", ["max_batch_tokens"], "'8192' is not of type 'integer'"),
        (
            "compute_precision",
            "fp8",
            ["compute_precision"],
            "'fp8' is not one of ['float16', 'bfloat16', 'float32']",
        ),
        ("adapter_options", {"loadtime": []}, ["adapter_options", "loadtime"], "[] is not of type 'object'"),
    ],
)
def test_type_errors_name_the_field(field: str, value: object, loc: list[str], message: str) -> None:
    config = _model()
    config["profiles"]["default"][field] = value

    assert model_config_schema_errors(config) == [{"loc": ["profiles", "default", *loc], "message": message}]


class TestConfigApiRejectsWorkerInvalidBodies:
    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        bundles_dir = tmp_path / "bundles"
        models_dir = tmp_path / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()
        bundle = {"name": "default", "priority": 10, "adapters": ["sie_server.adapters.bert_flash"]}
        (bundles_dir / "default.yaml").write_text(yaml.safe_dump(bundle))
        app = FastAPI()
        app.include_router(config_router)
        app.state.model_registry = ModelRegistry(bundles_dir, models_dir)
        app.state.nats_publisher = None
        app.state.config_store = ConfigStore(str(tmp_path / "store"))
        return TestClient(app)

    def test_post_with_unknown_profile_key_is_422_and_not_persisted(self, client: TestClient, tmp_path: Path) -> None:
        config = _model()
        config["profiles"]["default"]["max_output_token"] = 512

        resp = client.post("/v1/configs/models", content=yaml.safe_dump(config))

        assert resp.status_code == 422
        assert resp.json()["detail"] == {
            "error": "validation_error",
            "details": [{"loc": ["profiles", "default", "max_output_token"], "message": _UNKNOWN}],
        }
        assert client.get("/v1/configs/models/acme/bert").status_code == 404
        assert client.get("/v1/configs/epoch").json()["epoch"] == 0
        assert not (tmp_path / "store" / "models" / "acme__bert.yaml").exists()

    def test_post_append_with_type_error_keeps_existing_profiles(self, client: TestClient, tmp_path: Path) -> None:
        assert client.post("/v1/configs/models", content=yaml.safe_dump(_model())).status_code == 201
        append = {"sie_id": "acme/bert", "profiles": {"fast": {"extends": "default", "max_batch_tokens": "lots"}}}

        resp = client.post("/v1/configs/models", content=yaml.safe_dump(append))

        assert resp.status_code == 422
        assert resp.json()["detail"]["details"] == [
            {"loc": ["profiles", "fast", "max_batch_tokens"], "message": "'lots' is not of type 'integer'"}
        ]
        stored = yaml.safe_load((tmp_path / "store" / "models" / "acme__bert.yaml").read_text())
        assert set(stored["profiles"]) == {"default"}
        assert client.get("/v1/configs/epoch").json()["epoch"] == 1

    def test_put_with_unknown_top_level_key_is_422(self, client: TestClient) -> None:
        resp = client.put("/v1/configs/models/acme/bert", content=yaml.safe_dump(_model(description="x")))

        assert resp.status_code == 422
        assert resp.json()["detail"]["details"] == [{"loc": ["description"], "message": _UNKNOWN}]
        assert client.get("/v1/configs/models/acme/bert").status_code == 404


def test_config_service_accepts_generation_fallback_with_pre_output_handling() -> None:
    config = _model(
        hf_id="acme/bert",
        tasks={"generate": {"context_length": 8192, "max_output_tokens": 64}},
        profiles={
            "default": {"adapter_path": _ADAPTER, "max_batch_tokens": 4096, "kv_budget_tokens": 4096},
            "remote": {
                "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
                "max_batch_tokens": 8192,
                "kv_budget_tokens": 4096,
                "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "org/name"}},
            },
        },
        routing={"policy": "fallback", "fallback_profile": "remote"},
    )
    validate_model_routing(ModelConfig.model_validate(config))
    validate_routing_config(config)
