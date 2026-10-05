"""Only fresh deployment-owned exact contract evidence can admit a bridge."""

import contextlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from opentelemetry import trace
from pydantic import ValidationError
from sie_server import ipc_server
from sie_server.api.encode import router as encode_router
from sie_server.api.routing import route_request
from sie_server.config import hybrid_admission
from sie_server.config.equivalence import (
    EquivalenceRecord,
    canonical_digest,
    model_contract_digest,
    remote_profile_contract_digest,
    upstream_contract_digest,
)
from sie_server.config.fleet_equivalence import FleetEquivalenceRecord, equivalence_record_digest
from sie_server.config.model import ModelConfig
from sie_server.config.routing import validate_model_routing
from sie_server.config.upstreams import EquivalencePolicy, Upstream, install_upstreams
from sie_server.core.loader import expand_profile_variants
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_server import IpcServer
from sie_server.ipc_types import (
    BatchedF16MultivectorItem,
    BatchedF16MultivectorOutput,
    BatchOutcome,
    ItemOutcome,
    NumericalProfileSnapshotRequest,
    ProcessEncodeBatchRequest,
    ProcessScoreBatchRequest,
    ReplaceModelConfigEntry,
    ReplaceModelConfigsRequest,
    RunBatchRequest,
    WorkerCapabilitiesRequest,
    WorkerCapabilitiesResponse,
)
from sie_server.queue_executor import QueueExecutor

NOW = datetime(2026, 10, 3, tzinfo=UTC)
IDENTITY = "v1:sha256:" + "a" * 64
OTHER_IDENTITY = "v1:sha256:" + "e" * 64


@pytest.fixture
def admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ModelConfig, Upstream, Path, dict[str, Any]]]:
    path = tmp_path / "proof.json"
    upstream = Upstream.model_validate(
        {
            "kind": "openai",
            "base_url": "http://127.0.0.1:9",
            "endpoints": ["embeddings"],
            "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
            "equivalence": {"max_age_s": 60, "record_files": {"local/model": str(path)}},
        }
    )
    config = ModelConfig.model_validate(
        {
            "sie_id": "local/model",
            "hf_id": "BAAI/bge-m3",
            "hf_revision": "b" * 40,
            "max_sequence_length": 512,
            "tasks": {"encode": {"dense": {"dim": 1024}}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bge_m3:BGEM3Adapter",
                    "max_batch_tokens": 8192,
                    "compute_precision": "float32",
                },
                "remote": {
                    "adapter_path": "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"upstream": "vendor", "upstream_model": "vendor/model"}},
                },
            },
            "routing": {"policy": "fallback", "fallback_profile": "remote"},
        }
    )
    monkeypatch.setattr(hybrid_admission, "local_profile_identity", lambda *args, **kwargs: IDENTITY)
    install_upstreams({"vendor": upstream})
    data = {
        "version": 3,
        "measured_at": NOW.isoformat(),
        "upstream_name": "vendor",
        "upstream_model": "vendor/model",
        "upstream_contract_sha256": upstream_contract_digest(upstream),
        "model_contract_sha256": model_contract_digest(config),
        "probe_sources_sha256": "c" * 64,
        "remote_contract_sha256": remote_profile_contract_digest(config, "remote", {"vendor": upstream}),
        "remote_execution_sha256": hybrid_admission.serving_code_digest(),
        "local_observation_sha256": canonical_digest({"identity": IDENTITY, "revision": config.hf_revision}),
        "runtime_options_sha256": canonical_digest(dict(config.resolve_profile("default").runtime)),
        "output_dtype": "float32",
        "local_identity": IDENTITY,
        "model": config.sie_id,
        "remote_profile": "remote",
        "context_length": 512,
        "outputs": ["dense"],
        "cases": [
            {
                "operation": "encode",
                "category": category,
                "input_sha256": "d" * 64,
                "token_counts": [513 if category == "boundary_after" else 512],
                "outcomes": ["ok"] * 3,
                "measurements": {"dense": {"noise_floor": 0.0, "remote_error": 0.0, "values": 1024}},
            }
            for category in (
                "short",
                "long",
                "boundary_before",
                "boundary_after",
                "query_prefix",
                "query_default",
                "empty_prefix",
                "document_prefix",
            )
        ],
    }
    path.write_text(json.dumps(data))
    yield config, upstream, path, data
    install_upstreams({})


def refusal(config: ModelConfig, *, now: datetime = NOW) -> str | None:
    return hybrid_admission.openai_equivalence_refusal(config, device="cpu", now=now)


def test_matching_passing_evidence_is_admitted(admission: tuple) -> None:
    config, _, _, _ = admission
    assert refusal(config) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other/model"),
        ("remote_profile", "other"),
        ("upstream_name", "other"),
        ("upstream_model", "other"),
        ("context_length", 1024),
        ("local_identity", "v1:sha256:" + "f" * 64),
        ("model_contract_sha256", "f" * 64),
        ("upstream_contract_sha256", "f" * 64),
        ("remote_contract_sha256", "f" * 64),
        ("local_observation_sha256", "f" * 64),
        ("runtime_options_sha256", "f" * 64),
        ("remote_execution_sha256", "f" * 64),
    ],
)
def test_evidence_cannot_authorize_a_different_contract(admission: tuple, field: str, value: Any) -> None:
    config, _, path, data = admission
    data[field] = value
    path.write_text(json.dumps(data))
    assert refusal(config) is not None


@pytest.mark.parametrize("age_s", [-1, 61])
def test_future_or_expired_evidence_is_refused(admission: tuple, age_s: int) -> None:
    config, _, _, _ = admission
    assert refusal(config, now=NOW + timedelta(seconds=age_s)) is not None
    assert refusal(config, now=NOW + timedelta(seconds=60)) is None


def test_failed_measurement_cannot_admit(admission: tuple) -> None:
    config, _, path, data = admission
    data["cases"][0]["measurements"]["dense"]["remote_error"] = 0.1
    path.write_text(json.dumps(data))
    assert refusal(config) is not None


def test_record_is_reread_and_removal_revokes_admission(admission: tuple) -> None:
    config, _, path, data = admission
    assert refusal(config) is None
    path.unlink()
    assert refusal(config) is not None
    path.write_text(json.dumps(data))
    assert refusal(config) is None
    path.write_bytes(b"private" * (100 << 10))
    assert refusal(config) == "hybrid equivalence record cannot be validated"


def test_unidentified_local_execution_cannot_admit(admission: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _, _ = admission
    monkeypatch.setattr(hybrid_admission, "local_profile_identity", lambda *args, **kwargs: None)
    assert refusal(config) == "hybrid local execution cannot be identified"


def test_new_operator_policy_does_not_change_the_endpoint_contract(admission: tuple) -> None:
    config, upstream, _, _ = admission
    without_proof = upstream.model_copy(update={"equivalence": None})
    assert upstream_contract_digest(without_proof) == upstream_contract_digest(upstream)
    assert remote_profile_contract_digest(
        config, "remote", {"vendor": without_proof}
    ) == remote_profile_contract_digest(config, "remote", {"vendor": upstream})
    install_upstreams({"vendor": without_proof})
    assert refusal(config) is not None


def test_nonregular_record_refuses_without_waiting_for_a_writer(admission: tuple) -> None:
    config, _, path, _ = admission
    path.unlink()
    os.mkfifo(path)
    assert refusal(config) == "hybrid equivalence record cannot be validated"


@pytest.mark.parametrize("max_age", [0, -1, True, 604801])
def test_age_policy_is_bounded_and_strict(tmp_path: Path, max_age: Any) -> None:
    with pytest.raises(ValidationError):
        EquivalencePolicy(max_age_s=max_age, record_files={"model": str(tmp_path / "record.json")})
    assert (
        EquivalencePolicy(max_age_s=604800, record_files={"model": str(tmp_path / "record.json")}).max_age_s == 604800
    )


def test_record_paths_are_absolute_and_bounded() -> None:
    for name, path in (("model", "relative.json"), ("", "/record"), ("model", "/" + "a" * 4096)):
        with pytest.raises(ValidationError):
            EquivalencePolicy(max_age_s=60, record_files={name: path})


def _refresh_record(path: Path, data: dict[str, Any]) -> None:
    data["measured_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(data))


def test_registry_admits_current_proof_and_refuses_a_missing_record(admission: tuple, tmp_path: Path) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(config)
    assert registry.has_model(config.sie_id)
    assert registry.has_model(config.sie_id + ":remote")
    path.unlink()
    with pytest.raises(ValueError, match="record cannot be validated"):
        registry.add_config(config)


def test_directory_load_checks_the_same_runtime_evidence(admission: tuple, tmp_path: Path) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    registry = ModelRegistry(models_dir=models, device="cpu", enable_hot_reload=False)
    assert registry.has_model(config.sie_id)
    path.unlink()
    with pytest.raises(ValueError, match="record cannot be validated"):
        ModelRegistry(models_dir=models, device="cpu", enable_hot_reload=False)


def test_validator_without_a_runtime_cannot_admit_operator_proof(admission: tuple) -> None:
    config, _, _, _ = admission
    with pytest.raises(ValueError, match="refused until"):
        validate_model_routing(config)


@pytest.mark.parametrize("expired", [False, True])
async def test_prebridge_rechecks_freshness_while_preserving_local_warmup(admission: tuple, expired: bool) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    if expired:
        data["measured_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        path.write_text(json.dumps(data))
    registry = MagicMock(spec=ModelRegistry)
    registry.device = "cpu"
    registry.engine_config = None
    variants = expand_profile_variants([config])
    registry.get_config.side_effect = variants.__getitem__
    registry.has_model.return_value = True
    registry.is_unloading.return_value = False
    registry.is_loading.side_effect = lambda name: name == config.sie_id
    registry.is_loaded.side_effect = lambda name: name != config.sie_id
    registry.start_load_async = AsyncMock(return_value=True)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(registry=registry)),
        }
    )
    if expired:
        with pytest.raises(HTTPException) as refused:
            await route_request(request, config.sie_id, trace.INVALID_SPAN)
        assert refused.value.status_code == 503
        assert refused.value.detail["code"] == "MODEL_LOADING"
        assert refused.value.headers == {
            "Retry-After": "5",
            "X-SIE-Served-By": "local",
            "X-SIE-Fallback-Reason": "model_loading",
            "X-SIE-Fallback-Error": "INFERENCE_ERROR",
        }
        assert not hasattr(request.state, "serving_route")
    else:
        route = await route_request(request, config.sie_id, trace.INVALID_SPAN)
        assert route.key == config.sie_id + ":remote"
        assert route.upstream == "vendor"
    registry.start_load_async.assert_awaited_once_with(config.sie_id, "cpu")
    registry.load_now.assert_not_awaited()


@pytest.mark.parametrize("field", ["runtime_options_sha256", "output_dtype", "remote_execution_sha256"])
def test_previous_protocol_or_missing_runtime_binding_cannot_admit(admission: tuple, field: str) -> None:
    config, _, path, data = admission
    for version in (1, 2):
        path.write_text(json.dumps({**data, "version": version, "local_instance_id": "a" * 64}))
        assert refusal(config) == "hybrid equivalence record cannot be validated"
    data.pop(field)
    path.write_text(json.dumps(data))
    assert refusal(config) is not None


@pytest.mark.parametrize(
    "options", [{"normalize": False}, {"normalize": True}, {"pooling": "mean"}, {"overflow_policy": "truncate"}]
)
def test_unmeasured_runtime_overrides_cannot_bridge(admission: tuple, options: dict) -> None:
    config, _, _, _ = admission
    assert hybrid_admission.hybrid_request_refusal(config, options) is not None
    assert hybrid_admission.hybrid_request_refusal(config, {"is_query": True}) is None


@pytest.mark.parametrize(
    ("local", "remote", "admitted"),
    [
        ({}, {"normalize": False}, False),
        ({"normalize": True}, {"normalize": False}, False),
        ({"normalize": True}, {}, False),
        ({"normalize": True}, {"normalize": True}, True),
    ],
)
def test_a_remote_profile_must_run_with_exactly_the_measured_runtime(
    admission: tuple, local: dict[str, Any], remote: dict[str, Any], admitted: bool
) -> None:
    config, upstream, path, data = admission
    config.profiles["default"].adapter_options.runtime.update(local)
    config.profiles["remote"].adapter_options.runtime.update(remote)
    config._resolved_cache.clear()
    rebuilt = {
        **data,
        "model_contract_sha256": model_contract_digest(config),
        "remote_contract_sha256": remote_profile_contract_digest(config, "remote", {"vendor": upstream}),
        "runtime_options_sha256": canonical_digest(dict(config.resolve_profile("default").runtime)),
    }
    path.write_text(json.dumps(rebuilt))
    assert refusal(config) == (
        None if admitted else "hybrid remote profile runtime differs from the measured local runtime"
    )


def test_matching_explicit_defaults_are_allowed_but_numerical_type_aliases_are_not(admission: tuple) -> None:
    config, _, _, _ = admission
    config.profiles["default"].adapter_options.runtime["normalize"] = True
    config._resolved_cache.clear()
    assert hybrid_admission.hybrid_request_refusal(config, {"normalize": True}) is None
    assert hybrid_admission.hybrid_request_refusal(config, {"normalize": False}) is not None
    assert hybrid_admission.hybrid_request_refusal(config, {"normalize": 1}) is not None


@pytest.mark.parametrize("dtype", ["float16", "int8", "uint8", "binary", "ubinary"])
def test_quantized_output_does_not_inherit_float32_equivalence(admission: tuple, dtype: str) -> None:
    config, _, _, _ = admission
    assert hybrid_admission.hybrid_request_refusal(config, {"output_dtype": dtype}) is not None
    assert hybrid_admission.hybrid_request_refusal(config, {"output_dtype": "float32"}) is None
    config.profiles["default"].adapter_options.runtime["output_dtype"] = dtype
    config._resolved_cache.clear()
    assert refusal(config) == "hybrid output dtype differs from the measured float32 contract"


def test_a_different_execution_identity_is_not_covered(admission: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _, _ = admission
    assert refusal(config) is None
    monkeypatch.setattr(hybrid_admission, "local_profile_identity", lambda *args, **kwargs: OTHER_IDENTITY)
    assert refusal(config) == "hybrid equivalence record does not cover this execution identity"


def test_a_changed_remote_execution_cannot_reuse_evidence(admission: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _, _ = admission
    monkeypatch.setattr(hybrid_admission, "serving_code_digest", lambda: "f" * 64)
    assert refusal(config) == "hybrid equivalence record differs from the current serving contract"
    monkeypatch.setattr(hybrid_admission, "serving_code_digest", lambda: None)
    assert refusal(config) == "hybrid remote execution cannot be identified"


def _identity_record(data: dict[str, Any], identity: str, config: ModelConfig, **changes: Any) -> dict[str, Any]:
    local = canonical_digest({"identity": identity, "revision": config.hf_revision})
    return {**data, "local_identity": identity, "local_observation_sha256": local, **changes}


def _bundle(*records: dict[str, Any]) -> str:
    parsed = tuple(EquivalenceRecord.model_validate_json(json.dumps(record)) for record in records)
    return FleetEquivalenceRecord(records=parsed).model_dump_json()


def test_bundle_admits_each_measured_identity_and_names_them(admission: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, path, data = admission
    other = _identity_record(data, OTHER_IDENTITY, config, measured_at=(NOW - timedelta(seconds=30)).isoformat())
    path.write_text(_bundle(data, other))
    admitted = hybrid_admission.openai_admission(config, now=NOW)
    assert isinstance(admitted, hybrid_admission.NumericalAdmission)
    assert admitted.local_identities == {IDENTITY, OTHER_IDENTITY}
    assert admitted.outputs == {"dense"}
    assert admitted.model_contract_sha256 == model_contract_digest(config)
    assert admitted.expires_at == NOW + timedelta(seconds=30)
    assert refusal(config) is None
    monkeypatch.setattr(hybrid_admission, "local_profile_identity", lambda *args, **kwargs: OTHER_IDENTITY)
    assert refusal(config) is None


def test_admission_digest_names_exactly_the_admitted_records(admission: tuple) -> None:
    config, _, path, data = admission
    other = _identity_record(data, OTHER_IDENTITY, config)
    path.write_text(_bundle(data))
    single = hybrid_admission.openai_admission(config, now=NOW)
    path.write_text(_bundle(data, other))
    both = hybrid_admission.openai_admission(config, now=NOW)
    failed_cases = json.loads(json.dumps(other["cases"]))
    failed_cases[0]["measurements"]["dense"]["remote_error"] = 0.1
    path.write_text(_bundle(data, {**other, "cases": failed_cases}))
    with_failure = hybrid_admission.openai_admission(config, now=NOW)
    assert isinstance(single, hybrid_admission.NumericalAdmission)
    assert isinstance(both, hybrid_admission.NumericalAdmission)
    assert isinstance(with_failure, hybrid_admission.NumericalAdmission)
    assert single.sha256 != both.sha256
    assert with_failure.sha256 == single.sha256
    assert with_failure.local_identities == {IDENTITY}
    record = EquivalenceRecord.model_validate_json(json.dumps(data))
    assert single.sha256 == canonical_digest(
        {
            "version": 1,
            "kind": "openai",
            "model": config.sie_id,
            "remote_profile": "remote",
            "remote_contract_sha256": data["remote_contract_sha256"],
            "records": [equivalence_record_digest(record)],
        }
    )


def test_a_failed_measurement_for_this_identity_is_not_covered_by_another(
    admission: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, path, data = admission
    cases = json.loads(json.dumps(data["cases"]))
    cases[0]["measurements"]["dense"]["remote_error"] = 0.1
    path.write_text(_bundle({**data, "cases": cases}, _identity_record(data, OTHER_IDENTITY, config)))
    assert refusal(config) == "hybrid equivalence record does not cover this execution identity"
    monkeypatch.setattr(hybrid_admission, "local_profile_identity", lambda *args, **kwargs: OTHER_IDENTITY)
    assert refusal(config) is None


_RESTART_PROBE = """
import importlib, json, sys
from pathlib import Path
from sie_server.config.model import ModelConfig
from sie_server.config.routing import validate_model_routing
from sie_server.config.upstreams import install_upstreams, load_upstreams
from sie_server.core.profile_identity import local_profile_identity, runtime_instance_id, serving_code_digest
config = ModelConfig.model_validate_json(Path(sys.argv[1]).read_text())
install_upstreams(load_upstreams(sys.argv[2]))
if sys.argv[3] == "warm":
    importlib.import_module("transformers.models.xlm_roberta.modeling_xlm_roberta")
try:
    validate_model_routing(config, device="cpu")
    admitted = True
except ValueError:
    admitted = False
print(json.dumps({
    "identity": local_profile_identity(config, "default", device="cpu"),
    "code": serving_code_digest(),
    "instance": runtime_instance_id(),
    "admitted": admitted,
}))
"""


def _run_server_process(config_file: Path, upstreams_file: Path, state: str) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603 - executes the fixed local interpreter
        [sys.executable, "-c", _RESTART_PROBE, str(config_file), str(upstreams_file), state],
        capture_output=True,
        check=True,
        text=True,
        timeout=300,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_evidence_measured_in_one_process_admits_a_new_process_with_the_same_identity(
    admission: tuple, tmp_path: Path
) -> None:
    config, upstream, path, data = admission
    config_file, upstreams_file = tmp_path / "model.json", tmp_path / "upstreams.yaml"
    config_file.write_text(config.model_dump_json())
    upstreams_file.write_text(yaml.safe_dump({"upstreams": {"vendor": upstream.model_dump(mode="json")}}))
    path.unlink()
    measured = _run_server_process(config_file, upstreams_file, "warm")
    assert measured["identity"] is not None
    assert not measured["admitted"]
    path.write_text(
        json.dumps(
            _identity_record(
                data,
                measured["identity"],
                config,
                measured_at=datetime.now(UTC).isoformat(),
                remote_execution_sha256=measured["code"],
            )
        )
    )
    restarted = _run_server_process(config_file, upstreams_file, "cold")
    assert restarted["instance"] != measured["instance"]
    assert restarted["identity"] == measured["identity"]
    assert restarted["admitted"]


@pytest.mark.parametrize("expired", [False, True])
async def test_snapshot_admits_fresh_proof_and_retains_expired_model_without_blocking_others(
    admission: tuple, expired: bool
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    executor = QueueExecutor(registry)

    def entry(model: ModelConfig) -> ReplaceModelConfigEntry:
        return ReplaceModelConfigEntry(
            model_id=model.sie_id, model_config=yaml.safe_dump(model.model_dump(mode="json"))
        )

    def snapshot(models: list[ModelConfig]) -> ReplaceModelConfigsRequest:
        return ReplaceModelConfigsRequest(
            bundle_id="default", epoch=1, bundle_config_hash="", models=[entry(model) for model in models]
        )

    response = await executor.replace_model_configs(snapshot([config]))
    assert response.applied
    assert registry.has_model(config.sie_id)
    retained = registry.get_config(config.sie_id)
    if expired:
        path.unlink()
    other_data = config.model_dump(mode="json")
    other_data.update(sie_id="other/model", routing=None)
    other_data["profiles"].pop("remote")
    other = ModelConfig.model_validate(other_data)
    response = await executor.replace_model_configs(snapshot([config, other]))
    assert response.applied
    assert registry.has_model(other.sie_id)
    if expired:
        assert registry.get_config(config.sie_id) is retained
        assert refusal(config) is not None


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_top_level_encode_dtype_keeps_cold_request_local(admission: tuple, dtype: str) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    registry = MagicMock(spec=ModelRegistry)
    registry.device = "cpu"
    registry.engine_config = None
    registry.get_config.side_effect = expand_profile_variants([config]).__getitem__
    registry.has_model.return_value = True
    registry.is_unloading.return_value = False
    registry.is_loading.return_value = True
    registry.is_loaded.return_value = False
    registry.is_failed.return_value = False
    registry.start_load_async = AsyncMock(return_value=True)
    app = FastAPI()
    app.state.registry = registry
    app.include_router(encode_router)
    with TestClient(app) as client:
        response = client.post(
            f"/v1/encode/{config.sie_id}", json={"items": [{"text": "query"}], "params": {"output_dtype": dtype}}
        )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "MODEL_LOADING"
    assert response.headers["Retry-After"] == "5"
    registry.start_load_async.assert_awaited_once_with(config.sie_id, "cpu")
    registry.get_worker.assert_not_called()


@pytest.mark.parametrize(
    ("device", "devices", "expected"),
    [
        ("cpu", None, "cpu"),
        ("cuda", None, None),
        ("cuda", ["cuda:0"], "cuda:0"),
        ("cuda:0", ["cuda:0", "cuda:1"], None),
        ("cuda:0", ["cuda:1"], None),
        ("cpu", ["cuda:0"], None),
        ("cuda", ["cuda:1"], "cuda:1"),
    ],
)
def test_hybrid_device_authority_requires_stable_placement(
    device: str, devices: list[str] | None, expected: str | None
) -> None:
    registry = ModelRegistry(device=device, devices=devices, enable_hot_reload=False)
    assert registry.profile_execution_device("local/model") == expected
    registry._loaded["local/model"] = SimpleNamespace(device="cuda:9")
    assert registry.profile_execution_device("local/model") is None


def test_family_level_cuda_cannot_admit_hybrid(admission) -> None:
    config, _, _, _ = admission
    assert hybrid_admission.openai_equivalence_refusal(config, device="cuda") == "hybrid execution device is ambiguous"


async def test_queue_worker_bridge_still_requires_current_evidence(
    admission: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _, path, _ = admission
    monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(tmp_path / "ipc.sock"))
    validate_model_routing(config, device="cpu")
    path.unlink()
    validate_model_routing(config, device="cpu")
    registry = MagicMock(spec=ModelRegistry)
    registry.device = "cpu"
    registry.engine_config = None
    registry.profile_execution_device.return_value = "cpu"
    registry.get_config.side_effect = expand_profile_variants([config]).__getitem__
    registry.has_model.return_value = True
    registry.is_unloading.return_value = False
    registry.is_loading.side_effect = lambda name: name == config.sie_id
    registry.is_loaded.side_effect = lambda name: name != config.sie_id
    registry.start_load_async = AsyncMock(return_value=True)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(registry=registry)),
        }
    )
    with pytest.raises(HTTPException) as refused:
        await route_request(request, config.sie_id, trace.INVALID_SPAN)
    assert refused.value.status_code == 503
    assert refused.value.headers["X-SIE-Fallback-Error"] == "INFERENCE_ERROR"
    assert not hasattr(request.state, "serving_route")
    registry.start_load_async.assert_awaited_once_with(config.sie_id, "cpu")


def _remote_lane_server(config: ModelConfig, tmp_path: Path) -> IpcServer:
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(config)
    return IpcServer(str(tmp_path / "w.sock"), QueueExecutor(registry), worker_id="w")


async def _snapshot(server: IpcServer) -> dict[str, Any]:
    response = await server._handle_numerical_profile_snapshot(NumericalProfileSnapshotRequest())
    return {profile.model_id: profile for profile in response.profiles}


async def test_remote_lane_snapshot_advertises_the_current_admission(admission: tuple, tmp_path: Path) -> None:
    config, upstream, path, data = admission
    _refresh_record(path, data)
    server = _remote_lane_server(config, tmp_path)
    observed = await _snapshot(server)
    bare = observed[config.sie_id]
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    assert bare.remote_contract_sha256 == remote_profile_contract_digest(config, "remote", {"vendor": upstream})
    assert bare.remote_execution_sha256 == hybrid_admission.serving_code_digest()
    assert bare.admission is not None
    assert bare.admission.sha256 == current.sha256
    assert bare.admission.kind == "openai"
    assert bare.admission.local_identities == [IDENTITY]
    assert bare.admission.outputs == ["dense"]
    assert bare.admission.model_contract_sha256 == model_contract_digest(config)
    assert bare.admission.expires_at_unix_ms == int(current.expires_at.timestamp() * 1000)
    variant = observed[config.sie_id + ":remote"]
    assert (variant.admission, variant.remote_contract_sha256) == (None, None)
    data["measured_at"] = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps(data))
    expired = (await _snapshot(server))[config.sie_id]
    assert expired.admission is None
    assert expired.remote_contract_sha256 == bare.remote_contract_sha256
    path.unlink()
    assert (await _snapshot(server))[config.sie_id].admission is None


async def test_snapshot_omits_an_admission_naming_more_identities_than_the_bound(
    admission: tuple, tmp_path: Path
) -> None:
    config, _, path, data = admission
    now = datetime.now(UTC).isoformat()
    identities = [IDENTITY, *(f"v1:sha256:{index:064x}" for index in range(8))]
    path.write_text(
        _bundle(*(_identity_record(data, identity, config, measured_at=now) for identity in identities[:8]))
    )
    server = _remote_lane_server(config, tmp_path)
    assert len((await _snapshot(server))[config.sie_id].admission.local_identities) == 8
    path.write_text(_bundle(*(_identity_record(data, identity, config, measured_at=now) for identity in identities)))
    assert len(hybrid_admission.openai_admission(config).local_identities) == 9
    assert (await _snapshot(server))[config.sie_id].admission is None


async def test_process_without_the_upstream_reports_no_remote_contract_or_admission(
    admission: tuple, tmp_path: Path
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server = _remote_lane_server(config, tmp_path)
    install_upstreams({}, remote_serving=False)
    observation = (await _snapshot(server))[config.sie_id]
    assert (observation.remote_contract_sha256, observation.remote_execution_sha256, observation.admission) == (
        None,
        None,
        None,
    )
    assert set(msgspec.to_builtins(observation)) <= {"model_id", "local_identity", "model_contract_sha256"}


def _fence_server(config: ModelConfig, tmp_path: Path) -> tuple[IpcServer, str]:
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(config)
    executor = QueueExecutor(registry)
    server = IpcServer(str(tmp_path / "w.sock"), executor, worker_id="w", bundle_id="remote")
    return server, executor.bundle_config_view("remote").bundle_config_hash


def _bridged_batch(model_id: str, bundle_hash: str, *items: dict[str, Any]) -> RunBatchRequest:
    wrapped = []
    for index, item in enumerate(items):
        op = item.pop("op", "encode")
        payload = {
            "work_item_id": f"req.{index}",
            "request_id": "req",
            "item_index": index,
            "total_items": len(items),
            "timestamp": 1.0,
            "bundle_config_hash": bundle_hash,
            **(
                {"query_item": {"text": "query"}, "score_items": [{"text": "doc"}]}
                if op == "score"
                else {"item": {"text": "secret-input"}}
            ),
            **item,
        }
        wrapped.append(
            {"op": op, op: payload, "work_item_id": f"req.{index}", "request_id": "req", "item_index": index}
        )
    return msgspec.convert(
        {"model_id": model_id, "batch_id": 1, "lora_key": "", "total_cost": len(items), "items": wrapped},
        type=RunBatchRequest,
    )


def _served(req: Any) -> BatchOutcome:
    return BatchOutcome(
        outcomes=[
            ItemOutcome(
                work_item_id=item.work_item_id,
                request_id=item.request_id,
                item_index=item.item_index,
                disposition="publish_and_ack",
            )
            for item in req.items
        ]
    )


def _assert_refused_before_the_upstream(outcome: ItemOutcome) -> None:
    assert outcome.disposition == "nak_retry"
    assert outcome.error_code == "INFERENCE_ERROR"
    assert outcome.result_msgpack is None
    assert outcome.raw_output is None
    assert outcome.units is None


async def test_fence_serves_an_item_naming_the_current_admission(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    inference = AsyncMock(side_effect=_served)
    monkeypatch.setattr(server._executor, "process_encode_batch", inference)

    outcome = await server._handle_run_batch(
        _bridged_batch(
            config.sie_id + ":remote",
            bundle_hash,
            {"numerical_admission_sha256": current.sha256, "options": {"output_dtype": "float32"}},
        ),
        require_authority=True,
        require_admission=True,
    )

    inference.assert_awaited_once()
    assert [value.disposition for value in outcome.outcomes] == ["publish_and_ack"]


def _expire(path: Path, data: dict[str, Any]) -> None:
    data["measured_at"] = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("model_suffix", "item", "change"),
    [
        (":remote", {"numerical_admission_sha256": "0" * 64}, None),
        (":remote", {"numerical_admission_sha256": None}, None),
        (":remote", {"output_types": ["sparse"]}, None),
        (":remote", {"output_types": []}, None),
        (":remote", {"options": {"normalize": False}}, None),
        (":remote", {"options": {"output_dtype": "int8"}}, None),
        (":remote", {"op": "score"}, None),
        (":remote", {"op": "extract"}, None),
        ("", {}, None),
        (":other", {}, None),
        (":remote", {}, "expire"),
        (":remote", {}, "remove"),
    ],
)
async def test_fence_refuses_before_the_upstream_when_the_admission_does_not_cover_the_item(
    admission: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_suffix: str,
    item: dict[str, Any],
    change: str | None,
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    inference = AsyncMock(side_effect=_served)
    for op in ("encode", "score", "extract"):
        monkeypatch.setattr(server._executor, f"process_{op}_batch", inference)
    monkeypatch.setattr(server, "_execution_config", _always_current)
    if change == "expire":
        _expire(path, data)
    elif change == "remove":
        path.unlink()

    outcome = await server._handle_run_batch(
        _bridged_batch(
            config.sie_id + model_suffix, bundle_hash, {"numerical_admission_sha256": current.sha256, **item}
        ),
        require_authority=True,
        require_admission=True,
    )

    inference.assert_not_awaited()
    assert len(outcome.outcomes) == 1
    _assert_refused_before_the_upstream(outcome.outcomes[0])


@contextlib.asynccontextmanager
async def _always_current(_model_id: str, _hashes: Any) -> AsyncIterator[bool]:
    yield True


async def test_fence_serves_the_covered_items_of_a_mixed_batch_and_keeps_their_order(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    inference = AsyncMock(side_effect=_served)
    monkeypatch.setattr(server._executor, "process_encode_batch", inference)

    outcome = await server._handle_run_batch(
        _bridged_batch(
            config.sie_id + ":remote",
            bundle_hash,
            {"numerical_admission_sha256": "0" * 64},
            {"numerical_admission_sha256": current.sha256},
            {"numerical_admission_sha256": current.sha256, "options": {"normalize": False}},
        ),
        require_authority=True,
        require_admission=True,
    )

    inference.assert_awaited_once()
    assert [item.work_item_id for item in inference.await_args.args[0].items] == ["req.1"]
    assert [value.work_item_id for value in outcome.outcomes] == ["req.0", "req.1", "req.2"]
    assert outcome.outcomes[1].disposition == "publish_and_ack"
    _assert_refused_before_the_upstream(outcome.outcomes[0])
    _assert_refused_before_the_upstream(outcome.outcomes[2])


async def test_fence_requires_execution_authority_first(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, _ = _fence_server(config, tmp_path)
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    inference = AsyncMock(side_effect=_served)
    monkeypatch.setattr(server._executor, "process_encode_batch", inference)

    stale = await server._handle_run_batch(
        _bridged_batch(config.sie_id + ":remote", "stale", {"numerical_admission_sha256": current.sha256}),
        require_authority=True,
        require_admission=True,
    )

    inference.assert_not_awaited()
    assert [(value.disposition, value.error_code) for value in stale.outcomes] == [("nak_retry", "INFERENCE_ERROR")]


def test_capability_names_the_numerical_admission_method(admission: tuple, tmp_path: Path) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, _ = _fence_server(config, tmp_path)
    assert server._handle_worker_capabilities(WorkerCapabilitiesRequest()).supports_numerical_admission_v1 is True
    legacy = msgspec.convert({"supports_execution_authority_v1": True}, type=WorkerCapabilitiesResponse)
    assert legacy.supports_numerical_admission_v1 is False


def test_evidence_is_parsed_once_per_file_version_and_judged_fresh_on_every_call(
    admission: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, path, data = admission
    monkeypatch.setattr(hybrid_admission, "_evidence_cache", {})
    monkeypatch.setattr(hybrid_admission, "_SETTLED_EVIDENCE_NS", -1)
    read = hybrid_admission.read_equivalence_evidence
    reads: list[str] = []

    def counted(target: str) -> Any:
        reads.append(target)
        return read(target)

    monkeypatch.setattr(hybrid_admission, "read_equivalence_evidence", counted)

    first = hybrid_admission.openai_admission(config, now=NOW)
    assert isinstance(first, hybrid_admission.NumericalAdmission)
    assert hybrid_admission.openai_admission(config, now=NOW) == first
    expired = "hybrid equivalence record failed or is outside its configured age"
    assert hybrid_admission.openai_admission(config, now=NOW + timedelta(seconds=61)) == expired
    assert len(reads) == 1

    failed = {
        **data,
        "cases": [
            {**case, "measurements": {"dense": {"noise_floor": 0.0, "remote_error": 0.5, "values": 1024}}}
            for case in data["cases"]
        ],
    }
    replacement = path.with_name("replacement.json")
    replacement.write_text(json.dumps(failed))
    replacement.replace(path)
    assert hybrid_admission.openai_admission(config, now=NOW) == expired
    assert len(reads) == 2


def test_a_freshly_written_evidence_file_is_read_again(admission: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _, _ = admission
    monkeypatch.setattr(hybrid_admission, "_evidence_cache", {})
    read = hybrid_admission.read_equivalence_evidence
    reads: list[str] = []

    def counted(target: str) -> Any:
        reads.append(target)
        return read(target)

    monkeypatch.setattr(hybrid_admission, "read_equivalence_evidence", counted)
    for _ in range(2):
        assert isinstance(hybrid_admission.openai_admission(config, now=NOW), hybrid_admission.NumericalAdmission)
    assert len(reads) == 2


@pytest.mark.parametrize("require_authority", [False, True])
async def test_batch_methods_without_the_fence_refuse_items_naming_an_admission(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_authority: bool
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    inference = AsyncMock(side_effect=_served)
    monkeypatch.setattr(server._executor, "process_encode_batch", inference)
    monkeypatch.setattr(server, "_execution_config", _always_current)

    outcome = await server._handle_run_batch(
        _bridged_batch(config.sie_id + ":remote", bundle_hash, {"numerical_admission_sha256": current.sha256}, {}),
        require_authority=require_authority,
    )

    assert [item.work_item_id for item in inference.await_args.args[0].items] == ["req.1"]
    _assert_refused_before_the_upstream(outcome.outcomes[0])
    assert outcome.outcomes[0].error == "this method does not verify numerical admissions"
    assert outcome.outcomes[1].disposition == "publish_and_ack"


@pytest.mark.parametrize("op", ["encode", "score"])
async def test_op_scoped_methods_refuse_items_naming_an_admission(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    inference = AsyncMock(side_effect=_served)
    monkeypatch.setattr(server._executor, f"process_{op}_batch", inference)
    monkeypatch.setattr(server, "_execution_config", _always_current)
    batch = _bridged_batch(
        config.sie_id + ":remote", bundle_hash, {"op": op, "numerical_admission_sha256": "a" * 64}, {"op": op}
    )
    items = [getattr(item, op) for item in batch.items]

    if op == "encode":
        outcome = await server._handle_process_encode(ProcessEncodeBatchRequest(model_id=batch.model_id, items=items))
    else:
        outcome = await server._handle_process_score(ProcessScoreBatchRequest(model_id=batch.model_id, items=items))

    assert [item.work_item_id for item in inference.await_args.args[0].items] == ["req.1"]
    _assert_refused_before_the_upstream(outcome.outcomes[0])
    assert outcome.outcomes[1].disposition == "publish_and_ack"


async def test_a_failing_admission_check_refuses_every_item_and_logs_only_its_class(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    inference = AsyncMock(side_effect=_served)
    monkeypatch.setattr(server._executor, "process_encode_batch", inference)
    monkeypatch.setattr(server, "_execution_config", _always_current)

    def failing(_req: RunBatchRequest) -> list[str | None]:
        raise RuntimeError("secret-input")

    monkeypatch.setattr(server, "_numerical_admission_reasons", failing)
    with caplog.at_level(logging.INFO, logger="sie_server.ipc_server"):
        outcome = await server._handle_run_batch(
            _bridged_batch(
                config.sie_id + ":remote",
                bundle_hash,
                {"numerical_admission_sha256": "a" * 64},
                {"numerical_admission_sha256": "a" * 64},
            ),
            require_authority=True,
            require_admission=True,
        )

    inference.assert_not_awaited()
    for value in outcome.outcomes:
        _assert_refused_before_the_upstream(value)
        assert value.error == "numerical admission check failed"
    assert "RuntimeError" in caplog.text
    assert "secret-input" not in caplog.text


async def test_the_fence_keeps_shared_outputs_and_names_an_unanswered_item(
    admission: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, path, data = admission
    _refresh_record(path, data)
    server, bundle_hash = _fence_server(config, tmp_path)
    current = hybrid_admission.openai_admission(config)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    monkeypatch.setattr(server, "_execution_config", _always_current)
    shared = BatchedF16MultivectorOutput(
        values_f16=b"\x00\x00",
        items=[BatchedF16MultivectorItem(work_item_id="req.1", byte_offset=0, byte_len=2, num_tokens=1, token_dims=1)],
    )
    seen: list[RunBatchRequest] = []

    async def answers_one(_executor: Any, batch: RunBatchRequest) -> BatchOutcome:
        seen.append(batch)
        return BatchOutcome(
            outcomes=[ItemOutcome(work_item_id="req.1", request_id="req", item_index=1, disposition="publish_and_ack")],
            batched_f16_multivectors=[shared],
        )

    monkeypatch.setattr(ipc_server, "handle_run_batch", answers_one)

    outcome = await server._handle_run_batch(
        _bridged_batch(
            config.sie_id + ":remote",
            bundle_hash,
            {"numerical_admission_sha256": "0" * 64},
            {"numerical_admission_sha256": current.sha256},
            {"numerical_admission_sha256": current.sha256},
        ),
        require_authority=True,
        require_admission=True,
    )

    assert [item.work_item_id for item in seen[0].items] == ["req.1", "req.2"]
    assert outcome.batched_f16_multivectors == [shared]
    assert [value.work_item_id for value in outcome.outcomes] == ["req.0", "req.1", "req.2"]
    assert outcome.outcomes[1].disposition == "publish_and_ack"
    _assert_refused_before_the_upstream(outcome.outcomes[0])
    assert outcome.outcomes[0].error == "bridged item names a different numerical admission"
    _assert_refused_before_the_upstream(outcome.outcomes[2])
    assert outcome.outcomes[2].error == "the backend returned no outcome for this admitted item"
