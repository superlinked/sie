"""SIE hybrid authority is a fresh bounded SDK metadata observation."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yaml
from fastapi import HTTPException, Request
from opentelemetry import trace
from sie_server.api.routing import route_request
from sie_server.config import hybrid_admission, sie_identity
from sie_server.config.equivalence import model_contract_digest, remote_profile_contract_digest
from sie_server.config.model import ModelConfig
from sie_server.config.routing import validate_model_routing
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.loader import expand_profile_variants
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_server import IpcServer
from sie_server.ipc_types import NumericalProfileSnapshotRequest
from sie_server.queue_executor import QueueExecutor

IDENTITY = "v1:sha256:" + "b" * 64
REVISION = "a" * 40


def model(remote="operator/model:stable") -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sie_id": "local/model",
            "hf_id": "BAAI/bge-m3",
            "hf_revision": REVISION,
            "tasks": {"encode": {"dense": {"dim": 1024}}},
            "routing": {"policy": "fallback", "fallback_profile": "remote"},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.bge_m3:BGEM3Adapter",
                    "max_batch_tokens": 8192,
                    "compute_precision": "float32",
                },
                "remote": {
                    "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"upstream": "team", "upstream_model": remote}},
                },
            },
        }
    )


def metadata(**changes):
    return {
        "name": "operator/model",
        "revision": REVISION,
        "profiles": {"stable": {"identity": IDENTITY, "remote_contract_sha256": None}},
        **changes,
    }


@pytest.fixture
def remote(monkeypatch):
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "https://upstream.example.test/prefix",
            "api_key_secret": "TEAM_KEY",
            "proxy_url": "http://declared-proxy.example.test:3128",
            "rate_cap": {"requests_per_minute": 1000, "max_concurrency": 4},
        }
    )
    install_upstreams({"team": upstream})
    sie_identity._OBSERVATIONS.clear()
    monkeypatch.setenv("TEAM_KEY", "operator-key")
    monkeypatch.setenv("SIE_BASE_URL", upstream.base_url)
    monkeypatch.setenv("SIE_API_KEY", "ambient-key")
    monkeypatch.setenv("HTTP_PROXY", "http://ambient-proxy.example.test")
    monkeypatch.setattr(sie_identity, "local_profile_identity", lambda *args, **kwargs: IDENTITY)
    requests, constructions = [], []
    payload = [httpx.Response(200, json=metadata())]

    def handle(request):
        requests.append(request)
        return payload[0]

    def transport(**kwargs):
        constructions.append(kwargs)
        return httpx.MockTransport(handle)

    monkeypatch.setattr(sie_identity, "DeadlineTransport", transport)
    yield upstream, requests, constructions, payload
    sie_identity._OBSERVATIONS.clear()
    install_upstreams({})


def test_matching_profile_reads_metadata_via_sdk_with_declared_egress(remote) -> None:
    upstream, requests, constructions, _payload = remote
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is None
    assert len(requests) == 1
    assert requests[0].url.path == "/prefix/v1/models/operator/model"
    assert requests[0].headers["Authorization"] == "Bearer operator-key"
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert requests[0].headers["Accept"] == "application/json"
    assert constructions == [{"proxy": upstream.proxy_url}]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        metadata(name="different/model"),
        metadata(revision="c" * 40),
        metadata(revision=None),
        metadata(profiles={}),
        metadata(profiles={"stable": {"identity": None}}),
        metadata(profiles={"stable": {"identity": "v1:sha256:" + "c" * 64}}),
        metadata(profiles={"stable": {"identity": IDENTITY, "remote_contract_sha256": "d" * 64}}),
    ],
)
def test_missing_or_changed_identity_refuses(remote, payload) -> None:
    remote[3][0] = httpx.Response(200, json=payload)
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is not None


def test_mismatch_refusal_names_both_identities(remote) -> None:
    upstream_identity = "v1:sha256:" + "d" * 64
    remote[3][0] = httpx.Response(
        200, json=metadata(revision="c" * 40, profiles={"stable": {"identity": upstream_identity}})
    )
    assert sie_identity.sie_identity_refusal(model(), device="cpu") == (
        "hybrid upstream weights or execution profile differs from local: "
        f"upstream (hf_revision={'c' * 40}, identity={upstream_identity}), "
        f"local (hf_revision={REVISION}, identity={IDENTITY})"
    )


def test_upstream_revision_with_a_trailing_newline_is_not_an_identity(remote) -> None:
    remote[3][0] = httpx.Response(200, json=metadata(revision=REVISION + "\n"))
    assert sie_identity.sie_identity_refusal(model(), device="cpu") == (
        "hybrid upstream identity is unavailable or outside its age"
    )


def test_mismatch_refusal_prints_only_revisions_and_digests(remote, monkeypatch) -> None:
    hostile = "\x1b[2J" + "x" * 100_000 + "\r\n"
    monkeypatch.setattr(sie_identity, "_fresh_identity", lambda *args: (hostile, hostile))
    assert sie_identity.sie_identity_refusal(model(), device="cpu") == (
        "hybrid upstream weights or execution profile differs from local: "
        f"upstream (hf_revision=<invalid>, identity=<invalid>), local (hf_revision={REVISION}, identity={IDENTITY})"
    )


@pytest.mark.parametrize("status", [301, 302, 307, 400, 404, 429, 500, 503])
def test_failed_metadata_never_discloses_upstream_detail_or_follows_redirect(remote, status) -> None:
    remote[3][0] = httpx.Response(
        status, headers={"location": "https://attacker.example.test/private"}, content=b"PRIVATE_DETAIL"
    )
    refusal = sie_identity.sie_identity_refusal(model(), device="cpu")
    assert refusal is not None
    assert "PRIVATE" not in refusal
    assert "operator-key" not in refusal
    assert len(remote[1]) == 1


def test_identity_expires_and_divergence_closes_admission(remote, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(sie_identity.time, "monotonic", lambda: clock[0])
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is None
    remote[3][0] = httpx.Response(200, json=metadata(revision="c" * 40))
    clock[0] += 29
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is None
    assert len(remote[1]) == 1
    clock[0] += 2
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is not None
    assert len(remote[1]) == 2


def test_reinstalling_upstream_does_not_reuse_old_observation(remote) -> None:
    upstream, requests, _constructions, payload = remote
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is None
    install_upstreams({"team": upstream.model_copy()})
    payload[0] = httpx.Response(200, json=metadata(revision="c" * 40))
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is not None
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("field", "value"), [("set_params", {"options": {"normalize": False}}), ("strip_params", frozenset({"options"}))]
)
def test_operator_transforms_cannot_bypass_execution_identity(remote, field, value) -> None:
    install_upstreams({"team": remote[0].model_copy(update={field: value})})
    assert "transforms" in sie_identity.sie_identity_refusal(model(), device="cpu")
    assert not remote[1]


@pytest.mark.parametrize("device", ["", "cuda", "cuda:unknown"])
def test_ambiguous_device_never_reads_upstream(remote, device) -> None:
    assert "ambiguous" in sie_identity.sie_identity_refusal(model(), device=device)
    assert not remote[1]


def test_unknown_local_identity_never_reads_upstream(remote, monkeypatch) -> None:
    monkeypatch.setattr(sie_identity, "local_profile_identity", lambda *args, **kwargs: None)
    assert "local execution" in sie_identity.sie_identity_refusal(model(), device="cpu")
    assert not remote[1]


def test_unqualified_upstream_cannot_apply_another_bare_model_policy(remote) -> None:
    assert "explicit local profile" in sie_identity.sie_identity_refusal(model("operator/model"), device="cpu")
    assert not remote[1]


def test_changed_remote_runtime_cannot_override_matching_metadata(remote) -> None:
    config = model()
    config.profiles["remote"].adapter_options.runtime["normalize"] = False
    assert "runtime defaults" in sie_identity.sie_identity_refusal(config, device="cpu")
    assert not remote[1]


def test_non_float32_local_wire_contract_is_closed(remote) -> None:
    config = model()
    config.profiles["default"].adapter_options.runtime["output_dtype"] = "int8"
    assert "float32" in sie_identity.sie_identity_refusal(config, device="cpu")
    assert not remote[1]


class Body(httpx.SyncByteStream):
    def __init__(self, parts):
        self.parts, self.closed = parts, False

    def __iter__(self):
        yield from self.parts

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("headers", "parts"),
    [
        ({}, [b"x" * (sie_identity._MAX_METADATA_BYTES + 1)]),
        ({"content-encoding": "gzip"}, [b"PRIVATE"]),
        ({}, [b"not json"]),
    ],
)
def test_bounded_metadata_closes_invalid_stream(remote, headers, parts) -> None:
    stream = Body(parts)
    remote[3][0] = httpx.Response(200, headers=headers, stream=stream)
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is not None
    assert stream.closed


def test_deadline_refuses_trickling_metadata_and_closes(remote, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(sie_identity.time, "monotonic", lambda: clock[0])

    def trickle():
        clock[0] += sie_identity._METADATA_DEADLINE_S + 1
        yield json.dumps(metadata()).encode()

    stream = Body(trickle())
    remote[3][0] = httpx.Response(200, stream=stream)
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is not None
    assert stream.closed


def test_concurrent_refresh_refuses_without_waiting_or_duplicate_read(remote, monkeypatch) -> None:
    entered, release = threading.Event(), threading.Event()

    def read(*args):
        entered.set()
        assert release.wait(2)
        return REVISION, IDENTITY

    monkeypatch.setattr(sie_identity, "_read_identity", read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sie_identity._fresh_identity, "team", remote[0], "operator/model:stable")
        assert entered.wait(2)
        try:
            assert sie_identity._fresh_identity("team", remote[0], "operator/model:stable") is None
        finally:
            release.set()
        assert future.result() == (REVISION, IDENTITY)


def test_remote_serving_disabled_never_reads_metadata_or_uses_cached_identity(remote) -> None:
    assert sie_identity.sie_identity_refusal(model(), device="cpu") is None
    install_upstreams({"team": remote[0]}, remote_serving=False)
    assert "disabled" in sie_identity.sie_identity_refusal(model(), device="cpu")
    assert len(remote[1]) == 1


def test_muvera_defaults_cannot_be_applied_twice(remote) -> None:
    config = model()
    config.profiles["default"].adapter_options.runtime.update({"muvera": {"dim": 1024}, "output_types": ["dense"]})
    assert "postprocessing" in sie_identity.sie_identity_refusal(config, device="cpu")
    assert not remote[1]


def test_configuration_admission_requires_fresh_matching_runtime(remote, tmp_path) -> None:
    config = model()
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(config)
    assert registry.has_model(config.sie_id + ":remote")
    with pytest.raises(ValueError, match="refused until"):
        validate_model_routing(config)
    sie_identity._OBSERVATIONS.clear()
    remote[3][0] = httpx.Response(200, json=metadata(revision="c" * 40))
    with pytest.raises(ValueError, match="weights or execution profile"):
        registry.add_config(config)


def test_directory_load_uses_the_same_identity_admission(remote, tmp_path) -> None:
    config = model()
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    registry = ModelRegistry(models_dir=models, device="cpu", enable_hot_reload=False)
    assert registry.has_model(config.sie_id)
    sie_identity._OBSERVATIONS.clear()
    remote[3][0] = httpx.Response(200, json=metadata(profiles={}))
    with pytest.raises(ValueError, match="identity is unavailable"):
        ModelRegistry(models_dir=models, device="cpu", enable_hot_reload=False)


@pytest.mark.parametrize("changed", [False, True])
async def test_bridge_rechecks_expired_identity_and_preserves_warmup_and_refusal(remote, monkeypatch, changed) -> None:
    config = model()
    clock = [100.0]
    monkeypatch.setattr(sie_identity.time, "monotonic", lambda: clock[0])
    validate_model_routing(config, device="cpu")
    if changed:
        clock[0] += 31
        remote[3][0] = httpx.Response(200, json=metadata(revision="c" * 40))
    registry = MagicMock(spec=ModelRegistry)
    registry.device, registry.engine_config = "cpu", None
    registry.profile_execution_device.return_value = "cpu"
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
    if changed:
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
        assert route.upstream == "team"
    registry.start_load_async.assert_awaited_once_with(config.sie_id, "cpu")
    registry.load_now.assert_not_awaited()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("runtime", "normalize", False), ("loadtime", "upstream_model", "different/model:stable")],
)
def test_reused_mutated_config_cannot_admit_stale_profile_metadata(remote, section, key, value) -> None:
    config = model()
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(config)
    getattr(config.profiles["remote"].adapter_options, section)[key] = value
    with pytest.raises(ValueError, match="settings changed after resolution"):
        registry.add_config(config)
    with pytest.raises(ValueError, match="settings changed after resolution"):
        validate_model_routing(config, device="cpu")
    assert len(remote[1]) == 1


def test_sie_admission_admits_the_upstream_identity_for_matching_weights(remote) -> None:
    admitted = hybrid_admission.sie_admission(model())
    assert isinstance(admitted, hybrid_admission.NumericalAdmission)
    assert admitted.kind == "sie"
    assert admitted.local_identities == {IDENTITY}
    assert admitted.outputs == {"dense"}
    assert admitted.model_contract_sha256 == model_contract_digest(model())
    dispatched = hybrid_admission.remote_admission(model())
    assert isinstance(dispatched, hybrid_admission.NumericalAdmission)
    assert dispatched.sha256 == admitted.sha256
    _upstream, _requests, _constructions, payload = remote
    payload[0] = httpx.Response(200, json=metadata(revision="c" * 40))
    sie_identity._OBSERVATIONS.clear()
    assert hybrid_admission.sie_admission(model()) == "hybrid upstream weights differ from local"


def test_admission_without_waiting_never_reads_the_upstream_on_the_callers_thread(remote, monkeypatch) -> None:
    _upstream, requests, _constructions, _payload = remote
    entered, release = threading.Event(), threading.Event()
    read = sie_identity._read_identity

    def slow(*args):
        entered.set()
        assert release.wait(5)
        return read(*args)

    monkeypatch.setattr(sie_identity, "_read_identity", slow)
    assert (
        hybrid_admission.sie_admission(model(), wait=False)
        == "hybrid upstream identity is unavailable or outside its age"
    )
    assert entered.wait(5)
    assert (
        hybrid_admission.sie_admission(model(), wait=False)
        == "hybrid upstream identity is unavailable or outside its age"
    )
    release.set()
    deadline = time.monotonic() + 5
    while isinstance(hybrid_admission.sie_admission(model(), wait=False), str):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert len(requests) == 1


def test_admission_refreshes_ahead_of_expiry_without_losing_the_current_identity(remote, monkeypatch) -> None:
    _upstream, requests, _constructions, _payload = remote
    clock = [1000.0]
    monkeypatch.setattr(sie_identity.time, "monotonic", lambda: clock[0])
    first = hybrid_admission.sie_admission(model())
    assert isinstance(first, hybrid_admission.NumericalAdmission)
    clock[0] += 25.0
    refreshed = threading.Event()
    original = sie_identity._refresh

    def tracked(*args):
        original(*args)
        refreshed.set()

    monkeypatch.setattr(sie_identity, "_refresh", tracked)
    current = hybrid_admission.sie_admission(model(), wait=False)
    assert isinstance(current, hybrid_admission.NumericalAdmission)
    assert refreshed.wait(5)
    assert len(requests) == 2


def _tracked_refresh(monkeypatch) -> threading.Event:
    refreshed = threading.Event()
    original = sie_identity._refresh

    def tracked(*args):
        original(*args)
        refreshed.set()

    monkeypatch.setattr(sie_identity, "_refresh", tracked)
    return refreshed


def test_a_failed_refresh_keeps_the_current_identity_until_it_expires(remote, monkeypatch) -> None:
    _upstream, requests, _constructions, payload = remote
    clock = [1000.0]
    monkeypatch.setattr(sie_identity.time, "monotonic", lambda: clock[0])
    first = hybrid_admission.sie_admission(model())
    assert isinstance(first, hybrid_admission.NumericalAdmission)
    payload[0] = httpx.Response(503)
    refreshed = _tracked_refresh(monkeypatch)
    clock[0] += 25.0

    assert isinstance(hybrid_admission.sie_admission(model(), wait=False), hybrid_admission.NumericalAdmission)
    assert refreshed.wait(5)
    kept = sie_identity.sie_upstream_identity(model(), wait=False)
    assert kept == (REVISION, IDENTITY, 5.0)
    assert len(requests) == 2

    clock[0] += 6.0
    refreshed.clear()
    assert (
        hybrid_admission.sie_admission(model(), wait=False)
        == "hybrid upstream identity is unavailable or outside its age"
    )
    assert refreshed.wait(5)
    assert len(requests) == 3
    clock[0] += 3.0
    assert sie_identity.sie_upstream_identity(model()) == "hybrid upstream identity is unavailable or outside its age"
    assert len(requests) == 4


def test_a_completed_refusal_replaces_the_identity_at_once(remote, monkeypatch) -> None:
    _upstream, requests, _constructions, payload = remote
    clock = [1000.0]
    monkeypatch.setattr(sie_identity.time, "monotonic", lambda: clock[0])
    assert isinstance(hybrid_admission.sie_admission(model()), hybrid_admission.NumericalAdmission)
    payload[0] = httpx.Response(404, json={"detail": {"code": "MODEL_NOT_FOUND", "message": "gone"}})
    refreshed = _tracked_refresh(monkeypatch)
    clock[0] += 25.0

    hybrid_admission.sie_admission(model(), wait=False)
    assert refreshed.wait(5)

    assert (
        hybrid_admission.sie_admission(model(), wait=False)
        == "hybrid upstream identity is unavailable or outside its age"
    )
    assert len(requests) == 2


async def test_remote_lane_snapshot_reports_the_sie_admission(remote, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(tmp_path / "ipc.sock"))
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(model())
    assert isinstance(hybrid_admission.sie_admission(model()), hybrid_admission.NumericalAdmission)
    server = IpcServer(str(tmp_path / "w.sock"), QueueExecutor(registry), worker_id="w")
    response = await server._handle_numerical_profile_snapshot(NumericalProfileSnapshotRequest())
    observed = {profile.model_id: profile for profile in response.profiles}
    bare = observed["local/model"]
    assert bare.admission is not None
    assert bare.admission.kind == "sie"
    assert bare.admission.local_identities == [IDENTITY]
    assert bare.admission.sha256 == hybrid_admission.sie_admission(model()).sha256
    assert bare.remote_contract_sha256 == remote_profile_contract_digest(model(), "remote", {"team": remote[0]})
    assert observed["local/model:remote"].admission is None


async def test_snapshot_reports_no_admission_for_a_model_without_numerical_outputs(remote, tmp_path) -> None:
    data = model().model_dump(mode="json")
    data["tasks"] = {"encode": {}}
    empty = ModelConfig.model_validate(data)
    assert not set(empty.outputs) & {"dense", "sparse", "multivector", "score"}
    assert hybrid_admission.sie_admission(empty) == "hybrid model declares no numerical outputs"
    registry = ModelRegistry(device="cpu", enable_hot_reload=False)
    registry.add_config(empty)
    server = IpcServer(str(tmp_path / "w.sock"), QueueExecutor(registry), worker_id="w")
    response = await server._handle_numerical_profile_snapshot(NumericalProfileSnapshotRequest())
    bare = next(profile for profile in response.profiles if profile.model_id == "local/model")
    assert bare.admission is None
    assert bare.remote_contract_sha256 is not None
