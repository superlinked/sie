"""SIE hybrid authority is a fresh bounded SDK metadata observation."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from sie_server.config import sie_identity
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams

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
