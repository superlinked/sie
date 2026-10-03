"""Configured SDK transports retain egress ownership and origin confinement."""

from collections.abc import Generator

import httpx
import pytest
from sie_sdk import SIEClient

BASE = "https://upstream.example.test/prefix"


def test_metadata_preserves_dynamic_auth_hooks_timeout_and_cleanup(monkeypatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", BASE)
    monkeypatch.setenv("SIE_API_KEY", "ambient-key")
    observed = []
    hooks = []

    class DynamicAuth(httpx.Auth):
        def auth_flow(self, request) -> Generator[httpx.Request, httpx.Response, None]:
            request.headers["Authorization"] = "Bearer dynamic-key"
            yield request

    def handle(request):
        observed.append(request)
        return httpx.Response(200, json={"name": "model", "profiles": []})

    transport = httpx.Client(
        base_url=BASE,
        auth=DynamicAuth(),
        transport=httpx.MockTransport(handle),
        trust_env=False,
        timeout=3,
        follow_redirects=False,
        event_hooks={"request": [lambda request: hooks.append(request.url)]},
    )
    with SIEClient(BASE, api_key="", remote="forbid", http_client=transport) as client:
        assert client.get_model("model")["name"] == "model"
        assert client.get_model("model")["name"] == "model"
        assert len(hooks) == 2
        assert observed[0].url.path == "/prefix/v1/models/model"
        assert observed[0].headers["Authorization"] == "Bearer dynamic-key"
        assert observed[0].headers["X-SIE-Remote"] == "forbid"
        assert observed[0].headers["Accept"] == "application/json"
        assert observed[0].extensions["timeout"]["read"] == 3
        assert not transport.is_closed
    assert transport.is_closed


@pytest.mark.parametrize(
    "base",
    ["https://other.example.test/prefix", "https://upstream.example.test/other", "http://upstream.example.test/prefix"],
)
def test_mismatched_base_rejected_without_adopting_client(base) -> None:
    with httpx.Client(base_url=base) as transport:
        with pytest.raises(ValueError, match="base URL"):
            SIEClient(BASE, http_client=transport)
        assert not transport.is_closed


@pytest.mark.parametrize("kwargs", [{"max_connections": 4}, {"control_plane_url": "https://control.example.test"}])
def test_conflicting_configuration_is_rejected(kwargs) -> None:
    with httpx.Client(base_url=BASE) as transport:
        with pytest.raises(ValueError, match=r"connection limits|control plane"):
            SIEClient(BASE, http_client=transport, **kwargs)
        assert not transport.is_closed


def test_redirect_following_and_closed_client_rejected() -> None:
    with httpx.Client(base_url=BASE, follow_redirects=True) as transport:
        with pytest.raises(ValueError, match="redirects disabled"):
            SIEClient(BASE, http_client=transport)
    with pytest.raises(ValueError, match="must be open"):
        SIEClient(BASE, http_client=transport)


def test_origin_guard_prevents_cross_origin_transport_dispatch() -> None:
    calls = []
    transport = httpx.Client(base_url=BASE, transport=httpx.MockTransport(calls.append))
    with SIEClient(BASE, http_client=transport):
        with pytest.raises(httpx.RequestError, match="different origin"):
            transport.get("https://attacker.example.test/model")
        assert not calls


def test_existing_hook_cannot_change_origin_before_dispatch() -> None:
    calls = []

    def rewrite(request):
        request.url = httpx.URL("https://attacker.example.test/model")

    transport = httpx.Client(
        base_url=BASE,
        event_hooks={"request": [rewrite]},
        transport=httpx.MockTransport(calls.append),
    )
    with SIEClient(BASE, http_client=transport) as client:
        with pytest.raises(httpx.RequestError, match="different origin"):
            client.get_model("model")
        assert not calls
