"""``SIE_BASE_URL`` and ``SIE_API_KEY`` fallbacks for both client constructors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_BASE_URL", raising=False)
    monkeypatch.delenv("SIE_API_KEY", raising=False)


@pytest.fixture
def sync_headers() -> Iterator[dict[str, Any]]:
    captured: dict[str, Any] = {}
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.side_effect = lambda **kwargs: captured.update(kwargs) or client_cls.return_value
        yield captured


async def _async_session_kwargs(client: SIEAsyncClient) -> dict[str, Any]:
    with (
        patch("sie_sdk.client.async_.aiohttp.ClientSession") as session_cls,
        patch("sie_sdk.client.async_.aiohttp.TCPConnector"),
    ):
        session_cls.return_value.close = AsyncMock()
        client._ensure_session()
        kwargs = session_cls.call_args.kwargs
        await client.close()
    return kwargs


def test_sync_client_reads_base_url_and_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any]
) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com/")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    client = SIEClient()

    assert client.base_url == "https://gateway.example.com"
    assert sync_headers["base_url"] == "https://gateway.example.com"
    assert sync_headers["headers"]["Authorization"] == "Bearer env-key"


def test_sync_client_explicit_arguments_win_over_env(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any]
) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    client = SIEClient("http://localhost:8080", api_key="explicit-key")

    assert client.base_url == "http://localhost:8080"
    assert sync_headers["headers"]["Authorization"] == "Bearer explicit-key"


@pytest.mark.parametrize(
    "base_url",
    ["https://gateway.example.com", "https://GATEWAY.example.com:443/v1", "https://gateway.example.com/"],
)
def test_sync_client_sends_env_key_to_the_env_base_url_origin(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any], base_url: str
) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    SIEClient(base_url)

    assert sync_headers["headers"]["Authorization"] == "Bearer env-key"


@pytest.mark.parametrize(
    ("env_base_url", "base_url"),
    [
        (None, "https://gateway.example.com"),
        ("https://gateway.example.com", "https://other.example.com"),
        ("https://gateway.example.com", "http://gateway.example.com"),
        ("https://gateway.example.com", "https://gateway.example.com:8443"),
        ("not a url", "https://gateway.example.com"),
    ],
)
def test_sync_client_withholds_env_key_from_other_origins(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any], env_base_url: str | None, base_url: str
) -> None:
    if env_base_url is not None:
        monkeypatch.setenv("SIE_BASE_URL", env_base_url)
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    SIEClient(base_url)

    assert "Authorization" not in sync_headers["headers"]


def test_clients_require_an_explicit_key_for_a_cross_origin_control_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    with pytest.raises(ValueError, match="control_plane_url"):
        SIEClient(control_plane_url="https://control.example.com")
    with pytest.raises(ValueError, match="control_plane_url"):
        SIEAsyncClient(control_plane_url="https://control.example.com")


def test_sync_client_keeps_env_key_for_a_same_origin_control_plane(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any]
) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    SIEClient(control_plane_url="https://gateway.example.com/control")

    assert sync_headers["headers"]["Authorization"] == "Bearer env-key"


def test_sync_client_explicit_key_may_go_to_a_cross_origin_control_plane(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any]
) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    SIEClient(api_key="explicit-key", control_plane_url="https://control.example.com")

    assert sync_headers["headers"]["Authorization"] == "Bearer explicit-key"


def test_sync_client_empty_api_key_opts_out_of_env_key(
    monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any]
) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    SIEClient("http://localhost:8080", api_key="")

    assert "Authorization" not in sync_headers["headers"]


def test_sync_client_ignores_blank_env_key(monkeypatch: pytest.MonkeyPatch, sync_headers: dict[str, Any]) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("SIE_API_KEY", "  ")

    SIEClient("http://localhost:8080")

    assert "Authorization" not in sync_headers["headers"]


@pytest.mark.parametrize("env_value", [None, "", "   "])
def test_clients_require_a_base_url(monkeypatch: pytest.MonkeyPatch, env_value: str | None) -> None:
    if env_value is not None:
        monkeypatch.setenv("SIE_BASE_URL", env_value)
    with pytest.raises(ValueError, match="SIE_BASE_URL"):
        SIEClient()
    with pytest.raises(ValueError, match="SIE_BASE_URL"):
        SIEAsyncClient()


def test_clients_validate_the_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "localhost:8080")
    with pytest.raises(ValueError, match="http:// or https://"):
        SIEClient()
    with pytest.raises(ValueError, match="http:// or https://"):
        SIEAsyncClient()


@pytest.mark.asyncio
async def test_async_client_reads_base_url_and_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com/")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    client = SIEAsyncClient()
    kwargs = await _async_session_kwargs(client)

    assert client.base_url == "https://gateway.example.com"
    assert kwargs["base_url"] == "https://gateway.example.com"
    assert kwargs["headers"]["Authorization"] == "Bearer env-key"


@pytest.mark.asyncio
async def test_async_client_explicit_arguments_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    client = SIEAsyncClient("http://localhost:8080", api_key="explicit-key")
    kwargs = await _async_session_kwargs(client)

    assert client.base_url == "http://localhost:8080"
    assert kwargs["headers"]["Authorization"] == "Bearer explicit-key"


@pytest.mark.asyncio
async def test_async_client_scopes_env_key_to_the_env_base_url_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    same_origin = await _async_session_kwargs(SIEAsyncClient("https://gateway.example.com/v1"))
    other_origin = await _async_session_kwargs(SIEAsyncClient("https://other.example.com"))

    assert same_origin["headers"]["Authorization"] == "Bearer env-key"
    assert "Authorization" not in other_origin["headers"]


@pytest.mark.asyncio
async def test_async_client_empty_api_key_opts_out_of_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("SIE_API_KEY", "env-key")

    kwargs = await _async_session_kwargs(SIEAsyncClient("http://localhost:8080", api_key=""))

    assert "Authorization" not in kwargs["headers"]
