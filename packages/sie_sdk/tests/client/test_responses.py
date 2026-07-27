"""Tests for the typed, non-streaming Responses SDK surface."""

from __future__ import annotations

import json
from typing import Any, Self, get_args, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.errors import ProvisioningError, ServerError, SIEConnectionError
from sie_sdk.types import ResponseInputContentPart, ResponseInputMessage


def _payload() -> dict[str, Any]:
    return {
        "id": "resp-req-1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-req-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    }


def _sync_response(
    status: int,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.content = json.dumps(body).encode()
    response.json.return_value = body
    return response


def test_responses_input_types_match_public_openapi() -> None:
    part_types = get_args(get_type_hints(ResponseInputContentPart)["type"])
    roles = get_args(get_type_hints(ResponseInputMessage)["role"])

    assert set(part_types) == {"text", "input_text"}
    assert set(roles) == {"system", "user", "assistant", "developer"}


def test_responses_sends_complete_typed_body_and_attaches_request_metadata() -> None:
    response = _sync_response(
        200,
        _payload(),
        headers={
            "x-sie-request-id": "req-1",
            "x-sie-units-input-tokens": "2",
            "x-sie-units-output-tokens": "1",
            "x-sie-credits-debited": "9",
        },
    )
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = response
        client = SIEClient("https://gateway.example.test", base_url_headers={"Modal-Key": "wk"})

        result = client.responses(
            "m",
            [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            max_output_tokens=32,
            temperature=0.2,
            top_p=0.9,
            seed=-7,
            gpu="l4",
        )

        assert result["output"][0]["content"][0]["text"] == "Hello"
        assert result["request"] == {
            "id": "req-1",
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "credits_debited": 9,
        }
        call = client_cls.return_value.post.call_args
        assert call.args[0] == "/v1/responses"
        assert json.loads(call.kwargs["content"]) == {
            "model": "m",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "max_output_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": -7,
        }
        assert call.kwargs["headers"] == {
            "content-type": "application/json",
            "accept": "application/json",
            "X-SIE-MACHINE-PROFILE": "l4",
        }
        assert len(client_cls.call_args.kwargs["event_hooks"]["request"]) == 1
        client.close()


def test_responses_retries_pre_execution_provisioning_only() -> None:
    provisioning = _sync_response(
        503,
        {"error": {"code": "PROVISIONING", "message": "warming"}},
        headers={"Retry-After": "0.01"},
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep"),
    ):
        client_cls.return_value.post.side_effect = [provisioning, _sync_response(200, _payload())]
        client = SIEClient("http://localhost:8080")

        client.responses("m", "Hi", provision_timeout_s=5)

        assert client_cls.return_value.post.call_count == 2
        assert client.last_retry_count == 1
        client.close()


def test_responses_surfaces_provisioning_without_retry_when_waiting_is_disabled() -> None:
    provisioning = _sync_response(
        503,
        {"error": {"code": "PROVISIONING", "message": "warming"}},
        headers={"Retry-After": "0.01"},
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep") as sleep,
    ):
        client_cls.return_value.post.return_value = provisioning
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ProvisioningError):
            client.responses("m", "Hi", wait_for_capacity=False)

        assert client_cls.return_value.post.call_count == 1
        assert client.last_retry_count == 0
        sleep.assert_not_called()
        client.close()


def test_responses_does_not_retry_post_publish_timeout() -> None:
    timeout = _sync_response(
        504,
        {"error": {"code": "GATEWAY_TIMEOUT", "message": "timed out"}},
        headers={
            "x-sie-request-id": "req-timeout",
            "x-sie-units-output-tokens": "4",
            "x-sie-credits-debited": "11",
        },
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep") as sleep,
    ):
        client_cls.return_value.post.return_value = timeout
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ServerError) as exc_info:
            client.responses("m", "Hi")

        assert client_cls.return_value.post.call_count == 1
        sleep.assert_not_called()
        assert exc_info.value.request == {
            "id": "req-timeout",
            "usage": {"output_tokens": 4},
            "credits_debited": 11,
        }
        client.close()


def test_responses_does_not_retry_midflight_transport_failure() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.side_effect = httpx.ReadError("connection reset")
        client = SIEClient("http://localhost:8080")

        with pytest.raises(SIEConnectionError, match="Connection lost mid-request"):
            client.responses("m", "Hi")

        assert client_cls.return_value.post.call_count == 1
        client.close()


class _AsyncRaw:
    def __init__(
        self,
        status: int,
        body: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def read(self) -> bytes:
        return json.dumps(self._body).encode()


def _patch_async_session(
    client: SIEAsyncClient,
    *,
    post_returns: _AsyncRaw | None = None,
    post_side_effect: list[_AsyncRaw] | Exception | None = None,
) -> MagicMock:
    session = MagicMock()
    session.post = (
        MagicMock(side_effect=post_side_effect)
        if post_side_effect is not None
        else MagicMock(return_value=post_returns)
    )
    session.close = AsyncMock()
    client._session = session
    return session


@pytest.mark.asyncio
async def test_async_responses_matches_typed_body_metadata_and_origin_headers() -> None:
    raw = _AsyncRaw(
        200,
        _payload(),
        headers={
            "x-sie-request-id": "req-async",
            "x-sie-units-output-tokens": "1",
            "x-sie-credits-debited": "6",
        },
    )
    client = SIEAsyncClient("https://gateway.example.test", base_url_headers={"Modal-Key": "wk"})
    session = _patch_async_session(client, post_returns=raw)

    result = await client.responses("m", "Hi", max_output_tokens=8, temperature=0, seed=3)

    assert result["request"] == {
        "id": "req-async",
        "usage": {"output_tokens": 1},
        "credits_debited": 6,
    }
    call = session.post.call_args
    assert call.args[0] == "/v1/responses"
    assert json.loads(call.kwargs["data"]) == {
        "model": "m",
        "input": "Hi",
        "max_output_tokens": 8,
        "temperature": 0,
        "seed": 3,
    }
    assert call.kwargs["headers"]["accept"] == "application/json"
    assert call.kwargs["headers"]["Modal-Key"] == "wk"
    assert call.kwargs["allow_redirects"] is False
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_retries_pre_execution_provisioning_only() -> None:
    provisioning = _AsyncRaw(
        503,
        {"error": {"code": "PROVISIONING", "message": "warming"}},
        headers={"Retry-After": "0.01"},
    )
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(
        client,
        post_side_effect=[provisioning, _AsyncRaw(200, _payload())],
    )

    with patch("sie_sdk.client.async_.asyncio.sleep", new=AsyncMock()):
        await client.responses("m", "Hi", provision_timeout_s=5)

    assert session.post.call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_does_not_retry_midflight_transport_failure() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(
        client,
        post_side_effect=aiohttp.ClientPayloadError("truncated response"),
    )

    with pytest.raises(SIEConnectionError, match="Request failed"):
        await client.responses("m", "Hi")

    assert session.post.call_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_does_not_retry_post_publish_timeout() -> None:
    timeout = _AsyncRaw(
        504,
        {"error": {"code": "GATEWAY_TIMEOUT", "message": "timed out"}},
        headers={
            "x-sie-request-id": "req-async-timeout",
            "x-sie-units-output-tokens": "4",
            "x-sie-credits-debited": "12",
        },
    )
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(client, post_returns=timeout)

    with pytest.raises(ServerError) as exc_info:
        await client.responses("m", "Hi")

    assert session.post.call_count == 1
    assert exc_info.value.request == {
        "id": "req-async-timeout",
        "usage": {"output_tokens": 4},
        "credits_debited": 12,
    }
    await client.close()
