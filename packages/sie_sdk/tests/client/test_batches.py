"""client.batches namespace — request-shape assertions against a mocked transport.

Mirrors the OpenAI ``client.batches`` surface (create/retrieve/list/cancel) so
an ``openai`` → ``sie_sdk`` swap is mechanical.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient

GW = "http://gw.test:8080"
KEY = "sk-sie-testkey"

_BATCH = {
    "id": "batch-1",
    "object": "batch",
    "endpoint": "/v1/embeddings",
    "input_file_id": "file-in",
    "completion_window": "24h",
    "status": "completed",
    "output_file_id": "file-out",
    "error_file_id": None,
    "created_at": 1000,
    "finished_at": 1001,
    "request_counts": {"total": 2, "completed": 2, "failed": 0},
}


def _resp(status: int, body: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/json"}
    r.content = json.dumps(body).encode()
    r.json.return_value = body
    return r


class _FakeAio:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers = {"content-type": "application/json"}
        self._body = body

    def json(self) -> Any:
        return self._body


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_create_posts_v1_batches_with_openai_body() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, _BATCH))
        client = SIEClient(GW, api_key=KEY)
        out = client.batches.create(input_file_id="file-in", endpoint="/v1/embeddings")
        assert out["id"] == "batch-1"
        assert out["request_counts"]["total"] == 2
        method, url = mock_client.return_value.request.call_args.args
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert (method, url) == ("POST", "/v1/batches")
        assert body == {
            "input_file_id": "file-in",
            "endpoint": "/v1/embeddings",
            "completion_window": "24h",
        }
        client.close()


def test_create_includes_metadata_when_set() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, _BATCH))
        client = SIEClient(GW, api_key=KEY)
        client.batches.create(
            input_file_id="file-in",
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"run": "eval-7"},
        )
        body = mock_client.return_value.request.call_args.kwargs["json"]
        assert body["metadata"] == {"run": "eval-7"}
        client.close()


def test_retrieve_and_cancel_hit_expected_urls() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_resp(200, _BATCH))
        client = SIEClient(GW, api_key=KEY)
        client.batches.retrieve("batch-1")
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/batches/batch-1")
        client.batches.cancel("batch-1")
        assert mock_client.return_value.request.call_args.args == ("POST", "/v1/batches/batch-1/cancel")
        client.close()


def test_list_returns_data_array() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_resp(200, {"object": "list", "data": [{"id": "batch-1"}, {"id": "batch-2"}]})
        )
        client = SIEClient(GW, api_key=KEY)
        batches = client.batches.list()
        assert [b["id"] for b in batches] == ["batch-1", "batch-2"]
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/batches")
        client.close()


def test_list_encodes_cursor_arguments_and_preserves_sequence_return() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_resp(
                200,
                {
                    "object": "list",
                    "data": [{"id": "batch-2"}],
                    "first_id": "batch-2",
                    "last_id": "batch-2",
                    "has_more": False,
                },
            )
        )
        client = SIEClient(GW, api_key=KEY)
        batches = client.batches.list(after='batch/"a b"&limit=1', limit=9)
        assert [batch["id"] for batch in batches] == ["batch-2"]
        assert mock_client.return_value.request.call_args.args == (
            "GET",
            "/v1/batches?after=batch%2F%22a+b%22%26limit%3D1&limit=9",
        )
        client.close()


def test_list_page_exposes_cursor_metadata() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_resp(
                200,
                {
                    "object": "list",
                    "data": [{"id": "batch-1"}, {"id": "batch-2"}],
                    "first_id": "batch-1",
                    "last_id": "batch-2",
                    "has_more": True,
                },
            )
        )
        client = SIEClient(GW, api_key=KEY)
        page = client.batches.list_page()
        assert [batch["id"] for batch in page["data"]] == ["batch-1", "batch-2"]
        assert page["first_id"] == "batch-1"
        assert page["last_id"] == "batch-2"
        assert page["has_more"] is True
        client.close()


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_create_serializes_json_body() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(200, _BATCH))
    out = await client.batches.create(input_file_id="file-in", endpoint="/v1/embeddings")
    assert out["id"] == "batch-1"
    call = client._post.call_args
    assert call.args[0] == "/v1/batches"
    assert json.loads(call.kwargs["data"]) == {
        "input_file_id": "file-in",
        "endpoint": "/v1/embeddings",
        "completion_window": "24h",
    }
    await client.close()


@pytest.mark.asyncio
async def test_async_create_floors_long_running_timeout() -> None:
    """batches.create floors the per-call timeout to 120s (parity with the sync client)."""
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(200, _BATCH))
    await client.batches.create(input_file_id="file-in")
    assert client._post.call_args.kwargs["timeout_s"] == max(client._timeout, 120.0)
    await client.close()


@pytest.mark.asyncio
async def test_async_retrieve_and_cancel() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(return_value=_FakeAio(200, _BATCH))
    client._post = AsyncMock(return_value=_FakeAio(200, {"id": "batch-1", "status": "cancelling"}))
    got = await client.batches.retrieve("batch-1")
    assert got["status"] == "completed"
    assert client._get.call_args.args[0] == "/v1/batches/batch-1"
    out = await client.batches.cancel("batch-1")
    assert out["status"] == "cancelling"
    assert client._post.call_args.args[0] == "/v1/batches/batch-1/cancel"
    await client.close()


@pytest.mark.asyncio
async def test_async_list_and_list_page_share_cursor_arguments() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(
        return_value=_FakeAio(
            200,
            {
                "object": "list",
                "data": [{"id": "batch-2"}],
                "first_id": "batch-2",
                "last_id": "batch-2",
                "has_more": True,
            },
        )
    )
    batches = await client.batches.list(after='batch/"a b"', limit=9)
    assert [batch["id"] for batch in batches] == ["batch-2"]
    assert client._get.call_args.args[0] == "/v1/batches?after=batch%2F%22a+b%22&limit=9"
    page = await client.batches.list_page(limit=1)
    assert page["first_id"] == "batch-2"
    assert page["has_more"] is True
    assert client._get.call_args.args[0] == "/v1/batches?limit=1"
    await client.close()


@pytest.mark.asyncio
async def test_async_list_page_normalizes_empty_legacy_bare_array() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(return_value=_FakeAio(200, []))
    page = await client.batches.list_page()
    assert page["data"] == []
    assert page["first_id"] is None
    assert page["last_id"] is None
    assert page["has_more"] is False
    await client.close()
