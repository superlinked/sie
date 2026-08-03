"""client.files namespace — request-shape assertions against a mocked transport.

Mirrors the OpenAI ``client.files`` surface (upload/create/retrieve/content/
delete) so an ``openai`` → ``sie_sdk`` swap is mechanical.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import FileList, SIEAsyncClient, SIEClient
from sie_sdk.files import resolve_upload

GW = "http://gw.test:8080"
KEY = "sk-sie-testkey"

_FILE = {
    "id": "file-abc",
    "object": "file",
    "bytes": 42,
    "created_at": 1,
    "filename": "in.jsonl",
    "purpose": "batch",
}


def _json_resp(status: int, body: Any) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/json"}
    r.json.return_value = body
    r.content = b"{}"
    return r


def _bytes_resp(status: int, data: bytes) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": "application/jsonl"}
    r.content = data
    return r


class _FakeAio:
    def __init__(self, status: int, body: Any, *, raw: bytes | None = None) -> None:
        self.status_code = status
        self.content = raw if raw is not None else b"{}"
        self.headers = {"content-type": "application/json"}
        self._body = body

    def json(self) -> Any:
        return self._body


# ---------------------------------------------------------------------------
# resolve_upload (pure helper)
# ---------------------------------------------------------------------------


def test_file_list_is_exported_from_public_package() -> None:
    page: FileList = {
        "object": "list",
        "data": [_FILE],
        "first_id": "file-abc",
        "last_id": "file-abc",
        "has_more": False,
    }
    assert page["data"][0]["id"] == "file-abc"


def test_resolve_upload_bytes_defaults_filename() -> None:
    assert resolve_upload(b"hi") == (b"hi", "upload.jsonl")
    assert resolve_upload(b"hi", "x.jsonl") == (b"hi", "x.jsonl")


def test_resolve_upload_path_reads_bytes_and_basename(tmp_path: Any) -> None:
    p = tmp_path / "batch_input.jsonl"
    p.write_bytes(b'{"custom_id":"a"}\n')
    content, name = resolve_upload(p)
    assert content == b'{"custom_id":"a"}\n'
    assert name == "batch_input.jsonl"


def test_resolve_upload_file_like_uses_basename_of_name(tmp_path: Any) -> None:
    p = tmp_path / "sub.jsonl"
    p.write_bytes(b"data")
    with p.open("rb") as fh:
        content, name = resolve_upload(fh)
    assert content == b"data"
    assert name == "sub.jsonl"


def test_resolve_upload_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="path, bytes"):
        resolve_upload(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_upload_bytes_posts_to_v1_files_raw_body() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.post = MagicMock(return_value=_json_resp(200, _FILE))
        client = SIEClient(GW, api_key=KEY)
        out = client.files.upload(b'{"custom_id":"a"}\n', purpose="batch", filename="in.jsonl")
        assert out["id"] == "file-abc"
        url = mock_client.return_value.post.call_args.args[0]
        kwargs = mock_client.return_value.post.call_args.kwargs
        assert url.startswith("/v1/files?")
        assert "purpose=batch" in url
        assert "filename=in.jsonl" in url
        assert kwargs["content"] == b'{"custom_id":"a"}\n'
        assert kwargs["headers"]["Content-Type"] == "application/jsonl"
        client.close()


def test_upload_path_derives_filename(tmp_path: Any) -> None:
    p = tmp_path / "batch.jsonl"
    p.write_bytes(b"x")
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.post = MagicMock(return_value=_json_resp(200, _FILE))
        client = SIEClient(GW, api_key=KEY)
        client.files.upload(p)
        url = mock_client.return_value.post.call_args.args[0]
        assert "filename=batch.jsonl" in url
        assert mock_client.return_value.post.call_args.kwargs["content"] == b"x"
        client.close()


def test_create_is_openai_exact_alias() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.post = MagicMock(return_value=_json_resp(200, _FILE))
        client = SIEClient(GW, api_key=KEY)
        out = client.files.create(file=b"x", purpose="batch")
        assert out["id"] == "file-abc"
        assert "purpose=batch" in mock_client.return_value.post.call_args.args[0]
        client.close()


def test_retrieve_and_delete_hit_expected_urls() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_json_resp(200, _FILE))
        client = SIEClient(GW, api_key=KEY)
        client.files.retrieve("file-abc")
        assert mock_client.return_value.request.call_args.args == ("GET", "/v1/files/file-abc")
        mock_client.return_value.request.return_value = _json_resp(
            200, {"id": "file-abc", "object": "file", "deleted": True}
        )
        out = client.files.delete("file-abc")
        assert out["deleted"] is True
        assert mock_client.return_value.request.call_args.args == ("DELETE", "/v1/files/file-abc")
        client.close()


def test_list_encodes_openai_cursor_filters_and_preserves_sequence_return() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_json_resp(
                200,
                {
                    "object": "list",
                    "data": [_FILE],
                    "first_id": "file-abc",
                    "last_id": "file-abc",
                    "has_more": True,
                },
            )
        )
        client = SIEClient(GW, api_key=KEY)
        files = client.files.list(after='file/"a b"&purpose=x', limit=7, order="asc", purpose="batch")
        assert [file["id"] for file in files] == ["file-abc"]
        assert mock_client.return_value.request.call_args.args == (
            "GET",
            "/v1/files?after=file%2F%22a+b%22%26purpose%3Dx&limit=7&order=asc&purpose=batch",
        )
        client.close()


def test_list_page_exposes_cursor_metadata() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(
            return_value=_json_resp(
                200,
                {
                    "object": "list",
                    "data": [_FILE],
                    "first_id": "file-abc",
                    "last_id": "file-abc",
                    "has_more": True,
                },
            )
        )
        client = SIEClient(GW, api_key=KEY)
        page = client.files.list_page()
        assert [file["id"] for file in page["data"]] == ["file-abc"]
        assert page["first_id"] == "file-abc"
        assert page["last_id"] == "file-abc"
        assert page["has_more"] is True
        client.close()


def test_list_page_normalizes_legacy_bare_array() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.request = MagicMock(return_value=_json_resp(200, [_FILE, {**_FILE, "id": "file-def"}]))
        client = SIEClient(GW, api_key=KEY)
        page = client.files.list_page()
        assert [file["id"] for file in page["data"]] == ["file-abc", "file-def"]
        assert page["first_id"] == "file-abc"
        assert page["last_id"] == "file-def"
        assert page["has_more"] is False
        client.close()


def test_content_returns_raw_bytes() -> None:
    payload = b'{"custom_id":"a","response":{"status_code":200}}\n'
    with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
        mock_client.return_value.get = MagicMock(return_value=_bytes_resp(200, payload))
        client = SIEClient(GW, api_key=KEY)
        out = client.files.content("file-out")
        assert out == payload
        assert mock_client.return_value.get.call_args.args[0] == "/v1/files/file-out/content"
        client.close()


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_upload_sends_raw_body() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._post = AsyncMock(return_value=_FakeAio(200, _FILE))
    out = await client.files.upload(b"x", filename="in.jsonl")
    assert out["id"] == "file-abc"
    call = client._post.call_args
    assert call.args[0].startswith("/v1/files?")
    assert "filename=in.jsonl" in call.args[0]
    assert call.kwargs["data"] == b"x"
    assert call.kwargs["headers"]["Content-Type"] == "application/jsonl"
    await client.close()


@pytest.mark.asyncio
async def test_async_content_and_delete() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(return_value=_FakeAio(200, None, raw=b"jsonl-bytes"))
    client._delete = AsyncMock(return_value=_FakeAio(200, {"id": "file-1", "object": "file", "deleted": True}))
    assert await client.files.content("file-1") == b"jsonl-bytes"
    assert client._get.call_args.args[0] == "/v1/files/file-1/content"
    out = await client.files.delete("file-1")
    assert out["deleted"] is True
    assert client._delete.call_args.args[0] == "/v1/files/file-1"
    await client.close()


@pytest.mark.asyncio
async def test_async_list_and_list_page_share_cursor_arguments() -> None:
    client = SIEAsyncClient(GW, api_key=KEY)
    client._get = AsyncMock(
        return_value=_FakeAio(
            200,
            {
                "object": "list",
                "data": [_FILE],
                "first_id": "file-abc",
                "last_id": "file-abc",
                "has_more": False,
            },
        )
    )
    files = await client.files.list(after='file/"a b"', limit=7, order="desc", purpose="batch")
    assert [file["id"] for file in files] == ["file-abc"]
    assert client._get.call_args.args[0] == ("/v1/files?after=file%2F%22a+b%22&limit=7&order=desc&purpose=batch")
    page = await client.files.list_page(limit=1)
    assert page["first_id"] == "file-abc"
    assert page["has_more"] is False
    assert client._get.call_args.args[0] == "/v1/files?limit=1"
    await client.close()
