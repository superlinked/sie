import asyncio
import json
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import msgpack
import numpy as np
import pytest
from sie_sdk import RequestError, ServerError, SIEAsyncClient, SIEConnectionError
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import PoolError


def _make_response(status_code: int = 200, content: bytes = b"", headers: dict | None = None) -> _AioResponse:
    return _AioResponse(status_code, content, headers or {})


def _make_msgpack_response(data: dict, status_code: int = 200) -> _AioResponse:
    return _AioResponse(status_code, msgpack.packb(data, use_bin_type=True), {})


def _make_json_response(data: dict, status_code: int = 200) -> _AioResponse:
    return _AioResponse(status_code, json.dumps(data).encode(), {"content-type": "application/json"})


class TestSIEAsyncClientInit:
    """Tests for SIEAsyncClient initialization."""

    @pytest.mark.asyncio
    async def test_default_initialization(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession") as mock_session,
            patch("sie_sdk.client.async_.aiohttp.TCPConnector"),
        ):
            mock_session.return_value.close = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            client._ensure_session()
            mock_session.assert_called_once()
            call_kwargs = mock_session.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080"
            assert call_kwargs["headers"]["Content-Type"] == "application/msgpack"
            await client.close()

    @pytest.mark.asyncio
    async def test_custom_timeout(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession") as mock_session,
            patch("sie_sdk.client.async_.aiohttp.TCPConnector"),
        ):
            mock_session.return_value.close = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080", timeout_s=60.0)
            client._ensure_session()
            call_kwargs = mock_session.call_args.kwargs
            assert call_kwargs["timeout"].total == 60.0
            await client.close()

    @pytest.mark.asyncio
    async def test_api_key_sets_auth_header(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession") as mock_session,
            patch("sie_sdk.client.async_.aiohttp.TCPConnector"),
        ):
            mock_session.return_value.close = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080", api_key="secret")
            client._ensure_session()
            call_kwargs = mock_session.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer secret"
            await client.close()

    @pytest.mark.asyncio
    async def test_base_url_property(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession") as mock_session,
            patch("sie_sdk.client.async_.aiohttp.TCPConnector"),
        ):
            mock_session.return_value.close = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080")
            assert client.base_url == "http://localhost:8080"
            await client.close()

    @pytest.mark.asyncio
    async def test_max_connections(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession") as mock_session,
            patch("sie_sdk.client.async_.aiohttp.TCPConnector") as mock_connector,
        ):
            mock_session.return_value.close = AsyncMock()
            client = SIEAsyncClient("http://localhost:8080", max_connections=512)
            client._ensure_session()
            call_kwargs = mock_connector.call_args.kwargs
            assert call_kwargs["limit"] == 512
            assert call_kwargs["limit_per_host"] == 512
            await client.close()

    @pytest.mark.asyncio
    async def test_max_concurrency_creates_semaphore(self) -> None:
        client = SIEAsyncClient("http://localhost:8080", max_concurrency=10)
        assert client._semaphore is not None
        # asyncio.Semaphore internal value check
        assert client._semaphore._value == 10  # type: ignore[attr-defined]
        await client.close()

    @pytest.mark.asyncio
    async def test_max_concurrency_none_no_semaphore(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        assert client._semaphore is None
        await client.close()


class TestMaxConcurrency:
    """Tests for max_concurrency semaphore throttling."""

    @pytest.mark.asyncio
    async def test_max_concurrency_limits_inflight(self) -> None:
        """Verify that with max_concurrency=2, only 2 requests are in-flight simultaneously."""
        peak_concurrency = 0
        current_concurrency = 0
        lock = asyncio.Lock()

        resp_body = msgpack.packb(
            {"model": "m", "items": [{"dense": {"dims": 2, "dtype": "float32", "values": np.array([1.0, 2.0])}}]},
            use_bin_type=True,
        )

        class _FakeResp:
            """Fake aiohttp response context manager with a delay to measure concurrency."""

            def __init__(self) -> None:
                self.status = 200
                self.headers: dict[str, str] = {}

            async def read(self) -> bytes:
                return resp_body

            async def __aenter__(self) -> Self:
                nonlocal peak_concurrency, current_concurrency
                async with lock:
                    current_concurrency += 1
                    peak_concurrency = max(peak_concurrency, current_concurrency)
                await asyncio.sleep(0.05)
                return self

            async def __aexit__(self, *args: object) -> None:
                nonlocal current_concurrency
                async with lock:
                    current_concurrency -= 1

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=lambda *a, **kw: _FakeResp())
        mock_session.close = AsyncMock()

        client = SIEAsyncClient("http://localhost:8080", max_concurrency=2)
        client._session = mock_session

        tasks = [client.encode("m", {"text": f"item-{i}"}) for i in range(10)]
        await asyncio.gather(*tasks)

        assert peak_concurrency <= 2
        assert mock_session.post.call_count == 10
        await client.close()

    @pytest.mark.asyncio
    async def test_max_concurrency_none_no_limit(self) -> None:
        """Verify that when max_concurrency is not set, all requests run concurrently."""
        peak_concurrency = 0
        current_concurrency = 0
        lock = asyncio.Lock()

        resp_body = msgpack.packb(
            {"model": "m", "items": [{"dense": {"dims": 2, "dtype": "float32", "values": np.array([1.0, 2.0])}}]},
            use_bin_type=True,
        )

        class _FakeResp:
            def __init__(self) -> None:
                self.status = 200
                self.headers: dict[str, str] = {}

            async def read(self) -> bytes:
                return resp_body

            async def __aenter__(self) -> Self:
                nonlocal peak_concurrency, current_concurrency
                async with lock:
                    current_concurrency += 1
                    peak_concurrency = max(peak_concurrency, current_concurrency)
                await asyncio.sleep(0.05)
                return self

            async def __aexit__(self, *args: object) -> None:
                nonlocal current_concurrency
                async with lock:
                    current_concurrency -= 1

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=lambda *a, **kw: _FakeResp())
        mock_session.close = AsyncMock()

        client = SIEAsyncClient("http://localhost:8080")
        client._session = mock_session

        tasks = [client.encode("m", {"text": f"item-{i}"}) for i in range(10)]
        await asyncio.gather(*tasks)

        # Without concurrency limit, all 10 should run concurrently
        assert peak_concurrency == 10
        await client.close()


class TestAsyncEncode:
    """Tests for async encode() method."""

    @pytest.mark.asyncio
    async def test_encode_single_item_returns_single_result(self) -> None:
        resp = _make_msgpack_response(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "id": "doc-1",
                        "dense": {
                            "dims": 4,
                            "dtype": "float32",
                            "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                        },
                    }
                ],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        result = await client.encode("bge-m3", {"id": "doc-1", "text": "hello"})

        assert isinstance(result, dict)
        assert result["id"] == "doc-1"
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].shape == (4,)
        await client.close()

    @pytest.mark.asyncio
    async def test_encode_list_returns_list(self) -> None:
        resp = _make_msgpack_response(
            {
                "model": "bge-m3",
                "items": [
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([1.0, 2.0, 3.0, 4.0])}},
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([5.0, 6.0, 7.0, 8.0])}},
                ],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        results = await client.encode("bge-m3", [{"text": "hello"}, {"text": "world"}])

        assert isinstance(results, list)
        assert len(results) == 2
        await client.close()


class TestAsyncListModels:
    """Tests for async list_models() method."""

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self) -> None:
        resp = _make_json_response(
            {
                "models": [
                    {
                        "name": "bge-m3",
                        "loaded": True,
                        "capabilities": {
                            "inputs": ["text"],
                            "outputs": ["dense", "sparse", "multivector"],
                        },
                        "dims": {"dense": 1024, "sparse": 250002, "multivector": 1024},
                    }
                ]
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        models = await client.list_models()

        assert len(models) == 1
        assert models[0]["name"] == "bge-m3"
        assert models[0]["loaded"] is True
        await client.close()


class TestAsyncScore:
    """Tests for async score() method."""

    @pytest.mark.asyncio
    async def test_score_returns_score_result(self) -> None:
        resp = _make_msgpack_response(
            {
                "model": "bge-reranker-v2",
                "scores": [
                    {"item_id": "doc-1", "score": 0.95, "rank": 0},
                    {"item_id": "doc-2", "score": 0.72, "rank": 1},
                ],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        result = await client.score(
            "bge-reranker-v2",
            query={"text": "What is ML?"},
            items=[{"id": "doc-1", "text": "ML info"}, {"id": "doc-2", "text": "Other"}],
        )

        assert result["model"] == "bge-reranker-v2"
        assert len(result["scores"]) == 2
        assert result["scores"][0]["item_id"] == "doc-1"
        await client.close()


class TestAsyncExtract:
    """Tests for async extract() method."""

    @pytest.mark.asyncio
    async def test_extract_single_item_returns_single_result(self) -> None:
        resp = _make_msgpack_response(
            {
                "model": "gliner",
                "items": [
                    {
                        "entities": [
                            {"text": "Apple", "label": "org", "score": 0.98, "start": 0, "end": 5},
                        ]
                    }
                ],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        result = await client.extract(
            "gliner",
            {"text": "Apple founded by Steve Jobs."},
            labels=["org", "person"],
        )

        assert isinstance(result, dict)
        assert "entities" in result
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "Apple"
        await client.close()

    @pytest.mark.asyncio
    async def test_extract_list_returns_list(self) -> None:
        resp = _make_msgpack_response(
            {
                "model": "gliner",
                "items": [
                    {"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]},
                    {"entities": [{"text": "Tesla", "label": "org", "score": 0.95, "start": 0, "end": 5}]},
                ],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        results = await client.extract(
            "gliner",
            [{"text": "Apple info"}, {"text": "Tesla info"}],
            labels=["org"],
        )

        assert isinstance(results, list)
        assert len(results) == 2
        await client.close()


class TestAsyncContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession") as mock_session,
            patch("sie_sdk.client.async_.aiohttp.TCPConnector"),
        ):
            mock_session.return_value.close = AsyncMock()
            async with SIEAsyncClient("http://localhost:8080") as client:
                assert client is not None
                client._ensure_session()
            mock_session.return_value.close.assert_called_once()


class TestAsyncResourceCleanup:
    """Tests for async client resource cleanup warnings."""

    def test_unclosed_async_client_warns(self) -> None:
        import gc
        import warnings

        client = SIEAsyncClient("http://localhost:8080")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del client
            gc.collect()

        resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
        assert len(resource_warnings) == 1
        assert "Unclosed" in str(resource_warnings[0].message)

    @pytest.mark.asyncio
    async def test_closed_async_client_no_warning(self) -> None:
        import gc
        import warnings

        client = SIEAsyncClient("http://localhost:8080")
        await client.close()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del client
            gc.collect()

        resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
        assert len(resource_warnings) == 0


class TestAsyncErrorHandling:
    """Tests for async error handling."""

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="Failed to connect"):
            await client.encode("bge-m3", {"text": "hello"})
        await client.close()

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=TimeoutError("Timeout"))  # type: ignore[method-assign]

        with pytest.raises(SIEConnectionError, match="timed out"):
            await client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=False)
        await client.close()

    @pytest.mark.asyncio
    async def test_broad_client_error(self) -> None:
        """Non-connector aiohttp.ClientError (e.g. ServerDisconnectedError) is caught."""
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ServerDisconnectedError(),
        )

        with pytest.raises(SIEConnectionError, match="Failed to connect"):
            await client.encode("bge-m3", {"text": "hello"})
        await client.close()

    @pytest.mark.asyncio
    async def test_connection_error_on_create_pool(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="connection error"):
            await client.create_pool("my-pool", gpus={"l4": 1})

        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_request_error_400(self) -> None:
        resp = _make_json_response(
            {"detail": {"code": "INVALID_INPUT", "message": "Invalid text"}},
            status_code=400,
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        with pytest.raises(RequestError) as exc_info:
            await client.encode("bge-m3", {"text": "hello"})

        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "INVALID_INPUT"
        await client.close()

    @pytest.mark.asyncio
    async def test_server_error_500(self) -> None:
        resp = _make_json_response(
            {"error": {"code": "INFERENCE_ERROR", "message": "Model failed"}},
            status_code=500,
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        with pytest.raises(ServerError) as exc_info:
            await client.encode("bge-m3", {"text": "hello"})

        assert exc_info.value.status_code == 500
        assert exc_info.value.code == "INFERENCE_ERROR"
        await client.close()


class TestAsyncPoolOperations:
    """Tests for async pool create/get/delete."""

    @pytest.mark.asyncio
    async def test_create_pool_success(self) -> None:
        resp = _make_json_response(
            {"name": "my-pool", "status": {"state": "active"}},
            status_code=200,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        await client.create_pool("my-pool", gpus={"l4": 1})

        client._post.assert_called_once()
        call_kwargs = client._post.call_args.kwargs
        assert call_kwargs["json_data"]["name"] == "my-pool"
        assert call_kwargs["json_data"]["gpus"] == {"l4": 1}
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_duplicate_skipped(self) -> None:
        resp = _make_json_response(
            {"name": "my-pool", "status": {"state": "active"}},
            status_code=200,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        await client.create_pool("my-pool", gpus={"l4": 1})
        await client.create_pool("my-pool", gpus={"l4": 1})

        assert client._post.call_count == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_connection_error_cleans_sentinel(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="connection error"):
            await client.create_pool("my-pool", gpus={"l4": 1})

        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_http_error_raises_pool_error(self) -> None:
        resp = _make_json_response(
            {"detail": {"message": "Invalid machine profile"}},
            status_code=400,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        with pytest.raises(PoolError, match="Invalid machine profile"):
            await client.create_pool("my-pool", gpus={"l4": 1})

        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_negative_worker_count_raises(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")

        with pytest.raises(ValueError, match="minimum_worker_count must be >= 0"):
            await client.create_pool("my-pool", gpus={"l4": 1}, minimum_worker_count=-1)

        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_get_pool_success(self) -> None:
        resp = _make_json_response(
            {"name": "my-pool", "spec": {"gpus": {"l4": 1}}, "status": {"state": "active"}},
            status_code=200,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        result = await client.get_pool("my-pool")

        assert result is not None
        assert result["name"] == "my-pool"
        await client.close()

    @pytest.mark.asyncio
    async def test_get_pool_not_found_returns_none(self) -> None:
        resp = _make_json_response({}, status_code=404)
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        result = await client.get_pool("missing-pool")

        assert result is None
        await client.close()

    @pytest.mark.asyncio
    async def test_get_pool_connection_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="connection error"):
            await client.get_pool("my-pool")
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_pool_success(self) -> None:
        resp = _make_json_response({}, status_code=200)
        client = SIEAsyncClient("http://localhost:8080")
        client._delete = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        result = await client.delete_pool("my-pool")

        assert result is True
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_pool_not_found(self) -> None:
        resp = _make_json_response({}, status_code=404)
        client = SIEAsyncClient("http://localhost:8080")
        client._delete = AsyncMock(return_value=resp)  # type: ignore[method-assign]

        result = await client.delete_pool("missing-pool")

        assert result is False
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_pool_connection_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._delete = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="connection error"):
            await client.delete_pool("my-pool")
        await client.close()
