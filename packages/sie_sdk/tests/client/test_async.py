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


def _make_msgpack_response(data: dict, status_code: int = 200, headers: dict[str, str] | None = None) -> _AioResponse:
    return _AioResponse(status_code, msgpack.packb(data, use_bin_type=True), headers or {})


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
        assert client._semaphore._value == 10  # type: ignore
        await client.close()

    @pytest.mark.asyncio
    async def test_max_concurrency_none_no_semaphore(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        assert client._semaphore is None
        await client.close()


class TestAsyncCapacity:
    """Tests for async gateway capacity parsing."""

    @pytest.mark.asyncio
    async def test_get_capacity_preserves_worker_pressure_fields(self) -> None:
        payload = {
            "type": "gateway",
            "status": "healthy",
            "cluster": {"worker_count": 1, "gpu_count": 4, "models_loaded": 1},
            "workers": [
                {
                    "name": "worker-l4-0",
                    "url": "http://worker-l4-0:8080",
                    "gpu": "l4",
                    "gpu_count": 4,
                    "ready_gpu_slots": 3,
                    "healthy": True,
                    "queue_depth": 7,
                    "pending_cost": 128,
                    "inflight_batches": 2,
                    "loaded_models": ["BAAI/bge-m3"],
                    "memory_used_bytes": 1024,
                    "memory_total_bytes": 4096,
                    "bundle": "default",
                    "bundle_config_hash": "hash-1",
                }
            ],
        }
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(return_value=_make_json_response(payload))  # type: ignore

        capacity = await client.get_capacity()
        await client.close()

        worker = capacity["workers"][0]
        assert worker["gpu_count"] == 4
        assert worker["ready_gpu_slots"] == 3
        assert worker["queue_depth"] == 7
        assert worker["pending_cost"] == 128
        assert worker["inflight_batches"] == 2

    @pytest.mark.asyncio
    async def test_get_capacity_defaults_baseline_worker_to_one_ready_slot(self) -> None:
        payload = {
            "type": "gateway",
            "status": "healthy",
            "cluster": {"worker_count": 1, "gpu_count": 1, "models_loaded": 0},
            "workers": [
                {
                    "url": "http://worker-l4-0:8080",
                    "gpu": "l4",
                    "healthy": True,
                    "queue_depth": 0,
                    "loaded_models": [],
                }
            ],
        }
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(return_value=_make_json_response(payload))  # type: ignore

        capacity = await client.get_capacity()
        await client.close()

        worker = capacity["workers"][0]
        assert worker["ready_gpu_slots"] == 1
        assert worker["pending_cost"] == 0
        assert worker["inflight_batches"] == 0


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
            },
            headers={
                "X-SIE-Request-ID": "req-async-encode",
                "X-SIE-Units-Input-Tokens": "2",
                "X-SIE-Credits-Debited": "5",
                "X-SIE-Execution-Identity-SHA256": "b" * 64,
            },
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        result = await client.encode("bge-m3", {"id": "doc-1", "text": "hello"})

        assert isinstance(result, dict)
        assert result["id"] == "doc-1"
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].shape == (4,)
        assert result["request"] == {
            "id": "req-async-encode",
            "usage": {"input_tokens": 2},
            "credits_debited": 5,
            "execution_identity_sha256": "b" * 64,
        }
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
            },
            headers={
                "X-SIE-Request-ID": " bad ",
                "X-SIE-Units-Input-Tokens": "-1",
                "X-SIE-Credits-Debited": "1.5",
            },
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        results = await client.encode("bge-m3", [{"text": "hello"}, {"text": "world"}])

        assert isinstance(results, list)
        assert len(results) == 2
        assert all("request" not in result for result in results)
        await client.close()

    @pytest.mark.asyncio
    async def test_encode_single_item_empty_items_raises_server_error(self) -> None:
        """Empty ``items`` for a single input raises ``ServerError`` not ``IndexError`` (#1526)."""
        resp = _make_msgpack_response(
            {"model": "e5-mistral", "items": []},
            headers={
                "X-SIE-Request-ID": "req-async-encode-desync",
                "X-SIE-Units-Input-Tokens": "13",
                "X-SIE-Credits-Debited": "5",
            },
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        with pytest.raises(ServerError) as excinfo:
            await client.encode("e5-mistral", {"text": "a very long document"})
        assert "e5-mistral" in str(excinfo.value)
        assert "0 embedding(s) for 1 input item(s)" in str(excinfo.value)
        assert excinfo.value.code == "ENCODE_RESULT_COUNT_MISMATCH"
        assert excinfo.value.request == {
            "id": "req-async-encode-desync",
            "usage": {"input_tokens": 13},
            "credits_debited": 5,
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_encode_batch_short_items_raises_server_error(self) -> None:
        """A batch response that drops an item raises ``ServerError`` with counts (#1526)."""
        resp = _make_msgpack_response(
            {
                "model": "e5-mistral",
                "items": [{"dense": {"dims": 4, "dtype": "float32", "values": np.array([1.0, 2.0, 3.0, 4.0])}}],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        with pytest.raises(ServerError) as excinfo:
            await client.encode("e5-mistral", [{"text": "short"}, {"text": "over-length doc"}])
        assert "1 embedding(s) for 2 input item(s)" in str(excinfo.value)
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
        client._get = AsyncMock(return_value=resp)  # type: ignore
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
            },
            headers={
                "X-SIE-Request-ID": "req-async-score",
                "X-SIE-Units-Pairs": "2",
                "X-SIE-Credits-Debited": "7",
            },
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        result = await client.score(
            "bge-reranker-v2",
            query={"text": "What is ML?"},
            items=[{"id": "doc-1", "text": "ML info"}, {"id": "doc-2", "text": "Other"}],
        )

        assert result["model"] == "bge-reranker-v2"
        assert len(result["scores"]) == 2
        assert result["scores"][0]["item_id"] == "doc-1"
        assert result["request"] == {
            "id": "req-async-score",
            "usage": {"pairs": 2},
            "credits_debited": 7,
        }
        await client.close()

    @pytest.mark.asyncio
    async def test_score_converts_image_query_and_items_to_wire_format(self) -> None:
        """Async score converts image query/items before msgpack serialization."""
        resp = _make_msgpack_response(
            {
                "model": "qwen3-vl-reranker",
                "scores": [{"item_id": "page-1", "score": 0.8, "rank": 0}],
            }
        )

        query_image = b"\xff\xd8\xff\xe0query"
        item_image = b"\xff\xd8\xff\xe0item"
        query = {"text": "rocket nozzle", "images": [query_image]}
        item = {"id": "page-1", "images": [item_image]}

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        await client.score("qwen3-vl-reranker", query=query, items=[item])

        body = client._post.call_args.kwargs["data"]
        request_body = msgpack.unpackb(body, raw=False)
        query_wire = request_body["query"]["images"][0]
        item_wire = request_body["items"][0]["images"][0]
        assert query_wire == {"data": query_image, "format": "jpeg"}
        assert item_wire == {"data": item_image, "format": "jpeg"}
        assert query["images"][0] == query_image
        assert item["images"][0] == item_image
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
            },
            headers={
                "X-SIE-Request-ID": "req-async-extract",
                "X-SIE-Units-Pages": "1",
                "X-SIE-Units-Audio-Ms": "250",
                "X-SIE-Credits-Debited": "11",
            },
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        result = await client.extract(
            "gliner",
            {"text": "Apple founded by Steve Jobs."},
            labels=["org", "person"],
        )

        assert isinstance(result, dict)
        assert "entities" in result
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "Apple"
        assert result["request"] == {
            "id": "req-async-extract",
            "usage": {"pages": 1, "audio_ms": 250},
            "credits_debited": 11,
        }
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
        client._post = AsyncMock(return_value=resp)  # type: ignore
        results = await client.extract(
            "gliner",
            [{"text": "Apple info"}, {"text": "Tesla info"}],
            labels=["org"],
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all("request" not in result for result in results)
        await client.close()

    @pytest.mark.asyncio
    async def test_extract_converts_document_to_wire_format(self) -> None:
        """Async extract converts document inputs and returns parsed `data`."""
        resp = _make_msgpack_response(
            {
                "model": "docling",
                "items": [{"entities": [], "data": {"document": {"pages": []}}}],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore
        result = await client.extract(
            "docling",
            {"document": b"%PDF-1.4 fake content"},
        )

        body = client._post.call_args.kwargs["data"]
        request_body = msgpack.unpackb(body, raw=False)
        wire_doc = request_body["items"][0]["document"]
        assert wire_doc["data"] == b"%PDF-1.4 fake content"
        assert wire_doc["format"] is None
        assert result["data"] == {"document": {"pages": []}}
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
        # `wait_for_capacity=False` opts out of the issue-#95 retry path;
        # see test_transport_error_retry.py for retry coverage.
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="Failed to connect"):
            await client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=False)
        await client.close()

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=TimeoutError("Timeout"))  # type: ignore

        with pytest.raises(SIEConnectionError, match="timed out"):
            await client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=False)
        await client.close()

    @pytest.mark.asyncio
    async def test_broad_client_error(self) -> None:
        """Mid-flight `aiohttp.ServerDisconnectedError` is now in the retryable
        transport-error set (see `_RETRYABLE_TRANSPORT_ERRORS` in `async_.py`).
        With `wait_for_capacity=False` it surfaces immediately as
        `SIEConnectionError` with the new "Connection lost mid-request"
        message; with the default `wait_for_capacity=True` it would retry
        until the provision timeout, which is covered by
        `test_transport_error_retry.py`.
        """
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore
            side_effect=aiohttp.ServerDisconnectedError(),
        )

        with pytest.raises(SIEConnectionError, match="Connection lost mid-request"):
            await client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=False)
        await client.close()

    @pytest.mark.asyncio
    async def test_connection_error_on_create_pool(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore
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
        client._post = AsyncMock(return_value=resp)  # type: ignore

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
        client._post = AsyncMock(return_value=resp)  # type: ignore

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
        client._post = AsyncMock(return_value=resp)  # type: ignore

        await client.create_pool("my-pool", gpus={"l4": 1}, gpu_caps={"l4": 4})

        client._post.assert_called_once()
        call_kwargs = client._post.call_args.kwargs
        assert call_kwargs["json_data"]["name"] == "my-pool"
        assert call_kwargs["json_data"]["gpus"] == {"l4": 1}
        assert call_kwargs["json_data"]["gpu_caps"] == {"l4": 4}
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_gpu_caps_only_success(self) -> None:
        resp = _make_json_response(
            {"name": "my-pool", "status": {"state": "active"}},
            status_code=200,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore

        await client.create_pool("my-pool", gpu_caps={"l4": 4})

        client._post.assert_called_once()
        call_kwargs = client._post.call_args.kwargs
        assert call_kwargs["json_data"]["name"] == "my-pool"
        assert "gpus" not in call_kwargs["json_data"]
        assert call_kwargs["json_data"]["gpu_caps"] == {"l4": 4}
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_duplicate_posts_update_without_duplicate_renewal(self) -> None:
        resp = _make_json_response(
            {"name": "my-pool", "status": {"state": "active"}},
            status_code=200,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore

        await client.create_pool("my-pool", gpus={"l4": 1})
        await client.create_pool("my-pool", gpus={"l4": 1})

        assert client._post.call_count == 2
        assert len(client._pools) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_connection_error_cleans_inflight_entry(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore
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
    async def test_concurrent_create_pool_waits_for_inflight_creation(self) -> None:
        release = asyncio.Event()
        post_calls = 0

        async def gated_post(*_args: object, **_kwargs: object) -> _AioResponse:
            nonlocal post_calls
            post_calls += 1
            await release.wait()
            return _make_json_response({"name": "my-pool", "status": {"state": "active"}})

        client = SIEAsyncClient("http://localhost:8080")
        client._post = gated_post  # type: ignore

        first = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)
        second = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)

        assert post_calls == 1
        assert not second.done()

        release.set()
        await asyncio.gather(first, second)

        assert post_calls == 1
        assert len(client._pools) == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_create_pool_propagates_inflight_failure(self) -> None:
        release = asyncio.Event()
        post_calls = 0

        async def failing_post(*_args: object, **_kwargs: object) -> _AioResponse:
            nonlocal post_calls
            post_calls += 1
            await release.wait()
            raise aiohttp.ServerDisconnectedError

        client = SIEAsyncClient("http://localhost:8080")
        client._post = failing_post  # type: ignore

        first = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)
        second = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)

        assert post_calls == 1
        assert not second.done()

        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

        assert post_calls == 1
        assert all(isinstance(result, SIEConnectionError) for result in results)
        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_cancel_cleans_inflight_entry(self) -> None:
        async def slow_post(*_args: object, **_kwargs: object) -> _AioResponse:
            await asyncio.sleep(60)
            return _make_json_response({"name": "my-pool", "status": {"state": "active"}})

        client = SIEAsyncClient("http://localhost:8080")
        client._post = slow_post  # type: ignore

        task = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_close_during_create_pool_does_not_start_renewal(self) -> None:
        release = asyncio.Event()

        async def gated_post(*_args: object, **_kwargs: object) -> _AioResponse:
            await release.wait()
            return _make_json_response({"name": "my-pool", "status": {"state": "active"}})

        client = SIEAsyncClient("http://localhost:8080")
        client._post = gated_post  # type: ignore

        task = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)
        await client.close()
        release.set()
        await task

        assert "my-pool" not in client._pools

    @pytest.mark.asyncio
    async def test_create_pool_cancel_during_lease_start_cleans_inflight_entry(self) -> None:
        resp = _make_json_response({"name": "my-pool", "status": {"state": "active"}})
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore

        async def cancelled_start(*_args: object, **_kwargs: object) -> None:
            raise asyncio.CancelledError

        client._start_pool_lease_renewal = cancelled_start  # type: ignore

        with pytest.raises(asyncio.CancelledError):
            await client.create_pool("my-pool", gpus={"l4": 1})

        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_pool_waits_for_inflight_create_before_delete(self) -> None:
        release = asyncio.Event()
        events: list[str] = []

        async def gated_post(*_args: object, **_kwargs: object) -> _AioResponse:
            events.append("post-start")
            await release.wait()
            events.append("post-end")
            return _make_json_response({"name": "my-pool", "status": {"state": "active"}})

        async def tracked_delete(*_args: object, **_kwargs: object) -> _AioResponse:
            events.append("delete")
            return _make_json_response({}, status_code=200)

        client = SIEAsyncClient("http://localhost:8080")
        client._post = gated_post  # type: ignore
        client._delete = tracked_delete  # type: ignore

        create_task = asyncio.create_task(client.create_pool("my-pool", gpus={"l4": 1}))
        await asyncio.sleep(0)
        delete_task = asyncio.create_task(client.delete_pool("my-pool"))
        await asyncio.sleep(0)

        assert events == ["post-start"]
        assert not delete_task.done()

        release.set()
        assert await delete_task is True
        await create_task

        assert events == ["post-start", "post-end", "delete"]
        assert "my-pool" not in client._pools
        await client.close()

    @pytest.mark.asyncio
    async def test_create_pool_http_error_raises_pool_error(self) -> None:
        resp = _make_json_response(
            {"detail": {"message": "Invalid machine profile"}},
            status_code=400,
        )
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp)  # type: ignore

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
        client._get = AsyncMock(return_value=resp)  # type: ignore

        result = await client.get_pool("my-pool")

        assert result is not None
        assert result["name"] == "my-pool"
        await client.close()

    @pytest.mark.asyncio
    async def test_get_pool_not_found_returns_none(self) -> None:
        resp = _make_json_response({}, status_code=404)
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(return_value=resp)  # type: ignore

        result = await client.get_pool("missing-pool")

        assert result is None
        await client.close()

    @pytest.mark.asyncio
    async def test_get_pool_connection_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._get = AsyncMock(  # type: ignore
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
        client._delete = AsyncMock(return_value=resp)  # type: ignore

        result = await client.delete_pool("my-pool")

        assert result is True
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_pool_not_found(self) -> None:
        resp = _make_json_response({}, status_code=404)
        client = SIEAsyncClient("http://localhost:8080")
        client._delete = AsyncMock(return_value=resp)  # type: ignore

        result = await client.delete_pool("missing-pool")

        assert result is False
        await client.close()

    @pytest.mark.asyncio
    async def test_delete_pool_connection_error(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._delete = AsyncMock(  # type: ignore
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused"),
            ),
        )

        with pytest.raises(SIEConnectionError, match="connection error"):
            await client.delete_pool("my-pool")
        await client.close()
