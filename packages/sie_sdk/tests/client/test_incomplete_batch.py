"""Positional batch-contract guard (architecture-review finding U1).

The gateway's queue path answers a mixed-success batch with ``200`` carrying
only the *successful* items, and batch results are positional (item ``id`` is
optional). A shortened body therefore shifts every result after the dropped
item, so a zip-inputs-to-outputs consumer stores results against the wrong
inputs — silently. Both clients guard the 1:1 contract on encode and extract
and raise :class:`IncompleteBatchError` instead of returning a desynced list.

The pre-existing count assertions for encode live in ``test_sync.py`` /
``test_async.py`` (#1526); this module covers the typed error, its
diagnostic attributes, and the extract + async surfaces.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import IncompleteBatchError, ServerError


def _dense(value: float) -> dict[str, object]:
    return {"dims": 4, "dtype": "float32", "values": np.array([value, value, value, value])}


def _sync_response(payload: dict[str, object], headers: dict[str, str] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = headers or {}
    response.content = msgpack.packb(payload, use_bin_type=True)
    return response


def _async_response(payload: dict[str, object], headers: dict[str, str] | None = None) -> _AioResponse:
    return _AioResponse(200, msgpack.packb(payload, use_bin_type=True), headers or {})


class TestEncodeGuard:
    def test_dropped_item_raises_typed_error_with_counts(self) -> None:
        response = _sync_response(
            {"model": "e5-mistral", "items": [{"dense": _dense(1.0)}]},
            headers={"X-SIE-Request-ID": "req-drop"},
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(IncompleteBatchError) as excinfo:
                client.encode("e5-mistral", [{"text": "ok"}, {"text": "over-length"}])

            error = excinfo.value
            assert error.expected == 2
            assert error.received == 1
            assert error.model == "e5-mistral"
            assert error.code == "ENCODE_RESULT_COUNT_MISMATCH"
            assert error.request is not None
            assert error.request["id"] == "req-drop"
            client.close()

    def test_remains_a_server_error_for_existing_handlers(self) -> None:
        """Subclassing keeps pre-#3221-era ``except ServerError`` callers working."""
        response = _sync_response({"model": "e5-mistral", "items": []})

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(ServerError):
                client.encode("e5-mistral", {"text": "hello"})
            client.close()

    def test_names_the_dropped_ids_when_items_carry_them(self) -> None:
        response = _sync_response(
            {
                "model": "e5-mistral",
                "items": [{"id": "doc-a", "dense": _dense(1.0)}, {"id": "doc-c", "dense": _dense(3.0)}],
            }
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(IncompleteBatchError) as excinfo:
                client.encode(
                    "e5-mistral",
                    [
                        {"id": "doc-a", "text": "a"},
                        {"id": "doc-b", "text": "b"},
                        {"id": "doc-c", "text": "c"},
                    ],
                )

            assert excinfo.value.missing_ids == ["doc-b"]
            assert "doc-b" in str(excinfo.value)
            client.close()

    def test_missing_ids_is_none_when_ids_are_absent(self) -> None:
        """Without ids on both sides the set difference could mislabel a
        present-but-unnamed item, so the diagnostic degrades to counts only.
        """
        response = _sync_response({"model": "e5-mistral", "items": [{"dense": _dense(1.0)}]})

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(IncompleteBatchError) as excinfo:
                client.encode("e5-mistral", [{"text": "a"}, {"text": "b"}])

            assert excinfo.value.missing_ids is None
            client.close()

    def test_matched_counts_pass_through(self) -> None:
        response = _sync_response({"model": "e5-mistral", "items": [{"dense": _dense(1.0)}, {"dense": _dense(2.0)}]})

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            results = client.encode("e5-mistral", [{"text": "a"}, {"text": "b"}])

            assert len(results) == 2
            client.close()


class TestExtractGuard:
    def test_dropped_item_raises_typed_error(self) -> None:
        response = _sync_response(
            {
                "model": "gliner",
                "items": [{"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]}],
            }
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(IncompleteBatchError) as excinfo:
                client.extract("gliner", [{"text": "Apple info"}, {"text": "Tesla info"}], labels=["org"])

            error = excinfo.value
            assert error.expected == 2
            assert error.received == 1
            assert error.code == "EXTRACT_RESULT_COUNT_MISMATCH"
            assert "extraction result(s)" in str(error)
            client.close()

    def test_matched_counts_pass_through(self) -> None:
        response = _sync_response(
            {
                "model": "gliner",
                "items": [
                    {"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]},
                    {"entities": [{"text": "Tesla", "label": "org", "score": 0.95, "start": 0, "end": 5}]},
                ],
            }
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            results = client.extract("gliner", [{"text": "Apple info"}, {"text": "Tesla info"}], labels=["org"])

            assert len(results) == 2
            client.close()


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_encode_dropped_item_raises_typed_error(self) -> None:
        response = _async_response({"model": "e5-mistral", "items": [{"dense": _dense(1.0)}]})

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=response)  # type: ignore[method-assign]
        with pytest.raises(IncompleteBatchError) as excinfo:
            await client.encode("e5-mistral", [{"text": "a"}, {"text": "b"}])

        assert excinfo.value.expected == 2
        assert excinfo.value.received == 1
        assert excinfo.value.code == "ENCODE_RESULT_COUNT_MISMATCH"
        await client.close()

    @pytest.mark.asyncio
    async def test_extract_dropped_item_raises_typed_error(self) -> None:
        response = _async_response(
            {
                "model": "gliner",
                "items": [{"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]}],
            }
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=response)  # type: ignore[method-assign]
        with pytest.raises(IncompleteBatchError) as excinfo:
            await client.extract("gliner", [{"text": "a"}, {"text": "b"}], labels=["org"])

        assert excinfo.value.code == "EXTRACT_RESULT_COUNT_MISMATCH"
        await client.close()
