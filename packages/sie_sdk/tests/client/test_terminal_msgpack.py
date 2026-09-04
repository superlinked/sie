from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from msgpack.exceptions import UnpackException
from sie_sdk import ProvisioningError, RequestError, ServerError, SIEAsyncClient, SIEClient
from sie_sdk._msgpack import packb as pack_msgpack
from sie_sdk.client._shared import get_error_detail, handle_error, parse_terminal_msgpack_object
from sie_sdk.client.async_ import _AioResponse

_ATTACKER_BODY = b"attacker-private-response-token"


def _object_array_sentinel() -> dict[bytes, object]:
    return {
        b"nd": True,
        b"type": "|O",
        b"kind": b"O",
        b"shape": [1],
        b"data": b"attacker-controlled-pickle",
    }


def _assert_content_safe_error(error: RequestError, *, expected: str, status_code: int, request_id: str) -> None:
    message = str(error)
    assert message == expected
    assert _ATTACKER_BODY.decode() not in message
    assert error.__context__ is None
    assert error.status_code == status_code
    assert error.request == {"id": request_id}


def _terminal_payload(operation: str) -> dict[str, object]:
    if operation == "encode":
        return {
            "model": "m",
            "items": [{"dense": {"dims": 1, "values": np.array([0.5], dtype=np.float32)}}],
        }
    if operation == "score":
        return {"model": "m", "scores": [{"item_id": "0", "score": 0.5, "rank": 0}]}
    return {"model": "m", "items": [{"entities": []}]}


def _invoke_sync(client: SIEClient, operation: str) -> object:
    if operation == "encode":
        return client.encode("m", {"text": "input"})
    if operation == "score":
        return client.score("m", "query", ["candidate"])
    return client.extract("m", {"text": "input"}, labels=["entity"])


async def _invoke_async(client: SIEAsyncClient, operation: str) -> object:
    if operation == "encode":
        return await client.encode("m", {"text": "input"})
    if operation == "score":
        return await client.score("m", "query", ["candidate"])
    return await client.extract("m", {"text": "input"}, labels=["entity"])


class _AsyncResponseContext:
    def __init__(self, response: _AioResponse) -> None:
        self.status = response.status_code
        self.headers = response.headers
        self._content = response.content

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def read(self) -> bytes:
        return self._content


@pytest.mark.parametrize("operation", ["encode", "score", "extract"])
def test_sync_terminal_operation_consumes_same_origin_modal_continuation(operation: str) -> None:
    path = f"/v1/{operation}/m?__modal_attempt_token=opaque"
    redirect = MagicMock(status_code=303, content=b"", headers={"Location": path})
    terminal = MagicMock(
        status_code=200,
        content=pack_msgpack(_terminal_payload(operation)),
        headers={"content-type": "application/msgpack"},
    )

    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = redirect
        client_cls.return_value.get.return_value = terminal
        client = SIEClient(
            "https://gateway.example.test",
            api_key="sie-secret",
            base_url_headers={"Modal-Key": "edge-secret"},
        )
        try:
            with patch("sie_sdk.client.sync.time.monotonic", return_value=100):
                result = _invoke_sync(client, operation)
        finally:
            client.close()

    assert isinstance(result, dict)
    assert client_cls.call_args.kwargs["headers"]["Authorization"] == "Bearer sie-secret"
    assert "event_hooks" in client_cls.call_args.kwargs
    client_cls.return_value.post.assert_called_once()
    client_cls.return_value.get.assert_called_once_with(
        path,
        headers={"Accept": "application/msgpack"},
        timeout=30,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["encode", "score", "extract"])
async def test_async_terminal_operation_consumes_same_origin_modal_continuation(operation: str) -> None:
    path = f"/v1/{operation}/m?__modal_attempt_token=opaque"
    redirect = _AioResponse(303, b"", {"Location": path})
    terminal = _AioResponse(
        200,
        pack_msgpack(_terminal_payload(operation)),
        {"content-type": "application/msgpack"},
    )
    session = MagicMock()
    session.get.return_value = _AsyncResponseContext(terminal)
    session.close = AsyncMock()
    client = SIEAsyncClient(
        "https://gateway.example.test",
        api_key="sie-secret",
        base_url_headers={"Modal-Key": "edge-secret"},
    )
    client._session = session
    client._post = AsyncMock(return_value=redirect)  # type: ignore[method-assign]

    try:
        with patch("sie_sdk.client.async_.time.monotonic", return_value=100):
            result = await _invoke_async(client, operation)
    finally:
        await client.close()

    assert isinstance(result, dict)
    client._post.assert_awaited_once()  # type: ignore[attr-defined]
    get_call = session.get.call_args
    assert get_call.args == (path,)
    assert client._headers["Authorization"] == "Bearer sie-secret"
    assert get_call.kwargs["headers"]["Accept"] == "application/msgpack"
    assert get_call.kwargs["headers"]["Modal-Key"] == "edge-secret"
    assert get_call.kwargs["allow_redirects"] is False


def test_sync_terminal_continuation_exhausted_budget_performs_no_get() -> None:
    redirect = MagicMock(
        status_code=303,
        content=b"",
        headers={"Location": "/result?__modal_attempt_token=opaque"},
    )

    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client = SIEClient("https://gateway.example.test")
        try:
            with (
                patch("sie_sdk.client.sync.time.monotonic", return_value=31),
                pytest.raises(ProvisioningError, match="awaiting request result"),
            ):
                client._follow_modal_continuations(
                    redirect,
                    start_time=0,
                    budget_s=30,
                    accept="application/msgpack",
                )
        finally:
            client.close()

    client_cls.return_value.get.assert_not_called()


@pytest.mark.asyncio
async def test_async_terminal_continuation_exhausted_budget_performs_no_get() -> None:
    redirect = _AioResponse(303, b"", {"Location": "/result?__modal_attempt_token=opaque"})
    session = MagicMock()
    session.close = AsyncMock()
    client = SIEAsyncClient("https://gateway.example.test")
    client._session = session

    try:
        with (
            patch("sie_sdk.client.async_.time.monotonic", return_value=31),
            pytest.raises(ProvisioningError, match="awaiting request result"),
        ):
            await client._follow_modal_continuations(
                redirect,
                start_time=0,
                budget_s=30,
                accept="application/msgpack",
            )
    finally:
        await client.close()

    session.get.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "expected_prefix"),
    [
        (303, "Unexpected score HTTP response"),
        (200, "Malformed score MessagePack response"),
    ],
)
def test_sync_score_rejects_nonterminal_msgpack_content_safely(status_code: int, expected_prefix: str) -> None:
    request_id = f"req-sync-score-{status_code}"
    response = MagicMock(
        status_code=status_code,
        content=_ATTACKER_BODY,
        headers={"content-type": "application/msgpack", "x-sie-request-id": request_id},
    )

    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = response
        client = SIEClient("https://gateway.example.test")
        try:
            with pytest.raises(RequestError) as exc_info:
                client.score("Qwen/Qwen3-Reranker-4B", "query", ["candidate"])
        finally:
            client.close()

    _assert_content_safe_error(
        exc_info.value,
        expected=(
            f"{expected_prefix} "
            f"(status={status_code}, content_type=application/msgpack, body_bytes={len(_ATTACKER_BODY)})"
        ),
        status_code=status_code,
        request_id=request_id,
    )
    assert client_cls.call_args.kwargs["follow_redirects"] is False
    client_cls.return_value.post.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_prefix"),
    [
        (303, "Unexpected score HTTP response"),
        (200, "Malformed score MessagePack response"),
    ],
)
async def test_async_score_rejects_nonterminal_msgpack_content_safely(
    status_code: int,
    expected_prefix: str,
) -> None:
    request_id = f"req-async-score-{status_code}"
    response = _AioResponse(
        status_code,
        _ATTACKER_BODY,
        {"content-type": "application/msgpack", "x-sie-request-id": request_id},
    )
    client = SIEAsyncClient("https://gateway.example.test")
    client._post = AsyncMock(return_value=response)  # type: ignore[method-assign]
    try:
        with pytest.raises(RequestError) as exc_info:
            await client.score("Qwen/Qwen3-Reranker-4B", "query", ["candidate"])
    finally:
        await client.close()

    _assert_content_safe_error(
        exc_info.value,
        expected=(
            f"{expected_prefix} "
            f"(status={status_code}, content_type=application/msgpack, body_bytes={len(_ATTACKER_BODY)})"
        ),
        status_code=status_code,
        request_id=request_id,
    )
    client._post.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.parametrize("status_code", [200, 303])
def test_terminal_msgpack_never_unpickles_object_arrays(status_code: int) -> None:
    response = MagicMock(
        status_code=status_code,
        content=pack_msgpack({"items": [{"dense": {"values": _object_array_sentinel()}}]}),
        headers={"content-type": "application/msgpack", "x-sie-request-id": "req-object-array"},
    )

    with patch("msgpack_numpy.pickle.loads") as pickle_loads:
        with pytest.raises(RequestError) as exc_info:
            parse_terminal_msgpack_object(response, owner="encode")

    pickle_loads.assert_not_called()
    assert exc_info.value.status_code == status_code
    assert exc_info.value.request == {"id": "req-object-array"}
    assert exc_info.value.__context__ is None
    assert b"attacker-controlled-pickle" not in str(exc_info.value).encode()


def test_msgpack_unpack_exceptions_are_sanitized() -> None:
    terminal_response = MagicMock(
        status_code=200,
        content=b"",
        headers={"content-type": "application/msgpack", "x-sie-request-id": "req-truncated"},
    )
    error_response = MagicMock(
        status_code=500,
        content=b"",
        headers={"content-type": "application/msgpack", "x-sie-request-id": "req-truncated"},
        text="safe fallback",
    )

    with patch(
        "sie_sdk.client._shared.unpack_msgpack",
        side_effect=UnpackException("private parser context"),
    ) as unpack:
        with pytest.raises(RequestError) as terminal_error:
            parse_terminal_msgpack_object(terminal_response, owner="encode")
        assert get_error_detail(error_response) is None
        with pytest.raises(ServerError, match="safe fallback") as server_error:
            handle_error(error_response)

    assert unpack.call_count == 3
    assert terminal_error.value.__context__ is None
    assert "private parser context" not in str(terminal_error.value)
    assert "private parser context" not in str(server_error.value)


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(400, RequestError), (500, ServerError)],
)
def test_msgpack_error_envelopes_never_unpickle_object_arrays(
    status_code: int,
    error_type: type[RequestError | ServerError],
) -> None:
    response = MagicMock(
        status_code=status_code,
        content=pack_msgpack({"error": _object_array_sentinel()}),
        headers={"content-type": "application/msgpack"},
    )

    with patch("msgpack_numpy.pickle.loads") as pickle_loads:
        detail = get_error_detail(response)
        with pytest.raises(error_type):
            handle_error(response)

    pickle_loads.assert_not_called()
    assert detail == _object_array_sentinel()
