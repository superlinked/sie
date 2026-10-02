from collections.abc import AsyncIterator

import httpx
import pytest
from sie_server.adapters.remote._http import RemoteUpstreamError, sse_data


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("separator", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("split", [False, True])
async def test_sse_handles_split_lines_comments_and_multiline_data(separator: bytes, split: bool) -> None:
    wire = separator.join([b": keepalive", b"event: chunk", b"data: one", b"data: two", b"", b""])
    chunks = [wire[index : index + 1] for index in range(len(wire))] if split else [wire]
    response = httpx.Response(200, stream=BytesStream(chunks))
    assert [data async for data in sse_data(response, max_event_bytes=100, max_total_bytes=100)] == [b"one\ntwo"]


@pytest.mark.parametrize("wire", [b"data: partial", b"data: partial\n", b"data: partial\r"])
async def test_sse_discards_an_unterminated_event(wire: bytes) -> None:
    response = httpx.Response(200, stream=BytesStream([wire]))
    assert [data async for data in sse_data(response, max_event_bytes=100, max_total_bytes=100)] == []


@pytest.mark.parametrize("wire", [b"data: too large\n\n", b": too large\n\n", b"data: too large"])
async def test_sse_refuses_oversized_events_before_yielding(wire: bytes) -> None:
    response = httpx.Response(200, stream=BytesStream([wire]))
    with pytest.raises(RemoteUpstreamError, match="event over"):
        async for _data in sse_data(response, max_event_bytes=10, max_total_bytes=100):
            pytest.fail("an oversized event must not reach the caller")


async def test_sse_bounds_the_whole_stream() -> None:
    response = httpx.Response(200, stream=BytesStream([b"data: a\n\n", b"data: b\n\n"]))
    iterator = sse_data(response, max_event_bytes=10, max_total_bytes=10)
    assert await anext(iterator) == b"a"
    with pytest.raises(RemoteUpstreamError, match="stream exceeds"):
        await anext(iterator)
