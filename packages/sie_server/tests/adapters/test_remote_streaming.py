import asyncio
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


async def test_cr_terminated_event_is_delivered_while_upstream_stays_open() -> None:
    class HeldOpen(BytesStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"data: ready\r\r"
            await asyncio.Event().wait()

    response = httpx.Response(200, stream=HeldOpen([]))
    iterator = sse_data(response, max_event_bytes=100, max_total_bytes=100)
    try:
        async with asyncio.timeout(0.5):
            assert await anext(iterator) == b"ready"
    finally:
        await iterator.aclose()
        await response.aclose()


async def test_fragmented_long_line_and_many_short_lines() -> None:
    value = b"a" * 65_536
    wire = b"data: " + value + b"\n\n"
    chunks = [wire[index : index + 1] for index in range(len(wire))]
    chunks.append(b"data: b\n\n" * 4096)
    response = httpx.Response(200, stream=BytesStream(chunks))
    events = [data async for data in sse_data(response, max_event_bytes=70_000, max_total_bytes=120_000)]
    assert events == [value] + [b"b"] * 4096


async def test_lf_after_a_cr_terminated_nonempty_fragment_is_not_discarded() -> None:
    response = httpx.Response(200, stream=BytesStream([b"data: one\rdata: two", b"\n\n"]))
    assert [data async for data in sse_data(response, max_event_bytes=100, max_total_bytes=100)] == [b"one\ntwo"]
