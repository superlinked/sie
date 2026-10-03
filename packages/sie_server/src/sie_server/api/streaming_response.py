"""Own a primed SSE iterator through ASGI delivery and disconnects."""

from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from sie_server.adapters._generation_base import aclose_with_error_precedence


class PrefetchedStreamingResponse(StreamingResponse):
    def __init__(self, iterator: AsyncIterator[str | bytes], first: str | bytes, *, headers: dict[str, str]) -> None:
        self._iterator = iterator

        async def body() -> AsyncIterator[str | bytes]:
            yield first
            async for event in iterator:
                yield event

        super().__init__(body(), media_type="text/event-stream", headers=headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await aclose_with_error_precedence(
                self._iterator, outcome_selected=True, context="prefetched generation response"
            )


async def prefetched_sse_response(
    iterator: AsyncIterator[str | bytes], *, headers: dict[str, str]
) -> PrefetchedStreamingResponse:
    """Decide the first output/refusal before HTTP headers and retain ownership."""
    try:
        first = await anext(iterator)
        return PrefetchedStreamingResponse(iterator, first, headers=headers)
    except BaseException:
        await aclose_with_error_precedence(iterator, outcome_selected=True, context="generation pre-output refusal")
        raise
