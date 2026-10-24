"""``POST /v1/messages``: the Anthropic Messages API on the single-instance server.

The request is validated and translated into a chat completions body
(:mod:`sie_server.api.anthropic_translation`), this server's own
``/v1/chat/completions`` serves it in process, and the answer is translated
back: a chat completion into a Message, chat SSE chunks into Messages events,
and every error into the Anthropic envelope. Every chat gate, limit and timeout
therefore applies unchanged. ``POST /v1/messages/count_tokens`` is not served.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from http import HTTPStatus
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sie_server.api.anthropic_translation import (
    PING_INTERVAL_S,
    ErrorContext,
    MessagesError,
    StreamMachine,
    anthropic_error,
    chat_completion_to_message,
    envelope,
    error_type_for_status,
    is_forwarded_header,
    is_messages_path,
    json_text,
    messages_to_chat,
    parse_json,
    sie_code_for_status,
    stream_events,
)
from sie_server.api.openai_local import _MAX_BODY_BYTES, serve_chat_completions
from sie_server.api.routing import remote_routing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["anthropic-compat"], dependencies=[Depends(remote_routing)])

_MAX_ERROR_BODY_BYTES = 1024 * 1024
_END = object()
_INNER_SCOPE: Scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}}


class _ReleasedError(Exception):
    """Ends a response that is closed before it was read."""


async def _never() -> Message:
    return await asyncio.get_running_loop().create_future()


async def _refuse(_: Message) -> None:
    raise _ReleasedError


class _ResponseBody:
    """The body of a response, read through the response's own ASGI call.

    A chat answer releases what it holds, such as a generation slot or the
    child's stream, when its ASGI call ends. Reading it through that call and
    closing it by ending the call keeps that release with the chat route.
    """

    def __init__(self, response: Response) -> None:
        self._response = response
        self._chunks: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def __aiter__(self) -> _ResponseBody:
        return self

    async def __anext__(self) -> bytes:
        if self._task is None:
            self._task = asyncio.create_task(self._serve())
        item = await self._chunks.get()
        if item is _END:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return cast("bytes", item)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is None:
            try:
                await self._response(_INNER_SCOPE, _never, _refuse)
            except _ReleasedError:
                pass
            except Exception:  # noqa: BLE001 - closing must not mask the answer being sent
                logger.warning("closing an unread chat answer failed", exc_info=True)
            return
        self._task.cancel()
        await asyncio.wait({self._task})
        if not self._task.cancelled():
            self._task.exception()

    async def _serve(self) -> None:
        try:
            await self._response(_INNER_SCOPE, _never, self._forward)
        except Exception as error:  # noqa: BLE001 - a broken chat stream is reported to the reader
            await self._chunks.put(error)
            return
        await self._chunks.put(_END)

    async def _forward(self, message: Message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            await self._chunks.put(bytes(message["body"]))


def _header_pairs(response: Response) -> list[tuple[str, str]]:
    return [(name.decode("latin-1").lower(), value.decode("latin-1")) for name, value in response.raw_headers]


async def _read(response: Response) -> bytes:
    if not isinstance(response, StreamingResponse):
        return bytes(response.body)
    body = _ResponseBody(response)
    try:
        return b"".join([chunk async for chunk in body])
    finally:
        await body.aclose()


def _json_answer(status: int, headers: list[tuple[str, str]], content: bytes) -> Response:
    response = Response(content=content, status_code=status, media_type="application/json")
    for name, value in headers:
        if name != "content-type":
            response.headers.append(name, value)
    return response


def _refusal(error: MessagesError) -> Response:
    body = envelope(error_type_for_status(error.status), error.message, None)
    headers = [("x-sie-error-code", sie_code_for_status(error.status))]
    return _json_answer(error.status, headers, json_text(body).encode())


async def _error_answer(answer: Response, context: ErrorContext) -> Response:
    headers, content = anthropic_error(answer.status_code, _header_pairs(answer), await _read(answer), context)
    return _json_answer(answer.status_code, headers, content)


async def _read_body(request: Request) -> Any:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_BODY_BYTES:
        raise MessagesError(f"request body is larger than {_MAX_BODY_BYTES} bytes", status=413)
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > _MAX_BODY_BYTES:
            raise MessagesError(f"request body is larger than {_MAX_BODY_BYTES} bytes", status=413)
        raw.extend(chunk)
    try:
        return parse_json(bytes(raw))
    except ValueError as error:
        raise MessagesError(f"body: invalid JSON: {error}") from None


async def _buffered_message(answer: Response, model: str, request_id: str | None) -> Response:
    content = await _read(answer)
    try:
        chat = parse_json(content)
    except ValueError:
        return _refusal(MessagesError("the chat completion is not valid JSON", status=500))
    try:
        message = chat_completion_to_message(chat, model, request_id or uuid.uuid4().hex)
    except ValueError as error:
        return _refusal(MessagesError(str(error), status=500))
    headers = {name: value for name, value in _header_pairs(answer) if is_forwarded_header(name)}
    if request_id:
        headers["request-id"] = request_id
    return JSONResponse(content=message, headers=headers)


class _MessagesStream(StreamingResponse):
    """A chat SSE answer served as Messages events. Ending this answer ends the chat answer."""

    def __init__(self, chat: Response, machine: StreamMachine, request_id: str | None) -> None:
        self._body = _ResponseBody(chat)
        headers = {name: value for name, value in _header_pairs(chat) if name not in ("content-length", "content-type")}
        if request_id:
            headers["request-id"] = request_id
        super().__init__(
            stream_events(self._body, machine, PING_INTERVAL_S), media_type="text/event-stream", headers=headers
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._body.aclose()


@router.post(
    "/messages",
    response_model=None,
    responses={
        400: {"description": "Invalid or unsupported request"},
        404: {"description": "Model not found"},
        413: {"description": "Request body is too large"},
        500: {"description": "Generation failed"},
        502: {"description": "The generation backend failed the request"},
        503: {"description": "Model loading or temporarily unavailable; retry after Retry-After"},
        504: {"description": "Generation timed out"},
    },
)
async def messages(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> Response:
    """Anthropic Messages API, served by this server's chat completions route.

    The request is translated into a ``/v1/chat/completions`` request, served
    in process with all of that route's limits, and the answer is translated
    back, as a Message or, with ``stream``, as Messages events. Errors use the
    Anthropic envelope at the status the chat route answers.
    """
    try:
        translation = messages_to_chat(await _read_body(http_request), http_request.headers.get("anthropic-version"))
    except MessagesError as error:
        return _refusal(error)
    try:
        answer = await serve_chat_completions(http_request, x_machine_profile, translation.chat)
        if not HTTPStatus.OK <= answer.status_code < HTTPStatus.MULTIPLE_CHOICES:
            return await _error_answer(answer, ErrorContext(first_image=translation.first_image))
        request_id = next((value for name, value in _header_pairs(answer) if name == "x-sie-request-id"), None)
        if translation.stream:
            machine = StreamMachine(f"msg_{request_id or uuid.uuid4().hex}", translation.model)
            return _MessagesStream(answer, machine, request_id)
        return await _buffered_message(answer, translation.model, request_id)
    except Exception:
        logger.exception("the Messages translation of a chat answer failed")
        return _refusal(MessagesError("internal server error", status=500))


class AnthropicErrorEnvelopeMiddleware:
    """Give every error answer on ``/v1/messages`` and the paths below it the Anthropic envelope.

    Added outside the server's other middleware, so their refusals on these
    paths, such as a missing credential, take the same shape. An answer that is
    already in the Anthropic envelope passes through with ``Retry-After``
    capped.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_messages_path(scope["path"]):
            await self.app(scope, receive, send)
            return
        held: Message | None = None
        body = bytearray()
        oversized = False

        async def respond(message: Message) -> None:
            nonlocal held, oversized
            if message["type"] == "http.response.start" and message["status"] >= HTTPStatus.BAD_REQUEST:
                held = message
                return
            if held is None or message["type"] != "http.response.body":
                await send(message)
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > _MAX_ERROR_BODY_BYTES:
                oversized = True
            elif not oversized:
                body.extend(chunk)
            if message.get("more_body", False):
                return
            status = held["status"]
            inner = [(name.decode("latin-1"), value.decode("latin-1")) for name, value in held.get("headers", [])]
            headers, content = anthropic_error(status, inner, b"" if oversized else bytes(body), ErrorContext())
            raw = [(name.encode("latin-1"), value.encode("latin-1")) for name, value in headers]
            raw.append((b"content-length", str(len(content)).encode()))
            await send({"type": "http.response.start", "status": status, "headers": raw})
            await send({"type": "http.response.body", "body": content})

        await self.app(scope, receive, respond)
