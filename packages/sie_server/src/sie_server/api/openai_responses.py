"""``POST /v1/responses``: the OpenAI Responses API on the single-instance server.

The request is validated and translated into a chat completions body
(:mod:`sie_server.api.responses_translation`), this server's own
``/v1/chat/completions`` serves it in process, and the answer is translated
back: a chat completion into a Response, chat SSE chunks into Responses events,
and a chat error into the same error with the field the caller sent. Every chat
gate, limit and timeout therefore applies unchanged. Responses are not stored,
so ``/v1/responses/{id}`` and the routes below it are not served.
"""

from __future__ import annotations

import logging
import time
import uuid
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from sie_server.api.anthropic_messages import _header_pairs, _json_answer, _read, _ResponseBody
from sie_server.api.anthropic_translation import is_forwarded_header, parse_json
from sie_server.api.openai_local import _MAX_BODY_BYTES, serve_chat_completions
from sie_server.api.responses_translation import (
    ResponsesError,
    ResponseShape,
    StreamMachine,
    chat_completion_to_response,
    responses_error,
    responses_to_chat,
    stream_events,
)
from sie_server.api.routing import remote_routing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"], dependencies=[Depends(remote_routing)])


def _refusal(error: ResponsesError) -> Response:
    return JSONResponse(status_code=error.status, content=error.body())


def _server_error(message: str) -> Response:
    return _refusal(ResponsesError(message, param=None, code="server_error", status=500))


async def _error_answer(answer: Response) -> Response:
    headers, content = responses_error(_header_pairs(answer), await _read(answer))
    return _json_answer(answer.status_code, headers, content)


async def _read_body(request: Request) -> Any:
    declared = request.headers.get("content-length")
    too_large = ResponsesError("request body too large", param=None, code="INPUT_TOO_LONG", status=413)
    if declared is not None and declared.isdigit() and int(declared) > _MAX_BODY_BYTES:
        raise too_large
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > _MAX_BODY_BYTES:
            raise too_large
        raw.extend(chunk)
    try:
        return parse_json(bytes(raw))
    except (ValueError, UnicodeDecodeError):
        raise ResponsesError("body: invalid JSON", param="body", code="invalid_request") from None


async def _buffered_response(answer: Response, shape: ResponseShape, request_id: str) -> Response:
    try:
        chat = parse_json(await _read(answer))
    except ValueError:
        return _server_error("the chat completion is not valid JSON")
    try:
        response = chat_completion_to_response(chat, shape, request_id, int(time.time()))
    except ValueError:
        logger.warning("a chat completion could not be translated into a Response", exc_info=True)
        return _server_error("the chat completion could not be translated")
    headers = {name: value for name, value in _header_pairs(answer) if is_forwarded_header(name)}
    return JSONResponse(content=response, headers=headers)


class _ResponsesStream(StreamingResponse):
    """A chat SSE answer served as Responses events. Ending this answer ends the chat answer."""

    def __init__(self, chat: Response, machine: StreamMachine) -> None:
        self._body = _ResponseBody(chat)
        headers = {name: value for name, value in _header_pairs(chat) if name not in ("content-length", "content-type")}
        super().__init__(stream_events(self._body, machine), media_type="text/event-stream", headers=headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._body.aclose()


@router.post(
    "/responses",
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
async def responses(
    http_request: Request,
    x_machine_profile: Annotated[str | None, Header(alias="X-SIE-MACHINE-PROFILE")] = None,
) -> Response:
    """OpenAI Responses API, served by this server's chat completions route.

    The request is translated into a ``/v1/chat/completions`` request, served
    in process with all of that route's limits, and the answer is translated
    back, as a Response or, with ``stream``, as Responses events. Responses are
    not stored: ``store``, ``previous_response_id`` and ``conversation`` are
    refused, and the caller sends the whole conversation in ``input``. Function
    and custom tools are supported. Errors use the OpenAI envelope at the status
    the chat route answers.
    """
    try:
        translation = responses_to_chat(await _read_body(http_request))
    except ResponsesError as error:
        return _refusal(error)
    try:
        answer = await serve_chat_completions(http_request, x_machine_profile, translation.chat)
        if not HTTPStatus.OK <= answer.status_code < HTTPStatus.MULTIPLE_CHOICES:
            return await _error_answer(answer)
        request_id = next((value for name, value in _header_pairs(answer) if name == "x-sie-request-id"), None)
        request_id = request_id or uuid.uuid4().hex
        if translation.stream:
            return _ResponsesStream(answer, StreamMachine(translation.shape, request_id, int(time.time())))
        return await _buffered_response(answer, translation.shape, request_id)
    except Exception:
        logger.exception("the Responses translation of a chat answer failed")
        return _server_error("internal server error")
