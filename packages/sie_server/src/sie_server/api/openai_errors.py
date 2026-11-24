"""The OpenAI error envelope for every error answer on the OpenAI-compatible routes.

The routes' handlers answer in the OpenAI envelope. Refusals made outside them,
such as token authentication's, use the SIE ``{"detail": ...}`` envelope, which
OpenAI clients cannot read. :class:`OpenAIErrorEnvelopeMiddleware` re-renders
those on ``/v1/chat/completions``, ``/v1/completions``, ``/v1/responses``,
``/v1/embeddings`` and ``/v1/audio/transcriptions`` with the rule the gateway's
``openai_error_envelope`` applies.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

OPENAI_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
)
_MAX_ERROR_BODY_BYTES = 1024 * 1024
_ENVELOPE_FIELDS = frozenset({"message", "type", "param", "code"})
_STATUS_TYPES: dict[int, str] = {
    HTTPStatus.UNAUTHORIZED: "authentication_error",
    HTTPStatus.FORBIDDEN: "permission_error",
    HTTPStatus.TOO_MANY_REQUESTS: "rate_limit_error",
}
_STATUS_CODES: dict[int, str] = {
    HTTPStatus.UNAUTHORIZED: "UNAUTHORIZED",
    HTTPStatus.FORBIDDEN: "FORBIDDEN",
    HTTPStatus.NOT_FOUND: "NOT_FOUND",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "PAYLOAD_TOO_LARGE",
    HTTPStatus.TOO_MANY_REQUESTS: "RATE_LIMIT",
}


def is_openai_path(path: str) -> bool:
    """Whether a path is one of this server's OpenAI-compatible routes or below one."""
    return any(path == base or path.startswith(f"{base}/") for base in OPENAI_PATHS)


def _client_error(status: int) -> bool:
    return HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR


def openai_type_for_status(status: int) -> str:
    """The ``error.type`` of a re-rendered refusal: by status, as on the gateway."""
    if status in _STATUS_TYPES:
        return _STATUS_TYPES[status]
    return "invalid_request_error" if _client_error(status) else "server_error"


def _code_for_status(status: int) -> str:
    if status in _STATUS_CODES:
        return _STATUS_CODES[status]
    return "INVALID_REQUEST" if _client_error(status) else "INTERNAL_SERVER_ERROR"


def _reason(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def openai_error_from_detail(status: int, body: Any) -> dict[str, Any] | None:
    """The OpenAI envelope for a ``{"detail": ...}`` body, or ``None`` for any other body.

    ``detail.code`` is kept verbatim and a string ``detail`` gets the SIE code of
    its status; ``param`` is ``detail.param`` when it is a string; every other
    ``detail`` field is kept in ``error``.
    """
    if not isinstance(body, dict) or isinstance(body.get("error"), dict):
        return None
    detail = body.get("detail")
    error_type = openai_type_for_status(status)
    if isinstance(detail, str):
        return {"error": {"message": detail, "type": error_type, "param": None, "code": _code_for_status(status)}}
    if not isinstance(detail, dict):
        return None
    message = detail.get("message")
    param = detail.get("param")
    code = detail.get("code")
    error: dict[str, Any] = {
        "message": message if isinstance(message, str) else _reason(status),
        "type": error_type,
        "param": param if isinstance(param, str) else None,
        "code": code if isinstance(code, str) else _code_for_status(status),
    }
    error.update((key, value) for key, value in detail.items() if key not in _ENVELOPE_FIELDS)
    return {"error": error}


def _is_json(headers: list[tuple[bytes, bytes]]) -> bool:
    for name, value in headers:
        if name.lower() == b"content-type":
            mime = value.decode("latin-1").split(";", 1)[0].strip().lower()
            return mime == "application/json" or mime.endswith("+json")
    return False


def _render(status: int, body: bytes) -> bytes | None:
    try:
        parsed = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None
    envelope = openai_error_from_detail(status, parsed)
    return None if envelope is None else orjson.dumps(envelope)


class OpenAIErrorEnvelopeMiddleware:
    """Give every error answer on the OpenAI-compatible routes and the paths below them the OpenAI envelope.

    Added outside the server's other middleware, so their refusals on these
    paths, such as a missing credential, take the shape OpenAI clients read. A
    4xx or 5xx JSON answer whose body is ``{"detail": {...}}`` or
    ``{"detail": "..."}`` becomes ``{"error": {"message", "type", "param",
    "code", ...}}`` with its status and headers, and a recomputed
    ``Content-Length``. ``type`` follows the status: ``authentication_error``
    (401), ``permission_error`` (403), ``rate_limit_error`` (429),
    ``invalid_request_error`` (other 4xx) and ``server_error`` (5xx). Answers
    already in the OpenAI envelope, answers that are not JSON, and successes
    pass through untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_openai_path(scope["path"]):
            await self.app(scope, receive, send)
            return
        held: Message | None = None
        body = bytearray()

        async def respond(message: Message) -> None:
            nonlocal held
            if message["type"] == "http.response.start":
                if message["status"] >= HTTPStatus.BAD_REQUEST and _is_json(message.get("headers", [])):
                    held = message
                    return
                await send(message)
                return
            if held is None or message["type"] != "http.response.body":
                await send(message)
                return
            body.extend(message.get("body", b""))
            more_body = message.get("more_body", False)
            start = held
            if len(body) > _MAX_ERROR_BODY_BYTES:
                held = None
                await send(start)
                await send({"type": "http.response.body", "body": bytes(body), "more_body": more_body})
                return
            if more_body:
                return
            held = None
            rendered = _render(start["status"], bytes(body))
            if rendered is None:
                await send(start)
                await send({"type": "http.response.body", "body": bytes(body)})
                return
            headers = [(name, value) for name, value in start.get("headers", []) if name.lower() != b"content-length"]
            headers.append((b"content-length", str(len(rendered)).encode()))
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": rendered})

        await self.app(scope, receive, respond)
