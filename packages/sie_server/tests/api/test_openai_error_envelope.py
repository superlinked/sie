"""Every error answer on the server's OpenAI-compatible routes uses the OpenAI envelope."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sie_server.api.openai_errors import OpenAIErrorEnvelopeMiddleware, is_openai_path, openai_error_from_detail
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

TOKEN = "openai-envelope-token"  # noqa: S105 - a test credential
OPENAI_PATHS = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/responses/resp_1",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
]
OTHER_PATHS = [
    "/v1/messages",
    "/v1/encode/m",
    "/v1/score/m",
    "/v1/extract/m",
    "/v1/generate/m",
    "/v1/models",
    "/v1/models/m",
    "/v1/rerank",
    "/v1/systemone",
    "/v1/decisions",
    "/v1/completionsx",
    "/healthz",
]
MISSING = "Missing credentials: send Authorization: Bearer <token> or x-api-key: <token>"


@pytest.mark.parametrize("path", OPENAI_PATHS)
def test_openai_paths_cover_the_routes_and_the_paths_below_them(path: str) -> None:
    assert is_openai_path(path)


@pytest.mark.parametrize("path", OTHER_PATHS)
def test_other_paths_are_not_openai_paths(path: str) -> None:
    assert not is_openai_path(path)


@pytest.mark.parametrize(
    ("status", "error_type", "code"),
    [
        (400, "invalid_request_error", "INVALID_REQUEST"),
        (401, "authentication_error", "UNAUTHORIZED"),
        (402, "invalid_request_error", "INVALID_REQUEST"),
        (403, "permission_error", "FORBIDDEN"),
        (404, "invalid_request_error", "NOT_FOUND"),
        (405, "invalid_request_error", "INVALID_REQUEST"),
        (413, "invalid_request_error", "PAYLOAD_TOO_LARGE"),
        (429, "rate_limit_error", "RATE_LIMIT"),
        (500, "server_error", "INTERNAL_SERVER_ERROR"),
        (503, "server_error", "INTERNAL_SERVER_ERROR"),
    ],
)
def test_the_type_follows_the_status(status: int, error_type: str, code: str) -> None:
    assert openai_error_from_detail(status, {"detail": "refused"}) == {
        "error": {"message": "refused", "type": error_type, "param": None, "code": code}
    }
    envelope = openai_error_from_detail(status, {"detail": {"code": "SOME_SIE_CODE", "message": "refused"}})
    assert envelope is not None
    assert envelope["error"]["type"] == error_type
    assert envelope["error"]["code"] == "SOME_SIE_CODE"


def test_detail_fields_are_kept() -> None:
    detail = {
        "code": "KEY_REGION",
        "message": "this key belongs to another region",
        "param": "model",
        "key_region": "eu",
        "region_endpoint": "https://eu.example.test",
        "epoch": 7,
    }

    assert openai_error_from_detail(403, {"detail": detail}) == {
        "error": {
            "message": "this key belongs to another region",
            "type": "permission_error",
            "param": "model",
            "code": "KEY_REGION",
            "key_region": "eu",
            "region_endpoint": "https://eu.example.test",
            "epoch": 7,
        }
    }


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "no", "type": "invalid_request_error", "param": None, "code": "unsupported_field"}},
        {"detail": [{"loc": ["body"], "msg": "required"}]},
        {"message": "no"},
        ["detail"],
        "detail",
    ],
)
def test_other_bodies_are_not_rewritten(body: Any) -> None:
    assert openai_error_from_detail(400, body) is None


def _refusal(request: Request) -> Response:
    return JSONResponse(
        {"detail": {"code": "UNAUTHORIZED", "message": "no token"}},
        status_code=401,
        headers={
            "WWW-Authenticate": "Bearer",
            "Retry-After": "30",
            "X-SIE-Error-Code": "UNAUTHORIZED",
            "X-SIE-RateLimit-Rps": "5",
        },
    )


def _chunked(request: Request) -> Response:
    async def chunks() -> Any:
        yield b'{"detail": "Not '
        yield b'Found"}'

    return StreamingResponse(chunks(), status_code=404, media_type="application/json")


def _oversized(request: Request) -> Response:
    return JSONResponse({"detail": "x" * (1024 * 1024 + 1)}, status_code=400)


def _plain(request: Request) -> Response:
    return PlainTextResponse('{"detail": "no"}', status_code=401)


def _success(request: Request) -> Response:
    return JSONResponse({"detail": {"code": "OK", "message": "fine"}})


def _openai(request: Request) -> Response:
    return Response(
        b'{"error":{"message":"no","type":"x","param":null,"code":"y"}}', 400, media_type="application/json"
    )


def _app() -> TestClient:
    handlers = {
        "refusal": _refusal,
        "chunked": _chunked,
        "oversized": _oversized,
        "plain": _plain,
        "success": _success,
        "openai": _openai,
    }
    routes = [
        Route(f"{base}/{name}", handler)
        for base in ("/v1/chat/completions", "/v1/encode", "/v1/messages")
        for name, handler in handlers.items()
    ]
    app = Starlette(routes=routes)
    app.add_middleware(OpenAIErrorEnvelopeMiddleware)
    return TestClient(app)


def test_a_detail_refusal_keeps_its_status_and_headers() -> None:
    response = _app().get("/v1/chat/completions/refusal")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["retry-after"] == "30"
    assert response.headers["x-sie-error-code"] == "UNAUTHORIZED"
    assert response.headers["x-sie-ratelimit-rps"] == "5"
    assert response.headers["content-length"] == str(len(response.content))
    assert response.json() == {
        "error": {"message": "no token", "type": "authentication_error", "param": None, "code": "UNAUTHORIZED"}
    }


def test_a_body_sent_in_chunks_is_rewritten() -> None:
    response = _app().get("/v1/chat/completions/chunked")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"message": "Not Found", "type": "invalid_request_error", "param": None, "code": "NOT_FOUND"}
    }


@pytest.mark.parametrize("name", ["oversized", "plain", "success", "openai"])
def test_other_answers_pass_through(name: str) -> None:
    client = _app()
    expected = client.get(f"/v1/encode/{name}")
    response = client.get(f"/v1/chat/completions/{name}")

    assert response.status_code == expected.status_code
    assert response.content == expected.content
    assert response.headers.get("content-type") == expected.headers.get("content-type")


@pytest.mark.parametrize("base", ["/v1/encode", "/v1/messages"])
def test_other_paths_keep_the_detail_envelope(base: str) -> None:
    response = _app().get(f"{base}/refusal")

    assert response.status_code == 401
    assert response.json() == {"detail": {"code": "UNAUTHORIZED", "message": "no token"}}


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
    return TestClient(AppFactory.create_app(AppStateConfig(auth_mode="token")))


@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings", "/v1/audio/transcriptions"],
)
def test_a_missing_token_on_an_openai_route_gets_the_openai_envelope(served: TestClient, path: str) -> None:
    response = served.post(path, json={})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {
        "error": {"message": MISSING, "type": "authentication_error", "param": None, "code": "UNAUTHORIZED"}
    }


def test_native_routes_and_messages_keep_their_envelopes(served: TestClient) -> None:
    for path in ("/v1/encode/m", "/v1/models"):
        response = served.post(path, json={}) if path.startswith("/v1/encode") else served.get(path)
        assert response.status_code == 401, path
        assert response.json() == {"detail": {"code": "UNAUTHORIZED", "message": MISSING}}, path

    messages = served.post("/v1/messages", json={})
    assert messages.status_code == 401
    assert messages.json()["type"] == "error"
    assert messages.json()["error"]["type"] == "authentication_error"


def test_routing_refusals_on_openai_paths_get_the_openai_envelope(served: TestClient) -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    unknown = served.post("/v1/responses/resp_1", json={}, headers=headers)
    wrong_method = served.get("/v1/chat/completions", headers=headers)

    assert unknown.status_code == 404
    assert unknown.json() == {
        "error": {"message": "Not Found", "type": "invalid_request_error", "param": None, "code": "NOT_FOUND"}
    }
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "INVALID_REQUEST"
    assert wrong_method.headers["allow"] == "POST"
