"""Request metadata retained by response-derived SDK exceptions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sie_sdk.client._shared import handle_error
from sie_sdk.client.errors import RequestError, ServerError


def _error_response(status_code: int, headers: dict[str, str]) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": "application/json", **headers}
    response.json.return_value = {"error": {"code": "terminal_error", "message": "terminal failure"}}
    response.content = b'{"error":{"code":"terminal_error","message":"terminal failure"}}'
    response.text = response.content.decode()
    return response


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(400, RequestError), (500, ServerError)],
)
def test_response_errors_retain_canonical_request_metadata(
    status_code: int,
    error_type: type[RequestError | ServerError],
) -> None:
    response = _error_response(
        status_code,
        {
            "x-sie-request-id": "req-terminal",
            "x-sie-units-input-tokens": "11",
            "x-sie-units-output-tokens": "7",
            "x-sie-credits-debited": "23",
        },
    )

    with pytest.raises(error_type) as excinfo:
        handle_error(response)

    assert excinfo.value.request == {
        "id": "req-terminal",
        "usage": {"input_tokens": 11, "output_tokens": 7},
        "credits_debited": 23,
    }


def test_response_errors_omit_malformed_metadata_fields_independently() -> None:
    response = _error_response(
        500,
        {
            "x-sie-request-id": " req-with-whitespace ",
            "x-sie-units-input-tokens": "-1",
            "x-sie-units-output-tokens": "07",
            "x-sie-credits-debited": "0",
        },
    )

    with pytest.raises(ServerError) as excinfo:
        handle_error(response)

    assert excinfo.value.request == {"credits_debited": 0}


def test_error_constructor_positional_compatibility_and_missing_metadata() -> None:
    request_error = RequestError("bad request", "legacy_code", 400)
    server_error = ServerError("bad server", "legacy_code", 500)

    assert (request_error.code, request_error.status_code, request_error.request) == ("legacy_code", 400, None)
    assert (server_error.code, server_error.status_code, server_error.request) == ("legacy_code", 500, None)
