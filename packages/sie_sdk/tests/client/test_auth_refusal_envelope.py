"""An auth refusal raises the same SDK error in the SIE and the OpenAI error envelope.

The gateway and ``sie-server`` answer a missing or invalid token on the
OpenAI-compatible routes (``/v1/chat/completions``, ``/v1/responses``, ...) in
the OpenAI ``{"error": {...}}`` envelope, and on the SIE-native routes in
``{"detail": {...}}``. Both carry the same status, ``code`` and ``message``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_sdk import SIEClient
from sie_sdk.client.errors import RequestError

MESSAGE = "Missing Authorization or x-api-key header"
DETAIL = {"detail": {"code": "UNAUTHORIZED", "message": MESSAGE}}
OPENAI = {"error": {"message": MESSAGE, "type": "authentication_error", "param": None, "code": "UNAUTHORIZED"}}


def _refusal(body: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.status_code = 401
    response.headers = {"content-type": "application/json", "www-authenticate": "Bearer"}
    response.content = json.dumps(body).encode()
    response.text = json.dumps(body)
    response.json.return_value = body
    return response


CALLS: dict[str, Callable[[SIEClient], object]] = {
    "chat_completions": lambda client: client.chat_completions("m", [{"role": "user", "content": "hi"}]),
    "responses": lambda client: client.responses("m", "hi"),
}


@pytest.mark.parametrize("body", [DETAIL, OPENAI], ids=["detail", "openai"])
@pytest.mark.parametrize("call", sorted(CALLS))
def test_an_auth_refusal_raises_request_error_in_either_envelope(body: dict[str, Any], call: str) -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = _refusal(body)
        client = SIEClient("http://localhost:8080")

        with pytest.raises(RequestError) as raised:
            CALLS[call](client)

        assert type(raised.value) is RequestError
        assert raised.value.status_code == 401
        assert raised.value.code == "UNAUTHORIZED"
        assert str(raised.value) == MESSAGE
        assert client_cls.return_value.post.call_count == 1
        client.close()
