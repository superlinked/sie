"""The direct ``/v1/chat/completions`` route shows a json_schema response format to the SGLang child's model."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sie_server.api import openai_local

from .test_openai_local import _chat_completion, _cuda_chat_client

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"weakness": {"type": "string", "enum": ["cross-site scripting", "SQL injection", "path traversal"]}},
    "required": ["weakness"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "cvss", "strict": True, "schema": SCHEMA}}
SECTION = "# Response Formats\n\n## cvss\n\n" + json.dumps(SCHEMA, separators=(",", ":"))


def _forwarded(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion('{"weakness": "path traversal"}'))

    client, _ = _cuda_chat_client(monkeypatch, _handler)
    response = client.post("/v1/chat/completions", json={"model": "Qwen/Qwen3-0.6B", "max_tokens": 8, **body})
    assert response.status_code == 200, response.text
    return seen["body"]


def test_schema_is_appended_to_the_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "system", "content": "Classify."}, {"role": "user", "content": "../../etc/passwd"}]
    forwarded = _forwarded(monkeypatch, {"messages": messages, "response_format": RESPONSE_FORMAT})
    assert forwarded["messages"] == [
        {"role": "system", "content": "Classify.\n\n" + SECTION},
        {"role": "user", "content": "../../etc/passwd"},
    ]
    assert forwarded["response_format"] == RESPONSE_FORMAT
    assert "schema_in_prompt" not in forwarded


def test_schema_becomes_the_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "user", "content": "../../etc/passwd"}]
    forwarded = _forwarded(monkeypatch, {"messages": messages, "response_format": RESPONSE_FORMAT})
    assert forwarded["messages"][0] == {"role": "system", "content": SECTION}


def test_template_without_system_role_gets_the_schema_in_the_user_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoSystemTokenizer:
        def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
            if any(message["role"] == "system" for message in messages):
                raise ValueError("System role not supported")
            return "rendered"

    monkeypatch.setattr(openai_local, "_load_native_tokenizer_coalesced", lambda *_a: _NoSystemTokenizer())
    monkeypatch.setattr(openai_local, "_SYSTEM_ROLE_RENDERING", {})
    seen: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion('{"weakness": "path traversal"}'))

    client, registry = _cuda_chat_client(monkeypatch, _handler)
    config = registry.get_config.return_value
    config.hf_id, config.hf_revision, config.weights_path = "google/gemma-2-2b-it", "abc123", None
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "../../etc/passwd"}],
            "response_format": RESPONSE_FORMAT,
        },
    )
    assert response.status_code == 200, response.text
    assert seen["body"]["messages"] == [{"role": "user", "content": SECTION + "\n\n../../etc/passwd"}]


def test_request_opt_out_forwards_the_messages_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "user", "content": "../../etc/passwd"}]
    forwarded = _forwarded(
        monkeypatch, {"messages": messages, "response_format": RESPONSE_FORMAT, "schema_in_prompt": False}
    )
    assert forwarded["messages"] == messages
    assert forwarded["response_format"] == RESPONSE_FORMAT
    assert "schema_in_prompt" not in forwarded


def test_profile_opt_out_forwards_the_messages_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion('{"weakness": "path traversal"}'))

    client, registry = _cuda_chat_client(monkeypatch, _handler)
    registry.get_config.return_value.resolve_profile.return_value.runtime["schema_in_prompt"] = False
    messages = [{"role": "user", "content": "../../etc/passwd"}]
    body = {"model": "Qwen/Qwen3-0.6B", "max_tokens": 8, "messages": messages, "response_format": RESPONSE_FORMAT}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert seen["body"]["messages"] == messages


def test_schema_already_in_the_prompt_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    system = "Classify. Answer with JSON matching:\n" + json.dumps(SCHEMA, indent=2)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "../../etc/passwd"}]
    forwarded = _forwarded(monkeypatch, {"messages": messages, "response_format": RESPONSE_FORMAT})
    assert forwarded["messages"] == messages


def test_json_object_mode_is_not_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "user", "content": "Reply in JSON"}]
    forwarded = _forwarded(monkeypatch, {"messages": messages, "response_format": {"type": "json_object"}})
    assert forwarded["messages"] == messages


def test_non_boolean_opt_out_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _cuda_chat_client(monkeypatch, lambda _request: pytest.fail("must not reach the child"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": RESPONSE_FORMAT,
            "schema_in_prompt": "no",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "schema_in_prompt"
