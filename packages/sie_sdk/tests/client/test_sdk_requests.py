"""The requests both clients send must match the shared request fixture.

Every case of ``packages/wire-fixtures/sdk_requests.json`` is issued through
the sync and the async client against a local HTTP server, and the request
that reached it is compared with the fixture. The TypeScript SDK replays the
same cases in ``sdkRequests.test.ts``, so a call one SDK serializes
differently, or cannot express, fails here or there.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient

_FIXTURE: dict[str, Any] = json.loads((Path(__file__).parents[3] / "wire-fixtures" / "sdk_requests.json").read_text())
_CASES: list[dict[str, Any]] = _FIXTURE["cases"]
_MEDIA: dict[str, bytes] = {name: base64.b64decode(data) for name, data in _FIXTURE["media"].items()}
_MEDIA_NAMES: dict[bytes, str] = {data: name for name, data in _MEDIA.items()}

_SCHEMA = {
    "type": "object",
    "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
    "required": ["vendor", "total"],
}
_QUERY = {"text": "What is machine learning?"}
_DOCUMENTS = [
    {"id": "doc-1", "text": "Machine learning is a subfield of AI."},
    {"id": "doc-2", "text": "Python is a programming language."},
]
_ENTITY_TEXT = {"text": "Apple was founded by Steve Jobs."}
_STREAMS = frozenset({"stream_generate", "stream_chat_completions"})

# The call each fixture case stands for: (method, positional arguments, keyword arguments).
_CALLS: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {
    "encode_single_text": ("encode", ("BAAI/bge-m3", {"text": "hello"}), {}),
    "encode_batch_all_params": (
        "encode",
        ("BAAI/bge-m3", [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]),
        {
            "output_types": ["dense", "sparse"],
            "instruction": "Represent the query",
            "output_dtype": "float16",
            "is_query": True,
        },
    ),
    "encode_profile_option": (
        "encode",
        ("lightonai/GTE-ModernColBERT-v1", {"text": "a long document"}),
        {"output_types": ["multivector"], "is_query": False, "options": {"profile": "long_context"}},
    ),
    "encode_model_profile_id": ("encode", ("BAAI/bge-m3:sparse", {"text": "hello"}), {}),
    "encode_image": ("encode", ("google/siglip2-base-patch16-224", {"images": [_MEDIA["png"]]}), {}),
    "encode_pool_and_gpu": ("encode", ("BAAI/bge-m3", {"text": "hello"}), {"gpu": "eval-bench/l4"}),
    "encode_client_defaults": ("encode", ("BAAI/bge-m3", {"text": "hello"}), {}),
    "score_basic": ("score", ("BAAI/bge-reranker-v2-m3", _QUERY, _DOCUMENTS), {}),
    "score_instruction_and_options": (
        "score",
        ("Qwen/Qwen3-Reranker-4B", _QUERY, _DOCUMENTS),
        {"instruction": "Rank passages that answer the question", "options": {"profile": "default"}},
    ),
    "extract_labels": (
        "extract",
        ("urchade/gliner_multi-v2.1", _ENTITY_TEXT),
        {"labels": ["person", "organization"]},
    ),
    "extract_threshold": (
        "extract",
        ("urchade/gliner_multi-v2.1", _ENTITY_TEXT),
        {"labels": ["person", "organization"], "options": {"threshold": 0.75}},
    ),
    "extract_image_without_params": ("extract", ("lightonai/LightOnOCR-2-1B", {"images": [_MEDIA["png"]]}), {}),
    "extract_output_schema_and_instruction": (
        "extract",
        ("fastino/GLiNER2.5-Decide", {"text": "Get me a person. This is the third time."}),
        {
            "output_schema": {"handoff": {"type": "noul", "instructions": "Should a person take over?"}},
            "instruction": "Triage the support ticket.",
            "options": {"overflow_policy": "truncate_text"},
        },
    ),
    "extract_audio": (
        "extract",
        ("openai/whisper-large-v3-turbo", {"audio": {"data": _MEDIA["wav"], "format": "wav", "sample_rate": 16000}}),
        {},
    ),
    "extract_document": ("extract", ("docling", {"document": {"data": _MEDIA["pdf"], "format": "pdf"}}), {}),
    "generate_basic": (
        "generate",
        ("Qwen/Qwen3-4B-Instruct-2507", "Write a haiku about the sea."),
        {"max_new_tokens": 64, "temperature": 0.7},
    ),
    "generate_all_fields": (
        "generate",
        ("google/gemma-4-31B-it:h100-96k-hires-thinking-budget512-no-spec", "Extract the invoice."),
        {
            "max_new_tokens": 256,
            "temperature": 0,
            "top_p": 0.9,
            "stop": ["\n\n"],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.25,
            "grammar": {"json_schema": _SCHEMA, "strict": True},
            "seed": 7,
            "logit_bias": {"42": -1.5},
            "routing_key": "tenant-1",
            "prompt_cache_key": "invoice-v1",
            "safety_identifier": "user-9",
            "lora_adapter": "invoices",
            "options": {"overall_timeout_s": 30},
        },
    ),
    "generate_image": (
        "generate",
        ("Qwen/Qwen3-VL-8B-Instruct", "Describe the image."),
        {"max_new_tokens": 32, "images": [_MEDIA["png"]]},
    ),
    "stream_generate": (
        "stream_generate",
        ("Qwen/Qwen3-4B-Instruct-2507", "Write a haiku."),
        {"max_new_tokens": 64, "temperature": 0.7, "logprobs": True, "top_logprobs": 2},
    ),
    "chat_completions": (
        "chat_completions",
        (
            "Qwen/Qwen3-4B-Instruct-2507",
            [
                {"role": "system", "content": "Answer in JSON."},
                {"role": "user", "content": "Extract the invoice fields."},
            ],
        ),
        {
            "max_completion_tokens": 256,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "invoice", "strict": True, "schema": _SCHEMA},
            },
            "extra_body": {"schema_in_prompt": False},
        },
    ),
    "stream_chat_completions": (
        "stream_chat_completions",
        ("Qwen/Qwen3-4B-Instruct-2507", [{"role": "user", "content": "Count to ten."}]),
        {"max_completion_tokens": 64, "stream_options": {"include_usage": True}, "include_reasoning": True},
    ),
}


def _success(path: str, accept: str, item_count: int) -> tuple[bytes, str]:
    """A minimal success for the route, shaped by what the request asked for."""
    if path.startswith("/v1/encode/"):
        items = [{"dense": {"dims": 1, "values": np.zeros(1, dtype=np.float32)}} for _ in range(item_count)]
        return _pack({"model": "m", "items": items}), "application/msgpack"
    if path.startswith("/v1/score/"):
        scores = [{"item_id": "doc-1", "score": 0.5, "rank": 0}]
        return _pack({"model": "m", "scores": scores}), "application/msgpack"
    if path.startswith("/v1/extract/"):
        return _pack({"model": "m", "items": [{"entities": []} for _ in range(item_count)]}), "application/msgpack"
    if accept == "text/event-stream":
        return b"data: [DONE]\n\n", "text/event-stream"
    if path.startswith("/v1/generate/"):
        return json.dumps({"model": "m", "text": "ok", "finish_reason": "stop"}).encode(), "application/json"
    chat = {"id": "chat-1", "object": "chat.completion", "model": "m", "choices": []}
    return json.dumps(chat).encode(), "application/json"


def _pack(body: dict[str, Any]) -> bytes:
    # conftest patches msgpack with NumPy support, as the server's encoder has.
    return msgpack.packb(body, use_bin_type=True)


class _RecordingServer:
    """Threaded HTTP server that records each request and answers with a success."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        recorder = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                headers = {name.lower(): value for name, value in self.headers.items()}
                if headers.get("content-type") == "application/msgpack":
                    body = msgpack.unpackb(raw, raw=False)
                else:
                    body = json.loads(raw)
                path = unquote(urlsplit(self.path).path)
                recorder.requests.append({"method": self.command, "path": path, "headers": headers, "body": body})
                item_count = len(body["items"]) if isinstance(body.get("items"), list) else 1
                payload, content_type = _success(path, headers.get("accept", ""), item_count)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server() -> Iterator[_RecordingServer]:
    recording = _RecordingServer()
    try:
        yield recording
    finally:
        recording.close()


def _canonical(value: Any) -> Any:
    """Fixture view of a decoded body: bytes named by media, ``None`` members dropped."""
    if isinstance(value, bytes):
        name = _MEDIA_NAMES.get(value)
        return {"$bytes": base64.b64encode(value).decode()} if name is None else {"$media": name}
    if isinstance(value, dict):
        return {key: _canonical(member) for key, member in value.items() if member is not None}
    if isinstance(value, (list, tuple)):
        return [_canonical(member) for member in value]
    return value


def _expected_body(value: Any) -> Any:
    """The fixture's expected body with base64 media references resolved."""
    if isinstance(value, dict):
        if "$media_base64" in value:
            return _FIXTURE["media"][value["$media_base64"]]
        return {key: _expected_body(member) for key, member in value.items() if member is not None}
    if isinstance(value, list):
        return [_expected_body(member) for member in value]
    return value


def _assert_matches_fixture(case: dict[str, Any], sent: list[dict[str, Any]]) -> None:
    assert len(sent) == 1
    request, expected = sent[0], case["request"]
    assert request["method"] == expected["method"]
    assert request["path"] == expected["path"]
    for name in _FIXTURE["compared_headers"]:
        assert {name: request["headers"].get(name)} == {name: expected["headers"].get(name)}
    assert request["headers"].get("x-sie-sdk-version")

    body = _canonical(request["body"])
    for member, value in expected.get("may_send_default", {}).items():
        if member in body:
            assert {member: body.pop(member)} == {member: value}
    assert body == _expected_body(expected["body"])


def _client_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    return {"api_key": _FIXTURE["client"]["api_key"], **case.get("client", {})}


def test_every_fixture_case_has_a_call() -> None:
    assert sorted(_CALLS) == sorted(case["id"] for case in _CASES)


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_sync_client_sends_the_fixture_request(server: _RecordingServer, case: dict[str, Any]) -> None:
    method, args, kwargs = _CALLS[case["id"]]
    client = SIEClient(server.url, **_client_kwargs(case))
    try:
        result = getattr(client, method)(*args, **kwargs)
        if case["id"] in _STREAMS:
            list(result)
    finally:
        client.close()
    _assert_matches_fixture(case, server.requests)


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
async def test_async_client_sends_the_fixture_request(server: _RecordingServer, case: dict[str, Any]) -> None:
    method, args, kwargs = _CALLS[case["id"]]
    async with SIEAsyncClient(server.url, **_client_kwargs(case)) as client:
        if case["id"] in _STREAMS:
            async for _chunk in getattr(client, method)(*args, **kwargs):
                pass
        else:
            await getattr(client, method)(*args, **kwargs)
    _assert_matches_fixture(case, server.requests)
