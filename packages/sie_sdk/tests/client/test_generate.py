"""Tests for the ``generate()`` SDK method (sync + async).

Mocks the HTTP layer; exercises:

- Happy-path JSON envelope is parsed into a :class:`GenerateResult`.
- Request body uses JSON (not msgpack) and the documented field names.
- 503 ``MODEL_LOADING`` retries under ``provision_timeout_s``.
- ``RequestError`` surfaces non-dict response payloads.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sie_sdk import SIEAsyncClient, SIEClient, SIEConnectionError
from sie_sdk.client._shared import validate_generate_grammar
from sie_sdk.client.errors import ProvisioningError, RequestError, ServerError


def _ok_response(payload: dict, headers: dict[str, str] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = headers or {}
    response.content = json.dumps(payload).encode("utf-8")
    response.json.return_value = payload
    return response


def _resp_504() -> MagicMock:
    # A gateway-owned queue result timeout is retryable for idempotent encode
    # paths; generate() must NOT retry it (post-publish, non-idempotent).
    response = MagicMock()
    response.status_code = 504
    response.headers = {
        "Retry-After": "0.01",
        "content-type": "application/json",
        "X-SIE-Request-ID": "req-generate-timeout",
        "X-SIE-Units-Output-Tokens": "5",
        "X-SIE-Credits-Debited": "13",
    }
    response.json.return_value = {"detail": {"code": "GATEWAY_TIMEOUT", "message": "Timeout waiting for queue result"}}
    response.content = json.dumps(response.json.return_value).encode("utf-8")
    return response


def _resp_503_model_loading() -> MagicMock:
    response = MagicMock()
    response.status_code = 503
    response.headers = {"X-SIE-Error-Code": "MODEL_LOADING", "content-type": "application/json"}
    response.json.return_value = {"error": {"code": "MODEL_LOADING", "message": "loading"}}
    response.content = json.dumps(response.json.return_value).encode("utf-8")
    return response


def _resp_503_provisioning() -> MagicMock:
    response = MagicMock()
    response.status_code = 503
    response.headers = {"Retry-After": "0.01", "content-type": "application/json"}
    response.json.return_value = {"error": {"code": "PROVISIONING", "message": "provisioning"}}
    response.content = json.dumps(response.json.return_value).encode("utf-8")
    return response


def _resp_503_openai_provisioning() -> MagicMock:
    response = MagicMock()
    response.status_code = 503
    response.headers = {
        "Retry-After": "0.01",
        "X-SIE-Error-Code": "PROVISIONING",
        "content-type": "application/json",
    }
    response.json.return_value = {"error": {"code": "provisioning", "message": "provisioning"}}
    response.content = json.dumps(response.json.return_value).encode("utf-8")
    return response


def _ok_envelope() -> dict:
    return {
        "model": "m",
        "text": "ok",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "attempt_id": "a",
    }


def _aio_resp(status: int, body: object, headers: dict | None = None) -> object:
    from sie_sdk.client.async_ import _AioResponse

    return _AioResponse(status, json.dumps(body).encode("utf-8"), headers or {"content-type": "application/json"})


def test_openai_provisioning_header_is_classified_as_provisioning() -> None:
    from sie_sdk.client._shared import get_error_code, handle_error

    response = _resp_503_openai_provisioning()
    assert get_error_code(response) == "PROVISIONING"
    with pytest.raises(ProvisioningError):
        handle_error(response)


def _make_session_post(seq_source: AsyncMock):
    """Build a ``session.post`` that yields a raw aiohttp-like response per call.

    The async ``generate`` path posts inline via ``_ensure_session().post(...)``
    (not ``_post``), reading ``raw.status``, ``await raw.read()`` and
    ``raw.headers``. This adapter pulls each canned ``_AioResponse`` from
    ``seq_source`` (an ``AsyncMock`` carrying the desired ``side_effect``) so
    tests can assert ``seq_source.call_count`` for retry behaviour.
    """

    def _post(*_args: object, **_kwargs: object) -> MagicMock:
        ctx = MagicMock()

        async def _aenter() -> MagicMock:
            aio = await seq_source()
            raw = MagicMock()
            raw.status = aio.status_code
            raw.headers = aio.headers
            raw.read = AsyncMock(return_value=aio.content)
            return raw

        ctx.__aenter__ = AsyncMock(side_effect=_aenter)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    return MagicMock(side_effect=_post)


class TestSyncGenerate:
    def test_json_schema_mapping_is_normalized_for_json_serialization(self) -> None:
        grammar = validate_generate_grammar(
            {"json_schema": MappingProxyType({"type": "object", "additionalProperties": False})}
        )

        assert json.loads(json.dumps(grammar)) == {"json_schema": {"type": "object", "additionalProperties": False}}

    def test_generate_serializes_native_images_as_bounded_json_envelopes(self) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = _ok_response(_ok_envelope())
            client = SIEClient("http://localhost:8080")
            client.generate(
                "m",
                prompt="Read",
                images=[{"data": b"\x89PNG\r\n\x1a\npayload", "format": "png"}],
                max_new_tokens=8,
            )

            sent = json.loads(mock_client.return_value.post.call_args.kwargs["content"])
            assert sent["images"] == [{"data": "iVBORw0KGgpwYXlsb2Fk", "format": "png"}]
            client.close()

    @pytest.mark.parametrize(
        ("grammar", "exception_type", "expected_message"),
        [
            ({}, ValueError, "grammar must contain exactly one of json_schema, regex, or ebnf"),
            (
                {"json_schema": {}, "regex": "x"},
                ValueError,
                "grammar must contain exactly one of json_schema, regex, or ebnf",
            ),
            ({"regex": 123}, TypeError, "grammar.regex must be a string"),
            (
                {"ebnf": "root", "unknown": True},
                ValueError,
                "grammar contains unsupported field(s): unknown",
            ),
            ({"json_schema": []}, TypeError, "grammar.json_schema must be a mapping"),
            ({"regex": "x", "label": 123}, TypeError, "grammar.label must be a string"),
            ({"ebnf": "root", "strict": "yes"}, TypeError, "grammar.strict must be a boolean"),
        ],
    )
    def test_generate_rejects_invalid_grammar_before_request(
        self,
        grammar: dict[str, object],
        exception_type: type[Exception],
        expected_message: str,
    ) -> None:
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080")
            with pytest.raises(exception_type) as exc_info:
                client.generate("m", prompt="Hi", grammar=grammar, max_new_tokens=8)
            assert str(exc_info.value) == expected_message
            mock_client.return_value.post.assert_not_called()
            client.close()

    def test_generate_happy_path_parses_envelope(self) -> None:
        envelope = {
            "model": "Qwen__Qwen3-4B-Instruct-2507",
            "text": "Hello world!",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "attempt_id": "att-abc",
            "ttft_ms": 120.5,
            "tpot_ms": 45.2,
        }
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = _ok_response(
                envelope,
                {
                    "X-SIE-Request-ID": "req-generate",
                    "X-SIE-Units-Input-Tokens": "5",
                    "X-SIE-Units-Output-Tokens": "3",
                    "X-SIE-Credits-Debited": "13",
                },
            )
            client = SIEClient("http://localhost:8080")
            result = client.generate(
                "Qwen__Qwen3-4B-Instruct-2507",
                prompt="Hi",
                max_new_tokens=32,
            )
            assert result["model"] == "Qwen__Qwen3-4B-Instruct-2507"
            assert result["text"] == "Hello world!"
            assert result["finish_reason"] == "stop"
            assert result["usage"]["completion_tokens"] == 3
            assert result["attempt_id"] == "att-abc"
            assert result["ttft_ms"] == 120.5
            assert result["tpot_ms"] == 45.2
            assert result["request"] == {
                "id": "req-generate",
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "credits_debited": 13,
            }
            sent = json.loads(mock_client.return_value.post.call_args.kwargs["content"].decode("utf-8"))
            assert "temperature" not in sent
            assert "top_p" not in sent

            client.close()

    def test_generate_request_body_uses_json_shape(self) -> None:
        envelope = {
            "model": "m",
            "text": "x",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "attempt_id": "a",
        }
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = _ok_response(envelope)
            client = SIEClient("http://localhost:8080", options={"first_chunk_timeout_s": 10, "overall_timeout_s": 60})
            client.generate(
                "m",
                prompt="Hi",
                max_new_tokens=8,
                temperature=1.0,
                top_p=1.0,
                stop=["</s>"],
                frequency_penalty=0.25,
                presence_penalty=-0.5,
                grammar={"regex": "[a-z]+", "label": None, "strict": None},
                seed=-1,
                logit_bias={"123": 1.5},
                routing_key="tenant-7",
                prompt_cache_key="prompt-9",
                safety_identifier="safety-3",
                lora_adapter="sql-adapter",
                options={"overall_timeout_s": 30},
            )
            call = mock_client.return_value.post.call_args
            assert call.args[0] == "/v1/generate/m"
            sent = json.loads(call.kwargs["content"].decode("utf-8"))
            assert sent == {
                "prompt": "Hi",
                "max_new_tokens": 8,
                "temperature": 1.0,
                "options": {"first_chunk_timeout_s": 10, "overall_timeout_s": 30},
                "top_p": 1.0,
                "stop": ["</s>"],
                "frequency_penalty": 0.25,
                "presence_penalty": -0.5,
                "grammar": {"regex": "[a-z]+", "label": None, "strict": None},
                "seed": -1,
                "logit_bias": {"123": 1.5},
                "routing_key": "tenant-7",
                "prompt_cache_key": "prompt-9",
                "safety_identifier": "safety-3",
                "lora_adapter": "sql-adapter",
            }
            assert call.kwargs["headers"]["content-type"] == "application/json"
            assert call.kwargs["headers"]["accept"] == "application/json"
            client.close()

    def test_generate_normalizes_hf_model_id_to_sie_safe_path(self) -> None:
        envelope = {
            "model": "Qwen__Qwen3-4B-Instruct-2507",
            "text": "x",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "attempt_id": "a",
        }
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = _ok_response(envelope)
            client = SIEClient("http://localhost:8080")
            client.generate("Qwen/Qwen3-4B-Instruct-2507", prompt="Hi", max_new_tokens=8)
            call = mock_client.return_value.post.call_args
            assert call.args[0] == "/v1/generate/Qwen__Qwen3-4B-Instruct-2507"
            client.close()

    def test_generate_503_model_loading_retries(self) -> None:
        # First response: 503 MODEL_LOADING; second: 200 envelope.
        loading_response = MagicMock()
        loading_response.status_code = 503
        loading_response.headers = {
            "X-SIE-Error-Code": "MODEL_LOADING",
            "X-SIE-Request-ID": "retry-intermediate",
            "X-SIE-Credits-Debited": "999",
        }
        loading_response.content = b'{"error":{"code":"MODEL_LOADING","message":"loading"}}'
        loading_response.json.return_value = {"error": {"code": "MODEL_LOADING", "message": "loading"}}

        ok = _ok_response(
            {
                "model": "m",
                "text": "ok",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "attempt_id": "a",
            },
            {
                "X-SIE-Request-ID": "retry-terminal",
                "X-SIE-Credits-Debited": "3",
            },
        )
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client, patch("sie_sdk.client.sync.time.sleep"):
            mock_client.return_value.post.side_effect = [loading_response, ok]
            client = SIEClient("http://localhost:8080")
            result = client.generate("m", prompt="hi", max_new_tokens=8, provision_timeout_s=10)
            assert result["text"] == "ok"
            assert result["request"] == {"id": "retry-terminal", "credits_debited": 3}
            assert mock_client.return_value.post.call_count == 2
            assert client.last_retry_count == 1

            mock_client.return_value.post.side_effect = None
            mock_client.return_value.post.return_value = ok
            client.generate("m", prompt="again", max_new_tokens=8)
            assert client.last_retry_count == 0
            client.close()

    def test_generate_non_dict_response_raises_request_error(self) -> None:
        response = MagicMock()
        response.status_code = 200
        response.headers = {
            "X-SIE-Request-ID": "req-generate-malformed",
            "X-SIE-Units-Output-Tokens": "3",
            "X-SIE-Credits-Debited": "8",
        }
        response.content = b'"not a dict"'
        response.json.return_value = "not a dict"
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(RequestError) as excinfo:
                client.generate("m", prompt="hi", max_new_tokens=4)
            assert excinfo.value.request == {
                "id": "req-generate-malformed",
                "usage": {"output_tokens": 3},
                "credits_debited": 8,
            }
            client.close()

    @pytest.mark.parametrize("body", [{"text": "ok"}, {"model": "m"}])
    def test_generate_invalid_object_retains_request_metadata(self, body: dict) -> None:
        response = _ok_response(
            body,
            {
                "X-SIE-Request-ID": "req-generate-invalid-object",
                "X-SIE-Units-Output-Tokens": "3",
                "X-SIE-Credits-Debited": "8",
            },
        )
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(RequestError) as excinfo:
                client.generate("m", prompt="hi", max_new_tokens=4)
            assert excinfo.value.request == {
                "id": "req-generate-invalid-object",
                "usage": {"output_tokens": 3},
                "credits_debited": 8,
            }
            client.close()

    def test_generate_does_not_retry_504_and_raises(self) -> None:
        # B1b: a 504 is a post-publish gateway timeout. generate() is
        # non-idempotent (no dedup key), so retrying could double-bill an
        # inference. The SDK must surface a terminal ServerError on the
        # FIRST 504 — exactly one POST, no retry — even with
        # wait_for_capacity=True (which DOES make the idempotent paths retry).
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_504(), _ok_response(_ok_envelope())])
            client = SIEClient("http://localhost:8080")
            with pytest.raises(ServerError) as excinfo:
                client.generate("m", prompt="hi", max_new_tokens=8, wait_for_capacity=True, provision_timeout_s=10)
            # Exactly one attempt; the second (success) response is never reached.
            assert mock_client.return_value.post.call_count == 1
            # No backoff sleep happened for the 504.
            mock_sleep.assert_not_called()
            assert excinfo.value.status_code == 504
            assert excinfo.value.request == {
                "id": "req-generate-timeout",
                "usage": {"output_tokens": 5},
                "credits_debited": 13,
            }
            client.close()

    def test_generate_still_retries_503_provisioning(self) -> None:
        # 503 PROVISIONING is pre-execution (no capacity yet, nothing
        # generated), so it is safe to retry under wait_for_capacity even on
        # the non-idempotent path.
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(
                side_effect=[_resp_503_provisioning(), _ok_response(_ok_envelope())]
            )
            client = SIEClient("http://localhost:8080")
            result = client.generate("m", prompt="hi", max_new_tokens=8, provision_timeout_s=10)
            assert result["text"] == "ok"
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_generate_does_not_retry_mid_flight_transport_error(self) -> None:
        # Guard against regression of the documented non-idempotent behavior:
        # a mid-flight read/write timeout fires after the body was sent, so
        # the worker may already be generating. Surface, do not retry.
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(side_effect=httpx.ReadTimeout("read timed out"))
            client = SIEClient("http://localhost:8080")
            with pytest.raises(SIEConnectionError):
                client.generate("m", prompt="hi", max_new_tokens=8, wait_for_capacity=True, provision_timeout_s=10)
            assert mock_client.return_value.post.call_count == 1
            client.close()


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_generate_serializes_native_images_as_bounded_json_envelopes(self) -> None:
        raw = MagicMock()
        raw.status = 200
        raw.headers = {}
        raw.read = AsyncMock(return_value=json.dumps(_ok_envelope()).encode())
        post_ctx = MagicMock()
        post_ctx.__aenter__ = AsyncMock(return_value=raw)
        post_ctx.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.post = MagicMock(return_value=post_ctx)
        session.close = AsyncMock()

        with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
            client = SIEAsyncClient("http://localhost:8080")
            try:
                await client.generate(
                    "m",
                    prompt="Read",
                    images=[b"\xff\xd8\xff\xe0hello"],
                    max_new_tokens=8,
                )
            finally:
                await client.close()

        sent = json.loads(session.post.call_args.kwargs["data"])
        assert sent["images"] == [{"data": "/9j/4GhlbGxv", "format": "jpeg"}]

    @pytest.mark.asyncio
    async def test_generate_rejects_invalid_grammar_before_request(self) -> None:
        session = MagicMock()
        session.post = MagicMock()
        session.close = AsyncMock()
        with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
            client = SIEAsyncClient("http://localhost:8080")
            try:
                with pytest.raises(ValueError, match="exactly one"):
                    await client.generate(
                        "m",
                        prompt="Hi",
                        grammar={"regex": "x", "ebnf": 'root ::= "x"'},
                        max_new_tokens=8,
                    )
            finally:
                await client.close()
        session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_non_dict_response_retains_request_metadata(self) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            return_value=_aio_resp(
                200,
                ["not a dict"],
                {
                    "X-SIE-Request-ID": "req-async-generate-malformed",
                    "X-SIE-Units-Output-Tokens": "3",
                    "X-SIE-Credits-Debited": "8",
                },
            )
        )
        with patch.object(client, "_ensure_session") as ensure:
            ensure.return_value.post = _make_session_post(client._post)
            try:
                with pytest.raises(RequestError) as excinfo:
                    await client.generate("m", prompt="hi", max_new_tokens=4)
            finally:
                await client.close()
        assert excinfo.value.request == {
            "id": "req-async-generate-malformed",
            "usage": {"output_tokens": 3},
            "credits_debited": 8,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [{"text": "ok"}, {"model": "m"}])
    async def test_generate_invalid_object_retains_request_metadata(self, body: dict) -> None:
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            return_value=_aio_resp(
                200,
                body,
                {
                    "X-SIE-Request-ID": "req-async-generate-invalid-object",
                    "X-SIE-Units-Output-Tokens": "3",
                    "X-SIE-Credits-Debited": "8",
                },
            )
        )
        with patch.object(client, "_ensure_session") as ensure:
            ensure.return_value.post = _make_session_post(client._post)
            try:
                with pytest.raises(RequestError) as excinfo:
                    await client.generate("m", prompt="hi", max_new_tokens=4)
            finally:
                await client.close()
        assert excinfo.value.request == {
            "id": "req-async-generate-invalid-object",
            "usage": {"output_tokens": 3},
            "credits_debited": 8,
        }

    @pytest.mark.asyncio
    async def test_generate_happy_path(self) -> None:
        envelope = {
            "model": "m",
            "text": "Hello!",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "attempt_id": "att-1",
            "ttft_ms": 80.0,
            "tpot_ms": 25.0,
        }

        # Mock the aiohttp session and the post context manager.
        raw = MagicMock()
        raw.status = 200
        raw.headers = {
            "X-SIE-Request-ID": "req-async-generate",
            "X-SIE-Units-Input-Tokens": "1",
            "X-SIE-Units-Output-Tokens": "2",
            "X-SIE-Credits-Debited": "9",
        }
        raw.read = AsyncMock(return_value=json.dumps(envelope).encode("utf-8"))

        post_ctx = MagicMock()
        post_ctx.__aenter__ = AsyncMock(return_value=raw)
        post_ctx.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.post = MagicMock(return_value=post_ctx)
        session.close = AsyncMock()

        with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
            client = SIEAsyncClient("http://localhost:8080", options={"first_chunk_timeout_s": 10})
            try:
                result = await client.generate(
                    "Qwen/Qwen3-4B-Instruct-2507",
                    prompt="Hi",
                    max_new_tokens=16,
                    frequency_penalty=0.25,
                    presence_penalty=-0.5,
                    grammar={"regex": "[a-z]+", "label": None, "strict": None},
                    seed=-(1 << 63),
                    logit_bias={"123": 1.5},
                    routing_key="tenant-7",
                    prompt_cache_key="prompt-9",
                    safety_identifier="safety-3",
                    lora_adapter="sql-adapter",
                    options={"overall_timeout_s": 30},
                )
            finally:
                await client.close()
            assert result["text"] == "Hello!"
            assert result["usage"]["total_tokens"] == 3
            assert result["attempt_id"] == "att-1"
            assert result["ttft_ms"] == 80.0
            assert result["request"] == {
                "id": "req-async-generate",
                "usage": {"input_tokens": 1, "output_tokens": 2},
                "credits_debited": 9,
            }
            call = session.post.call_args
            # Async client normalises HF model IDs to the SIE-safe path
            # (mirrors the sync client) so the gateway route matches.
            assert call.args[0] == "/v1/generate/Qwen__Qwen3-4B-Instruct-2507"
            sent = json.loads(call.kwargs["data"].decode("utf-8"))
            assert sent["seed"] == -(1 << 63)
            assert sent["frequency_penalty"] == 0.25
            assert sent["presence_penalty"] == -0.5
            assert sent["grammar"] == {"regex": "[a-z]+", "label": None, "strict": None}
            assert sent["logit_bias"] == {"123": 1.5}
            assert sent["routing_key"] == "tenant-7"
            assert sent["prompt_cache_key"] == "prompt-9"
            assert sent["safety_identifier"] == "safety-3"
            assert sent["lora_adapter"] == "sql-adapter"
            assert call.kwargs["headers"]["content-type"] == "application/json"
            assert sent["options"] == {"first_chunk_timeout_s": 10, "overall_timeout_s": 30}
            assert "temperature" not in sent
            assert "top_p" not in sent
            assert call.kwargs["headers"]["accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_generate_does_not_retry_504_and_raises(self) -> None:
        # B1b (async): 504 is post-publish; the non-idempotent generate path
        # must surface a terminal ServerError on the first 504, no retry,
        # even with wait_for_capacity=True.
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                _aio_resp(
                    504,
                    {"error": {"code": "MODEL_LOADING", "message": "timeout"}},
                    {
                        "Retry-After": "0.01",
                        "content-type": "application/json",
                        "X-SIE-Request-ID": "req-async-generate-timeout",
                        "X-SIE-Units-Output-Tokens": "5",
                        "X-SIE-Credits-Debited": "13",
                    },
                ),
                _aio_resp(200, _ok_envelope()),
            ]
        )
        # The async generate path posts inline via _ensure_session(), so mock
        # that to delegate to client._post for the canned sequence.
        with patch.object(client, "_ensure_session") as ensure:
            ensure.return_value.post = _make_session_post(client._post)
            with patch("sie_sdk.client.async_.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                try:
                    with pytest.raises(ServerError) as excinfo:
                        await client.generate(
                            "m", prompt="hi", max_new_tokens=8, wait_for_capacity=True, provision_timeout_s=10
                        )
                finally:
                    await client.close()
        assert client._post.call_count == 1
        mock_sleep.assert_not_called()
        assert excinfo.value.status_code == 504
        assert excinfo.value.request == {
            "id": "req-async-generate-timeout",
            "usage": {"output_tokens": 5},
            "credits_debited": 13,
        }

    @pytest.mark.asyncio
    async def test_generate_still_retries_503_model_loading(self) -> None:
        # 503 MODEL_LOADING is pre-execution (cold-start) and must keep retrying.
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                _aio_resp(
                    503,
                    {"error": {"code": "MODEL_LOADING", "message": "loading"}},
                    {
                        "X-SIE-Error-Code": "MODEL_LOADING",
                        "X-SIE-Request-ID": "async-retry-intermediate",
                        "X-SIE-Credits-Debited": "999",
                    },
                ),
                _aio_resp(
                    200,
                    _ok_envelope(),
                    {
                        "X-SIE-Request-ID": "async-retry-terminal",
                        "X-SIE-Credits-Debited": "3",
                    },
                ),
            ]
        )
        with patch.object(client, "_ensure_session") as ensure:
            ensure.return_value.post = _make_session_post(client._post)
            with patch("sie_sdk.client.async_.asyncio.sleep", new_callable=AsyncMock):
                try:
                    result = await client.generate("m", prompt="hi", max_new_tokens=8, provision_timeout_s=10)
                finally:
                    await client.close()
        assert result["text"] == "ok"
        assert result["request"] == {"id": "async-retry-terminal", "credits_debited": 3}
        assert client._post.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_does_not_retry_mid_flight_transport_error(self) -> None:
        # generate() is non-idempotent: a mid-flight transport error (here a
        # generic OSError / ECONNRESET) may fire after the worker already
        # started generating, so retrying would double-bill an inference.
        # The SDK must surface the error instead of silently re-running.
        session = MagicMock()
        session.post = MagicMock(side_effect=OSError("connection reset"))
        session.close = AsyncMock()

        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session),
            patch("sie_sdk.client.async_.asyncio.sleep", new_callable=AsyncMock),
        ):
            client = SIEAsyncClient("http://localhost:8080")
            try:
                with pytest.raises(SIEConnectionError):
                    await client.generate("m", prompt="Hi", max_new_tokens=16, provision_timeout_s=10)
            finally:
                await client.close()

        # Exactly one attempt — no retry.
        assert session.post.call_count == 1


class TestParseGenerateResultUsageRobustness:
    """Fix 2: a non-numeric usage value must not raise an un-wrapped
    ``ValueError`` / ``TypeError`` outside the parser's ``RequestError``
    contract. It degrades to 0 (the field is optional/auxiliary), mirroring
    how the parser already silently skips a non-numeric ``ttft_ms``.
    """

    @staticmethod
    def _envelope_with_usage(usage: object) -> dict:
        return {"model": "m", "text": "ok", "usage": usage}

    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            ({"prompt_tokens": "n/a", "completion_tokens": None, "total_tokens": [1, 2]}, (0, 0, 0)),
            ({"prompt_tokens": 5, "completion_tokens": 3.0, "total_tokens": 8}, (5, 3, 8)),
            ({}, (0, 0, 0)),
            ({"prompt_tokens": True}, (1, 0, 0)),  # bool is an int subclass
        ],
    )
    def test_sync_parser_coerces_non_numeric_usage(self, usage: dict, expected: tuple[int, int, int]) -> None:
        from sie_sdk.client.sync import _parse_generate_result

        result = _parse_generate_result(self._envelope_with_usage(usage))
        assert result["usage"]["prompt_tokens"] == expected[0]
        assert result["usage"]["completion_tokens"] == expected[1]
        assert result["usage"]["total_tokens"] == expected[2]

    def test_async_parser_coerces_non_numeric_usage(self) -> None:
        from sie_sdk.client.async_ import _parse_generate_result_async

        result = _parse_generate_result_async(
            self._envelope_with_usage({"prompt_tokens": "x", "completion_tokens": object(), "total_tokens": None})
        )
        assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_non_numeric_usage_does_not_raise_through_generate(self) -> None:
        # End-to-end: a malformed usage block previously crashed generate()
        # with an un-wrapped ValueError. It must now parse cleanly.
        envelope = {"model": "m", "text": "ok", "usage": {"prompt_tokens": "garbage"}}
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = _ok_response(envelope)
            client = SIEClient("http://localhost:8080")
            result = client.generate("m", prompt="hi", max_new_tokens=4)
            assert result["usage"]["prompt_tokens"] == 0
            client.close()

    def test_non_finite_usage_does_not_raise_through_sync_generate(self) -> None:
        # BUG 11: non-finite usage tokens (NaN / Infinity) used to escape the
        # parser's RequestError-only contract: int(nan) -> ValueError,
        # int(inf) -> OverflowError. They must now degrade to 0 and NOT raise.
        # JSON itself has no NaN/Infinity literals, so the gateway delivers
        # them as Python floats post-deserialization; build the dict directly.
        envelope = {
            "model": "m",
            "text": "ok",
            "usage": {
                "prompt_tokens": float("nan"),
                "completion_tokens": float("inf"),
                "total_tokens": 3,
            },
        }
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.content = b"{}"
        response.json.return_value = envelope
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = response
            client = SIEClient("http://localhost:8080")
            result = client.generate("m", prompt="hi", max_new_tokens=4)
            assert result["usage"]["prompt_tokens"] == 0
            assert result["usage"]["completion_tokens"] == 0
            assert result["usage"]["total_tokens"] == 3
            client.close()

    @pytest.mark.asyncio
    async def test_non_finite_usage_does_not_raise_through_async_generate(self) -> None:
        # BUG 11 (async): same non-finite usage must degrade to 0 end-to-end.
        envelope = {
            "model": "m",
            "text": "ok",
            "usage": {
                "prompt_tokens": float("nan"),
                "completion_tokens": float("inf"),
                "total_tokens": 3,
            },
        }
        raw = MagicMock()
        raw.status = 200
        raw.headers = {}
        # _AioResponse.json() re-parses content; std json.dumps cannot encode
        # NaN/inf as valid JSON, so emit the non-strict literals the worker's
        # serializer would and let json.loads (which accepts them) parse back.
        raw.read = AsyncMock(return_value=json.dumps(envelope).encode("utf-8"))

        post_ctx = MagicMock()
        post_ctx.__aenter__ = AsyncMock(return_value=raw)
        post_ctx.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.post = MagicMock(return_value=post_ctx)
        session.close = AsyncMock()

        with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
            client = SIEAsyncClient("http://localhost:8080")
            try:
                result = await client.generate("m", prompt="hi", max_new_tokens=4)
            finally:
                await client.close()
            assert result["usage"]["prompt_tokens"] == 0
            assert result["usage"]["completion_tokens"] == 0
            assert result["usage"]["total_tokens"] == 3


class TestParseGenerateResultStrictContract:
    """Guard: the strict ``model`` / ``text`` contract must NOT be loosened
    by the usage-robustness change — both still raise ``RequestError`` when
    missing or non-string.
    """

    @pytest.mark.parametrize(
        "envelope",
        [
            {"text": "ok"},  # missing model
            {"model": None, "text": "ok"},  # null model
            {"model": 123, "text": "ok"},  # non-string model
            {"model": "m"},  # missing text
            {"model": "m", "text": None},  # null text
            {"model": "m", "text": 5},  # non-string text
        ],
    )
    def test_sync_parser_raises_on_missing_or_non_string_required_fields(self, envelope: dict) -> None:
        from sie_sdk.client.sync import _parse_generate_result

        with pytest.raises(RequestError):
            _parse_generate_result(envelope)

    @pytest.mark.parametrize(
        "envelope",
        [
            {"text": "ok"},
            {"model": 123, "text": "ok"},
            {"model": "m", "text": None},
        ],
    )
    def test_async_parser_raises_on_missing_or_non_string_required_fields(self, envelope: dict) -> None:
        from sie_sdk.client.async_ import _parse_generate_result_async

        with pytest.raises(RequestError):
            _parse_generate_result_async(envelope)
