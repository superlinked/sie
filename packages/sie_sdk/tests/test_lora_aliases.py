"""Real SDK request builders preserve selection and reject ambiguity before HTTP."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk._msgpack import unpackb
from sie_sdk.client._lora import LORA_UNSET, lora_selection, merge_runtime_lora_options


class CapturedRequestError(Exception):
    """The fake transport records the completed request without contacting a server."""


def native_call(client, kind, options):
    if kind == "score":
        return client.score("m", {"text": "query"}, [{"text": "item"}], options=options)
    if kind == "system_one":
        return client.system_one("m", "state", {"yes": {"type": "noul"}}, options=options)
    return getattr(client, kind)("m", {"text": "item"}, options=options)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["encode", "score", "extract", "system_one"])
@pytest.mark.parametrize("selector", ["lora_adapter", "lora", "lora_id"])
async def test_native_request_replaces_default_selector_without_mutating_options(asynchronous, kind, selector):
    defaults = {"lora": "default", "normalize": True}
    options = {selector: None}
    client = (
        SIEAsyncClient("http://localhost:8080", options=defaults)
        if asynchronous
        else SIEClient("http://localhost:8080", options=defaults)
    )
    post = AsyncMock(side_effect=CapturedRequestError) if asynchronous else MagicMock(side_effect=CapturedRequestError)
    if asynchronous:
        client._post = post
    else:
        client._client.post = post
    try:
        with pytest.raises(CapturedRequestError):
            result = native_call(client, kind, options)
            if asynchronous:
                await result
        sent = unpackb(post.call_args.kwargs["data" if asynchronous else "content"], numeric_arrays=False)
        forwarded = sent["options"] if kind == "score" else sent["params"]["options"]
        assert forwarded[selector] is None
        assert set(forwarded) & {"lora_adapter", "lora", "lora_id"} == {selector}
        assert forwarded["normalize"] is True
        assert defaults == {"lora": "default", "normalize": True}
        assert options == {selector: None}
        assert post.call_count == 1
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["encode", "score", "extract", "system_one"])
@pytest.mark.parametrize(
    "options",
    [
        {"lora_adapter": None, "lora": "legal"},
        {"lora": "legal", "lora_id": "other"},
        {"lora_adapter": False},
        {"lora_adapter": " "},
    ],
)
async def test_native_invalid_selection_never_reaches_http(asynchronous, kind, options):
    client = SIEAsyncClient("http://localhost:8080") if asynchronous else SIEClient("http://localhost:8080")
    post = (
        AsyncMock(side_effect=AssertionError("HTTP must not run"))
        if asynchronous
        else MagicMock(side_effect=AssertionError("HTTP must not run"))
    )
    if asynchronous:
        client._post = post
    else:
        client._client.post = post
    try:
        with pytest.raises(ValueError, match="must"):
            result = native_call(client, kind, options)
            if asynchronous:
                await result
        post.assert_not_called()
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


def generation_call(client, kind, kwargs):
    if "chat" in kind:
        return getattr(client, kind)("m", [{"role": "user", "content": "hi"}], **kwargs)
    return getattr(client, kind)("m", prompt="hi", max_new_tokens=1, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["generate", "stream_generate", "chat_completions", "stream_chat_completions"])
@pytest.mark.parametrize("kwargs", [{}, {"lora_adapter": None}, {"lora_adapter": "legal"}])
async def test_generation_preserves_canonical_omission_or_null(asynchronous, kind, kwargs, monkeypatch):
    client = SIEAsyncClient("http://localhost:8080") if asynchronous else SIEClient("http://localhost:8080")
    transport = MagicMock()
    transport.post.side_effect = CapturedRequestError
    transport.stream.side_effect = CapturedRequestError
    if asynchronous:
        monkeypatch.setattr(client, "_ensure_session", lambda: transport)
        client._post = AsyncMock(side_effect=CapturedRequestError)
    else:
        monkeypatch.setattr(client._client, "post", transport.post)
        monkeypatch.setattr(client._client, "stream", transport.stream)
    try:
        with pytest.raises(CapturedRequestError):
            result = generation_call(client, kind, kwargs)
            if kind.startswith("stream"):
                if asynchronous:
                    await anext(result)
                else:
                    next(result)
            elif asynchronous:
                await result
        calls = [mock for mock in (transport.post, transport.stream) if mock.called]
        if asynchronous and client._post.called:
            calls.append(client._post)
        assert len(calls) == 1
        call = calls[0].call_args
        raw = call.kwargs.get("content", call.kwargs.get("data"))
        body = json.loads(raw)
        assert {key: body[key] for key in ("lora_adapter", "lora", "lora_id") if key in body} == kwargs
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["stream_generate", "chat_completions", "stream_chat_completions"])
@pytest.mark.parametrize("extra_body", [{"lora_adapter": "other"}, {"lora": "other"}, {"lora_id": None}])
async def test_extra_body_cannot_silently_replace_typed_selection(asynchronous, kind, extra_body):
    client = SIEAsyncClient("http://localhost:8080") if asynchronous else SIEClient("http://localhost:8080")
    try:
        with pytest.raises(ValueError, match="conflicts"):
            result = generation_call(client, kind, {"lora_adapter": "legal", "extra_body": extra_body})
            if kind.startswith("stream"):
                if asynchronous:
                    await anext(result)
                else:
                    next(result)
            elif asynchronous:
                await result
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


def test_omission_and_identical_aliases_remain_distinct_from_null():
    assert lora_selection({}) is LORA_UNSET
    assert lora_selection({"lora_adapter": None, "lora": None}) is None
    assert lora_selection({"lora_adapter": "legal", "lora_id": "legal"}) == "legal"
    assert merge_runtime_lora_options({"lora_id": "inherited"}, {}) == {"lora_id": "inherited"}
