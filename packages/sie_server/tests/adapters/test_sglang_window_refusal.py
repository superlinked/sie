"""A request that cannot fit the model's window is a client error on every SGLang path."""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from sie_server.adapters._generation_base import (
    GenerationError,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
)
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.sglang._window import SGLangInputTooLongError, window_refusal_message
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import (
    SGLangGenerationAdapter,
    _raise_for_sglang_event_error,
    _raise_for_sglang_http_error,
)
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.queue_executor import _inference_exception_outcome
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput, InvalidInputError, Item

# The engine's four texts, with the tokens and limit each should report.
REFUSALS = [
    pytest.param(
        "The input (30042 tokens) is longer than the model's context length (24576 tokens).",
        "30042",
        "24576",
        id="tokenizer-input",
    ),
    pytest.param(
        "Requested token count exceeds the model's maximum context length of 8192 tokens. "
        "You requested a total of 8192 tokens: 8176 tokens from the input messages and 16 tokens for the "
        "completion. Please reduce the number of tokens in the input messages or the completion to fit within "
        "the limit.",
        "8192",
        "8192",
        id="tokenizer-total",
    ),
    pytest.param(
        "Input length (8189 tokens) exceeds the maximum allowed length (8186 tokens). "
        "Use a shorter input or enable --allow-auto-truncate.",
        "8189",
        "8186",
        id="scheduler-input",
    ),
    pytest.param(
        "Multimodal prompt is too long after expanding multimodal tokens. "
        "After expanding len(req.origin_input_ids_unpadded)=71 => 16454 >= 16378.",
        "16454",
        "16378",
        id="scheduler-multimodal",
    ),
]
OTHER_MESSAGES = [
    "vision processor rejected image",
    "The input (12 tokens) is longer than the model's context length (10 tokens). secret",
    "secret The input (12 tokens) is longer than the model's context length (10 tokens).",
    "The input (twelve tokens) is longer than the model's context length (10 tokens).",
    ["The input (12 tokens) is longer than the model's context length (10 tokens)."],
    None,
]
ONE_REFUSAL = "The input (30042 tokens) is longer than the model's context length (24576 tokens)."
REGEX_GRAMMAR = GrammarSpec(kind="regex", value="[a-z]+")


def _expected(tokens: str, limit: str) -> str:
    return f"The request needs {tokens} tokens, at or over this model's limit of {limit}"


@pytest.mark.parametrize(("message", "tokens", "limit"), REFUSALS)
def test_engine_refusals_become_fixed_text(message: str, tokens: str, limit: str) -> None:
    assert window_refusal_message(message) == _expected(tokens, limit)


@pytest.mark.parametrize("message", OTHER_MESSAGES)
def test_other_engine_messages_are_not_window_refusals(message: object) -> None:
    assert window_refusal_message(message) is None


# Generation: where the engine reports the refusal depends on where it noticed.


@pytest.mark.parametrize(("message", "tokens", "limit"), REFUSALS)
def test_in_band_refusal_is_input_too_long(message: str, tokens: str, limit: str) -> None:
    with pytest.raises(GenerationInputTooLongError) as error:
        _raise_for_sglang_event_error({"error": {"message": message}})
    assert error.value.code == "INPUT_TOO_LONG"
    assert str(error.value) == _expected(tokens, limit)


def test_in_band_refusal_is_recognised_in_the_newer_engine_event() -> None:
    # SGLang 0.5.20 reports the same text with these extra fields.
    event = {
        "error": {"message": ONE_REFUSAL, "type": "invalid_request_error", "code": 400, "retryable": False},
    }
    with pytest.raises(GenerationInputTooLongError) as error:
        _raise_for_sglang_event_error(event)
    assert str(error.value) == _expected("30042", "24576")


@pytest.mark.parametrize("grammar", [None, REGEX_GRAMMAR], ids=["no-grammar", "grammar"])
@pytest.mark.parametrize(("message", "tokens", "limit"), REFUSALS)
def test_abort_refusal_is_input_too_long_even_with_a_grammar(
    message: str, tokens: str, limit: str, grammar: GrammarSpec | None
) -> None:
    event = {"meta_info": {"finish_reason": {"type": "abort", "status_code": 400, "message": message}}}
    with pytest.raises(GenerationInputTooLongError) as error:
        _raise_for_sglang_event_error(event, grammar=grammar)
    assert str(error.value) == _expected(tokens, limit)


@pytest.mark.asyncio
@pytest.mark.parametrize(("message", "tokens", "limit"), REFUSALS)
async def test_http_400_refusal_is_input_too_long(message: str, tokens: str, limit: str) -> None:
    response = httpx.Response(
        400, json={"error": {"message": message}}, request=httpx.Request("POST", "http://localhost/generate")
    )
    with pytest.raises(GenerationInputTooLongError) as error:
        await _raise_for_sglang_http_error(response, grammar=None, grammar_backend=None)
    assert str(error.value) == _expected(tokens, limit)


@pytest.mark.parametrize("message", OTHER_MESSAGES)
def test_other_in_band_errors_stay_engine_failures(message: object) -> None:
    with pytest.raises(GenerationError) as error:
        _raise_for_sglang_event_error({"error": {"message": message}})
    assert not isinstance(error.value, GenerationInputTooLongError)
    assert error.value.code == "inference_error"
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_a_refusal_text_on_an_engine_failure_status_stays_a_failure() -> None:
    response = httpx.Response(
        500, json={"error": {"message": ONE_REFUSAL}}, request=httpx.Request("POST", "http://localhost/generate")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await _raise_for_sglang_http_error(response, grammar=None, grammar_backend=None)

    event = {"meta_info": {"finish_reason": {"type": "abort", "status_code": 500, "message": ONE_REFUSAL}}}
    with pytest.raises(GenerationError) as error:
        _raise_for_sglang_event_error(event)
    assert not isinstance(error.value, GenerationInputTooLongError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("n", "stream", "best_of"), [(1, False, None), (2, True, None), (2, False, None), (1, False, 2)]
)
async def test_generate_reports_a_refusal_on_every_request_shape(n: int, stream: bool, best_of: int | None) -> None:
    adapter = SGLangGenerationAdapter(model_name_or_path="org/model", max_seq_length=8192, mem_fraction_static=0.85)

    def respond(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["stream"]:
            line = json.dumps({"error": {"message": ONE_REFUSAL}})
            return httpx.Response(200, content=f"data: {line}\n\ndata: [DONE]\n\n".encode())
        return httpx.Response(400, json={"error": {"message": ONE_REFUSAL}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter._http_client = client
        adapter._server_url = "http://localhost:30005"
        with pytest.raises(GenerationInputTooLongError):
            async for _ in adapter.generate(prompt="word", max_new_tokens=8, n=n, stream=stream, best_of=best_of):
                pass


# Extract: the same refusal comes through a generation that extract callers do not know.


class _Processor:
    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "rendered prompt"


def _engine_stream(event: dict[str, Any]) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _: httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode()))


@pytest.fixture
def extract_adapter() -> Iterator[SGLangVisionExtractAdapter]:
    adapter = SGLangVisionExtractAdapter("org/ocr", max_concurrent_requests=2)
    adapter._processor = _Processor()
    adapter._server_url = "http://sglang.test"
    adapter._start_request_loop()
    yield adapter
    adapter.unload()


def _page() -> Item:
    return Item(images=[ImageInput(data=b"page", format="png")])


@pytest.mark.parametrize(
    "event",
    [
        pytest.param({"error": {"message": ONE_REFUSAL}}, id="in-band"),
        pytest.param(
            {"meta_info": {"finish_reason": {"type": "abort", "status_code": 400, "message": ONE_REFUSAL}}},
            id="abort",
        ),
    ],
)
def test_extract_reports_a_refusal_as_input_too_long(
    extract_adapter: SGLangVisionExtractAdapter, event: dict[str, Any]
) -> None:
    extract_adapter._http_client = httpx.AsyncClient(transport=_engine_stream(event))

    with pytest.raises(InputTooLongError) as error:
        extract_adapter.extract([_page(), _page()])

    assert str(error.value) == _expected("30042", "24576")
    # Also an input error, so a batch shared with other requests isolates this one.
    assert isinstance(error.value, InvalidInputError)


def test_extract_leaves_other_engine_failures_alone(extract_adapter: SGLangVisionExtractAdapter) -> None:
    event = {"error": {"message": "vision processor rejected image"}}
    extract_adapter._http_client = httpx.AsyncClient(transport=_engine_stream(event))

    with pytest.raises(GenerationError) as error:
        extract_adapter.extract([_page()])

    assert not isinstance(error.value, GenerationInvalidRequestError)
    assert not isinstance(error.value, ValueError)


def test_extract_refusal_keeps_the_input_too_long_code_on_the_queue_path() -> None:
    batch_item: Any = SimpleNamespace(work_item_id="w", request_id="r", item_index=0)
    outcome = _inference_exception_outcome(batch_item, SGLangInputTooLongError(_expected("30042", "24576")))

    assert outcome.error_code == "INPUT_TOO_LONG"
    assert outcome.error == _expected("30042", "24576")


# Encode: SGLang's OpenAI surface returns the same text in its own error body.


def _embedding_refusal(status_code: int = 400, message: object = ONE_REFUSAL) -> MagicMock:
    response = MagicMock(status_code=status_code, text="engine error body")
    response.json.return_value = {
        "object": "error",
        "message": message,
        "type": "BadRequestError",
        "param": None,
        "code": status_code,
    }
    response.raise_for_status.side_effect = RuntimeError(f"status {status_code}")
    return response


def test_encode_reports_a_refusal_as_input_too_long() -> None:
    adapter = SGLangEmbeddingAdapter("org/embedding", normalize=False, embed_concurrency=1)
    adapter._server_url = "http://localhost:30000"

    with patch("sie_server.adapters.sglang.embedding.requests.post", return_value=_embedding_refusal()):
        with pytest.raises(InputTooLongError) as error:
            adapter.encode([Item(text="hello"), Item(text="world")], output_types=["dense"])

    assert str(error.value) == _expected("30042", "24576")
    # Also an input error, so a batch shared with other requests isolates this one.
    assert isinstance(error.value, InvalidInputError)


def test_encode_reports_a_refusal_from_a_concurrent_shard() -> None:
    adapter = SGLangEmbeddingAdapter("org/embedding", normalize=False, embed_concurrency=2)
    adapter._server_url = "http://localhost:30000"
    adapter._session = MagicMock()
    adapter._session.post.return_value = _embedding_refusal()

    with ThreadPoolExecutor(2) as executor:
        adapter._post_executor = executor
        with pytest.raises(InputTooLongError):
            adapter.encode([Item(text="hello"), Item(text="world")], output_types=["dense"])

    assert adapter._session.post.call_count == 2


@pytest.mark.parametrize(
    ("status_code", "message"),
    [(400, "unknown model"), (400, ["not text"]), (500, ONE_REFUSAL)],
)
def test_encode_leaves_other_engine_errors_alone(status_code: int, message: object) -> None:
    adapter = SGLangEmbeddingAdapter("org/embedding", normalize=False, embed_concurrency=1)
    adapter._server_url = "http://localhost:30000"
    refused = _embedding_refusal(status_code, message)

    with patch("sie_server.adapters.sglang.embedding.requests.post", return_value=refused):
        with pytest.raises(RuntimeError, match=f"status {status_code}"):
            adapter.encode([Item(text="hello")], output_types=["dense"])


def test_encode_tolerates_an_engine_error_that_is_not_json() -> None:
    adapter = SGLangEmbeddingAdapter("org/embedding", normalize=False, embed_concurrency=1)
    adapter._server_url = "http://localhost:30000"
    refused = _embedding_refusal()
    refused.json.side_effect = ValueError("not json")

    with patch("sie_server.adapters.sglang.embedding.requests.post", return_value=refused):
        with pytest.raises(RuntimeError, match="status 400"):
            adapter.encode([Item(text="hello")], output_types=["dense"])
