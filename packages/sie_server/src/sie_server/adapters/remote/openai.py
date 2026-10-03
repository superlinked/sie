"""Remote adapter for an upstream of kind ``openai``: an OpenAI-compatible endpoint.

The profile names an upstream from the server's startup configuration and the
model id the upstream serves. ``encode`` sends text items to the upstream's
``/embeddings`` and returns its dense vectors. ``score`` sends a request's query
and documents to the upstream's Cohere-shape ``/rerank`` and returns the scores
in document order. Generation sends a raw prompt to ``/completions`` or a
validated message list to ``/chat/completions``. Loading makes no outbound call,
holds no weights and uses no accelerator.

This server builds every body it sends, so only validated contract fields
reach the upstream and caller credentials never do. The upstream's operator-set fields are added and its
operator-stripped fields removed, and the only credential sent is the
upstream's own. A request the upstream cannot serve is refused before anything
is sent: an output other than dense vectors, an item that is not plain text, an
extraction, or an operation whose endpoint the upstream does not declare.

Usage is the upstream's own count. One upstream request is sent for each item
encoded and one for each API request scored, as
:mod:`sie_server.adapters.remote._batching` describes, so each count is exact
for the work it is attributed to.

The upstream is outside this deployment, so its answer is treated as untrusted
by the rules in :mod:`sie_server.adapters.remote._http`.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import numpy as np

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationChunk,
    GenerationInputTooLongError,
    GenerationInvalidRequestError,
    GenerationPreflightResult,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters._utils import extract_texts
from sie_server.adapters.errors import InputTooLongError, UpstreamUnavailableError
from sie_server.adapters.remote._batching import (
    RequestScores,
    UpstreamUsage,
    call_each,
    requests_in_flight,
    score_each_request,
)
from sie_server.adapters.remote._chat_transport import chat_completion, chat_completion_stream
from sie_server.adapters.remote._http import RemoteUpstreamError, generation_error, open_stream, send_bounded, sse_data
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.adapters.remote._openai_completions import CompletionStreamParser
from sie_server.config.upstreams import (
    Upstream,
    UpstreamConfigError,
    UpstreamEndpoint,
    UpstreamKind,
    upstream_for_serving,
)
from sie_server.core.inference_output import EncodeOutput, ScoreOutput
from sie_server.core.upstream_client import upstream_client, upstream_sync_client
from sie_server.types.inputs import InvalidInputError

if TYPE_CHECKING:
    from sie_server.core.inference_output import ExtractOutput
    from sie_server.types.grammar import GrammarSpec
    from sie_server.types.inputs import ImageInput, Item, VideoInput

_JSON = "application/json"
REQUEST_DEADLINE_S = 60.0
READ_TIMEOUT_S = 10.0
_CONNECT_TIMEOUT_S = 5.0
_RESPONSE_OVERHEAD_BYTES = 1 << 20
_BYTES_PER_JSON_NUMBER = 32
_LARGEST_RESPONSE_BYTES = 64 << 20
_BYTES_PER_RERANK_RESULT = 256
_ESCAPED_ECHO_FACTOR = 6
_MAX_REPORTED_COUNT = 1 << 32
_USAGE_KEYS = ("prompt_tokens", "total_tokens")


class OpenAIUpstreamAdapter(BaseAdapter, GenerationAdapter):
    """Serve encode, score and generation from an OpenAI-compatible upstream.

    The outputs this adapter cannot produce are declared so that a model with
    them still loads; a request for one is refused when it arrives.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("dense", "sparse", "multivector", "score", "json", "tokens"),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        upstream: str,
        upstream_model: str,
        dense_dim: int | None = None,
        max_seq_length: int | None = None,
        compute_precision: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, max_seq_length, compute_precision, kwargs)
        self._upstream_name = upstream
        self._upstream_model = upstream_model
        self._dense_dim = dense_dim
        self._upstream: Upstream | None = None
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._closing: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        upstream = upstream_for_serving(self._upstream_name)
        if upstream.kind is not UpstreamKind.OPENAI:
            raise UpstreamConfigError(f"upstream {self._upstream_name!r} is not of kind 'openai'")
        self._upstream = upstream
        self._client = upstream_sync_client(upstream)
        self._executor = ThreadPoolExecutor(
            max_workers=requests_in_flight(upstream), thread_name_prefix=f"openai-upstream-{self._upstream_name}"
        )
        self._device = device

    def unload(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        client, self._client = self._client, None
        if client is not None:
            client.close()
        if self._async_client is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                async_client, self._async_client = self._async_client, None
                self._closing = loop.create_task(async_client.aclose())
        self._upstream = None
        super().unload()

    @property
    def upstream_name(self) -> str:
        return self._upstream_name

    @property
    def supports_raw_completions(self) -> bool:
        """Whether this deployment accepts locally rendered chat prompts."""
        return self._upstream is not None and UpstreamEndpoint.COMPLETIONS in self._upstream.endpoints

    async def aclose_client(self) -> None:
        """Close the loop-bound generation client during awaitable registry teardown."""
        if self._closing is not None:
            await self._closing
            self._closing = None
        client, self._async_client = self._async_client, None
        if client is not None:
            await client.aclose()

    def _generation_upstream(self, endpoint: UpstreamEndpoint) -> Upstream:
        if self._upstream is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if endpoint not in self._upstream.endpoints:
            raise GenerationUnsupportedFieldError(
                "messages" if endpoint is UpstreamEndpoint.CHAT else "prompt",
                "the upstream does not declare the required generation endpoint",
            )
        return self._upstream

    def _generation_client(self) -> httpx.AsyncClient:
        if self._upstream is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if self._async_client is None:
            self._async_client = upstream_client(self._upstream)
        return self._async_client

    def _generation_request(
        self, body: dict[str, Any], *, chat: bool, stream: bool
    ) -> tuple[httpx.AsyncClient, httpx.Request]:
        upstream = self._generation_upstream(UpstreamEndpoint.CHAT if chat else UpstreamEndpoint.COMPLETIONS)
        sent = upstream.apply_params({**body, "model": self._upstream_model, "stream": stream})
        if not chat:
            sent["echo"] = False
            if sent.get("best_of", 1) != 1:
                raise GenerationUnsupportedFieldError("best_of", "raw remote generation accepts one candidate only")
        # Both aliases must obey the caller ceiling, even when an operator
        # introduces the alias absent from the validated caller body.
        limits = [body[field] for field in ("max_tokens", "max_completion_tokens") if field in body]
        if limits:
            ceiling = min(limits)
            caps = {}
            for field in ("max_tokens", "max_completion_tokens"):
                if field in body or field in sent:
                    cap = sent.get(field, body.get(field, ceiling))
                    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
                        raise GenerationInvalidRequestError(field, "the upstream output limit is invalid")
                    caps[field] = cap
            effective = min(ceiling, *caps.values())
            sent.update(dict.fromkeys(caps, effective))
        if stream:
            sent["stream_options"] = {"include_usage": True}
        else:
            sent.pop("stream_options", None)
        client = self._generation_client()
        return client, client.build_request(
            "POST",
            "/chat/completions" if chat else "/completions",
            json=sent,
            headers={"Accept": "text/event-stream" if stream else "application/json", "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(300.0, connect=_CONNECT_TIMEOUT_S, write=30.0),
        )

    async def chat_completion(
        self, body: dict[str, Any], *, requested_model: str, max_response_bytes: int = 32 << 20
    ) -> dict[str, Any]:
        client, request = self._generation_request(body, chat=True, stream=False)
        return await chat_completion(
            client,
            request,
            upstream=self._upstream_name,
            requested_model=requested_model,
            choices=1 if body.get("n") is None else body["n"],
            max_response_bytes=max_response_bytes,
            error_body_timeout_s=REQUEST_DEADLINE_S,
        )

    def chat_completion_stream(
        self, body: dict[str, Any], *, requested_model: str, max_response_bytes: int = 32 << 20
    ) -> AsyncIterator[dict[str, Any]]:
        client, request = self._generation_request(body, chat=True, stream=True)
        return chat_completion_stream(
            client,
            request,
            upstream=self._upstream_name,
            requested_model=requested_model,
            choices=1 if body.get("n") is None else body["n"],
            max_response_bytes=max_response_bytes,
            error_body_timeout_s=REQUEST_DEADLINE_S,
        )

    def preflight_generate(self, parameters: Mapping[str, Any], *, stream: bool) -> GenerationPreflightResult | None:
        _ = stream
        _refuse_unforwarded(parameters)
        self._generation_upstream(UpstreamEndpoint.COMPLETIONS)
        return None

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
        grammar: GrammarSpec | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        images: list[ImageInput] | None = None,
        videos: list[VideoInput] | None = None,
    ) -> AsyncGenerator[GenerationChunk, None]:
        """Send an already rendered raw prompt; never replay an accepted stream."""
        _refuse_unforwarded(
            {
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "min_new_tokens": min_new_tokens,
                "grammar": grammar,
                "images": images,
                "videos": videos,
            }
        )
        body: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
        }
        optional = {
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "logit_bias": logit_bias,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        if logprobs:
            body["logprobs"] = 1 if top_logprobs is None else top_logprobs
        client, request = self._generation_request(body, chat=False, stream=True)
        parser = CompletionStreamParser(logprobs=logprobs)
        yielded = False
        try:
            async with open_stream(
                client, request, upstream=self._upstream_name, error_body_timeout_s=REQUEST_DEADLINE_S
            ) as response:
                if response.headers.get("content-type", "").partition(";")[0].strip().lower() != "text/event-stream":
                    raise RemoteUpstreamError("upstream did not stream its completion")
                async for data in sse_data(response, max_event_bytes=1 << 20, max_total_bytes=64 << 20):
                    chunk = parser.parse(data)
                    if chunk is not None:
                        yielded = True
                        yield chunk
                        if chunk.done:
                            return
                parser.finish()
        except UpstreamUnavailableError as error:
            if yielded:
                raise RemoteUpstreamError("the upstream failed during generation") from None
            raise generation_error(error) from None
        except InputTooLongError:
            raise GenerationInputTooLongError("the upstream refused the prompt as too long") from None
        except InvalidInputError:
            raise GenerationInvalidRequestError("prompt", "the upstream refused the request") from None

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        _ = prepared_items
        executor, upstream = self._serving(UpstreamEndpoint.EMBEDDINGS, "encode")
        unavailable = sorted(set(output_types) - {"dense"})
        if unavailable:
            raise InvalidInputError(
                "an OpenAI-compatible upstream returns dense vectors only; "
                f"this profile cannot produce {', '.join(unavailable)}"
            )
        for item in items:
            _plain_text(item)
        opts = options or {}
        query_template = opts.get("query_template")
        if instruction is None and is_query and query_template:
            instruction = opts.get("default_instruction")
        texts = extract_texts(
            items, instruction, is_query=is_query, query_template=query_template, doc_template=opts.get("doc_template")
        )
        with upstream_limiter(self._upstream_name).batch(len(texts), concurrency=requests_in_flight(upstream)):
            embedded = call_each(executor, lambda text: self._embed_one(upstream, text), texts)
        try:
            dense = np.concatenate([vector for vector, _ in embedded])
        except ValueError:
            raise RemoteUpstreamError("upstream returned embeddings of different widths") from None
        if opts.get("normalize") is True:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            dense = dense / np.where(norms == 0, 1, norms)
        extra: dict[str, Any] = {}
        token_counts = _reported([None if usage is None else usage.input_tokens for _, usage in embedded])
        if token_counts is not None:
            extra["input_token_counts"] = token_counts
        return EncodeOutput(dense=dense, batch_size=len(items), is_query=is_query, extra=extra)

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        output = self.score_pairs([query] * len(items), items, instruction=instruction, options=options)
        return [float(score) for score in output.scores]

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        _ = options
        executor, upstream = self._serving(UpstreamEndpoint.RERANK, "score")
        for item in (*queries, *docs):
            _plain_text(item)

        def score_request(query: Item, request_docs: list[Item]) -> RequestScores:
            query_text = _plain_text(query)
            if instruction:
                query_text = f"{instruction} {query_text}"
            documents = [_plain_text(doc) for doc in request_docs]
            body = upstream.apply_params(
                {"model": self._upstream_model, "query": query_text, "documents": documents, "top_n": len(documents)}
            )
            sent_bytes = len(query_text.encode()) + sum(len(document.encode()) for document in documents)
            answer = self._post(
                "/rerank",
                body,
                max_bytes=_RESPONSE_OVERHEAD_BYTES
                + len(documents) * _BYTES_PER_RERANK_RESULT
                + _ESCAPED_ECHO_FACTOR * sent_bytes,
            )
            return RequestScores(scores=_rerank_scores(answer, len(documents)), usage=_usage(answer))

        return score_each_request(
            executor,
            queries,
            docs,
            score_request,
            limiter=upstream_limiter(self._upstream_name),
            concurrency=requests_in_flight(upstream),
        )

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        _ = (items, labels, output_schema, instruction, options, prepared_items)
        raise InvalidInputError("an OpenAI-compatible upstream has no extraction endpoint")

    def _serving(self, endpoint: UpstreamEndpoint, operation: str) -> tuple[ThreadPoolExecutor, Upstream]:
        if self._client is None or self._executor is None or self._upstream is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if endpoint not in self._upstream.endpoints:
            raise InvalidInputError(
                f"upstream {self._upstream_name!r} declares no {endpoint.value} endpoint, "
                f"so this profile cannot {operation}"
            )
        return self._executor, self._upstream

    def _embed_one(self, upstream: Upstream, text: str) -> tuple[np.ndarray, UpstreamUsage | None]:
        body = upstream.apply_params({"model": self._upstream_model, "input": [text], "encoding_format": "float"})
        max_bytes = (
            _LARGEST_RESPONSE_BYTES
            if self._dense_dim is None
            else _RESPONSE_OVERHEAD_BYTES + self._dense_dim * _BYTES_PER_JSON_NUMBER
        )
        answer = self._post("/embeddings", body, max_bytes=max_bytes)
        vector = _embedding(answer)
        if self._dense_dim is not None and vector.shape[1] != self._dense_dim:
            raise RemoteUpstreamError(
                f"upstream returned {vector.shape[1]}-dimensional vectors, the model declares {self._dense_dim}"
            )
        return vector, _usage(answer)

    def _post(self, path: str, body: Mapping[str, Any], *, max_bytes: int) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise RuntimeError(ERR_NOT_LOADED)
        content = json.dumps(body, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = client.build_request(
            "POST",
            path,
            content=content,
            headers={"Content-Type": _JSON, "Accept": _JSON, "Accept-Encoding": "identity"},
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
        )
        raw = send_bounded(
            client, request, upstream=self._upstream_name, max_bytes=max_bytes, deadline_s=REQUEST_DEADLINE_S
        )
        try:
            answer = json.loads(raw)
        except ValueError:
            raise RemoteUpstreamError("upstream returned a body that is not JSON") from None
        if not isinstance(answer, dict):
            raise RemoteUpstreamError("upstream returned a body that is not a JSON object")
        return answer


def _plain_text(item: Item) -> str:
    if (
        not isinstance(item.text, str)
        or item.images
        or item.audio is not None
        or item.video is not None
        or item.document is not None
    ):
        raise InvalidInputError("an OpenAI-compatible upstream takes items of plain text only")
    return item.text


def _reported(counts: list[int | None]) -> list[int] | None:
    reported = [count for count in counts if count is not None]
    return reported if len(reported) == len(counts) else None


def _usage(answer: dict[str, Any]) -> UpstreamUsage | None:
    """The input tokens the upstream reported for its whole request, or ``None`` when it reported none."""
    usage = answer.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise RemoteUpstreamError("upstream reported malformed usage")
    for key in _USAGE_KEYS:
        value = usage.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < _MAX_REPORTED_COUNT:
            raise RemoteUpstreamError("upstream reported malformed usage")
        return UpstreamUsage(input_tokens=value)
    return None


def _embedding(answer: dict[str, Any]) -> np.ndarray:
    """The single embedding of a one-input answer, as a ``[1, dim]`` array."""
    data = answer.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise RemoteUpstreamError("upstream returned a different number of embeddings than inputs sent")
    entry = data[0]
    if not isinstance(entry, dict) or entry.get("index", 0) != 0 or isinstance(entry.get("index"), bool):
        raise RemoteUpstreamError("upstream returned an embedding for an input it was not sent")
    try:
        values = np.array([entry.get("embedding")])
    except (ValueError, TypeError, OverflowError):
        raise RemoteUpstreamError("upstream returned an embedding that is not a list of numbers") from None
    if values.ndim != 2 or values.shape[1] == 0 or values.dtype.kind not in "iuf":
        raise RemoteUpstreamError("upstream returned an embedding that is not a list of numbers")
    vector = values.astype(np.float32)
    if not np.isfinite(vector).all():
        raise RemoteUpstreamError("upstream returned non-finite values")
    return vector


def _rerank_scores(answer: dict[str, Any], count: int) -> list[float]:
    """The upstream's score for each document sent, in the order sent, matched by index."""
    results = answer.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise RemoteUpstreamError("upstream returned a different number of rerank results than documents sent")
    scores: list[float | None] = [None] * count
    for result in results:
        index = result.get("index") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < count
            or scores[index] is not None
        ):
            raise RemoteUpstreamError("upstream returned rerank results with a missing or repeated index")
        score = result.get("relevance_score")
        if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
            raise RemoteUpstreamError("upstream returned a relevance score that is not a finite number")
        scores[index] = float(score)
    return [score for score in scores if score is not None]


def _refuse_unforwarded(parameters: Mapping[str, Any]) -> None:
    for field in ("top_k", "repetition_penalty", "min_new_tokens", "grammar", "images", "videos"):
        value = parameters.get(field)
        if value is not None and value != []:
            raise GenerationUnsupportedFieldError(field, "the OpenAI completions endpoint cannot enforce this field")
