"""Remote adapter for an upstream of kind ``openai``: an OpenAI-compatible endpoint.

The profile names an upstream from the server's startup configuration and the
model id the upstream serves. ``encode`` sends text items to the upstream's
``/embeddings`` and returns its dense vectors. ``score`` sends a request's query
and documents to the upstream's Cohere-shape ``/rerank`` and returns the scores
in document order. Loading makes no outbound call, holds no weights and uses no
accelerator.

This server builds every body it sends, so no field or header of the caller
reaches the upstream. The upstream's operator-set fields are added and its
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

import json
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import numpy as np

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED
from sie_server.adapters._utils import extract_texts
from sie_server.adapters.remote._batching import (
    RequestScores,
    UpstreamUsage,
    call_each,
    requests_in_flight,
    score_each_request,
)
from sie_server.adapters.remote._http import RemoteUpstreamError, send_bounded
from sie_server.config.upstreams import (
    Upstream,
    UpstreamConfigError,
    UpstreamEndpoint,
    UpstreamKind,
    upstream_for_serving,
)
from sie_server.core.inference_output import EncodeOutput, ScoreOutput
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.types.inputs import InvalidInputError

if TYPE_CHECKING:
    from sie_server.core.inference_output import ExtractOutput
    from sie_server.types.inputs import Item

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


class OpenAIUpstreamAdapter(BaseAdapter):
    """Serve ``encode`` and ``score`` for a remote profile from an OpenAI-compatible upstream.

    The outputs this adapter cannot produce are declared so that a model with
    them still loads; a request for one is refused when it arrives.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("dense", "sparse", "multivector", "score", "json"),
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
        self._upstream = None
        if client is not None:
            client.close()
        super().unload()

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

        return score_each_request(executor, queries, docs, score_request)

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
