"""How a remote adapter splits the worker's fused batch into upstream requests.

The worker fuses items from several API requests into one adapter call, while
an upstream reports usage for a whole upstream request. A remote adapter
therefore sends one upstream request for each item it encodes or extracts
from, and one for each API request it scores. Every count it reports is then
the upstream's own measurement of exactly the work it is attributed to, with
nothing apportioned. Each of those upstream requests counts against the
upstream's ``requests_per_minute``.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, Executor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.core.inference_output import ScoreOutput

if TYPE_CHECKING:
    from sie_server.adapters.remote._limits import UpstreamLimiter
    from sie_server.config.upstreams import Upstream
    from sie_server.types.inputs import Item

MAX_REQUESTS_IN_FLIGHT = 8


@dataclass(frozen=True, slots=True)
class UpstreamUsage:
    """Counts an upstream reported for one upstream request. ``None`` means not reported."""

    input_tokens: int | None = None
    images: int | None = None
    content_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class RequestPairs:
    """The (query, document) pairs one API request contributed to a fused score batch."""

    query: Item
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RequestScores:
    """An upstream's scores for one API request's documents, in the order sent, and its usage."""

    scores: Sequence[float]
    usage: UpstreamUsage | None


def requests_in_flight(upstream: Upstream) -> int:
    """How many requests one adapter call may have open to ``upstream`` at once."""
    return min(MAX_REQUESTS_IN_FLIGHT, upstream.rate_cap.max_concurrency)


def call_each[RequestT, AnswerT](
    executor: Executor, call: Callable[[RequestT], AnswerT], requests: Sequence[RequestT]
) -> list[AnswerT]:
    """Answer every request through ``call``, as many at once as ``executor`` runs, in request order.

    Each call runs in a copy of the caller's context, so it counts against a
    batch the caller reserved from the upstream's limiter. After a failure no
    further request is started. Once the started ones have finished, the first
    failure in request order is raised.
    """
    if len(requests) == 1:
        return [call(requests[0])]

    def run(context: contextvars.Context, request: RequestT) -> AnswerT:
        return context.run(call, request)

    futures = [executor.submit(run, contextvars.copy_context(), request) for request in requests]
    wait(futures, return_when=FIRST_EXCEPTION)
    for future in futures:
        future.cancel()
    wait(futures)
    for future in futures:
        if not future.cancelled() and (error := future.exception()) is not None:
            raise error
    return [future.result() for future in futures]


def pairs_by_request(queries: Sequence[Item]) -> list[RequestPairs]:
    """Group the parallel (query, document) pairs of a fused score batch by API request.

    Every pair of one API request carries the same query object, and no two
    requests share a query object, so its identity marks the request.
    """
    positions: dict[int, list[int]] = {}
    for position, query in enumerate(queries):
        positions.setdefault(id(query), []).append(position)
    return [RequestPairs(query=queries[group[0]], positions=tuple(group)) for group in positions.values()]


def score_each_request(
    executor: Executor,
    queries: Sequence[Item],
    docs: Sequence[Item],
    score_request: Callable[[Item, list[Item]], RequestScores],
    *,
    limiter: UpstreamLimiter | None = None,
    concurrency: int | None = None,
) -> ScoreOutput:
    """Score a fused batch with one upstream request per API request.

    With ``limiter``, the upstream requests are reserved from its budget
    together before any is sent. Each request's reported totals are carried on
    its first pair and its other pairs carry zero, so every per-request sum is
    the upstream's own count. A count is reported only when every request
    reported it.
    """
    if len(queries) != len(docs):
        raise ValueError(f"queries and docs must be parallel; got {len(queries)} vs {len(docs)}")
    requests = pairs_by_request(queries)
    reservation = limiter.batch(len(requests), concurrency=concurrency) if limiter is not None else nullcontext()
    with reservation:
        answers = call_each(
            executor,
            lambda request: score_request(request.query, [docs[position] for position in request.positions]),
            requests,
        )
    scores = np.zeros(len(docs), dtype=np.float32)
    for request, answer in zip(requests, answers, strict=True):
        if len(answer.scores) != len(request.positions):
            raise RemoteUpstreamError("upstream returned a different number of scores than documents sent")
        scores[list(request.positions)] = answer.scores
    usages = [answer.usage for answer in answers]
    return ScoreOutput(
        scores=scores,
        input_token_counts=_on_first_pairs(
            len(docs), requests, [None if usage is None else usage.input_tokens for usage in usages]
        ),
        content_token_counts=_on_first_pairs(
            len(docs), requests, [None if usage is None else usage.content_tokens for usage in usages]
        ),
        input_image_counts=_on_first_pairs(
            len(docs), requests, [None if usage is None else usage.images for usage in usages]
        ),
    )


def _on_first_pairs(size: int, requests: Sequence[RequestPairs], totals: Sequence[int | None]) -> list[int] | None:
    if any(total is None for total in totals):
        return None
    counts = [0] * size
    for request, total in zip(requests, totals, strict=True):
        counts[request.positions[0]] = int(total or 0)
    return counts
