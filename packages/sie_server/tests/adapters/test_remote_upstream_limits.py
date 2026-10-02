"""Process-wide limits on the calls to an upstream: the request budget, the concurrency cap and the breaker."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sie_sdk._msgpack import packb
from sie_server.adapters.errors import UpstreamRefusedError, UpstreamUnavailableError
from sie_server.adapters.remote import _limits
from sie_server.adapters.remote import openai as remote_openai
from sie_server.adapters.remote import sie as remote_sie
from sie_server.adapters.remote._batching import call_each
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote._limits import UpstreamLimiter, upstream_limiter
from sie_server.config.upstreams import Upstream, UpstreamConfigError, install_upstreams, load_upstreams
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.ipc_types import EncodeBatchItem
from sie_server.queue_executor import _inference_exception_outcome
from sie_server.types.inputs import Item

UPSTREAM = "team-sie"


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Events:
    """The limiter's telemetry events."""

    def __init__(self) -> None:
        self.refused: list[tuple[str, str]] = []
        self.breaker: list[tuple[str, bool]] = []

    def upstream_refused(self, *, upstream: object, refusal: object) -> None:
        self.refused.append((str(upstream), str(refusal)))

    def upstream_breaker_changed(self, *, upstream: object, open: bool) -> None:
        self.breaker.append((str(upstream), open))


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> Events:
    recorder = Events()
    monkeypatch.setattr(_limits, "worker_telemetry", lambda: recorder)
    return recorder


@pytest.fixture
def clock() -> Clock:
    return Clock()


def upstream_config(rpm: int = 60, concurrency: int = 4, **breaker: float) -> Upstream:
    config: dict[str, Any] = {
        "kind": "sie",
        "base_url": "https://sie.example.internal",
        "rate_cap": {"requests_per_minute": rpm, "max_concurrency": concurrency},
    }
    if breaker:
        config["breaker"] = breaker
    return Upstream.model_validate(config)


@pytest.fixture
def limiter(events: Events, clock: Clock) -> Callable[..., UpstreamLimiter]:
    _ = events

    def build(rpm: int = 60, concurrency: int = 4, **breaker: float) -> UpstreamLimiter:
        return UpstreamLimiter(UPSTREAM, upstream_config(rpm, concurrency, **breaker), clock=clock)

    return build


def call(limiter: UpstreamLimiter) -> None:
    with limiter.call():
        pass


def refused(limiter: UpstreamLimiter, refusal: str) -> UpstreamRefusedError:
    with pytest.raises(UpstreamRefusedError) as raised:
        call(limiter)
    assert raised.value.refusal == refusal
    return raised.value


def unavailable() -> UpstreamUnavailableError:
    return UpstreamUnavailableError(UPSTREAM, "unavailable", retry_after_s=5, reason="answered 502")


def fail(limiter: UpstreamLimiter, error: Exception | None = None) -> None:
    failure = error or unavailable()
    with pytest.raises(type(failure)), limiter.call():
        raise failure


def test_the_budget_holds_a_minute_of_requests_and_refills_continuously(
    limiter: Callable[..., UpstreamLimiter], clock: Clock, events: Events
) -> None:
    budget = limiter(rpm=60)
    for _ in range(60):
        call(budget)

    refusal = refused(budget, "rate_cap")
    clock.advance(2.5)
    call(budget)
    call(budget)
    refused(budget, "rate_cap")

    assert (refusal.kind, refusal.retry_after_s) == ("busy", 1)
    assert str(refusal) == "upstream 'team-sie' is busy: its request rate cap is reached"
    assert events.refused == [(UPSTREAM, "rate_cap")] * 2


@pytest.mark.parametrize(("rpm", "batch", "retry_after_s"), [(6, 1, 10), (6, 3, 30), (1, 1, 60)])
def test_a_refusal_says_when_the_budget_allows_the_request(
    limiter: Callable[..., UpstreamLimiter], rpm: int, batch: int, retry_after_s: int
) -> None:
    budget = limiter(rpm=rpm)
    for _ in range(rpm):
        call(budget)

    with pytest.raises(UpstreamRefusedError) as raised, budget.batch(batch):
        pass

    assert raised.value.retry_after_s == retry_after_s


def test_a_batch_is_reserved_whole_or_refused_before_anything_is_sent(limiter: Callable[..., UpstreamLimiter]) -> None:
    budget = limiter(rpm=60)
    for _ in range(58):
        call(budget)

    with pytest.raises(UpstreamRefusedError, match="rate cap"), budget.batch(3):
        pytest.fail("a refused batch must not run")
    call(budget)
    call(budget)
    refused(budget, "rate_cap")


def test_calls_in_a_batch_use_its_reservation_and_unused_ones_return_to_the_budget(
    limiter: Callable[..., UpstreamLimiter],
) -> None:
    budget = limiter(rpm=10)

    with budget.batch(4):
        call(budget)
    with budget.batch(5), ThreadPoolExecutor(max_workers=4) as executor:
        call_each(executor, lambda _index: call(budget), list(range(5)))

    for _ in range(10 - 1 - 5):
        call(budget)
    refused(budget, "rate_cap")


def test_a_batch_larger_than_the_budget_passes_when_the_budget_is_full(
    limiter: Callable[..., UpstreamLimiter], clock: Clock
) -> None:
    budget = limiter(rpm=4)

    with budget.batch(10):
        for _ in range(10):
            call(budget)
    refusal = refused(budget, "rate_cap")
    clock.advance(7 * 60 / 4)
    call(budget)

    assert refusal.retry_after_s == 60


def test_a_call_over_the_concurrency_cap_is_refused_at_once_and_spends_nothing(
    limiter: Callable[..., UpstreamLimiter],
) -> None:
    budget = limiter(rpm=3, concurrency=2)

    with ExitStack() as held:
        held.enter_context(budget.call())
        held.enter_context(budget.call())
        refusal = refused(budget, "concurrency_cap")
    call(budget)
    refused(budget, "rate_cap")

    assert (refusal.kind, refusal.retry_after_s) == ("busy", 1)
    assert str(refusal) == "upstream 'team-sie' is busy: its concurrency cap is reached"


def test_a_batch_reserves_concurrency_before_any_item_is_sent(limiter: Callable[..., UpstreamLimiter]) -> None:
    budget = limiter(concurrency=4)
    with budget.call(), pytest.raises(UpstreamRefusedError, match="concurrency cap"), budget.batch(4):
        pytest.fail("a refused batch must not send any requests")


def test_reserved_batch_slots_cannot_be_taken_by_another_caller(limiter: Callable[..., UpstreamLimiter]) -> None:
    budget = limiter(concurrency=2)
    with budget.batch(4):
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(refused, budget, "concurrency_cap").result().retry_after_s == 1
        for _ in range(4):
            call(budget)
    call(budget)


def test_a_batch_reserves_only_its_executor_width(limiter: Callable[..., UpstreamLimiter]) -> None:
    budget = limiter(concurrency=4)
    with budget.call(), budget.batch(10, concurrency=3):
        with ThreadPoolExecutor(max_workers=3) as executor:
            call_each(executor, lambda _index: call(budget), list(range(10)))


def test_a_half_open_breaker_refuses_a_multi_request_batch_before_sending(
    limiter: Callable[..., UpstreamLimiter],
    clock: Clock,
) -> None:
    breaker = limiter(failures=1, cooldown_s=60)
    fail(breaker)
    clock.advance(60)
    with pytest.raises(UpstreamRefusedError, match="circuit breaker"), breaker.batch(2):
        pytest.fail("only one probe may be sent")
    with breaker.batch(1):
        call(breaker)
    call(breaker)


def test_consecutive_failures_within_the_window_open_the_breaker_for_the_cooldown(
    limiter: Callable[..., UpstreamLimiter], clock: Clock, events: Events
) -> None:
    breaker = limiter(failures=3, window_s=30, cooldown_s=60)
    for _ in range(3):
        fail(breaker)

    refusal = refused(breaker, "breaker_open")
    with pytest.raises(UpstreamRefusedError, match="circuit breaker"), breaker.batch(1):
        pass
    clock.advance(59.5)
    late = refused(breaker, "breaker_open")

    assert (refusal.kind, refusal.retry_after_s, late.retry_after_s) == ("unavailable", 60, 1)
    assert str(refusal) == "upstream 'team-sie' is unavailable: its circuit breaker is open after repeated failures"
    assert events.breaker == [(UPSTREAM, False), (UPSTREAM, True)]


def test_after_the_cooldown_one_probe_closes_the_breaker(
    limiter: Callable[..., UpstreamLimiter], clock: Clock, events: Events
) -> None:
    breaker = limiter(failures=1, cooldown_s=60)
    fail(breaker)
    clock.advance(60)

    with breaker.call():
        concurrent = refused(breaker, "breaker_open")
    call(breaker)
    call(breaker)

    assert concurrent.retry_after_s == 1
    assert events.breaker == [(UPSTREAM, False), (UPSTREAM, True), (UPSTREAM, False)]


def test_a_failed_probe_opens_the_breaker_again(
    limiter: Callable[..., UpstreamLimiter], clock: Clock, events: Events
) -> None:
    breaker = limiter(failures=1, cooldown_s=60)
    fail(breaker)
    clock.advance(60)

    fail(breaker)
    refusal = refused(breaker, "breaker_open")

    assert refusal.retry_after_s == 60
    assert events.breaker == [(UPSTREAM, False), (UPSTREAM, True)]


@pytest.mark.parametrize(
    "outcomes",
    [
        pytest.param(["fail", "fail", "wait", "fail", "fail"], id="spread-beyond-the-window"),
        pytest.param(["fail", "fail", "answer", "fail", "fail"], id="broken-by-an-answer"),
        pytest.param(["busy"] * 6, id="busy-upstream"),
        pytest.param(["not_ready"] * 6, id="upstream-still-loading"),
        pytest.param(["terminal"] * 6, id="upstream-answers-with-an-error"),
    ],
)
def test_only_unavailability_in_a_row_within_the_window_opens_the_breaker(
    limiter: Callable[..., UpstreamLimiter], clock: Clock, outcomes: list[str]
) -> None:
    breaker = limiter(failures=3, window_s=30)
    for outcome in outcomes:
        if outcome == "fail":
            fail(breaker)
        elif outcome == "wait":
            clock.advance(31)
        elif outcome == "answer":
            call(breaker)
        elif outcome == "terminal":
            fail(breaker, RemoteUpstreamError("upstream answered 401"))
        else:
            fail(breaker, UpstreamUnavailableError(UPSTREAM, outcome, retry_after_s=5, reason="answered 503"))  # type: ignore[arg-type]

    call(breaker)


def test_every_adapter_shares_one_limiter_per_upstream_until_the_upstreams_are_installed_again() -> None:
    try:
        install_upstreams({UPSTREAM: upstream_config()})
        first = upstream_limiter(UPSTREAM)
        same = upstream_limiter(UPSTREAM)
        install_upstreams({UPSTREAM: upstream_config()})
        fresh = upstream_limiter(UPSTREAM)

        with pytest.raises(RuntimeError, match="not defined"):
            upstream_limiter("nobody")
    finally:
        install_upstreams({})

    assert first is same
    assert fresh is not first


def test_a_breaker_defaults_to_five_failures_in_thirty_seconds_and_a_minute_of_cooldown(tmp_path: Path) -> None:
    path = tmp_path / "upstreams.yaml"
    path.write_text(
        "upstreams:\n"
        "  team-sie:\n"
        "    kind: sie\n"
        "    base_url: https://sie.example.internal\n"
        "    rate_cap: {requests_per_minute: 60, max_concurrency: 4}\n",
        encoding="utf-8",
    )

    breaker = load_upstreams(path)[UPSTREAM].breaker

    assert (breaker.failures, breaker.window_s, breaker.cooldown_s) == (5, 30.0, 60.0)


@pytest.mark.parametrize(
    "breaker",
    [
        {"failures": 0},
        {"failures": 1001},
        {"window_s": 0},
        {"window_s": 3601},
        {"cooldown_s": -1},
        {"cooldown_s": ".nan"},
        {"open_for": 30},
    ],
    ids=["no-failures", "too-many-failures", "no-window", "window-over-an-hour", "negative-cooldown", "nan", "unknown"],
)
def test_an_invalid_breaker_is_refused_at_startup(tmp_path: Path, breaker: dict[str, Any]) -> None:
    path = tmp_path / "upstreams.yaml"
    rendered = ", ".join(f"{key}: {value}" for key, value in breaker.items())
    path.write_text(
        "upstreams:\n"
        "  team-sie:\n"
        "    kind: sie\n"
        "    base_url: https://sie.example.internal\n"
        "    rate_cap: {requests_per_minute: 60, max_concurrency: 4}\n"
        f"    breaker: {{{rendered}}}\n",
        encoding="utf-8",
    )

    with pytest.raises(UpstreamConfigError, match="breaker"):
        load_upstreams(path)


def dense_answer(_request: httpx.Request, dim: int = 4) -> httpx.Response:
    body = {"model": "sie-fake", "items": [{"dense": {"values": np.ones(dim, dtype=np.float32)}}]}
    return httpx.Response(200, stream=_Body(packb(body)), headers={"Content-Type": "application/msgpack"})


def bad_gateway(_request: httpx.Request) -> httpx.Response:
    detail = json.dumps({"detail": {"code": "INFERENCE_ERROR", "message": "upstream failed"}}).encode()
    return httpx.Response(502, stream=_Body(detail), headers={"Content-Type": "application/json"})


class _Body(httpx.SyncByteStream):
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __iter__(self) -> Iterator[bytes]:
        yield self._content


class Recorder:
    def __init__(self, answer: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._answer = answer

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._answer(request)


@pytest.fixture
def adapter_over(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., remote_sie.SieUpstreamAdapter]]:
    loaded: list[remote_sie.SieUpstreamAdapter] = []

    def build(recorder: Recorder, *, rpm: int = 600, **breaker: float) -> remote_sie.SieUpstreamAdapter:
        install_upstreams({UPSTREAM: upstream_config(rpm, 4, **breaker)})
        monkeypatch.setattr(
            remote_sie,
            "upstream_sync_client",
            lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(recorder)),
        )
        adapter = remote_sie.SieUpstreamAdapter(upstream=UPSTREAM, upstream_model="sie-fake", dense_dim=4)
        adapter.load("cpu")
        loaded.append(adapter)
        return adapter

    yield build
    for adapter in loaded:
        adapter.unload()
    install_upstreams({})


def test_a_batch_over_the_budget_is_refused_before_any_item_is_sent(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(dense_answer)
    adapter = adapter_over(recorder, rpm=10)
    for _ in range(8):
        adapter.encode([Item(text="a")], ["dense"])

    with pytest.raises(UpstreamRefusedError) as raised:
        adapter.encode([Item(text="a"), Item(text="b"), Item(text="c")], ["dense"])

    assert len(recorder.requests) == 8
    assert (raised.value.refusal, raised.value.retry_after_s) == ("rate_cap", 6)


def test_a_batch_under_concurrency_contention_sends_no_items(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(dense_answer)
    adapter = adapter_over(recorder)
    with upstream_limiter(UPSTREAM).call():
        with pytest.raises(UpstreamRefusedError, match="concurrency cap"):
            adapter.encode([Item(text=str(index)) for index in range(4)], ["dense"])
    assert recorder.requests == []


@pytest.mark.parametrize("operation", ["encode", "score"])
@pytest.mark.parametrize("refusal", ["rate_cap", "concurrency_cap", "breaker_open"])
def test_openai_batches_are_refused_before_any_item_is_sent(
    monkeypatch: pytest.MonkeyPatch,
    clock: Clock,
    operation: str,
    refusal: str,
) -> None:
    upstream = Upstream.model_validate(
        {
            "kind": "openai",
            "base_url": "https://openai.example.internal/v1",
            "endpoints": ["embeddings", "rerank"],
            "rate_cap": {"requests_per_minute": 2, "max_concurrency": 2},
            "breaker": {"failures": 1, "cooldown_s": 60},
        }
    )
    budget = UpstreamLimiter(UPSTREAM, upstream, clock=clock)
    install_upstreams({UPSTREAM: upstream})
    monkeypatch.setitem(_limits._LIMITERS, UPSTREAM, (upstream, budget))
    recorder = Recorder(lambda _request: httpx.Response(502))
    monkeypatch.setattr(
        remote_openai,
        "upstream_sync_client",
        lambda config: upstream_sync_client(config, transport=httpx.MockTransport(recorder)),
    )
    adapter = remote_openai.OpenAIUpstreamAdapter(upstream=UPSTREAM, upstream_model="test-model", dense_dim=4)
    try:
        adapter.load("cpu")
        with ExitStack() as held:
            if refusal == "rate_cap":
                call(budget)
            elif refusal == "concurrency_cap":
                held.enter_context(budget.call())
            else:
                fail(budget)
                clock.advance(60)
            with pytest.raises(UpstreamRefusedError) as raised:
                if operation == "encode":
                    adapter.encode([Item(text="a"), Item(text="b")], ["dense"])
                else:
                    adapter.score_pairs([Item(text="q1"), Item(text="q2")], [Item(text="a"), Item(text="b")])
            assert raised.value.refusal == refusal
            assert recorder.requests == []
    finally:
        adapter.unload()
        install_upstreams({})


def test_a_score_batch_reserves_one_request_per_api_request(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(dense_answer)
    adapter = adapter_over(recorder, rpm=2)
    first, second = Item(text="q1"), Item(text="q2")
    adapter.encode([Item(text="spend one")], ["dense"])

    with pytest.raises(UpstreamRefusedError, match="rate cap"):
        adapter.score_pairs([first, first, second], [Item(text="a"), Item(text="b"), Item(text="c")])

    assert len(recorder.requests) == 1


def test_an_upstream_that_keeps_failing_stops_receiving_calls(
    adapter_over: Callable[..., remote_sie.SieUpstreamAdapter],
) -> None:
    recorder = Recorder(bad_gateway)
    adapter = adapter_over(recorder)
    for _ in range(5):
        with pytest.raises(UpstreamUnavailableError, match="answered 502"):
            adapter.encode([Item(text="a")], ["dense"])

    with pytest.raises(UpstreamRefusedError) as raised:
        adapter.encode([Item(text="a")], ["dense"])

    assert len(recorder.requests) == 5
    assert (raised.value.refusal, raised.value.retry_after_s) == ("breaker_open", 60)


def test_a_refusal_is_a_retryable_503_with_its_retry_after_and_the_disclosure_headers(
    monkeypatch: pytest.MonkeyPatch, remote_app: Callable[..., Any]
) -> None:
    recorder = Recorder(lambda request: dense_answer(request, dim=384))
    monkeypatch.setattr(
        remote_sie,
        "upstream_sync_client",
        lambda upstream: upstream_sync_client(upstream, transport=httpx.MockTransport(recorder)),
    )
    app = remote_app("https://sie.example.internal", rate_cap={"requests_per_minute": 1, "max_concurrency": 1})
    body = {"items": [{"text": "a"}]}

    with TestClient(app) as client:
        deadline = time.monotonic() + 30
        while (first := client.post("/v1/encode/acme/remote-fake", json=body)).status_code == 503:
            assert time.monotonic() < deadline, first.text
            time.sleep(0.1)
        second = client.post("/v1/encode/acme/remote-fake", json=body, headers={"Accept": "application/json"})

    assert first.status_code == 200, first.text
    assert second.status_code == 503, second.text
    assert second.json()["detail"] == {
        "code": "QUEUE_FULL",
        "message": "The upstream serving model 'acme/remote-fake' is busy, please retry",
    }
    assert second.headers["retry-after"] == "60"
    assert second.headers["x-sie-served-by"] == "remote"
    assert second.headers["x-sie-upstream"] == "fake-sie"
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("refusal", ["rate_cap", "concurrency_cap", "breaker_open"])
def test_the_queue_path_redelivers_a_refused_item_after_the_wait(refusal: str) -> None:
    item = EncodeBatchItem(
        work_item_id="w.0", request_id="w", item_index=0, total_items=1, timestamp=0.0, item={"text": "a"}
    )

    outcome = _inference_exception_outcome(item, UpstreamRefusedError(UPSTREAM, refusal, retry_after_s=40))  # type: ignore[arg-type]

    assert (outcome.disposition, outcome.nak_delay_ms, outcome.error) == ("nak_retry", 40_000, None)
