"""Process-wide limits on the calls this server makes to each upstream.

Every remote adapter, on a single node and on a cluster's remote lane alike,
sends its calls through one limiter per upstream name. The limiter enforces
three limits from the upstream's configuration:

- a budget of ``rate_cap.requests_per_minute`` requests, refilled continuously
  and holding at most one minute's worth;
- at most ``rate_cap.max_concurrency`` calls in flight;
- a circuit breaker that opens for ``breaker.cooldown_s`` once
  ``breaker.failures`` calls in a row, all within ``breaker.window_s``, found
  the upstream unavailable. After the cooldown one call probes the upstream:
  its success closes the breaker and its failure opens it again.

A call over a limit is refused at once, never queued. It raises
:class:`~sie_server.adapters.errors.UpstreamRefusedError` with the wait until
the budget, a free slot or the end of the cooldown would let it through. The
limits hold per process, so the traffic an upstream receives grows with the
number of server replicas.

An adapter that sends several upstream requests for one batch reserves them
together with :meth:`UpstreamLimiter.batch`, so that a batch is sent whole or
refused before anything is sent. A batch larger than the whole budget is let
through when the budget is full, and the budget then refills before anything
else is sent.

Background refreshes of an SIE upstream's identity metadata for hybrid
admission go through a second limiter per upstream name, never through the
first, so they never consume the inference rate cap. It has a budget of its
own: a tenth of ``rate_cap.requests_per_minute``, at least one read a minute,
with one read in flight. Its circuit breaker, set by the same ``breaker``
settings, is its own too. Inference calls never draw on it, and its refusals
and breaker state are not reported to the worker telemetry, whose upstream
instruments describe inference calls. An identity read that its caller waits
for, such as the check at configuration load, goes through the first limiter
like an inference call.
"""

from __future__ import annotations

import contextvars
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sie_server.adapters.errors import (
    RETRY_AFTER_MAX_S,
    RETRY_AFTER_MIN_S,
    UpstreamRefusal,
    UpstreamRefusedError,
    UpstreamUnavailableError,
)
from sie_server.config.upstreams import RateCap, Upstream, installed_upstreams
from sie_server.observability.worker_telemetry import worker_telemetry

_SLOT_RETRY_AFTER_S = 1


class _Reservation:
    """Requests a batch reserved from one upstream's budget, taken one call at a time."""

    def __init__(self, limiter: UpstreamLimiter, requests: int, slots: int, *, probe: bool) -> None:
        self.limiter = limiter
        self.slots = threading.BoundedSemaphore(slots)
        self.probe = probe
        self._remaining = requests
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    def unused(self) -> int:
        with self._lock:
            unused, self._remaining = self._remaining, 0
            return unused


_BATCH: contextvars.ContextVar[_Reservation | None] = contextvars.ContextVar("sie_upstream_batch", default=None)


class UpstreamLimiter:
    """The budget, the concurrency cap and the circuit breaker for one upstream.

    Its refusals and breaker changes are reported to the worker telemetry
    unless ``telemetry`` is false.
    """

    def __init__(
        self, name: str, upstream: Upstream, *, clock: Callable[[], float] = time.monotonic, telemetry: bool = True
    ) -> None:
        self.name = name
        self._telemetry = telemetry
        self._clock = clock
        self._lock = threading.Lock()
        self._capacity = float(upstream.rate_cap.requests_per_minute)
        self._refill_per_s = self._capacity / 60.0
        self._tokens = self._capacity
        self._refilled_at = clock()
        self._max_in_flight = upstream.rate_cap.max_concurrency
        self._in_flight = 0
        self._failures_to_open = upstream.breaker.failures
        self._window_s = upstream.breaker.window_s
        self._cooldown_s = upstream.breaker.cooldown_s
        self._failures: deque[float] = deque()
        self._open_until: float | None = None
        self._probing = False
        if telemetry:
            worker_telemetry().upstream_breaker_changed(upstream=name, open=False)

    @contextmanager
    def batch(self, requests: int, *, concurrency: int | None = None) -> Iterator[None]:
        """Reserve ``requests`` calls at once for the calls made inside the block, or refuse them all.

        Calls made from threads must run in a copy of the caller's context, as
        :func:`~sie_server.adapters.remote._batching.call_each` runs them.
        Reserve up to ``concurrency`` in-flight slots for the whole block.
        Reserved calls the block does not make are returned to the budget.
        """
        if requests < 0 or (concurrency is not None and concurrency < 1):
            raise ValueError("batch requests must be non-negative and concurrency must be positive")
        slots = min(requests, self._max_in_flight, concurrency or self._max_in_flight)
        with self._lock:
            now = self._clock()
            self._refuse_while_open(now, requests=requests)
            if self._open_until is not None and requests > 1:
                raise self._refusal("breaker_open", _SLOT_RETRY_AFTER_S, requests=requests)
            if self._in_flight + slots > self._max_in_flight:
                raise self._refusal("concurrency_cap", _SLOT_RETRY_AFTER_S, requests=requests)
            probe = self._admit(now, requests=requests) if requests else False
            try:
                self._take(requests, now)
            except UpstreamRefusedError:
                if probe:
                    self._probing = False
                raise
            self._in_flight += slots
        reservation = _Reservation(self, requests, slots, probe=probe)
        token = _BATCH.set(reservation)
        try:
            yield
        finally:
            _BATCH.reset(token)
            unused = reservation.unused()
            with self._lock:
                self._in_flight -= slots
                self._return(unused, self._clock())
                if probe and unused:
                    self._probing = False

    @contextmanager
    def call(self) -> Iterator[None]:
        """Hold one in-flight slot for one upstream call; the call's outcome feeds the breaker.

        The call counts against the batch reserved around it, or else against
        the budget. An :class:`~sie_server.adapters.errors.UpstreamUnavailableError`
        of kind ``unavailable`` raised inside the block is a failure. Anything
        else means the upstream answered.
        """
        reservation = _BATCH.get()
        prepaid = reservation is not None and reservation.limiter is self and reservation.take()
        probe = False
        if prepaid:
            assert reservation is not None
            reservation.slots.acquire()
            probe = reservation.probe
        else:
            with self._lock:
                now = self._clock()
                try:
                    probe = self._admit(now)
                    if self._in_flight >= self._max_in_flight:
                        raise self._refusal("concurrency_cap", _SLOT_RETRY_AFTER_S)
                    self._take(1, now)
                except UpstreamRefusedError:
                    if probe:
                        self._probing = False
                    raise
                self._in_flight += 1
        failed = False
        try:
            yield
        except UpstreamUnavailableError as error:
            failed = error.kind == "unavailable"
            raise
        finally:
            with self._lock:
                if not prepaid:
                    self._in_flight -= 1
                self._record(failed=failed, probe=probe, now=self._clock())
            if prepaid:
                assert reservation is not None
                reservation.slots.release()

    def _admit(self, now: float, *, requests: int = 1) -> bool:
        """Refuse while the breaker is open; after the cooldown, admit one call as the probe."""
        if self._open_until is None:
            return False
        if now < self._open_until:
            raise self._refusal("breaker_open", self._open_until - now, requests=requests)
        if self._probing:
            raise self._refusal("breaker_open", _SLOT_RETRY_AFTER_S, requests=requests)
        self._probing = True
        return True

    def _refuse_while_open(self, now: float, *, requests: int = 1) -> None:
        if self._open_until is None:
            return
        if now < self._open_until:
            raise self._refusal("breaker_open", self._open_until - now, requests=requests)
        if self._probing:
            raise self._refusal("breaker_open", _SLOT_RETRY_AFTER_S, requests=requests)

    def _take(self, requests: int, now: float) -> None:
        self._refill(now)
        affordable = self._tokens >= requests or (requests > self._capacity and self._tokens >= self._capacity)
        if not affordable:
            missing = min(float(requests), self._capacity) - self._tokens
            raise self._refusal("rate_cap", missing / self._refill_per_s, requests=requests)
        self._tokens -= requests

    def _return(self, requests: int, now: float) -> None:
        self._refill(now)
        self._tokens = min(self._capacity, self._tokens + requests)

    def _refill(self, now: float) -> None:
        self._tokens = min(self._capacity, self._tokens + (now - self._refilled_at) * self._refill_per_s)
        self._refilled_at = now

    def _record(self, *, failed: bool, probe: bool, now: float) -> None:
        if probe:
            self._probing = False
            if failed:
                self._open(now)
            else:
                self._close()
            return
        if self._open_until is not None:
            return
        if not failed:
            self._failures.clear()
            return
        self._failures.append(now)
        while now - self._failures[0] > self._window_s:
            self._failures.popleft()
        if len(self._failures) >= self._failures_to_open:
            self._open(now)

    def _open(self, now: float) -> None:
        was_closed = self._open_until is None
        self._open_until = now + self._cooldown_s
        self._failures.clear()
        if was_closed and self._telemetry:
            worker_telemetry().upstream_breaker_changed(upstream=self.name, open=True)

    def _close(self) -> None:
        self._open_until = None
        if self._telemetry:
            worker_telemetry().upstream_breaker_changed(upstream=self.name, open=False)

    def _refusal(self, refusal: UpstreamRefusal, wait_s: float, *, requests: int = 1) -> UpstreamRefusedError:
        if self._telemetry:
            worker_telemetry().upstream_refused(upstream=self.name, refusal=refusal, requests=requests)
        retry_after_s = min(RETRY_AFTER_MAX_S, max(RETRY_AFTER_MIN_S, math.ceil(wait_s)))
        return UpstreamRefusedError(self.name, refusal, retry_after_s=retry_after_s)


_REGISTRY_LOCK = threading.Lock()
_LIMITERS: dict[str, tuple[Upstream, UpstreamLimiter]] = {}
_IDENTITY_LIMITERS: dict[str, tuple[Upstream, UpstreamLimiter]] = {}


def upstream_limiter(name: str) -> UpstreamLimiter:
    """The process-wide limiter for upstream ``name``, built from its installed configuration.

    Installing the upstreams again, as a restart of the configuration does,
    starts every limiter afresh.
    """
    return _installed_limiter(_LIMITERS, name, lambda upstream: UpstreamLimiter(name, upstream))


def identity_limiter(name: str) -> UpstreamLimiter:
    """The process-wide limiter for background identity metadata refreshes from upstream ``name``.

    Its configuration is the upstream's, with ``rate_cap`` set to a tenth of
    ``requests_per_minute``, at least one, and a ``max_concurrency`` of one.
    Installing the upstreams again starts it afresh, as it does
    :func:`upstream_limiter`.
    """

    def build(upstream: Upstream) -> UpstreamLimiter:
        rate_cap = RateCap(requests_per_minute=max(1, upstream.rate_cap.requests_per_minute // 10), max_concurrency=1)
        return UpstreamLimiter(name, upstream.model_copy(update={"rate_cap": rate_cap}), telemetry=False)

    return _installed_limiter(_IDENTITY_LIMITERS, name, build)


def _installed_limiter(
    limiters: dict[str, tuple[Upstream, UpstreamLimiter]], name: str, build: Callable[[Upstream], UpstreamLimiter]
) -> UpstreamLimiter:
    upstream = installed_upstreams().get(name)
    if upstream is None:
        raise RuntimeError(f"upstream {name!r} is not defined in the startup configuration")
    with _REGISTRY_LOCK:
        entry = limiters.get(name)
        if entry is None or entry[0] is not upstream:
            entry = (upstream, build(upstream))
            limiters[name] = entry
        return entry[1]
