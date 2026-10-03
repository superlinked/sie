from typing import Literal

UpstreamFailureKind = Literal["not_ready", "busy", "unavailable"]
UpstreamRefusal = Literal["rate_cap", "concurrency_cap", "breaker_open"]

RETRY_AFTER_MIN_S = 1
RETRY_AFTER_MAX_S = 60

_KIND_TEXT: dict[UpstreamFailureKind, str] = {
    "not_ready": "is not ready",
    "busy": "is busy",
    "unavailable": "is unavailable",
}
_REFUSAL_TEXT: dict[UpstreamRefusal, tuple[UpstreamFailureKind, str]] = {
    "rate_cap": ("busy", "its request rate cap is reached"),
    "concurrency_cap": ("busy", "its concurrency cap is reached"),
    "breaker_open": ("unavailable", "its circuit breaker is open after repeated failures"),
}


class InputTooLongError(ValueError):
    pass


class UpstreamUnavailableError(RuntimeError):
    """An upstream did not serve the request, and the same request may succeed later.

    ``kind`` says why: the model is ``not_ready`` on the upstream, the upstream
    is ``busy``, or it is ``unavailable``. ``retry_after_s`` is the wait to
    suggest to the caller. The message is fixed text and never carries anything
    the upstream sent.
    """

    def __init__(self, upstream: str, kind: UpstreamFailureKind, *, retry_after_s: int, reason: str) -> None:
        super().__init__(f"upstream {upstream!r} {_KIND_TEXT[kind]}: {reason}")
        self.upstream = upstream
        self.kind: UpstreamFailureKind = kind
        self.retry_after_s = retry_after_s


class UpstreamRefusedError(UpstreamUnavailableError):
    """This server refused a call to an upstream without sending it.

    A rate or concurrency cap is reached, which makes the upstream ``busy``, or
    its circuit breaker is open, which makes it ``unavailable``. ``refusal``
    names the limit.
    """

    def __init__(self, upstream: str, refusal: UpstreamRefusal, *, retry_after_s: int) -> None:
        kind, reason = _REFUSAL_TEXT[refusal]
        super().__init__(upstream, kind, retry_after_s=retry_after_s, reason=reason)
        self.refusal: UpstreamRefusal = refusal
