from typing import Literal, cast

UpstreamFailureKind = Literal["not_ready", "busy", "unavailable"]
UpstreamRefusal = Literal["rate_cap", "concurrency_cap", "breaker_open"]
UpstreamFailureReason = Literal[
    "upstream_credential_unavailable",
    "upstream_credential_rejected",
    "upstream_payment_required",
    "upstream_model_not_found",
    "upstream_refused",
    "upstream_unavailable",
    "upstream_busy",
    "upstream_not_ready",
    "upstream_invalid_response",
]

RETRY_AFTER_MIN_S = 1
RETRY_AFTER_MAX_S = 60

UPSTREAM_FAILURE_MESSAGES: dict[UpstreamFailureReason, str] = {
    "upstream_credential_unavailable": "the server has no usable credential for its upstream",
    "upstream_credential_rejected": "the upstream rejected the server's credential",
    "upstream_payment_required": "the upstream account cannot pay for the request",
    "upstream_model_not_found": "the upstream has no route for the configured model",
    "upstream_refused": "the upstream refused the request",
    "upstream_unavailable": "the upstream could not be reached, timed out or failed",
    "upstream_busy": "the upstream is busy",
    "upstream_not_ready": "the model is not ready on the upstream",
    "upstream_invalid_response": "the upstream answered with something the server cannot accept",
}
_KIND_TEXT: dict[UpstreamFailureKind, str] = {
    "not_ready": "is not ready",
    "busy": "is busy",
    "unavailable": "is unavailable",
}
_KIND_REASON: dict[UpstreamFailureKind, UpstreamFailureReason] = {
    "not_ready": "upstream_not_ready",
    "busy": "upstream_busy",
    "unavailable": "upstream_unavailable",
}
_REFUSAL_TEXT: dict[UpstreamRefusal, tuple[UpstreamFailureKind, str]] = {
    "rate_cap": ("busy", "its request rate cap is reached"),
    "concurrency_cap": ("busy", "its concurrency cap is reached"),
    "breaker_open": ("unavailable", "its circuit breaker is open after repeated failures"),
}


def known_upstream_failure_reason(value: object) -> UpstreamFailureReason | None:
    """``value`` when it names a :data:`UpstreamFailureReason`, else ``None``."""
    return (
        cast("UpstreamFailureReason", value) if isinstance(value, str) and value in UPSTREAM_FAILURE_MESSAGES else None
    )


class InputTooLongError(ValueError):
    pass


class UpstreamUnavailableError(RuntimeError):
    """An upstream did not serve the request, and the same request may succeed later.

    ``kind`` says why: the model is ``not_ready`` on the upstream, the upstream
    is ``busy``, or it is ``unavailable``. ``upstream_reason`` is the caller's
    name for that kind. ``retry_after_s`` is the wait to suggest to the caller.
    The message is fixed text and never carries anything the upstream sent.
    """

    def __init__(self, upstream: str, kind: UpstreamFailureKind, *, retry_after_s: int, reason: str) -> None:
        super().__init__(f"upstream {upstream!r} {_KIND_TEXT[kind]}: {reason}")
        self.upstream = upstream
        self.kind: UpstreamFailureKind = kind
        self.upstream_reason: UpstreamFailureReason = _KIND_REASON[kind]
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
