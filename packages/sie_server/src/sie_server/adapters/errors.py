from typing import Literal

UpstreamFailureKind = Literal["not_ready", "busy", "unavailable"]

_KIND_TEXT: dict[UpstreamFailureKind, str] = {
    "not_ready": "is not ready",
    "busy": "is busy",
    "unavailable": "is unavailable",
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
