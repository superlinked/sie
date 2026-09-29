"""Single source of truth for turning a request's raw options into the
effective options an adapter sees.

A model profile declares ``adapter_options.runtime`` defaults (e.g. an
embedding model's ``query_template`` / ``default_instruction`` / ``pooling`` /
``normalize``). Those defaults must be merged under the per-request options so
the adapter receives them at inference time.

There are two ingress paths and they MUST agree:

* Single-server HTTP (``api.options.resolve_runtime_options`` →
  ``api.encode``) — used by ``mise run serve`` and local SDK calls.
* Cluster queue worker (``queue_executor.process_encode_batch``,
  ``process_score_batch``, and ``process_extract_batch``) — the queue carries
  raw request options, so the worker performs the same merge itself.

Historically some queue operations forwarded raw request options verbatim, silently dropping
``query_template`` / ``default_instruction`` / ``pooling`` / ``normalize`` for
queued requests. Routing both paths through this helper keeps them in lockstep.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from sie_server.types.inputs import InvalidInputError
from sie_server.types.overflow_policy import VALID_OVERFLOW_POLICIES

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig, ResolvedProfile

logger = logging.getLogger(__name__)


def _resolve_profile_or_raise(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
) -> ResolvedProfile:
    profile_name = request_options.get("profile") if request_options else None
    if profile_name is not None and (not isinstance(profile_name, str) or not profile_name.strip()):
        raise InvalidInputError("'options.profile' must be a non-empty string or null")

    try:
        return config.resolve_profile(profile_name or "default")
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc


class InvalidOverflowPolicyError(InvalidInputError):
    """``options.overflow_policy`` is not one of ``VALID_OVERFLOW_POLICIES``."""


def check_overflow_policy(options: dict[str, Any]) -> None:
    """Reject an ``overflow_policy`` that is not one of the valid policy names.

    Raises:
        InvalidOverflowPolicyError: The value is not a string naming a valid policy.
    """
    overflow_policy = options.get("overflow_policy")
    if overflow_policy is not None and (
        not isinstance(overflow_policy, str) or overflow_policy not in VALID_OVERFLOW_POLICIES
    ):
        raise InvalidOverflowPolicyError(
            f"Invalid overflow_policy: {overflow_policy!r}. Must be one of {sorted(VALID_OVERFLOW_POLICIES)}."
        )


def merge_runtime_options_with_profile(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
) -> tuple[dict[str, Any], ResolvedProfile]:
    """Return merged adapter options and the profile used to derive them.

    Raises:
        InvalidInputError: The request selects an unknown or malformed profile, or
            the merged ``overflow_policy`` is not a valid policy name.
    """
    resolved = _resolve_profile_or_raise(config, request_options)
    merged: dict[str, Any] = dict(resolved.runtime)
    if request_options:
        merged |= {key: value for key, value in request_options.items() if key != "profile"}
    check_overflow_policy(merged)
    return merged, resolved


def merge_runtime_options(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the selected profile and overlay request options on its runtime defaults.

    The ``"profile"`` key in ``request_options`` selects the profile and is
    consumed (not forwarded). Request-supplied values win over profile runtime
    defaults so per-request overrides still work.

    Args:
        config: The model configuration (carries the profiles).
        request_options: Raw options from the request (may be ``None`` or carry
            a ``"profile"`` key).

    Returns:
        The merged options dict ready to hand to the adapter.

    Raises:
        InvalidInputError: If ``request_options`` selects a malformed or
            unknown profile, or the merged ``overflow_policy`` is invalid.
    """
    merged, _ = merge_runtime_options_with_profile(config, request_options)
    return merged


_GENERATION_RUNTIME_KEYS = frozenset(
    {
        "default_sampling",
        "stop_tokens",
        "first_chunk_timeout_s",
        "inter_chunk_timeout_s",
        "overall_timeout_s",
    }
)
_GENERATION_SAMPLING_KEYS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "top_k": "top_k",
    "min_new_tokens": "min_tokens",
    "seed": "seed",
}


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def apply_generation_runtime_options(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
    generate_params: dict[str, Any],
) -> dict[str, Any]:
    """Apply governed generation runtime defaults below typed request fields.

    Generation adapters expose explicit sampler arguments rather than a generic
    ``**options`` seam. Validate the currently governed runtime surface and
    translate it here so unsupported options fail closed instead of leaking to
    adapter kwargs or being silently ignored.
    """
    if request_options is not None and not isinstance(request_options, dict):
        raise ValueError("'options' must be an object")

    if request_options:
        profile = request_options.get("profile")
        if profile not in (None, "default"):
            if not isinstance(profile, str):
                raise ValueError("'options.profile' must be a string")
            raise ValueError(
                f"non-default options.profile '{profile}' cannot select a routed model variant; "
                "use the 'model:profile' identity"
            )
        unknown = set(request_options) - _GENERATION_RUNTIME_KEYS - {"profile"}
        if unknown:
            raise ValueError(f"unsupported generation option(s): {sorted(unknown)}")
        if "default_sampling" in request_options and not isinstance(request_options["default_sampling"], dict):
            raise ValueError("'options.default_sampling' must be an object")
        if "stop_tokens" in request_options and not isinstance(request_options["stop_tokens"], list):
            raise ValueError("'options.stop_tokens' must be an array of non-empty strings")
        for key in ("first_chunk_timeout_s", "inter_chunk_timeout_s", "overall_timeout_s"):
            value = request_options.get(key)
            if key in request_options and (isinstance(value, bool) or not isinstance(value, int | float) or value <= 0):
                raise ValueError(f"'options.{key}' must be a positive number")

    runtime = merge_runtime_options(config, request_options)
    profile_sampling = config.resolve_profile("default").runtime.get("default_sampling")
    request_sampling = request_options.get("default_sampling") if request_options else None
    if isinstance(profile_sampling, dict) and isinstance(request_sampling, dict):
        runtime["default_sampling"] = {**profile_sampling, **request_sampling}
    result = dict(generate_params)

    # The typed request maximum is a hard caller limit. Reject an explicit
    # minimum that contradicts it even when this profile has no sampler
    # defaults; the adapter repeats this check before the engine boundary.
    max_new_tokens = result.get("max_new_tokens")
    has_integer_max = isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool)
    explicit_min_tokens = result.get("min_tokens")
    if (
        has_integer_max
        and isinstance(explicit_min_tokens, int)
        and not isinstance(explicit_min_tokens, bool)
        and explicit_min_tokens > max_new_tokens
    ):
        raise ValueError(f"min_tokens ({explicit_min_tokens}) must not exceed max_new_tokens ({max_new_tokens})")

    sampling = runtime.get("default_sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise ValueError("'options.default_sampling' must be an object")
        unknown_sampling = set(sampling) - set(_GENERATION_SAMPLING_KEYS)
        if unknown_sampling:
            raise ValueError(f"unsupported generation sampling option(s): {sorted(unknown_sampling)}")
        for key, value in sampling.items():
            if key == "seed":
                valid = isinstance(value, int) and not isinstance(value, bool) and -(1 << 63) <= value <= (1 << 63) - 1
            else:
                valid = _is_finite_number(value)
            if key == "temperature":
                valid = valid and value >= 0
            elif key == "top_p":
                valid = valid and 0 < value <= 1
            elif key in {"frequency_penalty", "presence_penalty"}:
                valid = valid and -2 <= value <= 2
            elif key == "top_k":
                valid = valid and isinstance(value, int) and not isinstance(value, bool) and value >= 1
            elif key == "min_new_tokens":
                valid = valid and isinstance(value, int) and not isinstance(value, bool) and value >= 0
            if not valid:
                raise ValueError(f"'options.default_sampling.{key}' has an invalid value")

        for source, target in _GENERATION_SAMPLING_KEYS.items():
            if source in sampling and result.get(target) is None:
                value = sampling[source]
                if source == "min_new_tokens" and has_integer_max and value > max_new_tokens:
                    if isinstance(request_sampling, dict) and source in request_sampling:
                        raise ValueError(
                            f"'options.default_sampling.min_new_tokens' ({value}) "
                            f"must not exceed max_new_tokens ({max_new_tokens})"
                        )
                    value = max_new_tokens
                result[target] = value

    stop_tokens = runtime.get("stop_tokens")
    if stop_tokens is not None:
        if not isinstance(stop_tokens, list) or not all(isinstance(item, str) and item for item in stop_tokens):
            raise ValueError("'options.stop_tokens' must be an array of non-empty strings")
        explicit_stop = result.get("stop")
        if explicit_stop is None:
            result["stop"] = list(stop_tokens)
        elif isinstance(explicit_stop, list):
            result["stop"] = [*explicit_stop, *(item for item in stop_tokens if item not in explicit_stop)]

    for key in ("first_chunk_timeout_s", "inter_chunk_timeout_s", "overall_timeout_s"):
        value = runtime.get(key)
        if value is not None and (not _is_finite_number(value) or value <= 0):
            raise ValueError(f"'options.{key}' must be a positive number")

    return result


_GENERATION_CLOSE_TIMEOUT_S = 2.0
_EXHAUSTED = object()
_GENERATION_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()


@dataclass(frozen=True, slots=True)
class GenerationTimeouts:
    """Governed generation timeouts in seconds; ``None`` leaves that bound off."""

    first_chunk_s: float | None = None
    overall_s: float | None = None


class GenerationTimeoutError(TimeoutError):
    """A buffered generation exceeded a governed timeout.

    ``code`` matches the gateway's generation timeout codes, so a direct server
    and a gateway report the same expiry the same way.
    """

    def __init__(self, kind: Literal["first_chunk", "overall"]) -> None:
        self.kind = kind
        self.code = f"{kind}_timeout"
        super().__init__(f"Generation aborted: {kind} timeout")


def _timeout_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not _is_finite_number(value):
        return None
    return float(value) if value > 0 else None


def resolve_generation_timeouts(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
) -> GenerationTimeouts:
    """Resolve the profile and request ``first_chunk_timeout_s`` / ``overall_timeout_s``.

    Call after :func:`apply_generation_runtime_options` has validated the same
    options.
    """
    runtime = merge_runtime_options(config, request_options)
    return GenerationTimeouts(
        first_chunk_s=_timeout_seconds(runtime.get("first_chunk_timeout_s")),
        overall_s=_timeout_seconds(runtime.get("overall_timeout_s")),
    )


async def bound_generation[ChunkT](
    chunks: AsyncIterator[ChunkT],
    timeouts: GenerationTimeouts,
) -> AsyncIterator[ChunkT]:
    """Yield ``chunks`` under the first-chunk and overall timeouts.

    On expiry the pending read is cancelled, which aborts the engine request,
    and :class:`GenerationTimeoutError` is raised. Engine cleanup is waited for
    at most ``_GENERATION_CLOSE_TIMEOUT_S``; cleanup still running then keeps
    going in the background, so a hung abort cannot hold back the response and
    is not itself cancelled.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    first_chunk_at = None if timeouts.first_chunk_s is None else started + timeouts.first_chunk_s
    overall_at = None if timeouts.overall_s is None else started + timeouts.overall_s
    received_first = False
    pending_read: asyncio.Task[Any] | None = None
    cleanup_by: float | None = None
    try:
        while True:
            pending: list[tuple[float, Literal["first_chunk", "overall"]]] = []
            if first_chunk_at is not None and not received_first:
                pending.append((first_chunk_at, "first_chunk"))
            if overall_at is not None:
                pending.append((overall_at, "overall"))
            if not pending:
                chunk = await _next_chunk(chunks)
            else:
                deadline, kind = min(pending, key=lambda entry: entry[0])
                read = asyncio.ensure_future(_next_chunk(chunks))
                pending_read = read
                try:
                    done, _ = await asyncio.wait({read}, timeout=max(0.0, deadline - loop.time()))
                except BaseException:
                    read.cancel()
                    raise
                if read not in done:
                    read.cancel()
                    cleanup_by = loop.time() + _GENERATION_CLOSE_TIMEOUT_S
                    await asyncio.wait({read}, timeout=_GENERATION_CLOSE_TIMEOUT_S)
                    if read.done():
                        _log_cleanup_failure(read, "cancelled generation read")
                    raise GenerationTimeoutError(kind)
                pending_read = None
                chunk = read.result()
            if chunk is _EXHAUSTED:
                return
            received_first = True
            yield cast("ChunkT", chunk)
    finally:
        if pending_read is not None and not pending_read.done():
            _finish_in_background(pending_read, "cancelled generation read")
        else:
            budget = _GENERATION_CLOSE_TIMEOUT_S if cleanup_by is None else cleanup_by - loop.time()
            await _close_within(chunks, budget)


async def _next_chunk(chunks: AsyncIterator[Any]) -> Any:
    try:
        return await anext(chunks)
    except StopAsyncIteration:
        return _EXHAUSTED


async def _aclose(chunks: Any) -> None:
    await chunks.aclose()


async def _close_within(chunks: AsyncIterator[Any], budget_s: float) -> None:
    if getattr(chunks, "aclose", None) is None:
        return
    close = asyncio.ensure_future(_aclose(chunks))
    done, _ = await asyncio.wait({close}, timeout=max(0.0, budget_s))
    if close not in done:
        _finish_in_background(close, "generation stream close")
        return
    _log_cleanup_failure(close, "generation stream close")


def _finish_in_background(task: asyncio.Task[Any], context: str) -> None:
    _GENERATION_CLEANUP_TASKS.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        _GENERATION_CLEANUP_TASKS.discard(finished)
        _log_cleanup_failure(finished, context)

    task.add_done_callback(_done)


def _log_cleanup_failure(task: asyncio.Task[Any], context: str) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None and not isinstance(error, asyncio.CancelledError):
        logger.warning("%s failed after the generation outcome was decided", context, exc_info=error)
