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

import math
from typing import TYPE_CHECKING, Any

from sie_server.types.inputs import InvalidInputError

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig, ResolvedProfile


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


def merge_runtime_options_with_profile(
    config: ModelConfig,
    request_options: dict[str, Any] | None,
) -> tuple[dict[str, Any], ResolvedProfile]:
    """Return merged adapter options and the profile used to derive them."""
    resolved = _resolve_profile_or_raise(config, request_options)
    merged: dict[str, Any] = dict(resolved.runtime)
    if request_options:
        merged |= {key: value for key, value in request_options.items() if key != "profile"}
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
            unknown profile.
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
    "presence_penalty": "presence_penalty",
    "top_k": "top_k",
    "min_new_tokens": "min_tokens",
}


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

    sampling = runtime.get("default_sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise ValueError("'options.default_sampling' must be an object")
        unknown_sampling = set(sampling) - set(_GENERATION_SAMPLING_KEYS)
        if unknown_sampling:
            raise ValueError(f"unsupported generation sampling option(s): {sorted(unknown_sampling)}")
        for key, value in sampling.items():
            numeric = isinstance(value, int | float) and not isinstance(value, bool)
            valid = numeric and math.isfinite(float(value))
            if key == "temperature":
                valid = valid and value >= 0
            elif key == "top_p":
                valid = valid and 0 < value <= 1
            elif key == "presence_penalty":
                valid = valid and -2 <= value <= 2
            elif key == "top_k":
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 1
            elif key == "min_new_tokens":
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
            if not valid:
                raise ValueError(f"'options.default_sampling.{key}' has an invalid value")
        for source, target in _GENERATION_SAMPLING_KEYS.items():
            if source in sampling and result.get(target) is None:
                result[target] = sampling[source]

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
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"'options.{key}' must be a positive number")

    return result
