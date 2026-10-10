"""Grammar-profile admission shared by every worker generation ingress.

A model may declare ``tasks.generate.grammar_profile``, and a profile may
declare a profile-scoped ``grammar_profile``, naming the sibling profile that
honours decode-time grammar enforcement. Speculative decoding bypasses the
grammar backend, so a grammar-constrained request must run on that sibling.

The gateway rewrites the dispatch model before queue publication. The direct
HTTP routes and the queue worker resolve the same rewrite here, so a
constrained request runs on a grammar-safe profile whichever ingress it used.
The rules mirror the gateway's ``ModelRegistry::grammar_route_variant``:

- a model without a declared grammar profile keeps its id;
- a default profile that is already grammar compatible keeps its id;
- a request that names the grammar-safe variant keeps its id;
- a variant that directly extends the grammar profile and is grammar
  compatible keeps its id;
- a non-speculative variant compatible with its parent's declared grammar
  profile keeps its id;
- otherwise the request moves to ``{base}:{grammar_profile}``, and fails with
  ``unsupported_field`` when that variant is not served.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

from sie_server.adapters._generation_base import GenerationUnsupportedFieldError

logger = logging.getLogger(__name__)

_RAW_SPECULATIVE_FLAGS = frozenset({"--enable-multi-layer-eagle", "--config"})
_GRAMMAR_SAFE_EXTRA_LAUNCH_ARGS = ("--quantization", "fp8")


class ModelConfigSource(Protocol):
    def has_model(self, name: str) -> bool: ...

    def get_config(self, name: str) -> Any: ...


def resolve_grammar_serving_model(registry: ModelConfigSource, model_id: str) -> str:
    """Return the registered model id that serves a grammar-constrained request.

    Call only when the request carries a grammar. An unknown ``model_id`` is
    returned unchanged so the caller's own not-found handling applies.

    Raises:
        GenerationUnsupportedFieldError: The model routes grammar requests to a
            profile that this worker does not serve.
    """
    target = _grammar_target(registry, model_id)
    if target is None:
        return model_id
    base_id, grammar_profile = target
    if model_id == base_id and _profile_is_grammar_compatible(registry.get_config(base_id), "default", grammar_profile):
        return model_id
    target_id = f"{base_id}:{grammar_profile}"
    if model_id == target_id:
        return model_id
    if registry.has_model(target_id):
        return target_id
    logger.error(
        "model %s routes grammar-constrained generation to %s, which is not served; rejecting the request",
        model_id,
        target_id,
    )
    raise GenerationUnsupportedFieldError(
        "grammar",
        f"grammar-constrained generation for model '{model_id}' requires profile "
        f"'{grammar_profile}', which is not served",
    )


def _grammar_target(registry: ModelConfigSource, model_id: str) -> tuple[str, str] | None:
    if not registry.has_model(model_id):
        return None
    config = registry.get_config(model_id)
    source = getattr(config, "synthetic_profile_variant_source", None)
    if not isinstance(source, tuple):
        grammar_profile = _model_grammar_profile(config)
        return None if grammar_profile is None else (model_id, grammar_profile)
    base_id, source_profile = source
    if not registry.has_model(base_id):
        return _variant_only_target(config, base_id, source_profile)
    base = registry.get_config(base_id)
    profiles: Mapping[str, Any] = base.profiles
    if any(profile.grammar_profile == source_profile for profile in profiles.values()):
        return base_id, source_profile
    source_config = profiles.get(source_profile)
    if source_config is None:
        return None
    if source_config.grammar_profile is not None:
        return base_id, source_config.grammar_profile
    parent = profiles.get(source_config.extends)
    parent_grammar_profile = None if parent is None else parent.grammar_profile
    if parent_grammar_profile is not None and _profile_is_grammar_compatible(
        base, source_profile, parent_grammar_profile
    ):
        return base_id, source_profile
    grammar_profile = _model_grammar_profile(base)
    if grammar_profile is None:
        return None
    if source_config.extends == grammar_profile and _profile_is_grammar_compatible(
        base, source_profile, grammar_profile
    ):
        return base_id, source_profile
    return base_id, grammar_profile


def _variant_only_target(config: Any, base_id: str, source_profile: str) -> tuple[str, str] | None:
    """Resolve a variant whose base model is not registered on this worker.

    Without sibling profiles only the variant's own resolved launch shape is
    available, so an explicitly non-speculative variant is kept and any other
    variant routes to its declared grammar profile.
    """
    own_profile = config.profiles.get("default")
    scoped = getattr(own_profile, "grammar_profile", None)
    if isinstance(scoped, str):
        return base_id, scoped
    grammar_profile = _model_grammar_profile(config)
    if grammar_profile is None:
        return None
    if _explicitly_disables_speculation(config.resolve_profile("default")):
        return base_id, source_profile
    return base_id, grammar_profile


def _model_grammar_profile(config: Any) -> str | None:
    generate = getattr(getattr(config, "tasks", None), "generate", None)
    grammar_profile = getattr(generate, "grammar_profile", None)
    return grammar_profile if isinstance(grammar_profile, str) else None


def _profile_is_grammar_compatible(config: Any, profile: str, grammar_profile: str) -> bool:
    try:
        candidate = config.resolve_profile(profile)
        grammar = config.resolve_profile(grammar_profile)
    except ValueError:
        return False
    return (
        candidate.adapter_path == grammar.adapter_path
        and _explicitly_disables_speculation(candidate)
        and candidate.loadtime.get("grammar_backend") == grammar.loadtime.get("grammar_backend")
        and _extra_launch_args_are_compatible(candidate.loadtime, grammar.loadtime)
    )


def _explicitly_disables_speculation(profile: Any) -> bool:
    speculative = profile.loadtime.get("speculative")
    return (
        isinstance(speculative, Mapping)
        and speculative.get("enabled") is False
        and _has_no_raw_speculative_overrides(profile.loadtime)
    )


def _has_no_raw_speculative_overrides(loadtime: Mapping[str, Any]) -> bool:
    args = loadtime.get("extra_launch_args")
    if args is None:
        return True
    if not isinstance(args, list | tuple):
        return False
    for arg in args:
        if not isinstance(arg, str):
            return False
        flag = arg.split("=", 1)[0]
        if flag.startswith("--speculative-") or flag in _RAW_SPECULATIVE_FLAGS:
            return False
    return True


def _extra_launch_args_are_compatible(candidate: Mapping[str, Any], grammar: Mapping[str, Any]) -> bool:
    candidate_args = candidate.get("extra_launch_args")
    if candidate_args == grammar.get("extra_launch_args"):
        return True
    return isinstance(candidate_args, list | tuple) and tuple(candidate_args) == _GRAMMAR_SAFE_EXTRA_LAUNCH_ARGS
