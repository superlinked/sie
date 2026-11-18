"""Which profile of a model serves a profile's grammar-constrained requests.

A model may declare ``tasks.generate.grammar_profile``, and a profile may
declare a profile-scoped ``grammar_profile``, naming the sibling profile that
honours decode-time grammar enforcement. Speculative decoding bypasses the
grammar backend, so a grammar-constrained request must run on that sibling.
The rules mirror the gateway's ``ModelRegistry::grammar_route_variant``.
"""

from collections.abc import Mapping
from typing import Any

_RAW_SPECULATIVE_FLAGS = frozenset({"--enable-multi-layer-eagle", "--config"})
_GRAMMAR_SAFE_EXTRA_LAUNCH_ARGS = ("--quantization", "fp8")


def model_grammar_profile(config: Any) -> str | None:
    """The model-wide grammar profile, ``None`` when the model declares none."""
    generate = getattr(getattr(config, "tasks", None), "generate", None)
    grammar_profile = getattr(generate, "grammar_profile", None)
    return grammar_profile if isinstance(grammar_profile, str) else None


def grammar_serving_profile(config: Any, profile: str) -> str | None:
    """The profile that serves grammar-constrained requests for the non-default variant ``profile``.

    ``profile`` itself when it keeps them, ``None`` when nothing routes them.
    """
    profiles: Mapping[str, Any] = config.profiles
    if any(candidate.grammar_profile == profile for candidate in profiles.values()):
        return profile
    source = profiles.get(profile)
    if source is None:
        return None
    if source.grammar_profile is not None:
        return source.grammar_profile
    parent = profiles.get(source.extends)
    parent_grammar_profile = None if parent is None else parent.grammar_profile
    if parent_grammar_profile is not None and profile_is_grammar_compatible(config, profile, parent_grammar_profile):
        return profile
    grammar_profile = model_grammar_profile(config)
    if grammar_profile is None:
        return None
    if source.extends == grammar_profile and profile_is_grammar_compatible(config, profile, grammar_profile):
        return profile
    return grammar_profile


def profile_is_grammar_compatible(config: Any, profile: str, grammar_profile: str) -> bool:
    """Whether ``profile`` already serves grammar requests the way ``grammar_profile`` does."""
    try:
        candidate = config.resolve_profile(profile)
        grammar = config.resolve_profile(grammar_profile)
    except ValueError:
        return False
    return (
        candidate.adapter_path == grammar.adapter_path
        and explicitly_disables_speculation(candidate)
        and candidate.loadtime.get("grammar_backend") == grammar.loadtime.get("grammar_backend")
        and _extra_launch_args_are_compatible(candidate.loadtime, grammar.loadtime)
    )


def explicitly_disables_speculation(profile: Any) -> bool:
    """Whether a resolved profile turns speculative decoding off with no raw override."""
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
