"""Configuration-load gates for a model's routing block.

``ModelConfig`` checks the shape of the block. These gates refuse a valid
block that this server does not serve, at the same points that load a model
config: the models directory, a config added at runtime, and a config snapshot.
"""

from __future__ import annotations

from pathlib import Path

from sie_server.adapters._generation_base import GenerationAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.loader import expand_profile_variants, resolve_adapter_class

_HYBRID_POLICIES = frozenset({"fallback", "threshold"})
_EQUIVALENCE_TASKS = ("encode", "score")


def hybrid_equivalence_refusal(config: ModelConfig) -> str | None:
    """Why ``config`` may not serve a primitive from both its local and its remote profile, or ``None``.

    Under ``fallback`` and ``threshold`` one model name is served by two
    backends. For ``encode`` a difference between them is permanent, because
    the vectors are stored; for ``score`` it changes the score scale. Both are
    refused until the remote profile is shown to be equivalent to the local one.
    """
    routing = config.routing
    if routing is None or routing.policy not in _HYBRID_POLICIES:
        return None
    tasks = [name for name in _EQUIVALENCE_TASKS if getattr(config.tasks, name) is not None]
    if not tasks:
        return None
    return (
        f"Model '{config.sie_id}' would serve {' and '.join(tasks)} from both its local profile and remote "
        f"profile '{routing.fallback_profile}' under routing policy '{routing.policy}'; that is refused until "
        "the remote profile is shown to be equivalent to the local one"
    )


def remote_output_refusal(config: ModelConfig) -> str | None:
    """Why the remote profile cannot serve every output the model declares, or ``None``.

    Under ``fallback`` and ``threshold`` a caller cannot know which profile
    serves a request, so the outputs a request may ask for cannot depend on
    it. The remote profile's adapter class must declare every output the
    model does.
    """
    routing = config.routing
    if routing is None or routing.policy not in _HYBRID_POLICIES or routing.fallback_profile is None:
        return None
    variant = expand_profile_variants([config])[f"{config.sie_id}:{routing.fallback_profile}"]
    try:
        adapter_class = resolve_adapter_class(variant, Path())
    except (ImportError, ValueError):
        return f"Model '{config.sie_id}': the adapter of remote profile '{routing.fallback_profile}' cannot be imported"
    if config.tasks.generate is not None and not issubclass(adapter_class, GenerationAdapter):
        return f"Model '{config.sie_id}': remote profile '{routing.fallback_profile}' requires a GenerationAdapter"
    spec = getattr(adapter_class, "spec", None)
    uncovered = sorted(set(config.outputs) - set(getattr(spec, "outputs", ())))
    if not uncovered:
        return None
    return (
        f"Model '{config.sie_id}' declares {', '.join(uncovered)}, which remote profile "
        f"'{routing.fallback_profile}' does not produce; under routing policy '{routing.policy}' either profile "
        "may serve a request, so the remote profile must produce every output the model declares"
    )


def validate_model_routing(config: ModelConfig) -> None:
    """Refuse a routing block this server cannot honour. Raises ``ValueError``."""
    routing = config.routing
    if routing is None:
        return
    if routing.policy == "threshold":
        msg = f"Model '{config.sie_id}': routing policy 'threshold' is not available yet"
        raise ValueError(msg)
    refusal = hybrid_equivalence_refusal(config) or remote_output_refusal(config)
    if refusal is not None:
        raise ValueError(refusal)
