"""Configuration-load gates for a model's routing block.

``ModelConfig`` checks the shape of the block. These gates refuse a valid
block that this server does not serve, at the same points that load a model
config: the models directory, a config added at runtime, and a config snapshot.
"""

from __future__ import annotations

from sie_server.config.model import ModelConfig

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


def validate_model_routing(config: ModelConfig) -> None:
    """Refuse a routing block this server cannot honour. Raises ``ValueError``."""
    routing = config.routing
    if routing is None:
        return
    if routing.policy == "threshold":
        msg = f"Model '{config.sie_id}': routing policy 'threshold' is not available yet"
        raise ValueError(msg)
    refusal = hybrid_equivalence_refusal(config)
    if refusal is not None:
        raise ValueError(refusal)
