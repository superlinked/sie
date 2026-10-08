"""Pool-isolation validator.

A ``GenerationAdapter`` model cannot share a worker pool with non-gen
(encode/score/extract) models. The check is worker-process-local
because ``SIE_POOL`` is read per-worker at boot — there is no central
cross-worker pool registry. The validator fires when configs are added
to a worker's :class:`ModelRegistry` (i.e. at startup config-load and
at hot-reload time).

Rationale: mixed-pool fairness across request shapes is a separate concern. For the current horizon
the simpler invariant — a single pool holds a single task class — is
what the gateway's routing and the worker's batch-shape assumptions
already assume. Reject loudly at config load so operators see the
misconfiguration before it surfaces as silent batch-budget thrash.

Only models the worker can serve take part. With remote serving off
(``SIE_REMOTE_SERVING``), the worker refuses every remote profile at load,
so a model served through a remote profile never runs in its pool.
"""

from __future__ import annotations

import logging

from sie_server.config.model import ModelConfig, is_remote_adapter_path
from sie_server.config.upstreams import remote_serving_enabled

logger = logging.getLogger(__name__)


def is_generation_model(config: ModelConfig) -> bool:
    """Return ``True`` if a config declares a generation task.

    Proxy for "this model's adapter is a :class:`GenerationAdapter`
    subclass" — adapter-path string-matching is fragile; checking
    ``tasks.generate`` is the same signal the rest of the server uses
    (e.g. ``ModelConfig.outputs`` already reads it).
    """
    return config.tasks.generate is not None


def counts_toward_pool_isolation(config: ModelConfig) -> bool:
    """Return ``True`` unless this worker refuses to load ``config``.

    With remote serving off, a model whose served profile is remote is
    refused at load, so it never shares the pool with another model.
    """
    return remote_serving_enabled() or not is_remote_adapter_path(config.resolve_profile("default").adapter_path)


class PoolIsolationError(ValueError):
    """Raised when a model would join a pool with an incompatible task class."""


def validate_pool_isolation(
    *,
    candidate_name: str,
    candidate_config: ModelConfig,
    existing_configs: dict[str, ModelConfig],
    pool_name: str,
    fairness_enabled: bool = False,
) -> None:
    """Validate that ``candidate_config`` can join the pool occupied by
    ``existing_configs``.

    Raises :class:`PoolIsolationError` if a generation model would join
    a pool containing non-generation models, or vice versa. Names both
    offending models in the error message and recommends a separate
    pool (different ``SIE_POOL``).

    When ``fairness_enabled`` is ``True`` (the operator opted into
    ``pool.fairness`` — see
    :class:`sie_server.processors.work_class_scheduler.WorkClassScheduler`),
    a mixed pool is *intended*: the conflict is logged at WARNING and
    allowed instead of raising, because the fair-queue scheduler shares the
    worker's slots between classes with per-class floors.

    A model this worker refuses to load (see
    :func:`counts_toward_pool_isolation`) neither conflicts nor is
    conflicted with.
    """
    if not counts_toward_pool_isolation(candidate_config):
        return
    candidate_is_gen = is_generation_model(candidate_config)
    # Accumulate every conflicting existing config rather than stopping
    # on the first hit. An operator with 4 misconfigured models would
    # otherwise have to fix-and-reload 4 times to surface each error;
    # collecting all conflicts up front lets a single bulk-edit pass
    # cover the whole batch.
    conflicts: list[str] = []
    for existing_name, existing_config in existing_configs.items():
        if existing_name == candidate_name:
            # Re-registration of the same model (hot reload) is fine.
            continue
        if not counts_toward_pool_isolation(existing_config):
            continue
        existing_is_gen = is_generation_model(existing_config)
        if candidate_is_gen == existing_is_gen:
            continue
        conflicts.append(existing_name)
    if conflicts:
        if candidate_is_gen:
            msg = (
                f"cannot register generation model {candidate_name!r} into pool "
                f"{pool_name!r} which already contains non-generation model(s) "
                f"{sorted(conflicts)!r}; configure these models on separate workers "
                "(different SIE_POOL values)"
            )
        else:
            msg = (
                f"cannot register non-generation model {candidate_name!r} into pool "
                f"{pool_name!r} which already contains generation model(s) "
                f"{sorted(conflicts)!r}; configure these models on separate workers "
                "(different SIE_POOL values)"
            )
        if fairness_enabled:
            logger.warning(
                "mixed-pool fairness enabled: allowing %s",
                msg,
            )
            return
        raise PoolIsolationError(msg)


# --- Legacy scalar lora_id exclusion -----------------------------------------
#
# Multi-LoRA on generation models *has shipped*: it is declared via
# ``profile.adapter_options.loadtime["lora_paths"]`` (list of adapter ids,
# wired into the SGLang LoRA slot machinery by the model loader). The
# remaining exclusion is the *legacy* scalar shape
# ``profile.adapter_options.runtime["lora_id"]`` — a pre-Multi-LoRA spelling
# that no longer maps to a supported code path on the generation primitive.
# Reject at config-load time (registry hot-reload path) so the legacy spelling
# surfaces as a clear, actionable error rather than silent missing-adapter
# behaviour at request time.
#
# Empty-string ``lora_id`` is treated as absent (matches the historical
# loader's ``if lora_id:`` truthiness check).


class LegacyScalarLoraIdError(ValueError):
    """Raised when a generation model declares the legacy scalar ``lora_id``.

    Multi-LoRA on generation has shipped via
    ``adapter_options.loadtime.lora_paths``; the scalar
    ``adapter_options.runtime.lora_id`` spelling is the legacy form and is
    rejected at config-load time so the misconfiguration is visible before
    request traffic.
    """


def _has_legacy_scalar_lora_id(config: ModelConfig) -> bool:
    """Return ``True`` iff any profile declares a non-empty scalar ``lora_id``.

    This specifically inspects ``profile.adapter_options.runtime["lora_id"]``
    — the legacy scalar spelling. It does *not* look at the shipped
    Multi-LoRA path (``adapter_options.loadtime["lora_paths"]``).
    Empty-string ``lora_id`` is treated as absent (matches the historical
    loader's truthiness check).
    """
    for profile in config.profiles.values():
        runtime = profile.adapter_options.runtime if profile.adapter_options else None
        if not runtime:
            continue
        lora_id = runtime.get("lora_id")
        if lora_id:  # truthiness check matches historical loader behaviour
            return True
    return False


def _legacy_scalar_lora_id_profile_names(config: ModelConfig) -> list[str]:
    """Names of profiles that declare a non-empty scalar ``lora_id``."""
    offending: list[str] = []
    for profile_name, profile in config.profiles.items():
        runtime = profile.adapter_options.runtime if profile.adapter_options else None
        if not runtime:
            continue
        if runtime.get("lora_id"):
            offending.append(profile_name)
    return offending


def validate_no_legacy_scalar_lora_id(
    *,
    name: str,
    config: ModelConfig,
) -> None:
    """Reject a generation model that uses the legacy scalar ``lora_id`` shape.

    Raises :class:`LegacyScalarLoraIdError` if ``config`` is a generation
    model AND any profile declares ``adapter_options.runtime["lora_id"]``.
    Non-generation models with the scalar spelling are accepted (the
    encode/score/extract LoRA path predates Multi-LoRA generation and
    continues to use the scalar form).

    The shipped Multi-LoRA generation path
    (``adapter_options.loadtime["lora_paths"]``) is *not* affected by this
    validator and is fully supported.

    Unlike :func:`validate_pool_isolation`, this check is not pool-scoped —
    it is a hard invariant that fires regardless of ``SIE_POOL``.
    """
    if not is_generation_model(config):
        return
    offending = _legacy_scalar_lora_id_profile_names(config)
    if not offending:
        return
    msg = (
        f"cannot register generation model {name!r}: profile(s) "
        f"{offending!r} declare the legacy scalar ``lora_id`` in "
        f"adapter_options.runtime, which is not supported on the "
        f"generation primitive. Multi-LoRA generation is shipped via "
        f"``adapter_options.loadtime.lora_paths`` (list of adapter ids); "
        f"migrate to that form, or remove the ``lora_id`` entry."
    )
    raise LegacyScalarLoraIdError(msg)
