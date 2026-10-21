"""Reject the legacy scalar ``lora_id`` spelling on generation models.

Multi-LoRA on generation models is declared through
``profile.adapter_options.loadtime["lora_paths"]``. The older scalar
``profile.adapter_options.runtime["lora_id"]`` no longer maps to a supported
code path on the generation primitive, so it is refused at config load.
"""

from __future__ import annotations

from sie_server.config.model import ModelConfig


def is_generation_model(config: ModelConfig) -> bool:
    """Return ``True`` if a config declares a generation task."""
    return config.tasks.generate is not None


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

    This is a hard invariant that fires regardless of ``SIE_POOL``.
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
