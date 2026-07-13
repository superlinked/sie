"""Configuration system for Dusk.

Configuration is resolved from three layers, lowest to highest precedence:

1. Dataclass defaults defined on :class:`Config`.
2. A ``dusk.yaml`` file in the current working directory, if present.
3. Environment variables prefixed ``DUSK_`` (e.g. ``DUSK_SWEEP_THRESHOLD``).

The active configuration is a process-wide singleton: :func:`get_config`
loads it once and caches it. Detections and the engine receive a
:class:`Config` instance rather than reading any hardcoded values, so every
threshold has a single, overridable source of truth.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Any, get_type_hints

import yaml

logger = logging.getLogger("dusk.config")

#: Prefix for environment-variable overrides.
ENV_PREFIX = "DUSK_"
#: Default config file looked up in the current working directory.
CONFIG_FILENAME = "dusk.yaml"


class ConfigError(Exception):
    """Raised when configuration values are invalid or cannot be loaded."""


@dataclass(frozen=True)
class Config:
    """Immutable Dusk configuration.

    Attributes:
        sweep_threshold: Unique destinations within the sweep window above
            which a source is considered to be sweeping.
        sweep_window_seconds: Length of the sweep sliding window, in seconds.
        sweep_timing_std_threshold: Inter-packet interval standard deviation
            (seconds) below which timing is judged machine-regular.
        boundary_port_threshold: Unique destination ports for one
            ``(src, dst)`` pair above which a port scan is flagged.
        boundary_window_seconds: Length of the boundary sliding window, in
            seconds.
        gate_block_threshold: Anomaly score (0..1) at or above which the agent
            action gate refuses an action (WOULD-BLOCK in watch mode, BLOCK in
            enforce mode).
        alert_log_path: Path the alert responder appends JSON entries to.
        log_level: Default logging level name (e.g. ``"WARNING"``).
        enforce: When ``True``, the ``/v1/gate`` HTTP service renders BLOCK
            for refused actions instead of WOULD-BLOCK. Watch mode (``False``)
            is the default.
        sie_endpoint: Base URL of the self-hosted SIE container the gate
            calls for encode/score/extract. No API key is needed for a
            self-hosted instance.
        sie_encode_model: Catalog model id SIE uses for the encode primitive.
        sie_score_model: Catalog model id SIE uses for the score (rerank)
            primitive.
        sie_extract_model: Catalog model id SIE uses for the extract
            primitive.
        sie_timeout_ms: Timeout in milliseconds for a single SIE call.
        n8n_alert_url: n8n webhook fired only when a gate verdict is refused.
        n8n_report_url: n8n webhook fired on every gate verdict.
        n8n_decision_url: n8n webhook fired on every gate verdict, kept on
            its own URL so automation and reporting can be routed separately.
        n8n_max_workers: Size of the bounded thread pool webhook firing uses,
            so a sustained burst of refused verdicts can't spawn unbounded
            OS threads.
        n8n_max_queued: Webhook sends allowed to be queued or in flight at
            once; past this, new ones are dropped and logged instead of
            growing the backlog without limit.
        offense_memory_path: File the gate persists its repeat-offense
            memory to, so it survives a service restart. Empty (the
            default) disables persistence -- the signal still scores within
            one process lifetime, from an in-memory-only store, but nothing
            is written to disk. Deliberately opt-in, matching the
            ``n8n_*_url`` fields: an operator running the service outside
            the Docker Compose stack (which sets this explicitly) shouldn't
            get a file silently dropped into whatever the process's current
            working directory happens to be.
        repeat_offense_max_contribution: Upper bound on how much the
            repeat-offense signal alone can add to an anomaly score, so it
            is one input among several rather than something that alone can
            force a BLOCK on an otherwise benign action.
        repeat_offense_half_life_days: Time, in days, after which a past
            offense's contribution to the repeat-offense signal is halved.
            Models the intuition that an agent blocked once six months ago
            is not the same risk as one blocked five minutes ago.
    """

    sweep_threshold: int = 15
    sweep_window_seconds: float = 10.0
    sweep_timing_std_threshold: float = 0.05
    boundary_port_threshold: int = 10
    boundary_window_seconds: float = 30.0
    gate_block_threshold: float = 0.6
    alert_log_path: str = "dusk-alerts.json"
    log_level: str = "WARNING"
    enforce: bool = False
    sie_endpoint: str = "http://sie:8080"
    sie_encode_model: str = "BAAI/bge-m3"
    sie_score_model: str = "BAAI/bge-reranker-v2-m3"
    sie_extract_model: str = "urchade/gliner_multi-v2.1"
    sie_timeout_ms: int = 10000
    n8n_alert_url: str = ""
    n8n_report_url: str = ""
    n8n_decision_url: str = ""
    n8n_max_workers: int = 8
    n8n_max_queued: int = 200
    offense_memory_path: str = ""
    repeat_offense_max_contribution: float = 0.3
    repeat_offense_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        """Validate values after construction.

        Raises:
            ConfigError: If any numeric threshold is non-positive, the log
                level is not a recognised name, or a configured URL is
                non-empty but missing an http(s) scheme.
        """
        positive_numeric = (
            "sweep_threshold",
            "sweep_window_seconds",
            "sweep_timing_std_threshold",
            "boundary_port_threshold",
            "boundary_window_seconds",
            "gate_block_threshold",
            "sie_timeout_ms",
            "n8n_max_workers",
            "n8n_max_queued",
            "repeat_offense_max_contribution",
            "repeat_offense_half_life_days",
        )
        for name in positive_numeric:
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"Config value '{name}' must be > 0, got {value!r}")

        if not isinstance(logging.getLevelName(self.log_level), int):
            raise ConfigError(
                f"Config value 'log_level' is not a valid level name: {self.log_level!r}"
            )

        url_fields = ("sie_endpoint", "n8n_alert_url", "n8n_report_url", "n8n_decision_url")
        for name in url_fields:
            value = getattr(self, name)
            if value and not value.startswith(("http://", "https://")):
                raise ConfigError(
                    f"Config value '{name}' must be empty or start with http:// or "
                    f"https://, got {value!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view of the configuration."""
        return dataclasses.asdict(self)


def _coerce(field_type: Any, raw: Any, source: str) -> Any:  # noqa: ANN401
    """Coerce ``raw`` to ``field_type``, raising :class:`ConfigError` on failure.

    Args:
        field_type: The dataclass field's declared type (``int``, ``float``
            or ``str``).
        raw: The value read from YAML or the environment.
        source: Human-readable origin used in error messages.

    Returns:
        ``raw`` converted to ``field_type``.

    Raises:
        ConfigError: If ``raw`` cannot be coerced to ``field_type``.
    """
    try:
        if field_type is bool:
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}
        if field_type is int:
            return int(raw)
        if field_type is float:
            return float(raw)
        return str(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value for {source}: {raw!r} ({exc})") from exc


def _load_yaml_overrides(path: str) -> dict[str, Any]:
    """Read overrides from a ``dusk.yaml`` file, returning {} if absent.

    Raises:
        ConfigError: If the file exists but cannot be parsed or is not a
            mapping.
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read config file '{path}': {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Config file '{path}' must contain a mapping, got {type(loaded).__name__}"
        )

    logger.debug("Loaded %d override(s) from %s", len(loaded), path)
    return loaded


def _env_overrides(field_names: set[str]) -> dict[str, str]:
    """Collect ``DUSK_*`` environment overrides for known fields."""
    overrides: dict[str, str] = {}
    for env_name, value in os.environ.items():
        if not env_name.startswith(ENV_PREFIX):
            continue
        field_name = env_name[len(ENV_PREFIX) :].lower()
        if field_name in field_names:
            overrides[field_name] = value
            logger.debug("Applied env override %s", env_name)
    return overrides


def load_config(config_path: str = CONFIG_FILENAME) -> Config:
    """Build a :class:`Config` from defaults, ``dusk.yaml`` and env vars.

    Args:
        config_path: Path to the YAML config file. Defaults to ``dusk.yaml``
            in the current working directory.

    Returns:
        A validated :class:`Config` instance.

    Raises:
        ConfigError: If a config file is malformed or a value is invalid.
    """
    # get_type_hints resolves real type objects even though this module uses
    # `from __future__ import annotations` (which would otherwise yield strings).
    type_by_field = get_type_hints(Config)
    field_names = set(type_by_field)

    merged: dict[str, Any] = {}

    for field_name, raw in _load_yaml_overrides(config_path).items():
        if field_name not in field_names:
            logger.warning("Ignoring unknown config key in %s: %s", config_path, field_name)
            continue
        merged[field_name] = _coerce(type_by_field[field_name], raw, f"{config_path}:{field_name}")

    for field_name, raw in _env_overrides(field_names).items():
        merged[field_name] = _coerce(
            type_by_field[field_name], raw, f"{ENV_PREFIX}{field_name.upper()}"
        )

    return Config(**merged)


_active_config: Config | None = None


def get_config() -> Config:
    """Return the process-wide singleton config, loading it on first use."""
    global _active_config
    if _active_config is None:
        _active_config = load_config()
    return _active_config


def set_config(config: Config) -> None:
    """Replace the active singleton config (primarily for tests and the CLI)."""
    global _active_config
    _active_config = config


def reset_config() -> None:
    """Clear the cached singleton so the next :func:`get_config` reloads it."""
    global _active_config
    _active_config = None
