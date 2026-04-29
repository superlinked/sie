from dataclasses import dataclass, field


@dataclass
class AuditEntry:
    """Structured audit log entry for an API request.

    Required fields are always present; optional fields default to None
    and are omitted from the serialised dict (via ``to_dict``).
    """

    event: str
    method: str
    endpoint: str
    status: int
    token_id: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    body_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a dict with None-valued fields omitted."""
        from dataclasses import asdict

        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ConfigDeltaNotification:
    """Notification payload for config changes published to NATS."""

    router_id: str  # identifies publisher; kept as router_id for Rust gateway compat
    model_id: str
    profiles_added: list[str]
    epoch: int
    model_config: str  # full YAML
    affected_bundles: list[str]
    bundle_config_hash: str  # set per-bundle subject publish


@dataclass
class ConfigSnapshotModel:
    """A single model entry in a config snapshot."""

    model_id: str
    model_config: dict
    raw_yaml: str | None
    affected_bundles: list[str] = field(default_factory=list)


@dataclass
class ConfigSnapshot:
    """Full config snapshot for gateway bootstrap."""

    snapshot_version: int  # always 1 for now
    epoch: int
    generated_at: str  # ISO 8601
    models: list[ConfigSnapshotModel] = field(default_factory=list)
