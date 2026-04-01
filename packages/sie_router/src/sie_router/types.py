"""Types for SIE Router."""

from dataclasses import dataclass, field
from enum import Enum

from sie_sdk.types import ModelSummary, WorkerInfo


@dataclass
class MachineProfile:
    """Machine profile configuration.

    Machine profiles abstract cloud-specific hardware configuration (GPU type,
    machine type, spot/preemptible). Used by router to validate pool creation
    and match workers to WorkerGroups.
    """

    name: str  # e.g., "l4-spot"
    gpu_type: str  # e.g., "nvidia-l4"
    machine_type: str = ""  # e.g., "g2-standard-8" (optional for validation)
    spot: bool = False  # Whether this uses spot/preemptible instances


class WorkerHealth(Enum):
    """Worker health status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class WorkerState:
    """State of a single worker, updated via WebSocket."""

    url: str
    name: str = ""

    # GPU info
    gpu_count: int = 1

    # Machine profile for routing.
    # - In K8s: Set via SIE_MACHINE_PROFILE env var on worker (e.g., "l4-spot")
    # - Standalone: Detected GPU type (e.g., "l4")
    machine_profile: str = ""

    # Bundle info (for multi-bundle routing)
    bundle: str = "default"

    # Loaded models
    models: list[str] = field(default_factory=list)

    # Load info
    queue_depth: int = 0
    # Memory in bytes (consistent with GPUMetrics from worker)
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0

    # Health
    health: WorkerHealth = WorkerHealth.UNKNOWN
    last_heartbeat: float = 0.0

    @property
    def healthy(self) -> bool:
        """Check if worker is healthy."""
        return self.health == WorkerHealth.HEALTHY

    @property
    def memory_utilization(self) -> float:
        """Get GPU memory utilization as percentage."""
        if self.memory_total_bytes <= 0:
            return 0.0
        return self.memory_used_bytes / self.memory_total_bytes


@dataclass
class ClusterStatus:
    """Aggregated cluster status for /ws/cluster-status endpoint."""

    timestamp: float
    worker_count: int
    gpu_count: int
    models_loaded: int
    total_qps: float
    workers: list[WorkerInfo]
    models: list[ModelSummary]


@dataclass
class ProvisioningResponse:
    """Response when hardware is not available (202 Accepted)."""

    status: str = "provisioning"
    gpu: str = ""
    estimated_wait_s: int = 180
    message: str = ""


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
    pool: str | None = None
    gpu: str | None = None
    worker: str | None = None
    latency_ms: float | None = None
    body_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a dict with None-valued fields omitted."""
        from dataclasses import asdict

        return {k: v for k, v in asdict(self).items() if v is not None}
