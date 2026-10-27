"""Host-RAM tier for evicted models.

When the registry evicts a model to free accelerator memory, it can park the
model's weights in host RAM instead of dropping them. The next request for the
model copies the weights back to the accelerator, which avoids rebuilding the
adapter and re-reading weights from the local disk cache, the cluster cache or
Hugging Face.

This module owns the bookkeeping and nothing else: the budget, the least
recently parked order, and which parked models still hold accelerator memory.
The registry decides when to park, wake or drop a model, and the adapters move
their own weights (``ModelAdapter.offload_to_host`` / ``restore_from_host``).

Budget accounting is in host bytes the tier is responsible for:

- a parked model is charged the host bytes its adapter reported when it parked;
- an adapter that keeps its host copy after waking (SGLang's memory saver reuses
  the buffer for the next park) stays charged while it is resident, until the
  model is fully unloaded. Those charges cannot be dropped to make room; only
  parked models can.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HOST_CACHE_ENV = "SIE_HOST_CACHE_BYTES"
AUTO = "auto"
# ``auto`` budgets this share of the container memory limit (or of physical RAM
# when the container has none). The rest stays for the worker process, engine
# subprocesses, the page cache and /dev/shm.
AUTO_FRACTION = 0.25

_CGROUP_V2_LIMIT = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
# cgroup v1 reports "no limit" as a huge page-aligned number.
_CGROUP_V1_UNLIMITED = 1 << 60

_SIZE_SUFFIXES: dict[str, int] = {
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KI": 1024,
    "MI": 1024**2,
    "GI": 1024**3,
    "TI": 1024**4,
}


def parse_host_cache_setting(value: object) -> int | None:
    """Parse ``SIE_HOST_CACHE_BYTES``: ``None`` for ``auto``, else a byte count.

    Accepts ``auto``, a plain integer byte count, or a number with a binary
    (``KiB``/``MiB``/``GiB``/``TiB``, also Kubernetes ``Ki``/``Mi``/``Gi``/``Ti``)
    or decimal (``KB``/``MB``/``GB``/``TB``) suffix. ``0`` disables the tier.

    Raises:
        ValueError: For anything else, so a malformed budget fails at startup
            instead of silently disabling or over-committing the tier.
    """
    if isinstance(value, bool):
        raise ValueError(f"invalid {HOST_CACHE_ENV} value {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{HOST_CACHE_ENV} must not be negative, got {value}")
        return value
    if not isinstance(value, str):
        raise ValueError(f"invalid {HOST_CACHE_ENV} value {value!r}")
    text = value.strip()
    if text.lower() == AUTO:
        return None
    number, multiplier = text, 1
    upper = text.upper()
    for suffix, factor in _SIZE_SUFFIXES.items():
        if upper.endswith(suffix):
            number, multiplier = text[: -len(suffix)].strip(), factor
            break
    try:
        parsed = float(number)
    except ValueError:
        parsed = math.nan
    if not math.isfinite(parsed) or parsed < 0 or (multiplier == 1 and not number.isdigit()):
        msg = f"invalid {HOST_CACHE_ENV} value {value!r}: expected 'auto', bytes, or a size such as 8GiB"
        raise ValueError(msg)
    return int(parsed * multiplier)


def host_cache_configured() -> bool:
    """Whether the process-level setting leaves the tier on (anything but ``0``).

    Engines that must decide at launch whether they can park later (SGLang's
    memory saver) read this. A malformed value counts as off here; the registry
    rejects it at startup.
    """
    try:
        return parse_host_cache_setting(os.environ.get(HOST_CACHE_ENV, AUTO)) != 0
    except ValueError:
        return False


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def physical_memory_bytes() -> int | None:
    """Physical RAM of the machine, or None when the platform does not say."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def container_memory_limit_bytes(
    *,
    cgroup_v2: Path = _CGROUP_V2_LIMIT,
    cgroup_v1: Path = _CGROUP_V1_LIMIT,
) -> int | None:
    """The memory limit of this container (cgroup v2, then v1), or physical RAM.

    The smaller of the cgroup limit and physical RAM wins: a limit larger than
    the machine is not memory the process can use.
    """
    physical = physical_memory_bytes()
    limit = _read_int(cgroup_v2)
    if limit is None:
        limit = _read_int(cgroup_v1)
        if limit is not None and limit >= _CGROUP_V1_UNLIMITED:
            limit = None
    if limit is None or limit <= 0:
        return physical
    return min(limit, physical) if physical else limit


def resolve_host_cache_budget(setting: object, devices: Iterable[str], *, memory_limit: int | None = None) -> int:
    """Resolve the configured setting to a budget in bytes; 0 disables the tier.

    ``auto`` enables the tier only for CUDA workers, at ``AUTO_FRACTION`` of the
    container memory limit. On CPU there is nothing to move, and on Apple
    silicon the accelerator shares host memory, so parking would free nothing.
    An explicit size applies to any device, so tests and fake engines can
    exercise the tier on CPU; real adapters still decline to park where it would
    not free accelerator memory.
    """
    explicit = parse_host_cache_setting(setting)
    if explicit is not None:
        return explicit
    if not any(device.strip().lower().startswith("cuda") for device in devices):
        return 0
    limit = memory_limit if memory_limit is not None else container_memory_limit_bytes()
    if not limit:
        return 0
    return int(limit * AUTO_FRACTION)


@dataclass
class ParkedModel:
    """One model whose weights live in host RAM.

    Attributes:
        name: Registry model id.
        loaded: The registry's ``LoadedModel`` for it. Its adapter has offloaded
            its weights; its worker is stopped and is replaced on wake.
        host_bytes: Host bytes charged to the tier while it is parked.
        retains_device_memory: Whether the parked adapter still holds some
            accelerator memory (an engine process keeps its CUDA context and
            workspaces). Such models are dropped when that device needs memory
            and nothing resident can be evicted.
        parked_at: ``time.monotonic()`` when it parked.
    """

    name: str
    loaded: Any
    host_bytes: int
    retains_device_memory: bool = False
    parked_at: float = field(default_factory=time.monotonic)

    @property
    def device(self) -> str:
        return str(self.loaded.device)


class HostCache:
    """Budgeted, least-recently-parked set of models parked in host RAM.

    Not thread-safe: the registry mutates it on its lifecycle event loop.
    """

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes < 0:
            raise ValueError(f"host cache budget must not be negative, got {budget_bytes}")
        self._budget = budget_bytes
        self._parked: OrderedDict[str, ParkedModel] = OrderedDict()
        self._retained: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self._budget > 0

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def used_bytes(self) -> int:
        return sum(entry.host_bytes for entry in self._parked.values()) + sum(self._retained.values())

    @property
    def names(self) -> list[str]:
        """Parked model ids, least recently parked first."""
        return list(self._parked)

    def __contains__(self, name: object) -> bool:
        return name in self._parked

    def __len__(self) -> int:
        return len(self._parked)

    def get(self, name: str) -> ParkedModel | None:
        return self._parked.get(name)

    def retained_bytes(self, name: str) -> int:
        """Host bytes a resident model's adapter still holds for the tier."""
        return self._retained.get(name, 0)

    def admission_victims(self, name: str, host_bytes: int) -> list[str] | None:
        """Parked models to drop, oldest first, so ``name`` can park ``host_bytes``.

        Returns an empty list when it fits as is, and None when it cannot fit
        even after dropping every parked model (its size, plus host memory that
        resident adapters keep, exceeds the budget).
        """
        if host_bytes <= 0 or not self.enabled:
            return None
        others = self.used_bytes - self._retained.get(name, 0)
        if name in self._parked:
            others -= self._parked[name].host_bytes
        overflow = others + host_bytes - self._budget
        victims: list[str] = []
        for parked_name, entry in self._parked.items():
            if overflow <= 0:
                break
            if parked_name == name:
                continue
            victims.append(parked_name)
            overflow -= entry.host_bytes
        return victims if overflow <= 0 else None

    def put(self, entry: ParkedModel) -> None:
        """Park ``entry`` as the most recently parked model.

        Its host bytes replace any charge the resident adapter was carrying.
        """
        self._retained.pop(entry.name, None)
        self._parked.pop(entry.name, None)
        self._parked[entry.name] = entry

    def take(self, name: str) -> ParkedModel | None:
        """Remove and return a parked model to wake it; its charge goes with it."""
        return self._parked.pop(name, None)

    def retain(self, name: str, host_bytes: int) -> None:
        """Charge host bytes a resident adapter keeps after waking."""
        if host_bytes > 0:
            self._retained[name] = host_bytes
        else:
            self._retained.pop(name, None)

    def release(self, name: str) -> None:
        """Forget a resident model's retained charge (it was fully unloaded)."""
        self._retained.pop(name, None)

    def holding_device_memory(self, device: str) -> list[str]:
        """Parked models that still hold memory on ``device``, oldest first."""
        return [name for name, entry in self._parked.items() if entry.retains_device_memory and entry.device == device]
