"""Bounded, durable history of refused actions, partitioned by agent."""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("dusk.actions.offense_memory")

#: Retention limits bound both memory use and the persisted file.
_MAX_OFFENSES_PER_AGENT = 50
_MAX_TRACKED_AGENTS = 500


@dataclass(frozen=True)
class OffenseRecord:
    """One persisted refusal: enough to cite it later and judge similarity to it.

    Attributes:
        trace_id: The gate's trace_id for the refused verdict, so a later
            citation ("repeat of trace <id>") is checkable against real logs.
        agent_id: The agent this offense belongs to.
        action_type: The refused action's normalised verb.
        target_class: The refused action's coarse target class (see
            :func:`dusk.actions.baseline.target_class`).
        tokens: The refused action's target tokens, for similarity matching
            against a new action.
        verdict: The verdict rendered, ``WOULD-BLOCK`` or ``BLOCK``.
        timestamp: When the offense was recorded, UTC, timezone-aware.
    """

    trace_id: str
    agent_id: str
    action_type: str
    target_class: str
    tokens: tuple[str, ...]
    verdict: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["tokens"] = list(self.tokens)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OffenseRecord:
        """Reconstruct a record from :meth:`to_dict` output.

        Raises:
            ValueError: If a required key is missing or malformed.
        """
        try:
            return cls(
                trace_id=data["trace_id"],
                agent_id=data["agent_id"],
                action_type=data["action_type"],
                target_class=data["target_class"],
                tokens=tuple(data["tokens"]),
                verdict=data["verdict"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid offense record: {exc}") from exc


@dataclass
class _AgentOffenses:
    records: list[OffenseRecord] = field(default_factory=list)


class OffenseMemory:
    """Thread-safe offense store with coalesced, single-writer persistence.

    The dirty flag prevents request bursts from creating an unbounded write
    queue. All scheduling state shares the data lock, preserving the
    single-writer invariant during concurrent first use.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """Create the store, loading any existing state from ``storage_path``.

        Args:
            storage_path: File to persist offenses to. When ``None``, the
                store is in-memory only for the life of this instance (used
                by tests that don't care about durability).
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._lock = threading.Lock()
        self._by_agent: OrderedDict[str, _AgentOffenses] = OrderedDict()
        if self._storage_path is not None:
            self._load(self._storage_path)
        self._executor: ThreadPoolExecutor | None = None
        self._write_in_flight = False
        self._dirty = False
        self._last_write: Future[None] | None = None
        self._last_persist_error: str | None = None
        self._closed = False

    def _touch(self, agent_id: str) -> _AgentOffenses:
        """Return the agent record and enforce LRU retention. Lock required."""
        offenses = self._by_agent.get(agent_id)
        if offenses is None:
            offenses = _AgentOffenses()
            self._by_agent[agent_id] = offenses
        else:
            self._by_agent.move_to_end(agent_id)
        while len(self._by_agent) > _MAX_TRACKED_AGENTS:
            evicted_id, _ = self._by_agent.popitem(last=False)
            logger.info("Evicted offense memory for agent %s (tracked-agent cap)", evicted_id)
        return offenses

    def _load(self, storage_path: Path) -> None:
        if not storage_path.exists():
            return
        try:
            with storage_path.open(encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read offense memory from %s, starting empty: %s",
                storage_path,
                exc,
            )
            return
        if not isinstance(raw, dict):
            logger.warning(
                "Offense memory file %s did not contain a mapping, starting empty",
                storage_path,
            )
            return
        for agent_id, entries in raw.items():
            if not isinstance(entries, list):
                continue
            records: list[OffenseRecord] = []
            for entry in entries:
                try:
                    records.append(OffenseRecord.from_dict(entry))
                except ValueError as exc:
                    logger.warning("Skipping malformed offense record: %s", exc)
            if records:
                capped = records[-_MAX_OFFENSES_PER_AGENT:]
                self._by_agent[agent_id] = _AgentOffenses(records=capped)
        while len(self._by_agent) > _MAX_TRACKED_AGENTS:
            self._by_agent.popitem(last=False)
        logger.info(
            "Loaded offense memory for %d agent(s) from %s",
            len(self._by_agent),
            storage_path,
        )

    def _write_to_disk(self, payload: dict[str, Any]) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._storage_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            tmp_path.replace(self._storage_path)
            self._last_persist_error = None
        except OSError as exc:
            self._last_persist_error = str(exc)
            logger.warning("Could not persist offense memory to %s: %s", self._storage_path, exc)

    def _persist_loop(self) -> None:
        """Persist snapshots until no changes occurred during the last write."""
        while True:
            with self._lock:
                self._dirty = False
                payload = {
                    agent_id: [r.to_dict() for r in offenses.records]
                    for agent_id, offenses in self._by_agent.items()
                }
            self._write_to_disk(payload)
            with self._lock:
                if not self._dirty:
                    self._write_in_flight = False
                    return

    def _schedule_persist(self) -> None:
        """Mark state dirty and start the writer if needed. Lock required."""
        if self._storage_path is None or self._closed:
            return
        self._dirty = True
        if self._write_in_flight:
            return
        self._write_in_flight = True
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="offense-memory-writer"
            )
        self._last_write = self._executor.submit(self._persist_loop)

    def flush(self, timeout: float | None = None) -> None:
        """Block until every write scheduled before this call is durable."""
        with self._lock:
            future = self._last_write
        if future is not None:
            future.result(timeout=timeout)

    def close(self, timeout: float | None = None) -> None:
        """Flush and stop the writer. Safe to call more than once."""
        self.flush(timeout=timeout)
        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    @property
    def last_persist_error(self) -> str | None:
        """Return the latest disk-write error for health reporting."""
        return self._last_persist_error

    def record(
        self,
        trace_id: str,
        agent_id: str,
        action_type: str,
        target_class: str,
        tokens: set[str],
        verdict: str,
    ) -> None:
        """Record a refused verdict for ``agent_id`` and schedule a write."""
        entry = OffenseRecord(
            trace_id=trace_id,
            agent_id=agent_id,
            action_type=action_type,
            target_class=target_class,
            tokens=tuple(sorted(tokens)),
            verdict=verdict,
            timestamp=datetime.now(UTC),
        )
        with self._lock:
            offenses = self._touch(agent_id)
            offenses.records.append(entry)
            if len(offenses.records) > _MAX_OFFENSES_PER_AGENT:
                del offenses.records[: len(offenses.records) - _MAX_OFFENSES_PER_AGENT]
            self._schedule_persist()

    def offenses_for(self, agent_id: str) -> list[OffenseRecord]:
        """Return the recorded offenses for one agent, oldest first."""
        with self._lock:
            offenses = self._by_agent.get(agent_id)
            return list(offenses.records) if offenses else []

    def clear(self) -> None:
        """Wipe all recorded offenses. Test-only / operator reset hook."""
        with self._lock:
            self._by_agent.clear()
            self._schedule_persist()
