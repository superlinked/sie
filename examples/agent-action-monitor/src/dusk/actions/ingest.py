"""Normalize JSON source records into AgentAction events."""

from __future__ import annotations

import json
import logging
import os

from dusk.actions.event import AgentAction

logger = logging.getLogger("dusk.actions.ingest")


def ingest_file(path: str, source: str) -> list[AgentAction]:
    """Read a JSON list of raw ``source`` records into AgentAction events.

    Args:
        path: Filesystem path to a JSON file containing a list of records.
        source: Canonical source name selecting the adapter (for example
            ``"azure"`` or ``"generic"``).

    Returns:
        The list of successfully normalised :class:`AgentAction` events.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty, not valid JSON, or its top-level
            JSON is not a list.
    """
    # Imported here so the adapter registry loads lazily and ingest stays free
    # of an import cycle (ingest -> normaliser -> adapters).
    from dusk.actions.adapters.base import AdapterError
    from dusk.actions.normaliser import normalise_record

    if not os.path.exists(path):
        logger.critical("Action file not found: %s", path)
        raise FileNotFoundError(f"Action file not found: {path}")

    if os.path.getsize(path) == 0:
        logger.warning("Action file is empty: %s", path)
        raise ValueError(f"Action file is empty: {path}")

    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.critical("Action file is not valid JSON: %s: %s", path, exc)
            raise ValueError(f"Action file is not valid JSON: {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"Action file must contain a JSON list of records, got {type(payload).__name__}: {path}"
        )

    actions: list[AgentAction] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            logger.warning("Skipping record %d in %s: not a JSON object", index, path)
            continue
        try:
            actions.append(normalise_record(source, record))
        except AdapterError as exc:
            logger.warning("Skipping record %d in %s: %s", index, path, exc)

    logger.info(
        "Ingested %d action(s) from %s via '%s' adapter (%d record(s) skipped)",
        len(actions),
        path,
        source,
        len(payload) - len(actions),
    )
    return actions
