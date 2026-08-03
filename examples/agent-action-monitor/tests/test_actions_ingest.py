"""Tests for ingest_file: reading a JSON list of records via an adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

LAB_DIR = os.path.join(os.path.dirname(__file__), "..", "lab", "actions")
sys.path.insert(0, os.path.abspath(LAB_DIR))

import generate_actions  # noqa: E402

from dusk.actions.event import AgentAction  # noqa: E402
from dusk.actions.ingest import ingest_file  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _ensure_fixtures() -> tuple[str, str]:
    """Return the (normal, mixed) fixture paths, generating them if absent."""
    normal = os.path.join(FIXTURES, "actions_normal.json")
    mixed = os.path.join(FIXTURES, "actions_mixed.json")
    if not (os.path.exists(normal) and os.path.exists(mixed)):
        generate_actions.generate(FIXTURES)
    return normal, mixed


def _record(**overrides: object) -> dict[str, object]:
    """Build one valid generic record."""
    base: dict[str, object] = {
        "agent_id": "a",
        "timestamp": "2023-11-14T22:13:20+00:00",
        "action_type": "route_change",
        "target": "rt-1",
        "change": {"before": None, "after": None},
        "source": "generic",
    }
    base.update(overrides)
    return base


def test_ingest_normal_fixture() -> None:
    """The normal fixture ingests every record as an AgentAction."""
    normal, _ = _ensure_fixtures()
    actions = ingest_file(normal, "generic")
    assert len(actions) == 15
    assert all(isinstance(a, AgentAction) for a in actions)


def test_ingest_mixed_fixture() -> None:
    """The mixed fixture ingests the routine plus out-of-pattern actions."""
    _, mixed = _ensure_fixtures()
    assert len(ingest_file(mixed, "generic")) == 18


def test_ingest_skips_one_malformed_record(tmp_path: Path) -> None:
    """One malformed record is skipped; the valid ones still return."""
    records = [_record(), {"agent_id": "a"}]  # second is missing fields
    path = tmp_path / "actions.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    assert len(ingest_file(str(path), "generic")) == 1


def test_ingest_empty_list(tmp_path: Path) -> None:
    """An empty JSON list yields no actions and does not raise."""
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    assert ingest_file(str(path), "generic") == []


def test_ingest_single_record(tmp_path: Path) -> None:
    """A single valid record yields one action."""
    path = tmp_path / "one.json"
    path.write_text(json.dumps([_record()]), encoding="utf-8")
    assert len(ingest_file(str(path), "generic")) == 1


def test_ingest_missing_file_raises() -> None:
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ingest_file("does-not-exist.json", "generic")


def test_ingest_not_a_list_raises(tmp_path: Path) -> None:
    """A top-level JSON object (not a list) raises ValueError."""
    path = tmp_path / "object.json"
    path.write_text('{"agent_id": "a"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON list"):
        ingest_file(str(path), "generic")


def test_ingest_unknown_source_skips_all(tmp_path: Path) -> None:
    """An unknown source means every record is skipped, returning empty."""
    path = tmp_path / "actions.json"
    path.write_text(json.dumps([_record()]), encoding="utf-8")
    assert ingest_file(str(path), "nope") == []
