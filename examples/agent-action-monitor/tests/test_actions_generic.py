"""Tests for the generic source adapter."""

from __future__ import annotations

import pytest

from dusk.actions.adapters.base import AdapterError
from dusk.actions.adapters.generic import GenericAdapter


def _record(**overrides: object) -> dict[str, object]:
    """Build one well-formed generic record, overriding fields as needed."""
    base: dict[str, object] = {
        "agent_id": "netops-agent",
        "timestamp": "2023-11-14T22:13:20+00:00",
        "action_type": "firewall_rule_change",
        "target": "fw-corp-https",
        "change": {"before": None, "after": {"port": 443}},
        "source": "generic",
        "raw_ref": "evt-1",
    }
    base.update(overrides)
    return base


def test_generic_builds_agent_action() -> None:
    """A well-formed generic record maps to the canonical fields."""
    action = GenericAdapter().parse(_record())
    assert action.agent_id == "netops-agent"
    assert action.action_type == "firewall_rule_change"
    assert action.target == "fw-corp-https"
    assert action.source == "generic"
    assert action.change["after"] == {"port": 443}
    assert action.timestamp.tzinfo is not None


def test_generic_defaults_change_when_absent() -> None:
    """A record without a change gets the empty before/after delta."""
    record = _record()
    del record["change"]
    action = GenericAdapter().parse(record)
    assert action.change == {"before": None, "after": None}


def test_generic_missing_field_raises_adapter_error() -> None:
    """A missing required field raises AdapterError, not a raw error."""
    record = _record()
    del record["target"]
    with pytest.raises(AdapterError, match="target"):
        GenericAdapter().parse(record)


def test_generic_empty_identity_raises() -> None:
    """An empty agent_id is rejected as an AdapterError."""
    with pytest.raises(AdapterError, match="agent_id"):
        GenericAdapter().parse(_record(agent_id=""))


def test_generic_naive_timestamp_raises() -> None:
    """A timezone-naive timestamp is rejected as an AdapterError."""
    with pytest.raises(AdapterError, match="timezone-aware"):
        GenericAdapter().parse(_record(timestamp="2023-11-14T22:13:20"))
