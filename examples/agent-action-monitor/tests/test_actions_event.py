"""Tests for the AgentAction schema: construction, validation, round-trip."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dusk.actions.event import AgentAction

_TS = datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)


def _action(**overrides: object) -> AgentAction:
    """Build a valid AgentAction, overriding fields as needed."""
    kwargs: dict[str, object] = {
        "agent_id": "netops-agent",
        "timestamp": _TS,
        "action_type": "firewall_rule_change",
        "target": "fw-corp-https",
        "change": {"before": None, "after": {"port": 443}},
        "source": "generic",
        "raw_ref": "evt-1",
    }
    kwargs.update(overrides)
    return AgentAction(**kwargs)  # type: ignore[arg-type]


def test_valid_agent_action_builds() -> None:
    """A well-formed AgentAction constructs and exposes its fields."""
    action = _action()
    assert action.agent_id == "netops-agent"
    assert action.action_type == "firewall_rule_change"
    assert action.target == "fw-corp-https"
    assert action.change["after"] == {"port": 443}


def test_empty_agent_id_raises() -> None:
    """An empty or whitespace agent_id is rejected."""
    with pytest.raises(ValueError, match="agent_id"):
        _action(agent_id="   ")


def test_empty_target_raises() -> None:
    """An empty target is rejected."""
    with pytest.raises(ValueError, match="target"):
        _action(target="")


def test_naive_timestamp_raises() -> None:
    """A timezone-naive timestamp is rejected, not silently coerced."""
    with pytest.raises(ValueError, match="timezone-aware"):
        _action(timestamp=datetime(2023, 11, 14, 22, 13, 20))


def test_unknown_action_type_raises() -> None:
    """An action_type outside the known set is rejected."""
    with pytest.raises(ValueError, match="action_type"):
        _action(action_type="totally_made_up")


def test_unknown_is_a_valid_action_type() -> None:
    """The explicit 'unknown' verb is accepted."""
    assert _action(action_type="unknown").action_type == "unknown"


def test_to_dict_is_json_safe() -> None:
    """to_dict renders the timestamp as an ISO 8601 string."""
    payload = _action().to_dict()
    assert payload["timestamp"] == _TS.isoformat()
    assert payload["action_type"] == "firewall_rule_change"


def test_to_from_dict_round_trips() -> None:
    """from_dict(to_dict(x)) reproduces the original exactly."""
    original = _action()
    restored = AgentAction.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_missing_field_raises() -> None:
    """from_dict rejects a mapping that is missing a required field."""
    payload = _action().to_dict()
    del payload["target"]
    with pytest.raises(ValueError, match="target"):
        AgentAction.from_dict(payload)
