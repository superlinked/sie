"""Tests for the Bedrock tool-call adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dusk.actions.adapters.base import AdapterError
from dusk.actions.adapters.bedrock import BedrockAdapter


def _firewall_tool_use(**overrides: object) -> dict[str, object]:
    """Build a sample Bedrock toolUse block proposing a firewall change."""
    base: dict[str, object] = {
        "toolUseId": "tooluse-abc123",
        "name": "update_firewall_rule",
        "input": {
            "target": "fw-corp-restricted-segment",
            "before": None,
            "after": {"port": 22, "cidr": "0.0.0.0/0"},
        },
    }
    base.update(overrides)
    return base


def test_firewall_tool_name_maps_to_firewall_rule_change() -> None:
    """A tool name containing 'firewall' maps to firewall_rule_change."""
    action = BedrockAdapter().parse_tool_use(
        _firewall_tool_use(),
        agent_id="ops-agent-1",
        timestamp=datetime(2026, 7, 10, tzinfo=UTC),
    )
    assert action.action_type == "firewall_rule_change"
    assert action.agent_id == "ops-agent-1"
    assert action.target == "fw-corp-restricted-segment"
    assert action.source == "bedrock"
    assert action.raw_ref == "tooluse-abc123"
    assert action.change["after"] == {"port": 22, "cidr": "0.0.0.0/0"}
    assert action.timestamp.tzinfo is not None


def test_route_tool_name_maps_to_route_change() -> None:
    """A tool name containing 'route' maps to route_change."""
    tool_use = _firewall_tool_use(name="update_route_table", input={"target": "rt-1"})
    action = BedrockAdapter().parse_tool_use(
        tool_use, agent_id="ops-agent-1", timestamp=datetime(2026, 7, 10, tzinfo=UTC)
    )
    assert action.action_type == "route_change"


def test_unrecognised_tool_name_maps_to_unknown() -> None:
    """A tool name that matches no rule maps to unknown, not an error."""
    tool_use = _firewall_tool_use(name="get_weather", input={"target": "n/a"})
    action = BedrockAdapter().parse_tool_use(
        tool_use, agent_id="ops-agent-1", timestamp=datetime(2026, 7, 10, tzinfo=UTC)
    )
    assert action.action_type == "unknown"


def test_missing_target_raises_adapter_error() -> None:
    """A toolUse input with no target cannot yield a valid AgentAction."""
    tool_use = _firewall_tool_use(input={"before": None, "after": {}})
    with pytest.raises(AdapterError, match="target"):
        BedrockAdapter().parse_tool_use(
            tool_use, agent_id="ops-agent-1", timestamp=datetime(2026, 7, 10, tzinfo=UTC)
        )


def test_missing_input_raises_adapter_error() -> None:
    """A toolUse block with no input block at all is rejected."""
    tool_use = {"toolUseId": "tooluse-x", "name": "update_firewall_rule"}
    with pytest.raises(AdapterError, match="input"):
        BedrockAdapter().parse_tool_use(
            tool_use, agent_id="ops-agent-1", timestamp=datetime(2026, 7, 10, tzinfo=UTC)
        )


def test_parse_satisfies_source_adapter_contract() -> None:
    """The registry-facing parse() accepts a raw dict with agent_id/timestamp attached."""
    raw = {
        "tool_use": _firewall_tool_use(),
        "agent_id": "ops-agent-1",
        "timestamp": "2026-07-10T00:00:00+00:00",
    }
    action = BedrockAdapter().parse(raw)
    assert action.action_type == "firewall_rule_change"
    assert action.agent_id == "ops-agent-1"


def test_parse_missing_tool_use_raises() -> None:
    """parse() rejects a raw record with no tool_use block."""
    with pytest.raises(AdapterError, match="tool_use"):
        BedrockAdapter().parse({"agent_id": "a", "timestamp": "2026-07-10T00:00:00+00:00"})


def test_registered_in_normaliser() -> None:
    """BedrockAdapter is registered under the 'bedrock' source name."""
    from dusk.actions.normaliser import get_adapter, known_sources

    assert "bedrock" in known_sources()
    assert isinstance(get_adapter("bedrock"), BedrockAdapter)
