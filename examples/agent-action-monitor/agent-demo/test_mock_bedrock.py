"""Tests for MockBedrock -- keyless clean/poisoned scenarios."""

from __future__ import annotations

from mock_bedrock import MockBedrock, extract_tool_use


def test_clean_scenario_proposes_route_change():
    response = MockBedrock(scenario="clean").converse(modelId="test-model", messages=[])
    tool_use = extract_tool_use(response)
    assert tool_use is not None
    assert tool_use["name"] == "update_route_table"
    assert tool_use["input"]["target"] == "rt-corp-prod"


def test_poisoned_scenario_proposes_firewall_rule_into_restricted_segment():
    response = MockBedrock(scenario="poisoned").converse(modelId="test-model", messages=[])
    tool_use = extract_tool_use(response)
    assert tool_use is not None
    assert tool_use["name"] == "update_firewall_rule"
    assert tool_use["input"]["target"] == "fw-corp-restricted-segment"
    assert tool_use["input"]["after"]["cidr"] == "0.0.0.0/0"


def test_default_scenario_is_clean():
    response = MockBedrock().converse(modelId="test-model", messages=[])
    tool_use = extract_tool_use(response)
    assert tool_use["name"] == "update_route_table"


def test_extract_tool_use_returns_none_when_no_tool_call():
    response = {"output": {"message": {"content": [{"text": "no tool call here"}]}}}
    assert extract_tool_use(response) is None
