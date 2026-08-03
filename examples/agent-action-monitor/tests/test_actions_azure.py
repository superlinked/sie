"""Tests for the Azure activity-log adapter."""

from __future__ import annotations

import pytest

from dusk.actions.adapters.azure import AzureAdapter
from dusk.actions.adapters.base import AdapterError


def _nsg_record(**overrides: object) -> dict[str, object]:
    """Build a sample Azure NSG security-rule activity-log record."""
    base: dict[str, object] = {
        "operationName": {
            "value": "Microsoft.Network/networkSecurityGroups/securityRules/write",
            "localizedValue": "Create or update security rule",
        },
        "caller": "netops-agent@example.com",
        "resourceId": (
            "/subscriptions/sub-1/resourceGroups/rg-net/providers/"
            "Microsoft.Network/networkSecurityGroups/nsg-corp/securityRules/allow-https"
        ),
        "eventTimestamp": "2023-11-14T22:13:20Z",
        "correlationId": "corr-123",
        "properties": {"before": None, "after": {"port": 443}},
    }
    base.update(overrides)
    return base


def test_azure_nsg_maps_to_firewall_rule_change() -> None:
    """An NSG security-rule write maps to firewall_rule_change."""
    action = AzureAdapter().parse(_nsg_record())
    assert action.action_type == "firewall_rule_change"
    assert action.agent_id == "netops-agent@example.com"
    assert action.target.endswith("securityRules/allow-https")
    assert action.source == "azure"
    assert action.raw_ref == "corr-123"
    assert action.change["after"] == {"port": 443}
    assert action.timestamp.tzinfo is not None


def test_azure_route_table_maps_to_route_change() -> None:
    """A routeTables operation maps to route_change."""
    record = _nsg_record(
        operationName="Microsoft.Network/routeTables/routes/write",
        resourceId="/subscriptions/sub-1/.../routeTables/rt-corp/routes/default",
    )
    assert AzureAdapter().parse(record).action_type == "route_change"


def test_azure_role_assignment_maps() -> None:
    """A roleAssignments operation maps to role_assignment."""
    record = _nsg_record(operationName="Microsoft.Authorization/roleAssignments/write")
    assert AzureAdapter().parse(record).action_type == "role_assignment"


def test_azure_unknown_operation_maps_to_unknown() -> None:
    """An unrecognised operation falls back to the unknown action_type."""
    record = _nsg_record(operationName="Microsoft.Storage/storageAccounts/write")
    assert AzureAdapter().parse(record).action_type == "unknown"


def test_azure_uses_authorization_action_fallback() -> None:
    """When operationName is absent, the RBAC authorization action is used."""
    record = _nsg_record()
    del record["operationName"]
    record["authorization"] = {"action": "Microsoft.Network/routeTables/write"}
    assert AzureAdapter().parse(record).action_type == "route_change"


def test_azure_missing_caller_raises_adapter_error() -> None:
    """A record with no identity cannot yield a valid AgentAction."""
    record = _nsg_record()
    del record["caller"]
    with pytest.raises(AdapterError):
        AzureAdapter().parse(record)


def test_azure_missing_timestamp_raises_adapter_error() -> None:
    """A record with no timestamp cannot yield a valid AgentAction."""
    record = _nsg_record()
    del record["eventTimestamp"]
    with pytest.raises(AdapterError):
        AzureAdapter().parse(record)
