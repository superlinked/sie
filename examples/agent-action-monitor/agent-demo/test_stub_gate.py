"""Tests for the local stub gate -- schema-shape only, no real analysis."""

from __future__ import annotations

import pytest
from stub_gate import app

CONTRACT_FIELDS = {
    "trace_id",
    "verdict",
    "score",
    "blast",
    "mitre_attack",
    "mitre_atlas",
    "reasons",
    "predicted_next",
    "similar_decision_ids",
}

VALID_ACTION = {
    "agent_id": "agent-1",
    "timestamp": "2026-07-10T00:00:00+00:00",
    "action_type": "route_change",
    "target": "rt-123",
    "change": {"before": None, "after": {"cidr": "10.0.0.0/24"}},
    "source": "generic",
}


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_default_action_allows(client):
    resp = client.post("/v1/gate", json=VALID_ACTION)
    assert resp.status_code == 200
    body = resp.get_json()
    assert CONTRACT_FIELDS <= body.keys()
    assert body["verdict"] == "ALLOW"


def test_firewall_rule_change_blocks(client):
    action = {**VALID_ACTION, "action_type": "firewall_rule_change"}
    resp = client.post("/v1/gate", json=action)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["verdict"] == "BLOCK"
    assert body["reasons"]


def test_missing_required_field_rejected(client):
    action = dict(VALID_ACTION)
    del action["agent_id"]
    resp = client.post("/v1/gate", json=action)
    assert resp.status_code == 400


def test_non_object_body_rejected(client):
    resp = client.post("/v1/gate", json=["not", "an", "object"])
    assert resp.status_code == 400
