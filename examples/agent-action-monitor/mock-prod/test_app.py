"""Tests for the mock-prod dummy downstream target."""

from __future__ import annotations

import pytest
from app import app, applied_log


@pytest.fixture(autouse=True)
def _clear_log():
    applied_log.clear()
    yield
    applied_log.clear()


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_apply_logs_the_action(client):
    action = {"agent_id": "agent-1", "action_type": "route_change", "target": "rt-123"}
    resp = client.post("/apply", json=action)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "applied"
    assert body["agent_id"] == "agent-1"

    log_resp = client.get("/log")
    log_body = log_resp.get_json()
    assert log_body["count"] == 1
    assert log_body["entries"][0]["target"] == "rt-123"


def test_apply_rejects_non_object_body(client):
    resp = client.post("/apply", json=["not", "an", "object"])
    assert resp.status_code == 400
