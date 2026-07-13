"""Tests for the agent harness -- the critical path.

Runs against MockBedrock but mocks the HTTP calls to /v1/gate and
mock-PROD, so these tests need no services running.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bedrock_client import DuskBlockedError
from harness import run_scenario, run_scenario_or_raise


def _mock_gate_response(verdict: str, reasons: list[str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "trace_id": "trace-1",
        "verdict": verdict,
        "score": 0.9 if verdict != "ALLOW" else 0.05,
        "blast": "high" if verdict != "ALLOW" else "low",
        "reasons": reasons or [],
    }
    return resp


def _mock_apply_response() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"status": "applied"}
    return resp


@patch("harness.requests.post")
def test_clean_scenario_allowed_and_applied(mock_post):
    mock_post.side_effect = [_mock_gate_response("ALLOW"), _mock_apply_response()]

    result = run_scenario("agent-1", "clean")

    assert result["verdict"] == "ALLOW"
    assert result["applied"] is True
    assert mock_post.call_count == 2
    gate_call, apply_call = mock_post.call_args_list
    assert gate_call.args[0] == "http://localhost:8000/v1/gate"
    assert apply_call.args[0] == "http://localhost:9000/apply"


@patch("harness.requests.post")
def test_poisoned_scenario_blocked_before_mock_prod(mock_post):
    mock_post.side_effect = [
        _mock_gate_response("BLOCK", reasons=["out of baseline"]),
    ]

    result = run_scenario("agent-1", "poisoned")

    assert result["verdict"] == "BLOCK"
    assert result["applied"] is False
    assert "out of baseline" in result["reasons"]
    # Only the gate was called -- mock-PROD never sees a blocked action.
    assert mock_post.call_count == 1


@patch("harness.requests.post")
def test_would_block_watch_mode_still_applies(mock_post):
    """Watch mode is observational: WOULD-BLOCK logs the reason but does not
    stop the action, unlike a real BLOCK in enforce mode."""
    mock_post.side_effect = [
        _mock_gate_response("WOULD-BLOCK", reasons=["anomalous"]),
        _mock_apply_response(),
    ]

    result = run_scenario("agent-1", "poisoned")

    assert result["verdict"] == "WOULD-BLOCK"
    assert result["applied"] is True
    assert "anomalous" in result["reasons"]
    assert mock_post.call_count == 2


@patch("harness.requests.post")
def test_run_scenario_or_raise_raises_on_block(mock_post):
    mock_post.side_effect = [_mock_gate_response("BLOCK", reasons=["out of baseline"])]

    try:
        run_scenario_or_raise("agent-1", "poisoned")
        raise AssertionError("expected DuskBlockedError")
    except DuskBlockedError as exc:
        assert "out of baseline" in str(exc)


@patch("harness.requests.post")
def test_run_scenario_or_raise_returns_on_allow(mock_post):
    mock_post.side_effect = [_mock_gate_response("ALLOW"), _mock_apply_response()]

    result = run_scenario_or_raise("agent-1", "clean")

    assert result["applied"] is True
