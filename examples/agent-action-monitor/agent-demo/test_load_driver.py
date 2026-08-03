"""Tests for the load-scenario driver."""

from __future__ import annotations

from unittest.mock import patch

from load_driver import LoadResult, run_load


def test_load_result_records_verdicts_and_latency():
    result = LoadResult()
    result.record("ALLOW", 12.5)
    result.record("BLOCK", 8.0)
    result.record("ALLOW", 15.0)

    assert result.verdicts == {"ALLOW": 2, "BLOCK": 1}
    assert result.errors == 0
    assert len(result.latencies_ms) == 3


def test_load_result_percentile():
    result = LoadResult()
    for latency in [10, 20, 30, 40, 50]:
        result.record("ALLOW", latency)

    assert result.percentile(0.0) == 10
    assert result.percentile(1.0) == 50


def test_load_result_records_errors():
    result = LoadResult()
    result.record(None, 5.0)

    assert result.errors == 1
    assert result.verdicts == {}


@patch("load_driver.run_scenario")
def test_run_load_fires_total_requests(mock_run_scenario):
    mock_run_scenario.return_value = {"verdict": "ALLOW", "applied": True, "action": {}}

    result = run_load(concurrency=4, total=20, poisoned_ratio=0.0)

    assert mock_run_scenario.call_count == 20
    assert result.verdicts.get("ALLOW") == 20
    assert result.errors == 0


@patch("load_driver.run_scenario")
def test_run_load_counts_exceptions_as_errors(mock_run_scenario):
    mock_run_scenario.side_effect = ConnectionError("gate unreachable")

    result = run_load(concurrency=2, total=5, poisoned_ratio=0.0)

    assert result.errors == 5
    assert result.verdicts == {}
