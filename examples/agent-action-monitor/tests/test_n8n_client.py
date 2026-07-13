"""Tests for the n8n webhook client (dusk.trace.n8n_client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dusk.config import Config
from dusk.trace import n8n_client

_real_get_executor = n8n_client._get_executor


class _ImmediateExecutor:
    """Runs submitted work synchronously instead of on the real pool, for deterministic tests."""

    def submit(self, fn, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
        fn(*args, **kwargs)


@pytest.fixture(autouse=True)
def _synchronous_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(n8n_client, "_get_executor", lambda: _ImmediateExecutor())


def _track_send_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, dict[str, object]]]:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def _fake_send(url: str, label: str, payload: dict[str, object]) -> None:
        calls.append((url, label, payload))

    monkeypatch.setattr(n8n_client, "_send", _fake_send)
    return calls


def test_fire_decision_reads_url_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _track_send_calls(monkeypatch)
    config = Config(n8n_decision_url="https://example.com/decision")
    n8n_client.fire_decision({"a": 1}, config=config)
    assert calls == [("https://example.com/decision", "decision", {"a": 1})]


def test_fire_report_reads_url_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _track_send_calls(monkeypatch)
    config = Config(n8n_report_url="https://example.com/report")
    n8n_client.fire_report({"a": 1}, config=config)
    assert calls == [("https://example.com/report", "report", {"a": 1})]


def test_fire_alert_reads_url_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _track_send_calls(monkeypatch)
    config = Config(n8n_alert_url="https://example.com/alert")
    n8n_client.fire_alert({"a": 1}, config=config)
    assert calls == [("https://example.com/alert", "alert", {"a": 1})]


def test_send_no_op_when_url_empty() -> None:
    with patch("urllib.request.urlopen") as mock_urlopen:
        n8n_client._send("", "decision", {"a": 1})
    mock_urlopen.assert_not_called()


def test_send_rejects_unsupported_scheme() -> None:
    with patch("urllib.request.urlopen") as mock_urlopen:
        n8n_client._send("ftp://example.com/hook", "decision", {"a": 1})
    mock_urlopen.assert_not_called()


def test_send_posts_to_configured_url() -> None:
    mock_response = MagicMock()
    mock_response.status = 200
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_response
    with patch("urllib.request.urlopen", return_value=mock_context) as mock_urlopen:
        n8n_client._send("https://example.com/hook", "decision", {"a": 1})
    mock_urlopen.assert_called_once()


def test_send_swallows_errors() -> None:
    with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
        n8n_client._send("https://example.com/hook", "decision", {"a": 1})


def test_webhook_dropped_when_backlog_at_cap(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Once n8n_max_queued sends are already queued or in flight, a new one is
    dropped and logged rather than growing the backlog without limit."""
    calls = _track_send_calls(monkeypatch)
    config = Config(n8n_decision_url="https://example.com/decision", n8n_max_queued=2)
    monkeypatch.setattr(n8n_client, "_pending_count", 2)

    with caplog.at_level("WARNING"):
        n8n_client.fire_decision({"a": 1}, config=config)

    assert calls == []
    assert any("dropped" in r.message for r in caplog.records)


def test_pending_count_returns_to_zero_after_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backlog counter must not leak -- a delivered webhook frees its slot."""
    _track_send_calls(monkeypatch)
    config = Config(n8n_decision_url="https://example.com/decision")
    monkeypatch.setattr(n8n_client, "_pending_count", 0)

    n8n_client.fire_decision({"a": 1}, config=config)

    assert n8n_client._pending_count == 0


def test_fire_with_empty_url_does_not_touch_the_backlog(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _track_send_calls(monkeypatch)
    config = Config(n8n_decision_url="")
    monkeypatch.setattr(n8n_client, "_pending_count", 0)

    n8n_client.fire_decision({"a": 1}, config=config)

    assert calls == []
    assert n8n_client._pending_count == 0


def test_webhook_backlog_bound_survives_a_real_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end proof against the real pool: firing far more webhooks than
    n8n_max_queued allows must not grow the backlog past the configured cap.

    Needs the real pool, not the autouse synchronous-executor fixture -- that
    fixture runs submitted work inline on the calling thread, which would
    make this test's own blocking send deadlock waiting on a release signal
    only the same (blocked) thread could ever send.
    """
    import threading
    import time

    monkeypatch.setattr(n8n_client, "_get_executor", _real_get_executor)
    monkeypatch.setattr(n8n_client, "_executor", None)
    monkeypatch.setattr(n8n_client, "_pending_count", 0)
    config = Config(
        n8n_decision_url="https://example.com/decision", n8n_max_workers=1, n8n_max_queued=5
    )

    release = threading.Event()
    max_seen = 0
    seen_lock = threading.Lock()

    def _blocking_send(url: str, label: str, payload: dict[str, object]) -> None:
        nonlocal max_seen
        with seen_lock:
            max_seen = max(max_seen, n8n_client._pending_count)
        release.wait(timeout=5)

    monkeypatch.setattr(n8n_client, "_send", _blocking_send)

    for i in range(50):
        n8n_client.fire_decision({"i": i}, config=config)
    time.sleep(0.1)  # let the single worker pick up the first send and start blocking

    with n8n_client._pending_lock:
        assert n8n_client._pending_count <= config.n8n_max_queued

    release.set()


def test_webhook_concurrency_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Firing many webhooks in a burst must not spawn one OS thread per call."""
    import threading
    import time

    # This test needs the real pool, not the autouse synchronous-executor fixture.
    monkeypatch.setattr(n8n_client, "_executor", None)
    monkeypatch.setattr(n8n_client, "get_config", lambda: Config(n8n_max_workers=3))

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def _slow_send(url: str, label: str, payload: dict[str, object]) -> None:
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1

    executor = _real_get_executor()
    futures = [
        executor.submit(_slow_send, "https://example.com/decision", "decision", {"i": i})
        for i in range(20)
    ]
    for f in futures:
        f.result(timeout=5)

    assert max_in_flight <= 3
