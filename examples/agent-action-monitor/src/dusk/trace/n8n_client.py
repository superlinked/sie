"""Non-blocking n8n decision, report, and alert webhooks."""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from dusk.config import Config, get_config

logger = logging.getLogger(__name__)

#: Pool size is fixed from process configuration on first use.
_executor: ThreadPoolExecutor | None = None

#: Queued-or-in-flight send count; bounds the pool's own unbounded task queue.
_pending_lock = threading.Lock()
_pending_count = 0


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=get_config().n8n_max_workers, thread_name_prefix="n8n-webhook"
        )
    return _executor


def _submit(url: str, label: str, payload: dict[str, object], config: Config) -> None:
    if not url:
        return
    global _pending_count
    with _pending_lock:
        if _pending_count >= config.n8n_max_queued:
            logger.warning(
                "n8n webhook (%s) dropped: %d already queued or in flight (n8n_max_queued=%d)",
                label,
                _pending_count,
                config.n8n_max_queued,
            )
            return
        _pending_count += 1
    _get_executor().submit(_send_and_release, url, label, payload)


def _send_and_release(url: str, label: str, payload: dict[str, object]) -> None:
    global _pending_count
    try:
        _send(url, label, payload)
    finally:
        with _pending_lock:
            _pending_count -= 1


def fire_decision(payload: dict[str, object], config: Config | None = None) -> None:
    """Fire the decision webhook on the bounded pool -- never blocks the caller."""
    cfg = config or get_config()
    _submit(cfg.n8n_decision_url, "decision", payload, cfg)


def fire_report(payload: dict[str, object], config: Config | None = None) -> None:
    """Fire the report webhook on the bounded pool -- never blocks the caller."""
    cfg = config or get_config()
    _submit(cfg.n8n_report_url, "report", payload, cfg)


def fire_alert(payload: dict[str, object], config: Config | None = None) -> None:
    """Fire the alert webhook on the bounded pool -- never blocks the caller."""
    cfg = config or get_config()
    _submit(cfg.n8n_alert_url, "alert", payload, cfg)


def _send(url: str, label: str, payload: dict[str, object]) -> None:
    if not url:
        return
    if not url.startswith(("https://", "http://")):
        logger.warning("n8n webhook (%s) has unsupported scheme, skipping", label)
        return
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(  # noqa: S310
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310  # nosec B310
            logger.info("n8n webhook (%s) fired, status=%s", label, resp.status)
    except urllib.error.URLError as exc:
        logger.warning("n8n webhook (%s) failed: %s", label, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("n8n webhook (%s) error: %s", label, exc)
