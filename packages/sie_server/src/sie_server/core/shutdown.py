"""Graceful shutdown handling for spot instance preemption.

Handles SIGTERM signals to gracefully drain in-flight requests before shutdown.
This is critical for spot/preemptible instances which receive 30s warning.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


@dataclass
class ShutdownState:
    """Tracks shutdown state and in-flight requests.

    Attributes:
        shutting_down: True when SIGTERM received, stops accepting new requests.
        in_flight: Count of currently processing requests.
        drain_timeout_s: Maximum time to wait for requests to drain.
    """

    shutting_down: bool = False
    in_flight: int = 0
    drain_timeout_s: float = 25.0  # Leave 5s buffer before 30s preemption
    _drain_event: asyncio.Event = field(default_factory=asyncio.Event)
    _shutdown_time: float | None = None

    def start_shutdown(self) -> None:
        """Called when SIGTERM received. Stops accepting new requests."""
        if not self.shutting_down:
            self.shutting_down = True
            self._shutdown_time = time.time()
            logger.warning(
                "Shutdown initiated, draining %d in-flight requests (timeout: %.1fs)",
                self.in_flight,
                self.drain_timeout_s,
            )
            # If no requests in flight, signal immediately
            if self.in_flight == 0:
                self._drain_event.set()

    def request_started(self) -> None:
        """Called when a request starts processing.

        No lock needed: asyncio is single-threaded, and this method
        contains no await points between the read and write of in_flight.
        """
        self.in_flight += 1

    def request_finished(self) -> None:
        """Called when a request finishes processing.

        No lock needed: asyncio is single-threaded, and the decrement +
        check is atomic from the event loop's perspective (no await points).
        """
        self.in_flight -= 1
        if self.shutting_down and self.in_flight == 0:
            logger.info("All requests drained, ready for shutdown")
            self._drain_event.set()

    async def wait_for_drain(self, drain_timeout: float | None = None) -> bool:
        """Wait for all in-flight requests to complete.

        Args:
            drain_timeout: Maximum seconds to wait. Uses drain_timeout_s if None.

        Returns:
            True if all requests drained, False if timeout.
        """
        if drain_timeout is None:
            drain_timeout = self.drain_timeout_s

        if self.in_flight == 0:
            return True

        logger.info("Waiting for %d requests to drain (timeout: %.1fs)", self.in_flight, drain_timeout)

        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=drain_timeout)
        except TimeoutError:
            logger.warning(
                "Drain timeout after %.1fs, %d requests still in flight",
                drain_timeout,
                self.in_flight,
            )
            return False
        else:
            return True


def setup_signal_handlers(shutdown_state: ShutdownState) -> None:
    """Install SIGTERM handler for graceful shutdown.

    Args:
        shutdown_state: The shutdown state to update on signal.

    Note:
        Signal handlers can only be set in the main thread. In test environments
        or when running under certain frameworks, this may silently skip setup.
    """

    def handle_sigterm(_signum: int, _frame: object) -> None:
        logger.info("Received SIGTERM, initiating graceful shutdown")
        shutdown_state.start_shutdown()

    # Only set up signal handlers on Unix (not Windows)
    if sys.platform != "win32":
        try:
            signal.signal(signal.SIGTERM, handle_sigterm)
            logger.debug("SIGTERM handler installed for graceful shutdown")
        except ValueError:
            # Signal handlers can only be set in main thread
            # This is expected in test environments
            logger.debug("Skipping SIGTERM handler (not in main thread)")


class ShutdownMiddleware:
    """Pure ASGI middleware for graceful shutdown.

    Rejects new requests with 503 during shutdown and tracks in-flight requests.
    Uses raw ASGI protocol instead of BaseHTTPMiddleware to avoid response body
    wrapping and internal task overhead on the hot path.
    """

    def __init__(self, app: ASGIApp, shutdown_state: ShutdownState) -> None:
        self.app = app
        self.shutdown_state = shutdown_state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self.shutdown_state.shutting_down:
            response = JSONResponse(
                status_code=503,
                content={
                    "error": "Service shutting down",
                    "detail": "Server is draining requests before shutdown. Retry on another instance.",
                },
                headers={"Retry-After": "5"},
            )
            await response(scope, receive, send)
            return

        self.shutdown_state.request_started()
        try:
            await self.app(scope, receive, send)
        finally:
            self.shutdown_state.request_finished()
