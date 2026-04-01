"""Tests for graceful shutdown handling."""

from __future__ import annotations

import asyncio

import pytest
from sie_server.core.shutdown import ShutdownState


class TestShutdownState:
    """Tests for ShutdownState class."""

    @pytest.fixture
    def shutdown_state(self) -> ShutdownState:
        """Create a fresh shutdown state for each test."""
        return ShutdownState()

    def test_initial_state(self, shutdown_state: ShutdownState) -> None:
        """Shutdown state starts with correct defaults."""
        assert shutdown_state.shutting_down is False
        assert shutdown_state.in_flight == 0

    def test_start_shutdown_sets_flag(self, shutdown_state: ShutdownState) -> None:
        """start_shutdown sets shutting_down flag."""
        shutdown_state.start_shutdown()
        assert shutdown_state.shutting_down is True

    def test_start_shutdown_idempotent(self, shutdown_state: ShutdownState) -> None:
        """Multiple calls to start_shutdown are idempotent."""
        shutdown_state.start_shutdown()
        shutdown_state.start_shutdown()
        assert shutdown_state.shutting_down is True

    def test_request_tracking(self, shutdown_state: ShutdownState) -> None:
        """Request tracking increments and decrements correctly."""
        assert shutdown_state.in_flight == 0

        shutdown_state.request_started()
        assert shutdown_state.in_flight == 1

        shutdown_state.request_started()
        assert shutdown_state.in_flight == 2

        shutdown_state.request_finished()
        assert shutdown_state.in_flight == 1

        shutdown_state.request_finished()
        assert shutdown_state.in_flight == 0

    @pytest.mark.asyncio
    async def test_drain_completes_immediately_when_no_requests(self, shutdown_state: ShutdownState) -> None:
        """wait_for_drain returns immediately when no requests in flight."""
        result = await shutdown_state.wait_for_drain(drain_timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_drain_waits_for_requests(self, shutdown_state: ShutdownState) -> None:
        """wait_for_drain waits for in-flight requests to complete."""
        # Start a request
        shutdown_state.request_started()

        # Start drain in background
        async def drain() -> bool:
            return await shutdown_state.wait_for_drain(drain_timeout=5.0)

        drain_task = asyncio.create_task(drain())

        # Give drain task time to start
        await asyncio.sleep(0.02)
        assert not drain_task.done()

        # Initiate shutdown (needed for drain_event to be set)
        shutdown_state.start_shutdown()

        # Complete the request
        shutdown_state.request_finished()

        # Drain should complete now
        result = await asyncio.wait_for(drain_task, timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_drain_timeout(self, shutdown_state: ShutdownState) -> None:
        """wait_for_drain returns False on timeout."""
        # Start a request that won't complete
        shutdown_state.request_started()
        shutdown_state.start_shutdown()

        # Drain should timeout
        result = await shutdown_state.wait_for_drain(drain_timeout=0.01)
        assert result is False

    @pytest.mark.asyncio
    async def test_middleware_rejects_during_shutdown(self, shutdown_state: ShutdownState) -> None:
        """Middleware returns 503 when shutting down."""
        from unittest.mock import AsyncMock

        from sie_server.core.shutdown import ShutdownMiddleware

        # Track responses sent via ASGI send
        sent: list[dict] = []

        async def mock_send(message: dict) -> None:
            sent.append(message)

        inner_app = AsyncMock()
        middleware = ShutdownMiddleware(inner_app, shutdown_state)

        # Start shutdown
        shutdown_state.start_shutdown()

        # Call middleware with HTTP scope
        scope = {"type": "http"}
        await middleware(scope, AsyncMock(), mock_send)

        # Should return 503 without calling inner app
        inner_app.assert_not_called()
        # ASGI sends http.response.start then http.response.body
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 503

    @pytest.mark.asyncio
    async def test_middleware_tracks_requests(self, shutdown_state: ShutdownState) -> None:
        """Middleware tracks in-flight requests."""
        from unittest.mock import AsyncMock

        from sie_server.core.shutdown import ShutdownMiddleware

        async def inner_app(scope, receive, send):
            # While the inner app is running, request should be tracked
            assert shutdown_state.in_flight == 1

        middleware = ShutdownMiddleware(inner_app, shutdown_state)

        assert shutdown_state.in_flight == 0

        # Call middleware with HTTP scope
        scope = {"type": "http"}
        await middleware(scope, AsyncMock(), AsyncMock())

        # Request should be complete
        assert shutdown_state.in_flight == 0

    @pytest.mark.asyncio
    async def test_middleware_tracks_requests_on_error(self, shutdown_state: ShutdownState) -> None:
        """Middleware decrements counter even on error."""
        from unittest.mock import AsyncMock

        from sie_server.core.shutdown import ShutdownMiddleware

        async def inner_app(scope, receive, send):
            raise RuntimeError("boom")

        middleware = ShutdownMiddleware(inner_app, shutdown_state)

        assert shutdown_state.in_flight == 0

        # Call middleware - should raise but still clean up
        with pytest.raises(RuntimeError, match="boom"):
            scope = {"type": "http"}
            await middleware(scope, AsyncMock(), AsyncMock())

        # Request should still be complete
        assert shutdown_state.in_flight == 0

    @pytest.mark.asyncio
    async def test_middleware_passes_non_http_through(self, shutdown_state: ShutdownState) -> None:
        """Middleware passes non-HTTP scopes directly to inner app."""
        from unittest.mock import AsyncMock

        from sie_server.core.shutdown import ShutdownMiddleware

        inner_app = AsyncMock()
        middleware = ShutdownMiddleware(inner_app, shutdown_state)

        scope = {"type": "websocket"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        inner_app.assert_called_once_with(scope, receive, send)
        assert shutdown_state.in_flight == 0


class TestSignalHandler:
    """Tests for signal handler setup."""

    def test_setup_signal_handlers_does_not_crash(self) -> None:
        """setup_signal_handlers handles not being in main thread gracefully."""
        from sie_server.core.shutdown import setup_signal_handlers

        shutdown_state = ShutdownState()
        # Should not raise even in test thread
        setup_signal_handlers(shutdown_state)
