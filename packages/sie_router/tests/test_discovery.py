"""Tests for worker discovery and watch loop recovery."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from sie_router.discovery import (
    DiscoveryEvent,
    DiscoveryEventType,
    WorkerConnectionManager,
    WorkerDiscovery,
)
from sie_router.registry import WorkerRegistry


def _make_fast_sleep():
    """Create a fast sleep mock that returns immediately for short delays."""
    original_sleep = asyncio.sleep

    async def fast_sleep(delay: float):
        # Allow very short sleeps to proceed normally
        if delay <= 0.1:
            await original_sleep(delay)
        else:
            # For longer sleeps (backoff), just yield control
            await original_sleep(0.01)

    return fast_sleep


class FlakyWatchDiscovery(WorkerDiscovery):
    """Discovery that raises on first N watch() calls, then yields events."""

    def __init__(self, fail_count: int = 1, error: Exception | None = None) -> None:
        self._fail_count = fail_count
        self._call_count = 0
        self._error = error or RuntimeError("Connection refused")
        self.events_yielded = asyncio.Event()
        self._stop = asyncio.Event()

    async def get_worker_urls(self) -> list[str]:
        return []

    @property
    def supports_watch(self) -> bool:
        return True

    async def watch(self) -> AsyncIterator[DiscoveryEvent]:
        self._call_count += 1

        if self._call_count <= self._fail_count:
            raise self._error

        # On successful call, yield a worker added event
        yield DiscoveryEvent(DiscoveryEventType.ADDED, "http://10.1.6.6:8080")
        self.events_yielded.set()
        # Wait until stopped (will be cancelled by manager.stop())
        await self._stop.wait()


class TestWatchLoopRecovery:
    """Test that the watch loop recovers from transient failures."""

    @pytest.mark.asyncio
    async def test_watch_loop_retries_after_runtime_error(self) -> None:
        """Watch loop retries after a RuntimeError from the watch generator."""
        registry = WorkerRegistry()
        discovery = FlakyWatchDiscovery(fail_count=1)
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        # Patch _connect_to_worker to avoid real WebSocket connections
        connected_urls: list[str] = []

        async def fake_connect(url: str) -> None:
            connected_urls.append(url)

        manager._connect_to_worker = fake_connect  # type: ignore[assignment]

        # Mock asyncio.sleep to speed up backoff delays
        with patch("sie_router.discovery.asyncio.sleep", new=_make_fast_sleep()):
            await manager.start()

            try:
                await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("Watch loop did not recover after transient error")
            finally:
                await manager.stop()

        assert "http://10.1.6.6:8080" in connected_urls

    @pytest.mark.asyncio
    async def test_watch_loop_retries_after_multiple_failures(self) -> None:
        """Watch loop retries multiple times with increasing backoff."""
        registry = WorkerRegistry()
        discovery = FlakyWatchDiscovery(fail_count=3)
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)
        manager._connect_to_worker = lambda url: asyncio.sleep(0)  # type: ignore[assignment]

        # Mock asyncio.sleep to speed up backoff delays
        with patch("sie_router.discovery.asyncio.sleep", new=_make_fast_sleep()):
            await manager.start()

            try:
                await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("Watch loop did not recover after 3 failures")
            finally:
                await manager.stop()

        # Verify all failures were encountered before success
        assert discovery._call_count == 4  # 3 failures + 1 success

    @pytest.mark.asyncio
    async def test_watch_loop_handles_max_retry_error(self) -> None:
        """Watch loop recovers from urllib3 MaxRetryError (the production bug)."""
        # Simulate the exact error type that caused the production failure
        error = Exception(
            "HTTPSConnectionPool(host='10.2.0.1', port=443): "
            "Max retries exceeded with url: /api/v1/namespaces/sie/endpoints"
        )
        registry = WorkerRegistry()
        discovery = FlakyWatchDiscovery(fail_count=1, error=error)
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)
        manager._connect_to_worker = lambda url: asyncio.sleep(0)  # type: ignore[assignment]

        # Mock asyncio.sleep to speed up backoff delays
        with patch("sie_router.discovery.asyncio.sleep", new=_make_fast_sleep()):
            await manager.start()

            try:
                await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("Watch loop did not recover from MaxRetryError")
            finally:
                await manager.stop()

    @pytest.mark.asyncio
    async def test_watch_loop_discovers_worker_after_recovery(self) -> None:
        """After recovery, newly discovered workers get connection tasks started."""
        registry = WorkerRegistry()
        discovery = FlakyWatchDiscovery(fail_count=2)
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        connected_urls: list[str] = []

        async def fake_connect(url: str) -> None:
            connected_urls.append(url)

        manager._connect_to_worker = fake_connect  # type: ignore[assignment]

        # Mock asyncio.sleep to speed up backoff delays
        with patch("sie_router.discovery.asyncio.sleep", new=_make_fast_sleep()):
            await manager.start()

            try:
                await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail("Watch loop did not recover")
            finally:
                await manager.stop()

        assert "http://10.1.6.6:8080" in connected_urls

    @pytest.mark.asyncio
    async def test_watch_loop_cancellation_is_clean(self) -> None:
        """Cancelling the watch loop (via stop()) works cleanly."""
        registry = WorkerRegistry()

        class NeverSucceedsDiscovery(WorkerDiscovery):
            """Always fails to ensure the loop keeps retrying."""

            call_count = 0

            async def get_worker_urls(self) -> list[str]:
                return []

            @property
            def supports_watch(self) -> bool:
                return True

            async def watch(self) -> AsyncIterator[DiscoveryEvent]:
                self.call_count += 1
                raise RuntimeError("Always fails")
                yield  # type: ignore[misc]

        discovery = NeverSucceedsDiscovery()
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        # Mock asyncio.sleep to speed up backoff delays
        with patch("sie_router.discovery.asyncio.sleep", new=_make_fast_sleep()):
            await manager.start()
            # Wait just enough for at least 1 retry attempt
            await asyncio.sleep(0.05)
            # Stop should be clean (no unhandled exceptions)
            await manager.stop()

        # At least one retry happened (proving the loop runs and can be cancelled)
        assert discovery.call_count >= 1


class AddThenRemoveDiscovery(WorkerDiscovery):
    """Discovery that yields ADDED then REMOVED for the same worker."""

    def __init__(self) -> None:
        self.events_yielded = asyncio.Event()
        self._stop = asyncio.Event()

    async def get_worker_urls(self) -> list[str]:
        return []

    @property
    def supports_watch(self) -> bool:
        return True

    async def watch(self) -> AsyncIterator[DiscoveryEvent]:
        url = "http://10.1.6.6:8080"
        yield DiscoveryEvent(DiscoveryEventType.ADDED, url)
        # Small delay to let connection task start
        await asyncio.sleep(0.01)
        yield DiscoveryEvent(DiscoveryEventType.REMOVED, url)
        self.events_yielded.set()
        await self._stop.wait()


class ReAddAfterRemoveDiscovery(WorkerDiscovery):
    """Discovery that yields ADDED, REMOVED, then ADDED again for the same URL."""

    def __init__(self) -> None:
        self.events_yielded = asyncio.Event()
        self._stop = asyncio.Event()

    async def get_worker_urls(self) -> list[str]:
        return []

    @property
    def supports_watch(self) -> bool:
        return True

    async def watch(self) -> AsyncIterator[DiscoveryEvent]:
        url = "http://10.1.6.6:8080"
        yield DiscoveryEvent(DiscoveryEventType.ADDED, url)
        await asyncio.sleep(0.01)
        yield DiscoveryEvent(DiscoveryEventType.REMOVED, url)
        await asyncio.sleep(0.01)
        yield DiscoveryEvent(DiscoveryEventType.ADDED, url)
        self.events_yielded.set()
        await self._stop.wait()


class TestRemovedWorkerReconnection:
    """Test that removed workers are not reconnected to."""

    @pytest.mark.asyncio
    async def test_removed_worker_is_tracked(self) -> None:
        """After REMOVED event, the URL is in _removed_urls."""
        registry = WorkerRegistry()
        discovery = AddThenRemoveDiscovery()
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        connected_urls: list[str] = []

        async def fake_connect(url: str) -> None:
            connected_urls.append(url)

        manager._connect_to_worker = fake_connect  # type: ignore[assignment]

        await manager.start()

        try:
            await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
            # Give watch loop time to process REMOVED event
            await asyncio.sleep(0.01)
        except TimeoutError:
            pytest.fail("Discovery events were not yielded in time")
        finally:
            await manager.stop()

        assert "http://10.1.6.6:8080" in manager._removed_urls

    @pytest.mark.asyncio
    async def test_readded_worker_clears_removed(self) -> None:
        """Re-adding a worker clears it from _removed_urls."""
        registry = WorkerRegistry()
        discovery = ReAddAfterRemoveDiscovery()
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        connected_urls: list[str] = []

        async def fake_connect(url: str) -> None:
            connected_urls.append(url)

        manager._connect_to_worker = fake_connect  # type: ignore[assignment]

        await manager.start()

        try:
            await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
            await asyncio.sleep(0.01)
        except TimeoutError:
            pytest.fail("Discovery events were not yielded in time")
        finally:
            await manager.stop()

        # URL was re-added, so should NOT be in _removed_urls
        assert "http://10.1.6.6:8080" not in manager._removed_urls
        # Should have been connected twice (once for initial ADD, once for re-ADD)
        assert connected_urls.count("http://10.1.6.6:8080") == 2


class AddThenRemoveWithDelayDiscovery(WorkerDiscovery):
    """Discovery that emits ADDED, waits for the connection to fail, then emits REMOVED.

    The delay between ADDED and REMOVED is configurable to allow the connection
    task to attempt (and fail) a WebSocket connection before the removal occurs.
    """

    def __init__(self, *, pre_remove_delay: float = 0.02) -> None:
        self.events_yielded = asyncio.Event()
        self._stop = asyncio.Event()
        self._pre_remove_delay = pre_remove_delay

    async def get_worker_urls(self) -> list[str]:
        return []

    @property
    def supports_watch(self) -> bool:
        return True

    async def watch(self) -> AsyncIterator[DiscoveryEvent]:
        url = "http://10.1.6.6:8080"
        yield DiscoveryEvent(DiscoveryEventType.ADDED, url)
        # Wait long enough for the connection task to attempt and fail
        await asyncio.sleep(self._pre_remove_delay)
        yield DiscoveryEvent(DiscoveryEventType.REMOVED, url)
        self.events_yielded.set()
        await self._stop.wait()


class TestReconnectionBlocking:
    """Test that _connect_to_worker exits early when the URL is in _removed_urls.

    These tests exercise the actual reconnection-blocking code path:
        if url in self._removed_urls:
            logger.info("Worker %s was removed; stopping reconnection", url)

    Return:
    The existing TestRemovedWorkerReconnection tests verify that _removed_urls
    is populated correctly but don't test the blocking behavior itself because
    they mock out _connect_to_worker entirely.
    """

    @pytest.mark.asyncio
    async def test_connect_to_worker_exits_when_url_in_removed_urls(self) -> None:
        """_connect_to_worker returns without calling mark_unhealthy when URL is removed.

        This directly tests the reconnection-blocking code path by:
        1. Pre-populating _removed_urls with the worker URL
        2. Mocking websockets.connect to raise OSError (simulating connection failure)
        3. Calling _connect_to_worker and verifying it returns without mark_unhealthy
        """
        registry = WorkerRegistry()
        registry.mark_unhealthy = AsyncMock(wraps=registry.mark_unhealthy)  # type: ignore[method-assign]

        discovery = AddThenRemoveDiscovery()
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        url = "http://10.1.6.6:8080"
        manager._running = True
        manager._removed_urls.add(url)

        # Mock websockets.connect to raise OSError (connection refused)
        with patch("sie_router.discovery.websockets.connect", side_effect=OSError("Connection refused")):
            # _connect_to_worker should return promptly because it sees
            # url in _removed_urls after the first failed connection attempt
            await asyncio.wait_for(manager._connect_to_worker(url), timeout=2.0)

        # mark_unhealthy must NOT have been called -- the early return skips it
        registry.mark_unhealthy.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_to_worker_calls_mark_unhealthy_when_url_not_removed(self) -> None:
        """Contrast test: mark_unhealthy IS called when URL is NOT in _removed_urls.

        This ensures our mocking strategy is correct and that the only thing
        preventing mark_unhealthy is the _removed_urls check.
        """
        registry = WorkerRegistry()
        registry.mark_unhealthy = AsyncMock(wraps=registry.mark_unhealthy)  # type: ignore[method-assign]

        discovery = AddThenRemoveDiscovery()
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        url = "http://10.1.6.6:8080"
        manager._running = True
        # Crucially: do NOT add url to _removed_urls

        connect_count = 0

        def fake_connect(*args: object, **kwargs: object) -> None:
            nonlocal connect_count
            connect_count += 1
            # After first attempt triggers mark_unhealthy + reconnect,
            # stop the loop so the test doesn't run forever
            if connect_count >= 2:
                manager._running = False
            raise OSError("Connection refused")

        with patch("sie_router.discovery.websockets.connect", side_effect=fake_connect):
            await asyncio.wait_for(manager._connect_to_worker(url), timeout=5.0)

        # mark_unhealthy SHOULD have been called since URL is not removed
        registry.mark_unhealthy.assert_called_with(url)

    @pytest.mark.asyncio
    async def test_removed_worker_does_not_reconnect_end_to_end(self) -> None:
        """End-to-end test: ADDED -> connection fails -> REMOVED -> no reconnection.

        Uses the real _connect_to_worker (not mocked) with a fake WebSocket
        layer to verify the full flow from discovery events through to the
        reconnection-blocking behavior.
        """
        registry = WorkerRegistry()
        registry.mark_unhealthy = AsyncMock(wraps=registry.mark_unhealthy)  # type: ignore[method-assign]
        discovery = AddThenRemoveWithDelayDiscovery(pre_remove_delay=0.01)
        manager = WorkerConnectionManager(registry, discovery, reconnect_delay_s=0.01)

        url = "http://10.1.6.6:8080"

        # Mock websockets.connect to always raise OSError so _connect_to_worker
        # falls through to the reconnection logic each time
        with patch("sie_router.discovery.websockets.connect", side_effect=OSError("Connection refused")):
            await manager.start()

            try:
                # Wait for discovery to emit both ADDED and REMOVED
                await asyncio.wait_for(discovery.events_yielded.wait(), timeout=2.0)
                # Give watch loop time to process the REMOVED event (cancel task, etc.)
                await asyncio.sleep(0.01)
            except TimeoutError:
                pytest.fail("Discovery events were not yielded in time")
            finally:
                await manager.stop()

        # The URL should be in _removed_urls
        assert url in manager._removed_urls

        # The connection task for this URL should have been cleaned up
        assert url not in manager._tasks

        # mark_unhealthy should NOT have been called after the removal.
        # It may or may not have been called before removal (depending on timing),
        # but the key assertion is that after REMOVED, the task did not proceed
        # to mark_unhealthy + reconnect. Since the REMOVED handler cancels the
        # task, and _removed_urls prevents any reconnect loop, the worker should
        # have been removed cleanly via remove_worker, not marked unhealthy.
        # We verify remove_worker was called by checking the registry is empty.
        assert registry.get_worker(url) is None
