from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from sie_server.core.timing import RequestTiming
from sie_server.nats_pull_loop import (
    _DEFAULT_BATCH_BUDGET,
    _MAX_CONCURRENT_BATCHES,
    _NAK_DELAY_S,
    _THRASH_THRESHOLD,
    NatsPullLoop,
)


def _make_work_item(**overrides) -> dict:
    """Build a minimal WorkItem dict with required fields."""
    wi = {
        "work_item_id": "req-1.0",
        "request_id": "req-1",
        "item_index": 0,
        "total_items": 1,
        "operation": "encode",
        "model_id": "test/model",
        "profile_id": "default",
        "pool_name": "_default",
        "router_id": "router-1",
        "reply_subject": "_INBOX.router-1.req-1",
        "timestamp": time.time(),
    }
    wi.update(overrides)
    return wi


def _make_loop(
    *,
    registry: MagicMock | None = None,
    nc: AsyncMock | None = None,
    js: AsyncMock | None = None,
    payload_store_url: str | None = None,
) -> NatsPullLoop:
    """Create a NatsPullLoop with mocked dependencies."""
    if nc is None:
        nc = AsyncMock()
    if js is None:
        js = AsyncMock()
    if registry is None:
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
    return NatsPullLoop(
        nc=nc,
        js=js,
        registry=registry,
        bundle_id="default",
        pool_name="_default",
        payload_store_url=payload_store_url,
    )


def _make_msg(wi: dict) -> AsyncMock:
    """Wrap a WorkItem dict into a mock NATS message with serialized data."""
    msg = AsyncMock()
    msg.data = msgpack.packb(wi, use_bin_type=True)
    return msg


def _published_results(nc_mock: AsyncMock) -> list[dict]:
    """Extract all published WorkResults from the mock nc.publish calls."""
    results = []
    for call in nc_mock.publish.call_args_list:
        _subject, data = call.args
        results.append(msgpack.unpackb(data, raw=False))
    return results


class TestEncodeBatching:
    """Verify encode items are sub-grouped by encode params for separate pipeline calls."""

    @pytest.mark.asyncio
    async def test_encode_batch_groups_by_output_types(self) -> None:
        """Items with different output_types should be processed in separate sub-batches."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)

        fake_timing = RequestTiming()

        # 2 items with ["dense"], 1 with ["dense", "sparse"]
        messages = [
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.0",
                    item_index=0,
                    item={"text": "item A"},
                    output_types=["dense"],
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.1",
                    item_index=1,
                    item={"text": "item B"},
                    output_types=["dense"],
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.2",
                    item_index=2,
                    item={"text": "item C"},
                    output_types=["dense", "sparse"],
                )
            ),
        ]

        call_items_counts: list[int] = []

        async def mock_run_encode(**kwargs):
            n = len(kwargs["items"])
            call_items_counts.append(n)
            fake_outputs = [{"dense": [0.1]} for _ in range(n)]
            return fake_outputs, fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            await loop._process_messages("test/model", messages)

            assert mock_encode.call_count == 2
            assert sorted(call_items_counts) == [1, 2]

    @pytest.mark.asyncio
    async def test_encode_batch_groups_by_instruction(self) -> None:
        """Items with different instruction values should be in separate sub-batches."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        messages = [
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.0",
                    item_index=0,
                    item={"text": "a"},
                    output_types=["dense"],
                    instruction="query:",
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.1",
                    item_index=1,
                    item={"text": "b"},
                    output_types=["dense"],
                    instruction="query:",
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.2",
                    item_index=2,
                    item={"text": "c"},
                    output_types=["dense"],
                    instruction="passage:",
                )
            ),
        ]

        call_instructions: list[str | None] = []

        async def mock_run_encode(**kwargs):
            call_instructions.append(kwargs.get("instruction"))
            n = len(kwargs["items"])
            return [{"dense": [0.1]} for _ in range(n)], fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            await loop._process_messages("test/model", messages)

            assert mock_encode.call_count == 2
            assert sorted(call_instructions) == ["passage:", "query:"]

    @pytest.mark.asyncio
    async def test_encode_batch_groups_by_is_query(self) -> None:
        """Items with is_query=True vs False should be in separate sub-batches."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        messages = [
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.0",
                    item_index=0,
                    item={"text": "query"},
                    output_types=["dense"],
                    is_query=True,
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.1",
                    item_index=1,
                    item={"text": "doc1"},
                    output_types=["dense"],
                    is_query=False,
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.2",
                    item_index=2,
                    item={"text": "doc2"},
                    output_types=["dense"],
                    is_query=False,
                )
            ),
        ]

        call_is_query_values: list[bool] = []

        async def mock_run_encode(**kwargs):
            call_is_query_values.append(kwargs.get("is_query", False))
            n = len(kwargs["items"])
            return [{"dense": [0.1]} for _ in range(n)], fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            await loop._process_messages("test/model", messages)

            assert mock_encode.call_count == 2
            assert sorted(call_is_query_values) == [False, True]

    @pytest.mark.asyncio
    async def test_encode_batch_groups_by_options(self) -> None:
        """Items with different options dicts (e.g., different LoRA adapters) get separate calls."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        messages = [
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.0",
                    item_index=0,
                    item={"text": "a"},
                    output_types=["dense"],
                    options={"lora": "adapter-a"},
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.1",
                    item_index=1,
                    item={"text": "b"},
                    output_types=["dense"],
                    options={"lora": "adapter-b"},
                )
            ),
        ]

        call_options: list[dict] = []

        async def mock_run_encode(**kwargs):
            call_options.append(kwargs.get("options", {}))
            n = len(kwargs["items"])
            return [{"dense": [0.1]} for _ in range(n)], fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            await loop._process_messages("test/model", messages)

            assert mock_encode.call_count == 2
            assert {"lora": "adapter-a"} in call_options
            assert {"lora": "adapter-b"} in call_options

    @pytest.mark.asyncio
    async def test_encode_batch_homogeneous_single_call(self) -> None:
        """All items with identical params should result in a single run_encode call."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        messages = []
        for i in range(5):
            messages.append(
                _make_msg(
                    _make_work_item(
                        work_item_id=f"req-1.{i}",
                        item_index=i,
                        item={"text": f"text {i}"},
                        output_types=["dense"],
                        instruction="query:",
                        is_query=True,
                        options={"pooling": "mean"},
                    )
                )
            )

        async def mock_run_encode(**kwargs):
            n = len(kwargs["items"])
            return [{"dense": [float(i)]} for i in range(n)], fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            await loop._process_messages("test/model", messages)

            # All 5 items have identical params — single call
            mock_encode.assert_called_once()
            assert len(mock_encode.call_args.kwargs["items"]) == 5

        # All 5 results published
        assert nc.publish.await_count == 5


class TestConfigHash:
    """Verify bundle_config_hash validation behavior."""

    @pytest.mark.asyncio
    async def test_config_hash_mismatch_logs_but_processes(self) -> None:
        """Items with wrong bundle_config_hash are still processed (soft check)."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)

        # Pre-seed a known config hash
        loop._config_hash_cache = "correct-hash-abc"
        loop._config_hash_cache_time = time.monotonic()

        fake_timing = RequestTiming()

        wi = _make_work_item(
            item={"text": "hello"},
            output_types=["dense"],
            bundle_config_hash="wrong-hash-xyz",
        )
        msg = _make_msg(wi)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"dense": [0.1]}], fake_timing),
        ) as mock_encode:
            await loop._process_messages("test/model", [msg])

            # Config hash mismatch is now a soft check — item is still processed
            mock_encode.assert_called_once()

        # Message should be ACKed (processed, not NAKed)
        msg.ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_config_hash_empty_passes_through(self) -> None:
        """Items without bundle_config_hash pass through (backward compat)."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        loop._config_hash_cache = "some-hash"
        loop._config_hash_cache_time = time.monotonic()

        fake_timing = RequestTiming()

        # No bundle_config_hash field in the work item
        wi = _make_work_item(
            item={"text": "hello"},
            output_types=["dense"],
        )
        msg = _make_msg(wi)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"dense": [0.1]}], fake_timing),
        ) as mock_encode:
            await loop._process_messages("test/model", [msg])

            # Should pass through — no hash means backward compat
            mock_encode.assert_called_once()

        msg.ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_config_hash_cached(self) -> None:
        """Hash is only computed once per TTL window, not per-message."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        # Force cache to be stale so _check_config_hash will recompute
        loop._config_hash_cache = None
        loop._config_hash_cache_time = 0.0

        # Create 3 messages with matching hash
        messages = []
        for i in range(3):
            wi = _make_work_item(
                work_item_id=f"req-1.{i}",
                item_index=i,
                item={"text": f"text {i}"},
                output_types=["dense"],
                bundle_config_hash="hash-abc",
            )
            messages.append(_make_msg(wi))

        with (
            patch(
                "sie_server.api.ws._compute_bundle_config_hash",
                return_value="hash-abc",
            ) as mock_compute_hash,
            patch(
                "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
                new_callable=AsyncMock,
                return_value=([{"dense": [0.1]} for _ in range(3)], fake_timing),
            ),
        ):
            await loop._process_messages("test/model", messages)

            # Hash should be computed only once (cached for subsequent checks)
            mock_compute_hash.assert_called_once()


class TestConcurrencyLimits:
    """Verify batch semaphore limits concurrent processing."""

    @pytest.mark.asyncio
    async def test_batch_semaphore_limits_concurrency(self) -> None:
        """_MAX_CONCURRENT_BATCHES limits how many batches process simultaneously."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)

        # Verify the semaphore was initialized with the correct value
        assert loop._batch_sem._value == _MAX_CONCURRENT_BATCHES

        # Acquire all permits
        for _ in range(_MAX_CONCURRENT_BATCHES):
            acquired = loop._batch_sem.acquire()
            # Semaphore.acquire is a coroutine
            await acquired

        # Next acquire should block (semaphore exhausted)
        # We test this by checking the value is 0
        assert loop._batch_sem._value == 0

        # Release one and verify it becomes available
        loop._batch_sem.release()
        assert loop._batch_sem._value == 1


class TestGracefulDrain:
    """Verify stop() drains in-flight tasks properly."""

    @pytest.mark.asyncio
    async def test_graceful_drain_waits_for_inflight(self) -> None:
        """stop() waits for in-flight tasks to complete before returning."""
        nc = AsyncMock()
        js = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, js=js, registry=registry)
        loop._running = True
        loop._pull_task = None  # No main pull task to cancel

        task_completed = False

        async def slow_task():
            nonlocal task_completed
            await asyncio.sleep(0.05)
            task_completed = True

        task = asyncio.create_task(slow_task())
        loop._in_flight_tasks.add(task)
        task.add_done_callback(loop._in_flight_tasks.discard)

        await loop.stop()

        # Task should have completed (waited, not cancelled)
        assert task_completed

    @pytest.mark.asyncio
    async def test_graceful_drain_timeout_cancels(self) -> None:
        """If tasks don't complete within drain timeout, they are cancelled."""
        nc = AsyncMock()
        js = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, js=js, registry=registry)
        loop._running = True
        loop._pull_task = None

        task_cancelled = False

        async def stuck_task():
            nonlocal task_cancelled
            try:
                # This task runs longer than drain timeout
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                task_cancelled = True
                raise

        task = asyncio.create_task(stuck_task())
        loop._in_flight_tasks.add(task)
        task.add_done_callback(loop._in_flight_tasks.discard)

        # Temporarily reduce drain timeout for test speed
        with patch("sie_server.nats_pull_loop._DRAIN_TIMEOUT_S", 0.01):
            await loop.stop()

        assert task_cancelled


class TestMixedOperations:
    """Verify batches with mixed operation types are routed correctly."""

    @pytest.mark.asyncio
    async def test_mixed_operations_processed_separately(self) -> None:
        """A pull batch containing encode, score, and extract items triggers all processors."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        encode_msg = _make_msg(
            _make_work_item(
                work_item_id="req-1.0",
                item_index=0,
                operation="encode",
                item={"text": "encode me"},
                output_types=["dense"],
            )
        )
        score_msg = _make_msg(
            _make_work_item(
                work_item_id="req-2.0",
                item_index=0,
                operation="score",
                query_item={"text": "q"},
                score_items=[{"text": "d"}],
            )
        )
        extract_msg = _make_msg(
            _make_work_item(
                work_item_id="req-3.0",
                item_index=0,
                operation="extract",
                item={"text": "extract me"},
                labels=["PER"],
            )
        )

        with (
            patch.object(
                loop,
                "_process_encode_batch",
                new_callable=AsyncMock,
            ) as mock_encode_batch,
            patch.object(
                loop,
                "_process_score_batch",
                new_callable=AsyncMock,
            ) as mock_score_batch,
            patch.object(
                loop,
                "_process_extract_batch",
                new_callable=AsyncMock,
            ) as mock_extract_batch,
        ):
            await loop._process_messages("test/model", [encode_msg, score_msg, extract_msg])

            mock_encode_batch.assert_called_once()
            mock_score_batch.assert_called_once()
            mock_extract_batch.assert_called_once()

            # Verify correct items were routed to each processor
            encode_items = mock_encode_batch.call_args[0][1]
            assert len(encode_items) == 1
            assert encode_items[0][0]["operation"] == "encode"

            score_items = mock_score_batch.call_args[0][1]
            assert len(score_items) == 1
            assert score_items[0][0]["operation"] == "score"

            extract_items = mock_extract_batch.call_args[0][1]
            assert len(extract_items) == 1
            assert extract_items[0][0]["operation"] == "extract"


class TestBatchBudget:
    """Verify _get_batch_budget returns correct values."""

    def test_batch_budget_from_worker_config(self) -> None:
        """_get_batch_budget returns the worker's max_batch_requests."""
        registry = MagicMock()
        registry.model_names = ["test/model"]

        # Mock a worker with a batch config
        mock_worker = MagicMock()
        mock_worker._batch_config.max_batch_requests = 128
        registry.get_worker.return_value = mock_worker

        loop = _make_loop(registry=registry)
        budget = loop._get_batch_budget("test/model")

        assert budget == 128

    def test_batch_budget_default_when_no_worker(self) -> None:
        """When worker is not loaded yet, returns _DEFAULT_BATCH_BUDGET."""
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_worker.return_value = None

        loop = _make_loop(registry=registry)
        budget = loop._get_batch_budget("test/model")

        assert budget == _DEFAULT_BATCH_BUDGET

    def test_batch_budget_default_on_key_error(self) -> None:
        """When get_worker raises KeyError, returns _DEFAULT_BATCH_BUDGET."""
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_worker.side_effect = KeyError("test/model")

        loop = _make_loop(registry=registry)
        budget = loop._get_batch_budget("test/model")

        assert budget == _DEFAULT_BATCH_BUDGET


class TestReactiveModelLoading:
    """Verify reactive model loading, eviction handling, backoff, and thrashing detection."""

    @pytest.mark.asyncio
    async def test_unloaded_model_naks_items_with_delay(self) -> None:
        """All items for an unloaded model are NAKed with delay; background load task created."""
        loop = _make_loop()

        msg1 = AsyncMock()
        msg2 = AsyncMock()
        msg3 = AsyncMock()

        await loop._handle_unloaded_model("test/model", [msg1, msg2, msg3])

        # All 3 messages NAKed with delay
        msg1.nak.assert_awaited_once_with(delay=_NAK_DELAY_S)
        msg2.nak.assert_awaited_once_with(delay=_NAK_DELAY_S)
        msg3.nak.assert_awaited_once_with(delay=_NAK_DELAY_S)

        # No messages ACKed
        msg1.ack.assert_not_awaited()
        msg2.ack.assert_not_awaited()
        msg3.ack.assert_not_awaited()

        # Background load task was created
        assert "test/model" in loop._loading_models
        assert "test/model" in loop._load_tasks

    @pytest.mark.asyncio
    async def test_unloaded_model_triggers_background_load(self) -> None:
        """_handle_unloaded_model calls registry.load_async for the model."""
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.device = "cpu"
        registry.load_async = AsyncMock()

        loop = _make_loop(registry=registry)

        msg = AsyncMock()
        await loop._handle_unloaded_model("test/model", [msg])

        # Wait for the background load task to complete
        load_task = loop._load_tasks["test/model"]
        await load_task

        registry.load_async.assert_awaited_once_with("test/model", "cpu")

        # After task completes and reap runs, model is cleaned up
        loop._reap_load_tasks()
        assert "test/model" not in loop._loading_models

    @pytest.mark.asyncio
    async def test_second_demand_does_not_double_load(self) -> None:
        """A second _handle_unloaded_model call while loading does not trigger a second load."""
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.device = "cpu"

        # Make load_async block until we release it
        load_started = asyncio.Event()
        load_release = asyncio.Event()

        async def slow_load(model_id: str, device: str) -> None:
            load_started.set()
            await load_release.wait()

        registry.load_async = AsyncMock(side_effect=slow_load)

        loop = _make_loop(registry=registry)

        msg1 = AsyncMock()
        msg2 = AsyncMock()

        # First demand — starts loading
        await loop._handle_unloaded_model("test/model", [msg1])
        await load_started.wait()

        # Second demand — should NOT trigger another load
        await loop._handle_unloaded_model("test/model", [msg2])

        # load_async called only once
        registry.load_async.assert_awaited_once()

        # Both messages were still NAKed
        msg1.nak.assert_awaited_once()
        msg2.nak.assert_awaited_once()

        # Clean up
        load_release.set()
        await loop._load_tasks["test/model"]

    @pytest.mark.asyncio
    async def test_failed_load_allows_retry(self) -> None:
        """A failed background load is cleaned up and allows a retry."""
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.device = "cpu"
        registry.load_async = AsyncMock(side_effect=RuntimeError("GPU OOM"))

        loop = _make_loop(registry=registry)

        msg1 = AsyncMock()
        await loop._handle_unloaded_model("test/model", [msg1])

        # Wait for the load task to finish (it will fail)
        load_task = loop._load_tasks["test/model"]
        with pytest.raises(RuntimeError, match="GPU OOM"):
            await load_task

        # Reap completed (failed) tasks
        loop._reap_load_tasks()
        assert "test/model" not in loop._loading_models
        assert "test/model" not in loop._load_tasks

        # Reset the mock to succeed this time and try again
        registry.load_async = AsyncMock()
        msg2 = AsyncMock()
        await loop._handle_unloaded_model("test/model", [msg2])

        # A new load task was created
        assert "test/model" in loop._loading_models
        assert "test/model" in loop._load_tasks

        # Clean up
        await loop._load_tasks["test/model"]

    def test_thrashing_detection(self, caplog: pytest.LogCaptureFixture) -> None:
        """_check_thrashing logs a warning when a model is loaded too frequently."""
        loop = _make_loop()

        now = time.monotonic()
        # Populate load history with THRASH_THRESHOLD entries within the window
        loop._load_history = [("test/model", now - i) for i in range(_THRASH_THRESHOLD)]

        with caplog.at_level(logging.WARNING, logger="sie_server.nats_pull_loop"):
            loop._check_thrashing("test/model")

        assert any("thrashing" in record.message.lower() for record in caplog.records)
        assert any("test/model" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_encode_cancelled_error_naks(self) -> None:
        """CancelledError during encode NAKs items instead of ACKing with error."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)

        msg = _make_msg(
            _make_work_item(
                operation="encode",
                item={"text": "hello"},
                output_types=["dense"],
            )
        )

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ):
            await loop._process_messages("test/model", [msg])

        # Message should be NAKed (eviction path), not ACKed
        msg.nak.assert_awaited()
        # No results published (error or otherwise)
        nc.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_score_model_evicted_naks(self) -> None:
        """KeyError from start_worker (model evicted) NAKs the score message."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(side_effect=KeyError("test/model"))

        loop = _make_loop(nc=nc, registry=registry)

        msg = _make_msg(
            _make_work_item(
                operation="score",
                query_item={"text": "q"},
                score_items=[{"text": "d"}],
            )
        )

        await loop._process_messages("test/model", [msg])

        msg.nak.assert_awaited()

    @pytest.mark.asyncio
    async def test_reap_load_tasks_cleans_completed(self) -> None:
        """_reap_load_tasks removes completed tasks from both tracking structures."""
        loop = _make_loop()

        # Create and await a task so it's done
        async def noop() -> None:
            pass

        task = asyncio.create_task(noop())
        await task  # Let it complete

        loop._load_tasks["test/model"] = task
        loop._loading_models.add("test/model")

        loop._reap_load_tasks()

        assert "test/model" not in loop._loading_models
        assert "test/model" not in loop._load_tasks

    @pytest.mark.asyncio
    async def test_reap_load_tasks_handles_cancelled_task(self) -> None:
        """_reap_load_tasks cleans up a cancelled-but-done task without raising."""
        loop = _make_loop()

        async def long_sleep() -> None:
            await asyncio.sleep(999)

        task = asyncio.create_task(long_sleep())
        # Cancel and let the cancellation propagate
        task.cancel()
        # Gather with return_exceptions so we don't re-raise
        await asyncio.gather(task, return_exceptions=True)

        # Task is now done + cancelled
        assert task.done()
        assert task.cancelled()

        loop._load_tasks["cancelled/model"] = task
        loop._loading_models.add("cancelled/model")

        # _reap_load_tasks should handle the CancelledError branch without raising
        loop._reap_load_tasks()

        assert "cancelled/model" not in loop._loading_models
        assert "cancelled/model" not in loop._load_tasks

    @pytest.mark.asyncio
    async def test_stop_cancels_load_tasks(self) -> None:
        """stop() cancels in-progress background load tasks."""
        nc = AsyncMock()
        js = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, js=js, registry=registry)
        loop._running = True
        loop._pull_task = None

        task_cancelled = False

        async def long_running_load() -> None:
            nonlocal task_cancelled
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                task_cancelled = True
                raise

        load_task = asyncio.create_task(long_running_load())
        loop._load_tasks["test/model"] = load_task
        loop._loading_models.add("test/model")

        # Yield to let the task start running before we stop
        await asyncio.sleep(0)

        await loop.stop()

        # Yield to let cancellation propagate fully
        await asyncio.sleep(0)

        assert task_cancelled
        assert len(loop._load_tasks) == 0
        assert len(loop._loading_models) == 0


class TestPayloadStoreFactory:
    """Tests for _create_payload_store factory function."""

    def test_gs_url_returns_gcs_store(self) -> None:
        """gs:// URLs return a _GCSPayloadStore."""
        from sie_server.nats_pull_loop import _create_payload_store, _GCSPayloadStore

        store = _create_payload_store("gs://my-bucket/payloads")
        assert isinstance(store, _GCSPayloadStore)
        assert store._bucket_name == "my-bucket"
        assert store._prefix == "payloads"

    def test_s3_url_returns_s3_store(self) -> None:
        """s3:// URLs return an _S3PayloadStore."""
        from sie_server.nats_pull_loop import _create_payload_store, _S3PayloadStore

        store = _create_payload_store("s3://my-bucket/payloads")
        assert isinstance(store, _S3PayloadStore)
        assert store._bucket == "my-bucket"
        assert store._prefix == "payloads"

    def test_local_path_returns_local_store(self) -> None:
        """Local paths return a _LocalPayloadStore."""
        from sie_server.nats_pull_loop import _create_payload_store, _LocalPayloadStore

        store = _create_payload_store("/tmp/payloads")  # noqa: S108
        assert isinstance(store, _LocalPayloadStore)

    def test_none_returns_none(self) -> None:
        """None URL returns None."""
        from sie_server.nats_pull_loop import _create_payload_store

        assert _create_payload_store(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string URL returns None."""
        from sie_server.nats_pull_loop import _create_payload_store

        assert _create_payload_store("") is None


class TestOptionsGroupingNestedDicts:
    """Verify options grouping handles nested dicts (e.g., adapter_options with lists)."""

    @pytest.mark.asyncio
    async def test_nested_options_group_correctly(self) -> None:
        """Items with nested options dicts (lists, nested dicts) group correctly and don't crash."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        # Items with nested options containing lists and nested dicts
        messages = [
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.0",
                    item_index=0,
                    item={"text": "a"},
                    output_types=["dense"],
                    options={"adapter_options": {"lora": [1, 2, 3]}, "pooling": "mean"},
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.1",
                    item_index=1,
                    item={"text": "b"},
                    output_types=["dense"],
                    options={"adapter_options": {"lora": [1, 2, 3]}, "pooling": "mean"},
                )
            ),
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.2",
                    item_index=2,
                    item={"text": "c"},
                    output_types=["dense"],
                    options={"adapter_options": {"lora": [4, 5]}, "pooling": "cls"},
                )
            ),
        ]

        call_counts: list[int] = []

        async def mock_run_encode(**kwargs):
            n = len(kwargs["items"])
            call_counts.append(n)
            return [{"dense": [0.1]} for _ in range(n)], fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            await loop._process_messages("test/model", messages)

            # 2 groups: first two items share options, third has different options
            assert mock_encode.call_count == 2
            assert sorted(call_counts) == [1, 2]

    @pytest.mark.asyncio
    async def test_deeply_nested_options_do_not_crash(self) -> None:
        """Options with deeply nested structures (dict of dict of list) don't TypeError."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        loop = _make_loop(nc=nc, registry=registry)
        fake_timing = RequestTiming()

        deep_options = {
            "adapter": {
                "weights": {"layer_0": [0.1, 0.2], "layer_1": [0.3, 0.4]},
                "mode": "inference",
            }
        }

        messages = [
            _make_msg(
                _make_work_item(
                    work_item_id="req-1.0",
                    item_index=0,
                    item={"text": "deep"},
                    output_types=["dense"],
                    options=deep_options,
                )
            ),
        ]

        async def mock_run_encode(**kwargs):
            n = len(kwargs["items"])
            return [{"dense": [0.1]} for _ in range(n)], fake_timing

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=mock_run_encode,
        ) as mock_encode:
            # Should not raise TypeError from tuple(sorted(...)) on nested dicts
            await loop._process_messages("test/model", messages)
            mock_encode.assert_called_once()
