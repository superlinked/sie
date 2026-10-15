import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import InvalidInputError, Item


class TestModelWorkerLoRABatching:
    """Tests for LoRA-aware batching."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )
        return mock

    @pytest.mark.asyncio
    async def test_different_loras_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Requests with different LoRA adapters are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # Two requests with different LoRA adapters
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"lora": "legal"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                options={"lora": "medical"},
            )

            await asyncio.gather(future1, future2)

            # Should have 2 separate encode calls (different LoRAs)
            assert mock_adapter.encode.call_count == 2

            # Verify set_active_lora was called for each LoRA
            loras = [call.args[0] for call in mock_adapter.set_active_lora.call_args_list]
            assert "legal" in loras
            assert "medical" in loras

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_a_lora_the_adapter_cannot_activate_fails_its_requests(self, mock_adapter: MagicMock) -> None:
        def set_active_lora(name: str | None) -> None:
            if name is not None:
                msg = f"LoRA '{name}' is not loaded. Available: {{'org/private-lora'}}"
                raise ValueError(msg)

        mock_adapter.set_active_lora.side_effect = set_active_lora
        worker = ModelWorker(mock_adapter, WorkerConfig(max_batch_wait_ms=1))
        await worker.start()
        try:
            failing = await worker.submit(
                [make_text_item([1, 2, 3], 0)], [Item(text="hello")], ["dense"], options={"lora": "gone"}
            )
            with pytest.raises(InvalidInputError) as raised:
                await asyncio.wait_for(failing, 5)
            served = await worker.submit([make_text_item([4, 5, 6], 0)], [Item(text="world")], ["dense"])
            result = await asyncio.wait_for(served, 5)
        finally:
            await worker.stop()

        assert "gone" in str(raised.value)
        assert "org/private-lora" not in str(raised.value)
        assert result.output.dense is not None

    @pytest.mark.asyncio
    async def test_same_lora_batched_together(self, mock_adapter: MagicMock) -> None:
        """Requests with the same LoRA adapter are batched together."""
        # Return 2 embeddings for batched call
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            batch_size=2,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # Two requests with the SAME LoRA adapter
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"lora": "legal"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                options={"lora": "legal"},
            )

            await asyncio.gather(future1, future2)

            # Should batch together - only 1 encode call
            assert mock_adapter.encode.call_count == 1

            # Verify set_active_lora was called with the LoRA
            mock_adapter.set_active_lora.assert_called_with("legal")

            # Verify 2 items were batched
            call_args = mock_adapter.encode.call_args.args
            assert len(call_args[0]) == 2  # 2 items

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_no_lora_batched_together(self, mock_adapter: MagicMock) -> None:
        """Requests without LoRA are batched together."""
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            batch_size=2,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # Two requests without LoRA
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
            )

            await asyncio.gather(future1, future2)

            # Should batch together
            assert mock_adapter.encode.call_count == 1

            # Verify set_active_lora was called with None (base model)
            mock_adapter.set_active_lora.assert_called_with(None)

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_lora_vs_no_lora_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Requests with LoRA and without LoRA are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            # One request with LoRA, one without
            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"lora": "legal"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                # No lora
            )

            await asyncio.gather(future1, future2)

            # Should have 2 separate encode calls
            assert mock_adapter.encode.call_count == 2

            # Verify set_active_lora was called for both LoRA and base model
            loras = [call.args[0] for call in mock_adapter.set_active_lora.call_args_list]
            assert "legal" in loras
            assert None in loras

        finally:
            await worker.stop()


class TestDrainActiveBatcherFairness:
    """The continuous-batching drain must be bounded to the backlog present at
    entry, so a saturated LoRA that keeps refilling its own batcher cannot loop
    forever and starve sibling LoRAs / the base model. See #1606.
    """

    @pytest.mark.asyncio
    async def test_drain_is_bounded_by_entry_backlog(self) -> None:
        worker = ModelWorker(MagicMock(), WorkerConfig())
        worker._process_batch = AsyncMock()  # don't run real inference

        # A batcher whose try_drain ALWAYS returns a size-1 batch simulates a
        # LoRA refilling itself during every forward pass. Unbounded, this loops
        # forever; bounded by the entry backlog it stops after pending_count.
        entry_backlog = 5
        drain_batch = MagicMock()
        drain_batch.size = 1
        drain_batch.total_tokens = 1

        batcher = MagicMock()
        batcher.pending_count = entry_backlog
        batcher.try_drain = AsyncMock(return_value=drain_batch)
        worker._batchers = {"A": batcher}

        # wait_for is a safety net: if the bound regressed, fail (not hang).
        drained = await asyncio.wait_for(worker._drain_active_batcher("A"), timeout=2.0)

        assert drained is True
        assert worker._process_batch.await_count == entry_backlog

    @pytest.mark.asyncio
    async def test_drain_noop_when_batcher_absent(self) -> None:
        worker = ModelWorker(MagicMock(), WorkerConfig())
        assert await worker._drain_active_batcher("no-such-lora") is False
