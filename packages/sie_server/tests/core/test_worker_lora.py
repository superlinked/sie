import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import Item


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
