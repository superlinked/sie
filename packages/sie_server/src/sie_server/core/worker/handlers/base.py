"""Base protocol for operation handlers.

Operation handlers encapsulate the operation-specific logic for inference:
- make_config_key: Creates hashable key for batching items together
- run_inference: Calls the adapter with correct parameters
- slice_output: Extracts single item from batch output
- assemble_output: Combines partial results into full output

This enables the ModelWorker to remain simple while handlers contain
the complexity of each operation type (encode, extract, score).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.batcher import HasCost
    from sie_server.core.worker.types import RequestMetadata
    from sie_server.types.inputs import Item

# Output type bound to our three output types
TOutput = TypeVar("TOutput")


class OperationHandler[TOutput](ABC):
    """Abstract base class for operation-specific inference logic.

    Each operation (encode, extract, score) has its own handler that
    knows how to group items, run inference, and fan out results.

    Type parameter TOutput is the typed output (EncodeOutput, ExtractOutput, ScoreOutput).
    """

    @abstractmethod
    def make_config_key(self, metadata: RequestMetadata) -> tuple[Any, ...]:
        """Create hashable key for grouping requests by inference config.

        Items with matching config keys can be batched together in a single
        inference call for GPU efficiency.

        Args:
            metadata: Request metadata containing operation parameters.

        Returns:
            Hashable tuple that uniquely identifies the inference configuration.
        """

    @abstractmethod
    def run_inference(
        self,
        adapter: ModelAdapter,
        items: list[Item],
        config_key: tuple[Any, ...],
        prepared_items: list[HasCost] | None,
        metadata_list: list[RequestMetadata],
    ) -> TOutput:
        """Run inference for a group of items with the same config.

        This is called once per unique config key in a batch.
        Runs synchronously (called from thread pool executor).

        Args:
            adapter: The model adapter to use for inference.
            items: Items to process (original Item objects).
            config_key: The config key for this group (from make_config_key).
            prepared_items: Pre-processed items (tokenized, etc.).
            metadata_list: Metadata for items (needed by score to get queries).

        Returns:
            Typed output (EncodeOutput, ExtractOutput, or ScoreOutput).
        """

    @abstractmethod
    def slice_output(self, output: TOutput, index: int) -> TOutput:
        """Extract single item from batched output.

        Used for fanning out batch results to individual request futures.

        Args:
            output: Batched output from run_inference.
            index: Index of item to extract.

        Returns:
            Single-item output (batch_size=1).
        """

    @abstractmethod
    def assemble_output(
        self,
        partials: dict[int, TOutput],
        batch_size: int,
    ) -> TOutput:
        """Assemble partial outputs into full batch.

        Called when all items of a request have been processed.

        Args:
            partials: Dict mapping original_index to single-item output.
            batch_size: Expected total batch size.

        Returns:
            Full batched output with all items in original order.
        """


def make_hashable(value: Any) -> Any:
    """Recursively convert a value to a hashable type for use as dict key.

    Lists become tuples, dicts become sorted tuples of (key, value) pairs.
    Other values are returned as-is.

    Args:
        value: Value to convert.

    Returns:
        Hashable version of the value.
    """
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    return value
