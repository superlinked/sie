"""Score operation handler.

Handles reranking/scoring of (query, document) pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from sie_server.core.inference_output import ScoreOutput
from sie_server.core.worker.handlers.base import OperationHandler, make_hashable
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.batcher import HasCost
    from sie_server.core.worker.types import RequestMetadata


class ScoreHandler(OperationHandler[ScoreOutput]):
    """Handler for score (reranking) operations.

    Supports:
    - Scoring (query, document) pairs
    - Optional instructions for instruction-tuned rerankers
    """

    def make_config_key(self, metadata: RequestMetadata) -> tuple[Any, ...]:
        """Create config key for batching score requests.

        Items with the same instruction and options can be batched together.
        Note: Queries are extracted from metadata during inference.

        Args:
            metadata: Request metadata.

        Returns:
            Hashable tuple for grouping.
        """
        options_key = make_hashable(metadata.options) if metadata.options else None
        return (metadata.instruction, options_key)

    def run_inference(
        self,
        adapter: ModelAdapter,
        items: list[Item],
        config_key: tuple[Any, ...],
        prepared_items: list[HasCost] | None,
        metadata_list: list[RequestMetadata],
    ) -> ScoreOutput:
        """Run score inference.

        Args:
            adapter: The model adapter.
            items: Document items to score.
            config_key: Config key tuple.
            prepared_items: Pre-processed items (unused for score).
            metadata_list: Metadata containing query items.

        Returns:
            ScoreOutput with scores.
        """
        instruction, _options_key = config_key
        # Read original options from metadata (avoids lossy tuple reconstruction)
        options = metadata_list[0].options if metadata_list else None
        queries = self._extract_queries(metadata_list)
        return adapter.score_pairs(
            queries,
            items,
            instruction=instruction,
            options=options,
        )

    def _extract_queries(self, metadata_list: list[RequestMetadata]) -> list[Item]:
        queries = [m.query for m in metadata_list]
        if any(q is None for q in queries):
            raise ValueError("All metadata must have a query for score operations")
        return cast("list[Item]", queries)

    def slice_output(self, output: ScoreOutput, index: int) -> ScoreOutput:
        """Extract single score from batched score output.

        Args:
            output: Batched output.
            index: Index to extract.

        Returns:
            Single-item ScoreOutput.
        """
        return ScoreOutput(
            scores=output.scores[index : index + 1],
            batch_size=1,
        )

    def assemble_output(
        self,
        partials: dict[int, ScoreOutput],
        batch_size: int,
    ) -> ScoreOutput:
        """Assemble partial outputs into full score output.

        Args:
            partials: Dict mapping index to single-item output.
            batch_size: Total batch size.

        Returns:
            Full ScoreOutput.
        """
        if not partials:
            return ScoreOutput(scores=np.array([], dtype=np.float32), batch_size=0)

        scores = np.concatenate([partials[i].scores for i in range(batch_size)])
        return ScoreOutput(scores=scores, batch_size=batch_size)
