from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from sie_server.core.inference_output import EncodeOutput
from sie_server.core.worker.handlers.base import OperationHandler, make_hashable

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.batcher import HasCost
    from sie_server.core.postprocessor_registry import PostprocessorRegistry
    from sie_server.core.worker.types import RequestMetadata
    from sie_server.types.inputs import Item

# ``EncodeOutput.extra`` keys whose values are per-item lists aligned 1:1 with
# the batch, and which therefore MUST be sliced and reassembled with the item.
# These are the unit meter's authoritative bases (§7): real tokenizer counts
# from adapters that own their tokenization, and the images/video frames an
# adapter actually processed — a number the wire item cannot express. Dropping
# either here silently strips the meter's evidence whenever the worker fuses
# several API requests into one GPU batch, which then bills the frames as zero
# or faults settlement outright. Every other ``extra`` key has no defined
# per-item semantics and is intentionally not propagated.
# ``text_tower_skipped`` (#2538) is the third: the modality routing a vision
# adapter actually applied, which is how the pipeline knows an item's zero text
# tokens is a MEASUREMENT ("it took the image tower") and not an absent count.
# It must be sliced with its item for the same reason the counts are.
PER_ITEM_EXTRA_KEYS: Final[tuple[str, ...]] = (
    "input_token_counts",
    "input_image_counts",
    "text_tower_skipped",
)


class EncodeHandler(OperationHandler[EncodeOutput]):
    """Handler for encode (embedding) operations.

    Supports:
    - Dense, sparse, and multivector embeddings
    - Instruction-tuned models (query vs document)
    - LoRA adapters (batched by LoRA name)
    - Postprocessing (MUVERA, quantization, etc.)
    """

    def __init__(
        self,
        model_name: str | None = None,
        postprocessor_registry: PostprocessorRegistry | None = None,
    ) -> None:
        """Initialize the encode handler.

        Args:
            model_name: Name of the model (for postprocessor lookup).
            postprocessor_registry: Registry for postprocessors (optional).
        """
        self._model_name = model_name
        self._postprocessor_registry = postprocessor_registry

    def make_config_key(self, metadata: RequestMetadata) -> tuple[Any, ...]:
        """Create config key for batching encode requests.

        Items with the same (output_types, instruction, is_query, lora, options)
        can be batched together.

        Args:
            metadata: Request metadata.

        Returns:
            Hashable tuple for grouping.
        """
        lora = metadata.options.get("lora") if metadata.options else None
        options_key = make_hashable(metadata.options) if metadata.options else None
        return (
            tuple(sorted(metadata.output_types)),
            metadata.instruction,
            metadata.is_query,
            lora,
            options_key,
        )

    def run_inference(
        self,
        adapter: ModelAdapter,
        items: list[Item],
        config_key: tuple[Any, ...],
        prepared_items: list[HasCost] | None,
        metadata_list: list[RequestMetadata],
    ) -> EncodeOutput:
        """Run encode inference.

        Args:
            adapter: The model adapter.
            items: Items to encode.
            config_key: Config key tuple.
            prepared_items: Pre-processed items (tokenized).
            metadata_list: Metadata for timing updates.

        Returns:
            EncodeOutput with embeddings.
        """
        output_types_tuple, instruction, is_query, _lora, _options_tuple = config_key
        # Read original options from metadata (avoids lossy tuple reconstruction)
        options = metadata_list[0].options or {} if metadata_list else {}
        output = self.encode(
            adapter=adapter,
            items=items,
            output_types=list(output_types_tuple),
            is_query=is_query,
            options=options,
            instruction=instruction,
            prepared_items=prepared_items,
        )
        postprocess_ms = self.post_process(is_query, options, output)
        for metadata in metadata_list:
            metadata.timing.add_postprocessing_ms(postprocess_ms)
        return output

    def post_process(self, is_query: bool, options: dict[str, Any], encode_output: EncodeOutput) -> float:
        if self._postprocessor_registry and self._model_name:
            return self._postprocessor_registry.transform_sync(
                self._model_name, encode_output, options, is_query=is_query
            )
        return 0

    def encode(
        self,
        adapter: ModelAdapter,
        items: list[Item],
        output_types: list[str],
        is_query: bool,
        options: dict[str, Any],
        instruction: str | None = None,
        prepared_items: list[Any] | None = None,
    ) -> EncodeOutput:
        return adapter.encode(
            items,
            output_types,
            instruction=instruction,
            is_query=is_query,
            prepared_items=prepared_items,
            options=options,
        )

    def slice_output(self, output: EncodeOutput, index: int) -> EncodeOutput:
        """Extract single item from batched encode output.

        Args:
            output: Batched output.
            index: Index to extract.

        Returns:
            Single-item EncodeOutput.
        """
        # Per-item unit counts are positional like dense/sparse, so they must
        # be sliced with the item — see ``PER_ITEM_EXTRA_KEYS``.
        extra: dict[str, Any] = {}
        for key in PER_ITEM_EXTRA_KEYS:
            counts = output.extra.get(key) if output.extra else None
            if isinstance(counts, list) and 0 <= index < len(counts):
                extra[key] = [counts[index]]
        return EncodeOutput(
            dense=output.dense[index : index + 1] if output.dense is not None else None,
            sparse=[output.sparse[index]] if output.sparse is not None else None,
            multivector=[output.multivector[index]] if output.multivector is not None else None,
            batch_size=1,
            is_query=output.is_query,
            dense_dim=output.dense_dim,
            multivector_token_dim=output.multivector_token_dim,
            extra=extra,
        )

    def assemble_output(
        self,
        partials: dict[int, EncodeOutput],
        batch_size: int,
    ) -> EncodeOutput:
        """Assemble partial outputs into full encode output.

        Args:
            partials: Dict mapping index to single-item output.
            batch_size: Total batch size.

        Returns:
            Full EncodeOutput.
        """
        if not partials:
            return EncodeOutput(batch_size=0)

        # Fast path: single item — skip concatenation entirely
        if batch_size == 1:
            return partials[0]

        # Fast path: single partial covering all items — skip slice→assemble round-trip
        if len(partials) == 1:
            only_output = next(iter(partials.values()))
            if only_output.batch_size == batch_size:
                return only_output

        # Get first partial to determine which fields are present
        first = next(iter(partials.values()))

        # Assemble dense: stack [1, dim] arrays into [batch, dim]
        dense = None
        if first.dense is not None:
            dense_list = [partials[i].dense for i in range(batch_size)]
            dense = np.concatenate(dense_list, axis=0)  # ty: ignore[no-matching-overload]

        # Assemble sparse: concatenate lists
        sparse = None
        if first.sparse is not None:
            sparse = [partials[i].sparse[0] for i in range(batch_size)]  # ty: ignore[not-subscriptable]

        # Assemble multivector: concatenate lists
        multivector = None
        if first.multivector is not None:
            multivector = [partials[i].multivector[0] for i in range(batch_size)]  # ty: ignore[not-subscriptable]

        # Reassemble per-item unit counts (see slice_output). All-or-nothing
        # per key: a partial without a count means the meter cannot attribute
        # the request exactly, so that dimension is not surfaced (metering then
        # falls back to its reserve estimate rather than under-counting). The
        # keys are independent — a video encode carries image counts with no
        # token counts at all.
        extra: dict[str, Any] = {}
        for key in PER_ITEM_EXTRA_KEYS:
            assembled_counts: list[int] = []
            for i in range(batch_size):
                partial_counts = partials[i].extra.get(key) if partials[i].extra else None
                if not (isinstance(partial_counts, list) and len(partial_counts) == 1):
                    assembled_counts = []
                    break
                assembled_counts.append(partial_counts[0])
            if assembled_counts:
                extra[key] = assembled_counts

        return EncodeOutput(
            dense=dense,
            sparse=sparse,
            multivector=multivector,
            batch_size=batch_size,
            is_query=first.is_query,
            dense_dim=first.dense_dim,
            multivector_token_dim=first.multivector_token_dim,
            extra=extra,
        )

    @classmethod
    def format_output(
        cls,
        output: EncodeOutput,
        output_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Convert EncodeOutput to per-item dicts for API response.

        Args:
            output: Batched encode output from adapter.encode() or postprocessors.
            **kwargs: Accepts 'output_types' (list[str] | None) to filter which
                outputs to include. If None/missing, includes all available.

        Returns:
            List of dicts, one per item, with requested output types.
            Keys: "dense", "sparse", "multivector"
        """
        results: list[dict[str, Any]] = [{} for _ in range(output.batch_size)]

        # Dense: unbatch [batch, dim] -> list of [dim]
        if output.dense is not None and (output_types is None or "dense" in output_types):
            for i in range(output.batch_size):
                results[i]["dense"] = output.dense[i]

        # Sparse: already per-item, convert SparseVector to dict format
        if output.sparse is not None and (output_types is None or "sparse" in output_types):
            for i, sv in enumerate(output.sparse):
                results[i]["sparse"] = {"indices": sv.indices, "values": sv.values}

        # Multivector: already per-item
        if output.multivector is not None and (output_types is None or "multivector" in output_types):
            for i, mv in enumerate(output.multivector):
                results[i]["multivector"] = mv

        return results
