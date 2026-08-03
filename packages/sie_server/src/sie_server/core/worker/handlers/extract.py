"""Extract operation handler.

Handles entity extraction (NER, RE) from text and images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sie_server.core.inference_output import ExtractOutput
from sie_server.core.worker.handlers.base import OperationHandler, make_hashable

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.batcher import HasCost
    from sie_server.core.worker.types import RequestMetadata
    from sie_server.types.inputs import Item
    from sie_server.types.responses import Classification, DetectedObject, Relation


class ExtractHandler(OperationHandler[ExtractOutput]):
    """Handler for extract (NER, RE) operations.

    Supports:
    - Entity extraction with configurable labels
    - Structured extraction with output schema
    - Optional instructions for instruction-tuned models
    """

    def make_config_key(self, metadata: RequestMetadata) -> tuple[Any, ...]:
        """Create config key for batching extract requests.

        Items with the same (labels, output_schema, instruction, options) can be batched together.

        Args:
            metadata: Request metadata.

        Returns:
            Hashable tuple for grouping.
        """
        # Label order is part of the model input for label-conditioned extractors.
        labels_key = tuple(metadata.labels) if metadata.labels else None
        schema_key = make_hashable(metadata.output_schema) if metadata.output_schema is not None else None
        options_key = make_hashable(metadata.options) if metadata.options else None
        return (
            labels_key,
            schema_key,
            metadata.instruction,
            options_key,
        )

    def run_inference(
        self,
        adapter: ModelAdapter,
        items: list[Item],
        config_key: tuple[Any, ...],
        prepared_items: list[HasCost] | None,
        metadata_list: list[RequestMetadata],
    ) -> ExtractOutput:
        """Run extract inference.

        Args:
            adapter: The model adapter.
            items: Items to extract from.
            config_key: Config key tuple.
            prepared_items: Pre-processed items.
            metadata_list: Metadata (unused for extract).

        Returns:
            ExtractOutput with entities.
        """
        labels_tuple, _schema_key, instruction, options_tuple = config_key
        labels = list(labels_tuple) if labels_tuple else None
        options = dict(options_tuple) if options_tuple else None
        output_schema = metadata_list[0].output_schema

        return adapter.extract(
            items,
            labels=labels,
            output_schema=output_schema,
            instruction=instruction,
            options=options,
            prepared_items=prepared_items,
        )

    def slice_output(self, output: ExtractOutput, index: int) -> ExtractOutput:
        """Extract single item from batched extract output.

        Args:
            output: Batched output.
            index: Index to extract.

        Returns:
            Single-item ExtractOutput.
        """
        classifications = [output.classifications[index]] if output.classifications is not None else None
        relations = [output.relations[index]] if output.relations is not None else None
        objects = [output.objects[index]] if output.objects is not None else None
        data = [output.data[index]] if output.data is not None else None
        errors = [output.errors[index]] if output.errors is not None else None
        # Unit-meter counts are positional like ``entities``; slice them with
        # the item so fused cross-request batches keep each doc's real token
        # count attributed to the right work item (see EncodeHandler).
        counts = output.input_token_counts
        sliced_counts = [counts[index]] if counts is not None and 0 <= index < len(counts) else None
        # Per-item page counts are positional like token counts (see above):
        # slice them with the item so the parse/OCR page dimension (§7) stays
        # attributed to the right work item under cross-request fusion.
        pages = output.pages
        sliced_pages = [pages[index]] if pages is not None and 0 <= index < len(pages) else None
        return ExtractOutput(
            entities=[output.entities[index]],
            classifications=classifications,
            relations=relations,
            objects=objects,
            data=data,
            errors=errors,
            batch_size=1,
            input_token_counts=sliced_counts,
            pages=sliced_pages,
        )

    def assemble_output(
        self,
        partials: dict[int, ExtractOutput],
        batch_size: int,
    ) -> ExtractOutput:
        """Assemble partial outputs into full extract output.

        Args:
            partials: Dict mapping index to single-item output.
            batch_size: Total batch size.

        Returns:
            Full ExtractOutput.
        """
        if not partials:
            return ExtractOutput(entities=[], batch_size=0)

        entities = [partials[i].entities[0] for i in range(batch_size)]

        # Reassemble classifications if any partial has them
        has_classifications = any(p.classifications is not None for p in partials.values())
        classifications: list[list[Classification]] | None = None
        if has_classifications:
            classifications = []
            for i in range(batch_size):
                p_cls = partials[i].classifications
                classifications.append(p_cls[0] if p_cls is not None else [])

        # Reassemble relations if any partial has them
        has_relations = any(p.relations is not None for p in partials.values())
        relations: list[list[Relation]] | None = None
        if has_relations:
            relations = []
            for i in range(batch_size):
                p_rel = partials[i].relations
                relations.append(p_rel[0] if p_rel is not None else [])

        # Reassemble objects if any partial has them
        has_objects = any(p.objects is not None for p in partials.values())
        objects: list[list[DetectedObject]] | None = None
        if has_objects:
            objects = []
            for i in range(batch_size):
                p_obj = partials[i].objects
                objects.append(p_obj[0] if p_obj is not None else [])

        # Reassemble structured data if any partial has it
        has_data = any(p.data is not None for p in partials.values())
        data: list[dict[str, Any]] | None = None
        if has_data:
            data = []
            for i in range(batch_size):
                p_data = partials[i].data
                data.append(p_data[0] if p_data is not None else {})

        # Unlike unit counts, item errors may cover only failed positions.
        # Preserve successful siblings as None entries in the aligned list.
        has_errors = any(p.errors is not None for p in partials.values())
        errors = None
        if has_errors:
            errors = []
            for i in range(batch_size):
                partial_errors = partials[i].errors
                errors.append(partial_errors[0] if partial_errors is not None else None)

        # Reassemble per-item unit counts (see slice_output). All-or-nothing:
        # a partial without a count means the meter cannot attribute the item
        # exactly, so no counts are surfaced (metering falls back to its
        # reserve estimate rather than under-counting).
        assembled_counts: list[int] = []
        for i in range(batch_size):
            partial_counts = partials[i].input_token_counts
            if not (isinstance(partial_counts, list) and len(partial_counts) == 1):
                assembled_counts = []
                break
            assembled_counts.append(partial_counts[0])

        # Reassemble per-item page counts (same all-or-nothing contract as the
        # token counts): a partial missing its page count means the parse
        # dimension cannot be attributed exactly, so no pages are surfaced.
        assembled_pages: list[int] = []
        for i in range(batch_size):
            partial_pages = partials[i].pages
            if not (isinstance(partial_pages, list) and len(partial_pages) == 1):
                assembled_pages = []
                break
            assembled_pages.append(partial_pages[0])

        return ExtractOutput(
            entities=entities,
            classifications=classifications,
            relations=relations,
            objects=objects,
            data=data,
            errors=errors,
            batch_size=batch_size,
            input_token_counts=assembled_counts or None,
            pages=assembled_pages or None,
        )

    @classmethod
    def format_output(cls, output: ExtractOutput) -> list[dict[str, Any]]:
        """Convert ExtractOutput to per-item dicts for API response."""
        results: list[dict[str, Any]] = []
        for i, ents in enumerate(output.entities):
            item: dict[str, Any] = {"entities": list(ents)}
            if output.classifications is not None:
                item["classifications"] = list(output.classifications[i])
            else:
                item["classifications"] = []
            if output.relations is not None:
                item["relations"] = list(output.relations[i])
            else:
                item["relations"] = []
            if output.objects is not None:
                item["objects"] = list(output.objects[i])
            else:
                item["objects"] = []
            if output.data is not None:
                item["data"] = output.data[i]
            else:
                item["data"] = {}
            if output.errors is not None:
                error = output.errors[i]
                if error is not None:
                    item["error"] = {"code": error.code, "message": error.message}
            results.append(item)
        return results
