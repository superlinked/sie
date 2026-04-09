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

        Items with the same (labels, instruction, options) can be batched together.

        Args:
            metadata: Request metadata.

        Returns:
            Hashable tuple for grouping.
        """
        labels_key = tuple(sorted(metadata.labels)) if metadata.labels else None
        options_key = make_hashable(metadata.options) if metadata.options else None
        return (
            labels_key,
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
        labels_tuple, instruction, options_tuple = config_key
        labels = list(labels_tuple) if labels_tuple else None
        options = dict(options_tuple) if options_tuple else None

        return adapter.extract(
            items,
            labels=labels,
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
        return ExtractOutput(
            entities=[output.entities[index]],
            classifications=classifications,
            relations=relations,
            objects=objects,
            batch_size=1,
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

        return ExtractOutput(
            entities=entities,
            classifications=classifications,
            relations=relations,
            objects=objects,
            batch_size=batch_size,
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
            results.append(item)
        return results
