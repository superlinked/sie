"""SIE entity extraction for LanceDB table enrichment.

Provides entity extraction that integrates with LanceDB's lakehouse workflows.
Results are stored as Arrow struct arrays, enabling filtered search on
extracted entities.

The extractor supports two usage patterns:
1. Direct extraction: extract(texts, labels=...) for manual workflows
2. Table enrichment: enrich_table() for the read-transform-merge pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import lancedb
    from sie_sdk import SIEClient

# Arrow struct types for extracted results.
# These are exported so users can reference them in custom schemas.

ENTITY_STRUCT = pa.struct(
    [
        ("text", pa.utf8()),
        ("label", pa.utf8()),
        ("score", pa.float32()),
        ("start", pa.int32()),
        ("end", pa.int32()),
        ("bbox", pa.list_(pa.int32())),
    ]
)
"""Arrow struct for extracted entities.

Fields:
    text: Extracted text span or detected object label.
    label: Entity type (e.g., "person", "organization").
    score: Confidence score.
    start: Character offset start (null for image-based models).
    end: Character offset end (null for image-based models).
    bbox: Bounding box [x, y, w, h] in pixels (null for text-based models).
"""


class SIEExtractor:
    """Entity extraction with LanceDB table enrichment support.

    Wraps SIEClient.extract() to provide entity extraction that integrates
    with LanceDB's read-transform-merge enrichment pattern. Results are
    stored as Arrow struct arrays for efficient filtered search.

    Example (direct extraction):
        >>> extractor = SIEExtractor(model="urchade/gliner_multi-v2.1")
        >>> entities = extractor.extract(
        ...     ["John Smith works at Apple Inc."],
        ...     labels=["person", "organization"],
        ... )

    Example (table enrichment):
        >>> extractor = SIEExtractor(model="urchade/gliner_multi-v2.1")
        >>> extractor.enrich_table(
        ...     table,
        ...     source_column="text",
        ...     target_column="entities",
        ...     labels=["person", "organization", "location"],
        ...     id_column="id",
        ... )

    Args:
        base_url: URL of the SIE server.
        model: Entity extraction model name/ID.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        options: Runtime options dict for model adapter overrides.
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        model: str = "urchade/gliner_multi-v2.1",
        client: SIEClient | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE extractor."""
        self._base_url = base_url
        self._model = model
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._client = client

    @property
    def client(self) -> SIEClient:
        """Lazily initialize the SIE client."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    def extract(
        self,
        texts: list[str],
        *,
        labels: list[str],
        batch_size: int = 100,
    ) -> list[list[dict[str, Any]]]:
        """Extract entities from a list of texts.

        Args:
            texts: Texts to extract entities from.
            labels: Entity labels to extract (e.g., ["person", "organization"]).
            batch_size: Number of texts per SIE request.

        Returns:
            List of entity lists, one per input text. Each entity is a dict
            with keys: text, label, score, start, end, bbox.
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        all_entities: list[list[dict[str, Any]]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            items = [Item(text=t) for t in batch]
            results = self.client.extract(self._model, items, labels=labels)

            for result in results:
                entities = result.get("entities", []) if isinstance(result, dict) else getattr(result, "entities", [])
                all_entities.append(_format_entities(entities))

        return all_entities

    def enrich_table(
        self,
        table: lancedb.table.Table,
        *,
        source_column: str,
        target_column: str,
        labels: list[str],
        id_column: str,
        batch_size: int = 100,
    ) -> None:
        """Extract entities from a table column and merge results back.

        Reads the source column, runs entity extraction in batches, and
        adds results as a new Arrow struct column via table.merge().

        The target column type is ``pa.list_(ENTITY_STRUCT)`` where each
        entity has: text, label, score, start, end, bbox.

        Args:
            table: LanceDB table to enrich.
            source_column: Column containing text to extract from.
            target_column: Name for the new entities column.
            labels: Entity labels to extract (e.g., ["person", "organization"]).
            id_column: Column to join on when merging results back.
            batch_size: Number of texts per SIE request.

        Example:
            >>> extractor = SIEExtractor(model="urchade/gliner_multi-v2.1")
            >>> extractor.enrich_table(
            ...     table,
            ...     source_column="text",
            ...     target_column="entities",
            ...     labels=["person", "organization", "location"],
            ...     id_column="id",
            ... )
            >>> # Table now has an "entities" column:
            >>> table.search("Apple").where("entities.label = 'organization'").to_list()
        """
        # Stream batch-by-batch: read only needed columns, extract, merge back.
        # Scanner uses Lance's default batch size (~8K rows) to minimize merge
        # operations, while self.extract() internally batches SIE requests at
        # batch_size (default 100) which suits NER model throughput.
        scanner = table.to_lance().scanner(columns=[source_column, id_column])
        for batch in scanner.to_batches():
            texts = batch.column(source_column).to_pylist()
            entities_per_row = self.extract(texts, labels=labels, batch_size=batch_size)
            entities_array = _build_entities_array(entities_per_row)
            enrichment = pa.table(
                {
                    id_column: batch.column(id_column),
                    target_column: entities_array,
                }
            )
            table.merge(enrichment, id_column)


def _format_entities(entities: list) -> list[dict[str, Any]]:
    """Format raw entity results into normalized dicts.

    Handles both dict and object responses from the SDK.
    """
    formatted = []
    for e in entities:
        if isinstance(e, dict):
            formatted.append(
                {
                    "text": e.get("text", ""),
                    "label": e.get("label", ""),
                    "score": float(e.get("score", 0.0)),
                    "start": e.get("start"),
                    "end": e.get("end"),
                    "bbox": e.get("bbox"),
                }
            )
        else:
            formatted.append(
                {
                    "text": getattr(e, "text", ""),
                    "label": getattr(e, "label", ""),
                    "score": float(getattr(e, "score", 0.0)),
                    "start": getattr(e, "start", None),
                    "end": getattr(e, "end", None),
                    "bbox": getattr(e, "bbox", None),
                }
            )
    return formatted


def _build_entities_array(
    entities_per_row: list[list[dict[str, Any]]],
) -> pa.Array:
    """Convert entity lists to a PyArrow array of ENTITY_STRUCT lists."""
    return pa.array(entities_per_row, type=pa.list_(ENTITY_STRUCT))
