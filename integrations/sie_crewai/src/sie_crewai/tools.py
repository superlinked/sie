"""SIE tools for CrewAI agents.

Provides tools for reranking and entity extraction using SIE.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sie_sdk import RequestError


def _raise_for_item_error(result: Any) -> None:
    """Raise when SIE reports that extraction failed for this item."""
    error = result.get("error") if isinstance(result, Mapping) else None
    if isinstance(error, Mapping):
        msg = f"Extraction failed: {error.get('message')}"
        raise RequestError(msg, code=error.get("code"), request=result.get("request"))


def _scores_by_index(results: Mapping[str, Any], count: int) -> list[float]:
    """Map ScoreResult entries back to input positions by item_id.

    Parses each entry's ``item_id`` defensively: a missing, non-integer,
    negative, or out-of-range id is skipped (that document keeps its 0.0
    default) so a malformed entry can neither crash the rerank nor mis-assign a
    score to the wrong document.

    Args:
        results: ScoreResult envelope from ``SIEClient.score()``.
        count: Number of input documents.

    Returns:
        Scores indexed by input position (0.0 for any unscored/invalid item).
    """
    scores = [0.0] * count
    for entry in results.get("scores", []):
        item_id = entry.get("item_id", entry.get("index"))
        # Accept only a genuine integer or an integer string. Reject bool (an int
        # subclass, int(True) == 1), float (int(1.5) == 1), and non-integer
        # strings, so a malformed id cannot silently overwrite the wrong position.
        if isinstance(item_id, bool):
            continue
        if isinstance(item_id, int):
            idx = item_id
        elif isinstance(item_id, str):
            try:
                idx = int(item_id)
            except ValueError:
                continue
        else:
            continue
        if 0 <= idx < count:
            scores[idx] = float(entry.get("score", 0.0))
    return scores


class RerankerInput(BaseModel):
    """Input schema for SIERerankerTool."""

    query: str = Field(..., description="The query to rank documents against.")
    documents: list[str] = Field(..., description="List of document texts to rerank.")
    top_k: int | None = Field(default=None, description="Number of top documents to return.")


class SIERerankerTool(BaseTool):
    """Rerank documents by relevance to a query using SIE.

    This tool uses a cross-encoder model to score each document against
    the query and returns them sorted by relevance.

    Example:
        >>> reranker = SIERerankerTool(
        ...     base_url="http://localhost:8080",
        ...     model="jinaai/jina-reranker-v2-base-multilingual",
        ... )
        >>> result = reranker._run(
        ...     query="What is machine learning?",
        ...     documents=["ML uses statistical models.", "The weather is sunny."],
        ...     top_k=1,
        ... )
    """

    name: str = "sie_reranker"
    description: str = (
        "Rerank a list of documents by their relevance to a query. "
        "Use this when you have multiple documents and need to find the most relevant ones. "
        "Input: query string and list of document texts. Output: ranked documents with scores."
    )
    args_schema: type[BaseModel] = RerankerInput

    # SIE configuration
    base_url: str = Field(default="http://localhost:8080", description="SIE server URL")
    model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual",
        description="Model name for reranking",
    )
    gpu: str | None = Field(default=None, description="GPU type for routing")
    options: dict[str, Any] | None = Field(default=None, description="Model-specific options")
    timeout_s: float = Field(default=180.0, description="Request timeout in seconds")

    _client: Any = None

    @property
    def client(self) -> Any:
        """Lazily initialize the SIE client."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._client

    def _run(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> str:
        """Rerank documents by relevance to the query.

        Args:
            query: The query to rank against.
            documents: List of document texts to rerank.
            top_k: Number of top documents to return.

        Returns:
            Formatted string with ranked documents and scores.
        """
        if not documents:
            return "No documents provided to rerank."

        from sie_sdk.types import Item

        query_item = Item(text=query)
        doc_items = [Item(text=doc) for doc in documents]

        results = self.client.score(self.model, query_item, doc_items)

        # ``results`` is a ScoreResult envelope; ranked entries are under
        # ``results["scores"]`` (each a ScoreEntry keyed by ``item_id`` = input
        # position, plus ``score``). Map scores back to input order by item_id
        # rather than zipping positionally — the entries are sorted by relevance,
        # not by input order. ``results["request"]`` (request id) and
        # ``results["usage"]`` (token usage) are also available but intentionally
        # not surfaced here.
        score_by_index = _scores_by_index(results, len(documents))

        # Build scored documents
        scored = [(score_by_index[idx], doc) for idx, doc in enumerate(documents)]

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k
        if top_k is not None:
            scored = scored[:top_k]

        # Format output
        lines = ["Ranked documents (most relevant first):"]
        max_preview_len = 200
        for i, (score, doc) in enumerate(scored, 1):
            # Truncate long documents for readability
            doc_preview = doc[:max_preview_len] + "..." if len(doc) > max_preview_len else doc
            lines.append(f"{i}. [Score: {score:.4f}] {doc_preview}")

        return "\n".join(lines)


class ExtractorInput(BaseModel):
    """Input schema for SIEExtractorTool."""

    text: str = Field(..., description="The text to extract entities from.")
    labels: list[str] | None = Field(
        default=None,
        description="Entity types to extract (e.g., ['person', 'organization']). Uses default labels if not provided.",
    )


class SIEExtractorTool(BaseTool):
    """Extract entities from text using SIE.

    This tool uses a GLiNER or similar model to identify and extract
    named entities from text.

    Example:
        >>> extractor = SIEExtractorTool(
        ...     base_url="http://localhost:8080",
        ...     model="urchade/gliner_multi-v2.1",
        ...     labels=["person", "organization", "location"],
        ... )
        >>> result = extractor._run(text="John Smith works at Google in New York.")
    """

    name: str = "sie_extractor"
    description: str = (
        "Extract named entities from text. "
        "Identifies people, organizations, locations, and other entity types. "
        "Input: text to analyze. Output: list of extracted entities with their types and positions."
    )
    args_schema: type[BaseModel] = ExtractorInput

    # SIE configuration
    base_url: str = Field(default="http://localhost:8080", description="SIE server URL")
    model: str = Field(
        default="urchade/gliner_multi-v2.1",
        description="Model name for extraction",
    )
    labels: list[str] = Field(
        default=["person", "organization", "location"],
        description="Default entity labels to extract",
    )
    gpu: str | None = Field(default=None, description="GPU type for routing")
    options: dict[str, Any] | None = Field(default=None, description="Model-specific options")
    timeout_s: float = Field(default=180.0, description="Request timeout in seconds")

    _client: Any = None

    @property
    def client(self) -> Any:
        """Lazily initialize the SIE client."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._client

    def _run(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> str:
        """Extract structured information from text.

        Args:
            text: The text to extract from.
            labels: Labels to extract (overrides default labels).

        Returns:
            Formatted string with extracted entities, relations, classifications, and objects.
        """
        from sie_sdk.types import Item

        effective_labels = labels if labels is not None else self.labels

        result = self.client.extract(
            self.model,
            Item(text=text),
            labels=effective_labels,
        )
        _raise_for_item_error(result)

        entities = self._parse_items(result, "entities")
        relations = self._parse_items(result, "relations")
        classifications = self._parse_items(result, "classifications")
        objects = self._parse_items(result, "objects")

        lines: list[str] = []

        if entities:
            lines.append("Extracted entities:")
            lines.extend(
                f"- {e.get('text', '')} ({e.get('label', '')}) [confidence: {e.get('score', 0.0):.2f}]"
                for e in entities
            )

        if relations:
            lines.append("Extracted relations:")
            lines.extend(
                f"- {r.get('head', '')} --[{r.get('relation', '')}]--> {r.get('tail', '')} "
                f"[confidence: {r.get('score', 0.0):.2f}]"
                for r in relations
            )

        if classifications:
            lines.append("Classifications:")
            lines.extend(f"- {c.get('label', '')} [confidence: {c.get('score', 0.0):.2f}]" for c in classifications)

        if objects:
            lines.append("Detected objects:")
            lines.extend(
                f"- {o.get('label', '')} at {o.get('bbox', [])} [confidence: {o.get('score', 0.0):.2f}]"
                for o in objects
            )

        if not lines:
            return f"No extraction results found for labels: {effective_labels}"

        return "\n".join(lines)

    def _parse_items(self, result: Any, field: str) -> list[dict[str, Any]]:
        """Parse a field from SDK result into list of dicts."""
        if isinstance(result, dict):
            items = result.get(field, [])
        else:
            items = getattr(result, field, [])

        parsed = []
        for item in items:
            if isinstance(item, dict):
                parsed.append(item)
            else:
                parsed.append(vars(item) if hasattr(item, "__dict__") else {"value": str(item)})
        return parsed
