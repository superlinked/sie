"""SIE tools for CrewAI agents.

Provides tools for reranking and entity extraction using SIE.
"""

from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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

        # Build scored documents
        scored = []
        for doc, result in zip(documents, results, strict=True):
            score = result.get("score", 0.0) if isinstance(result, dict) else getattr(result, "score", 0.0)
            scored.append((float(score), doc))

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
        """Extract entities from text.

        Args:
            text: The text to extract entities from.
            labels: Entity types to extract (overrides default labels).

        Returns:
            Formatted string with extracted entities.
        """
        from sie_sdk.types import Item

        effective_labels = labels if labels is not None else self.labels

        result = self.client.extract(
            self.model,
            Item(text=text),
            labels=effective_labels,
        )

        # Parse entities from result
        entities = self._parse_entities(result)

        if not entities:
            return f"No entities found in the text for labels: {effective_labels}"

        # Format output
        lines = ["Extracted entities:"]
        lines.extend(
            f"- {entity['text']} ({entity['label']}) [confidence: {entity['score']:.2f}]" for entity in entities
        )

        return "\n".join(lines)

    def _parse_entities(self, result: Any) -> list[dict[str, Any]]:
        """Parse entities from SDK result."""
        entities = []

        # Result could be a dict with 'entities' key or a list directly
        if isinstance(result, dict):
            items = result.get("entities", [])
        elif isinstance(result, list):
            items = result
        else:
            items = []

        for item in items:
            if isinstance(item, dict):
                entities.append(
                    {
                        "text": item.get("text", ""),
                        "label": item.get("label", ""),
                        "score": float(item.get("score", 0.0)),
                        "start": item.get("start"),
                        "end": item.get("end"),
                    }
                )
            else:
                entities.append(
                    {
                        "text": getattr(item, "text", ""),
                        "label": getattr(item, "label", ""),
                        "score": float(getattr(item, "score", 0.0)),
                        "start": getattr(item, "start", None),
                        "end": getattr(item, "end", None),
                    }
                )

        return entities
