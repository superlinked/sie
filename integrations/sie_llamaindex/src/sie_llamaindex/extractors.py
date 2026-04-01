"""SIE entity extraction tool for LlamaIndex.

Provides NER extraction using SIE's extract endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.tools import FunctionTool

if TYPE_CHECKING:
    from sie_sdk import SIEClient


def create_sie_extractor_tool(
    base_url: str = "http://localhost:8080",
    model: str = "urchade/gliner_multi-v2.1",
    labels: list[str] | None = None,
    options: dict[str, Any] | None = None,
    gpu: str | None = None,
    timeout_s: float = 180.0,
    name: str = "sie_extract_entities",
    description: str | None = None,
) -> FunctionTool:
    """Create a LlamaIndex FunctionTool for entity extraction.

    Creates a tool that wraps SIE's extract endpoint for use with
    LlamaIndex agents and workflows.

    Example:
        >>> from llama_index.core.agent import ReActAgent
        >>> from sie_llamaindex import create_sie_extractor_tool
        >>>
        >>> extractor = create_sie_extractor_tool(
        ...     base_url="http://localhost:8080",
        ...     model="urchade/gliner_multi-v2.1",
        ...     labels=["person", "organization", "location"],
        ... )
        >>>
        >>> agent = ReActAgent.from_tools([extractor], llm=llm)
        >>> response = agent.chat("Extract entities from: John works at Google")

    Args:
        base_url: URL of the SIE server.
        model: Entity extraction model name/ID.
        labels: Entity labels to extract.
        options: Runtime options dict for model adapter overrides.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
        name: Tool name for the agent.
        description: Tool description for the agent.

    Returns:
        FunctionTool wrapping SIE entity extraction.
    """
    if labels is None:
        labels = ["person", "organization", "location"]

    if description is None:
        description = (
            "Extract named entities from text. "
            f"Finds entities of types: {', '.join(labels)}. "
            "Returns a list of entities with their labels, positions, and confidence scores."
        )

    # Create extractor instance to hold state
    extractor = _SIEExtractor(
        base_url=base_url,
        model=model,
        labels=labels,
        options=options,
        gpu=gpu,
        timeout_s=timeout_s,
    )

    def extract_entities(text: str) -> list[dict[str, Any]]:
        """Extract named entities from text.

        Args:
            text: Text to extract entities from.

        Returns:
            List of extracted entities with text, label, score, start, end.
        """
        return extractor.extract(text)

    return FunctionTool.from_defaults(
        fn=extract_entities,
        name=name,
        description=description,
    )


class _SIEExtractor:
    """Internal class to hold SIE extractor state."""

    def __init__(
        self,
        base_url: str,
        model: str,
        labels: list[str],
        options: dict[str, Any] | None,
        gpu: str | None,
        timeout_s: float,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._labels = labels
        self._options = options
        self._gpu = gpu
        self._timeout_s = timeout_s
        self._client: SIEClient | None = None

    @property
    def client(self) -> SIEClient:
        """Get or create the sync SIEClient."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from.

        Returns:
            List of extracted entities.
        """
        from sie_sdk.types import Item

        result = self.client.extract(
            self._model,
            Item(text=text),
            labels=self._labels,
        )

        return self._format_entities(result)

    def _format_entities(self, result: object) -> list[dict[str, Any]]:
        """Format extraction result into entity list.

        Args:
            result: ExtractResult from SIE.

        Returns:
            List of entity dictionaries.
        """
        # Handle dict or object result
        entities = result.get("entities", []) if isinstance(result, dict) else getattr(result, "entities", [])

        formatted = []
        for entity in entities:
            if isinstance(entity, dict):
                formatted.append(
                    {
                        "text": entity.get("text", ""),
                        "label": entity.get("label", ""),
                        "score": float(entity.get("score", 0.0)),
                        "start": int(entity.get("start", 0)),
                        "end": int(entity.get("end", 0)),
                    }
                )
            else:
                formatted.append(
                    {
                        "text": getattr(entity, "text", ""),
                        "label": getattr(entity, "label", ""),
                        "score": float(getattr(entity, "score", 0.0)),
                        "start": int(getattr(entity, "start", 0)),
                        "end": int(getattr(entity, "end", 0)),
                    }
                )

        return formatted


# Alias for consistency with other integrations
SIEExtractorTool = create_sie_extractor_tool
