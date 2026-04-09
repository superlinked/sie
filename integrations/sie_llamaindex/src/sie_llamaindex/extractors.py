"""SIE extraction tool for LlamaIndex.

Provides entity, relation, classification, and object extraction using SIE's extract endpoint.
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
    name: str = "sie_extract",
    description: str | None = None,
) -> FunctionTool:
    """Create a LlamaIndex FunctionTool for extraction.

    Creates a tool that wraps SIE's extract endpoint for use with
    LlamaIndex agents and workflows. Returns entities, relations,
    classifications, and detected objects.

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
        model: Extraction model name/ID.
        labels: Labels to extract.
        options: Runtime options dict for model adapter overrides.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
        name: Tool name for the agent.
        description: Tool description for the agent.

    Returns:
        FunctionTool wrapping SIE extraction.
    """
    if labels is None:
        labels = ["person", "organization", "location"]

    if description is None:
        description = (
            "Extract structured information from text. "
            f"Finds entities of types: {', '.join(labels)}. "
            "Returns entities, relations, classifications, and detected objects."
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

    def extract(text: str) -> dict[str, list[dict[str, Any]]]:
        """Extract structured information from text.

        Args:
            text: Text to extract from.

        Returns:
            Dict with entities, relations, classifications, and objects.
        """
        return extractor.extract(text)

    return FunctionTool.from_defaults(
        fn=extract,
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

    def extract(self, text: str) -> dict[str, list[dict[str, Any]]]:
        """Extract structured information from text.

        Args:
            text: Text to extract from.

        Returns:
            Dict with entities, relations, classifications, and objects.
        """
        from sie_sdk.types import Item

        result = self.client.extract(
            self._model,
            Item(text=text),
            labels=self._labels,
        )

        return self._format_result(result)

    def _format_result(self, result: object) -> dict[str, list[dict[str, Any]]]:
        """Format extraction result into multi-type dict."""

        def _get(key: str) -> list:
            if isinstance(result, dict):
                return result.get(key, [])
            return getattr(result, key, [])

        def _entity(e: Any) -> dict:
            if isinstance(e, dict):
                return {
                    "text": e.get("text", ""),
                    "label": e.get("label", ""),
                    "score": float(e.get("score", 0.0)),
                    "start": int(e.get("start", 0)),
                    "end": int(e.get("end", 0)),
                }
            return {
                "text": getattr(e, "text", ""),
                "label": getattr(e, "label", ""),
                "score": float(getattr(e, "score", 0.0)),
                "start": int(getattr(e, "start", 0)),
                "end": int(getattr(e, "end", 0)),
            }

        def _relation(r: Any) -> dict:
            if isinstance(r, dict):
                return {
                    "head": r.get("head", ""),
                    "tail": r.get("tail", ""),
                    "relation": r.get("relation", ""),
                    "score": float(r.get("score", 0.0)),
                }
            return {
                "head": getattr(r, "head", ""),
                "tail": getattr(r, "tail", ""),
                "relation": getattr(r, "relation", ""),
                "score": float(getattr(r, "score", 0.0)),
            }

        def _classification(c: Any) -> dict:
            if isinstance(c, dict):
                return {"label": c.get("label", ""), "score": float(c.get("score", 0.0))}
            return {"label": getattr(c, "label", ""), "score": float(getattr(c, "score", 0.0))}

        def _object(o: Any) -> dict:
            if isinstance(o, dict):
                return {"label": o.get("label", ""), "score": float(o.get("score", 0.0)), "bbox": o.get("bbox", [])}
            return {
                "label": getattr(o, "label", ""),
                "score": float(getattr(o, "score", 0.0)),
                "bbox": getattr(o, "bbox", []),
            }

        return {
            "entities": [_entity(e) for e in _get("entities")],
            "relations": [_relation(r) for r in _get("relations")],
            "classifications": [_classification(c) for c in _get("classifications")],
            "objects": [_object(o) for o in _get("objects")],
        }


# Alias for consistency with other integrations
SIEExtractorTool = create_sie_extractor_tool
