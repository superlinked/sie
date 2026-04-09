"""SIE extraction tool for LangChain.

Provides entity, relation, classification, and object extraction using SIE's extract endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field
from sie_sdk.types import Item

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from sie_sdk import SIEAsyncClient, SIEClient


class SIEExtractor(BaseTool):
    """LangChain tool for extraction using SIE.

    Wraps SIEClient.extract() to implement BaseTool for use in agents.
    Returns entities, relations, classifications, and detected objects.

    Example:
        >>> extractor = SIEExtractor(
        ...     base_url="http://localhost:8080",
        ...     model="urchade/gliner_multi-v2.1",
        ...     labels=["person", "organization", "location"],
        ... )
        >>> result = extractor.invoke("John Smith works at Acme Corp in NYC")
        >>> result["entities"]  # list of entity dicts
        >>> result["relations"]  # list of relation dicts

    Args:
        base_url: URL of the SIE server.
        model: Extraction model name/ID.
        labels: Labels to extract (entity types, relation types, or classification labels).
        client: Optional pre-configured SIEClient instance.
        async_client: Optional pre-configured SIEAsyncClient instance.
        options: Runtime options dict for model adapter overrides.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    # Pydantic v2 config
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = "sie_extract"
    description: str = (
        "Extract structured information from text. "
        "Input should be text to analyze. "
        "Returns entities, relations, classifications, and detected objects."
    )

    base_url: str = Field(default="http://localhost:8080")
    model: str = Field(default="urchade/gliner_multi-v2.1")
    labels: list[str] = Field(default_factory=lambda: ["person", "organization", "location"])
    options: dict[str, Any] | None = Field(default=None)
    gpu: str | None = Field(default=None)
    timeout_s: float = Field(default=180.0)

    # Private attributes for clients
    _client: Any = None
    _async_client: Any = None

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        model: str = "urchade/gliner_multi-v2.1",
        labels: list[str] | None = None,
        client: SIEClient | None = None,
        async_client: SIEAsyncClient | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        timeout_s: float = 180.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SIE extractor."""
        if labels is None:
            labels = ["person", "organization", "location"]
        super().__init__(
            base_url=base_url, model=model, labels=labels, options=options, gpu=gpu, timeout_s=timeout_s, **kwargs
        )
        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> SIEClient:
        """Get or create the sync SIEClient."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._client

    @property
    def async_client(self) -> SIEAsyncClient:
        """Get or create the async SIEClient."""
        if self._async_client is None:
            from sie_sdk import SIEAsyncClient

            self._async_client = SIEAsyncClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._async_client

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract structured information from text.

        Args:
            text: Text to extract from.
            run_manager: Optional callback manager (not used).

        Returns:
            Dict with entities, relations, classifications, and objects lists.
        """
        result = self.client.extract(
            self.model,
            Item(text=text),
            labels=self.labels,
        )

        return self._format_result(result)

    async def _arun(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> dict[str, list[dict[str, Any]]]:
        """Async extract structured information from text.

        Args:
            text: Text to extract from.
            run_manager: Optional callback manager (not used).

        Returns:
            Dict with entities, relations, classifications, and objects lists.
        """
        result = await self.async_client.extract(
            self.model,
            Item(text=text),
            labels=self.labels,
        )

        return self._format_result(result)

    def _format_result(self, result: object) -> dict[str, list[dict[str, Any]]]:
        """Format extraction result into multi-type dict.

        Args:
            result: ExtractResult from SIE.

        Returns:
            Dict with entities, relations, classifications, and objects.
        """

        def _get(key: str) -> list:
            if isinstance(result, dict):
                return result.get(key, [])
            return getattr(result, key, [])

        entities = []
        for e in _get("entities"):
            if isinstance(e, dict):
                entities.append(
                    {
                        "text": e.get("text", ""),
                        "label": e.get("label", ""),
                        "score": float(e.get("score", 0.0)),
                        "start": int(e.get("start", 0)),
                        "end": int(e.get("end", 0)),
                    }
                )
            else:
                entities.append(
                    {
                        "text": getattr(e, "text", ""),
                        "label": getattr(e, "label", ""),
                        "score": float(getattr(e, "score", 0.0)),
                        "start": int(getattr(e, "start", 0)),
                        "end": int(getattr(e, "end", 0)),
                    }
                )

        relations = []
        for r in _get("relations"):
            if isinstance(r, dict):
                relations.append(
                    {
                        "head": r.get("head", ""),
                        "tail": r.get("tail", ""),
                        "relation": r.get("relation", ""),
                        "score": float(r.get("score", 0.0)),
                    }
                )
            else:
                relations.append(
                    {
                        "head": getattr(r, "head", ""),
                        "tail": getattr(r, "tail", ""),
                        "relation": getattr(r, "relation", ""),
                        "score": float(getattr(r, "score", 0.0)),
                    }
                )

        classifications = []
        for c in _get("classifications"):
            if isinstance(c, dict):
                classifications.append({"label": c.get("label", ""), "score": float(c.get("score", 0.0))})
            else:
                classifications.append({"label": getattr(c, "label", ""), "score": float(getattr(c, "score", 0.0))})

        objects = []
        for o in _get("objects"):
            if isinstance(o, dict):
                objects.append(
                    {"label": o.get("label", ""), "score": float(o.get("score", 0.0)), "bbox": o.get("bbox", [])}
                )
            else:
                objects.append(
                    {
                        "label": getattr(o, "label", ""),
                        "score": float(getattr(o, "score", 0.0)),
                        "bbox": getattr(o, "bbox", []),
                    }
                )

        return {
            "entities": entities,
            "relations": relations,
            "classifications": classifications,
            "objects": objects,
        }
