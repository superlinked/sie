"""SIE entity extraction tool for LangChain.

Provides NER extraction using SIE's extract endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from sie_sdk import SIEAsyncClient, SIEClient


class SIEExtractor(BaseTool):
    """LangChain tool for entity extraction using SIE.

    Wraps SIEClient.extract() to implement BaseTool for use in agents.

    Example:
        >>> extractor = SIEExtractor(
        ...     base_url="http://localhost:8080",
        ...     model="urchade/gliner_multi-v2.1",
        ...     labels=["person", "organization", "location"],
        ... )
        >>> entities = extractor.invoke("John Smith works at Acme Corp in NYC")

    Args:
        base_url: URL of the SIE server.
        model: Entity extraction model name/ID.
        labels: Entity labels to extract.
        client: Optional pre-configured SIEClient instance.
        async_client: Optional pre-configured SIEAsyncClient instance.
        options: Runtime options dict for model adapter overrides.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    # Pydantic v2 config
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = "sie_extract_entities"
    description: str = (
        "Extract named entities from text. "
        "Input should be text to analyze. "
        "Returns a list of entities with their labels and positions."
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
    ) -> list[dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from.
            run_manager: Optional callback manager (not used).

        Returns:
            List of extracted entities with text, label, score, start, end.
        """
        from sie_sdk.types import Item

        result = self.client.extract(
            self.model,
            Item(text=text),
            labels=self.labels,
        )

        return self._format_entities(result)

    async def _arun(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Async extract entities from text.

        Args:
            text: Text to extract entities from.
            run_manager: Optional callback manager (not used).

        Returns:
            List of extracted entities with text, label, score, start, end.
        """
        from sie_sdk.types import Item

        result = await self.async_client.extract(
            self.model,
            Item(text=text),
            labels=self.labels,
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
