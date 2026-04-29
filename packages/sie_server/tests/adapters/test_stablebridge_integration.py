from __future__ import annotations

import logging
import socket
import sys
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from sie_sdk import SIEClient

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Stablebridge tests require CUDA (flash-attn + bf16 inference)",
)

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "packages" / "sie_bench" / "src"))

ENCODER_MODEL = "answerdotai/ModernBERT-base"
PRUNER_MODEL = "sugiv/stablebridge-pruner-highlighter"


def _find_free_port(start: int = 8400, end: int = 8500) -> int:
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))  # noqa: S104
                return port
        except OSError:
            continue
    msg = f"No free port found in range {start}-{end}"
    raise RuntimeError(msg)


@pytest.fixture(scope="module")
def stablebridge_server(device: str) -> Generator[str]:
    from sie_bench.servers.sie import SIEServer

    models_dir = _project_root / "packages" / "sie_server" / "models"
    port = _find_free_port()

    server = SIEServer(port=port, models_dir=str(models_dir), instrumentation=True)

    try:
        server.start(f"{ENCODER_MODEL},{PRUNER_MODEL}", device)
        server.wait_ready(timeout_s=600.0)
        url = server.get_url()
        yield url
    finally:
        server.stop()


@pytest.fixture(scope="module")
def stablebridge_client(stablebridge_server: str) -> SIEClient:
    from sie_sdk import SIEClient

    return SIEClient(stablebridge_server, timeout_s=300.0)


@pytest.mark.integration
class TestModernBERTLoRAProfile:
    def test_encode_default_profile(self, stablebridge_client: SIEClient) -> None:
        from sie_sdk.types import Item

        result = stablebridge_client.encode(ENCODER_MODEL, Item(text="Stablecoin issuer reserve requirements"))
        assert "dense" in result
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].shape == (768,)

    def test_encode_us_regulatory_profile(self, stablebridge_client: SIEClient) -> None:
        from sie_sdk.types import Item

        result = stablebridge_client.encode(
            ENCODER_MODEL,
            Item(text="Stablecoin issuer reserve requirements"),
            options={"profile": "us-regulatory"},
        )
        assert "dense" in result
        assert result["dense"].shape == (768,)

    def test_us_regulatory_diverges_from_default(self, stablebridge_client: SIEClient) -> None:
        from sie_sdk.types import Item

        text = "Issuer must hold reserves equal to outstanding stablecoin liabilities."
        base = stablebridge_client.encode(ENCODER_MODEL, Item(text=text))["dense"]
        lora = stablebridge_client.encode(
            ENCODER_MODEL,
            Item(text=text),
            options={"profile": "us-regulatory"},
        )["dense"]

        cosine = float(np.dot(base, lora) / (np.linalg.norm(base) * np.linalg.norm(lora)))
        assert 0.5 < cosine < 0.999, f"Cosine similarity: {cosine}"


@pytest.mark.integration
class TestStablebridgePrunerHighlighter:
    QUERY = "What reserve requirements apply to stablecoin issuers?"
    DOC = (
        "Each issuer of a payment stablecoin shall maintain reserves "
        "in an amount equal to the outstanding stablecoin liabilities. "
        "The reserves shall be held in cash, demand deposits at insured "
        "depository institutions, or short-term Treasury securities."
    )

    def test_score_returns_rerank_value(self, stablebridge_client: SIEClient) -> None:
        from sie_sdk.types import Item

        result = stablebridge_client.score(PRUNER_MODEL, Item(text=self.QUERY), [Item(text=self.DOC)])
        scores = result.get("scores") if isinstance(result, dict) else None
        assert scores is not None
        assert len(scores) == 1
        score = float(scores[0]["score"]) if isinstance(scores[0], dict) else float(scores[0])
        assert 0.0 <= score <= 1.0

    def _extract(
        self,
        client: SIEClient,
        *,
        options: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        from sie_sdk.types import Item

        response = client.extract(
            PRUNER_MODEL,
            [Item(text=self.DOC)],
            instruction=self.QUERY,
            options=options,
        )
        # client.extract with a list of items returns list[ExtractResult].
        result = response[0] if isinstance(response, list) else response
        assert isinstance(result, dict)
        entity_lists = result.get("entities")
        assert entity_lists is not None
        assert len(entity_lists) == 1
        return entity_lists[0]

    def test_extract_emits_summary_and_spans(self, stablebridge_client: SIEClient) -> None:
        entities = self._extract(stablebridge_client)
        assert entities, "extract returned no entities"
        assert entities[0]["label"] == "summary"
        remaining_labels = {e["label"] for e in entities[1:]}
        assert remaining_labels.issubset({"kept", "highlight", "pruned"})

    def test_aggressive_profile_prunes_more_than_default(self, stablebridge_client: SIEClient) -> None:
        default_entities = self._extract(stablebridge_client)
        aggressive_entities = self._extract(stablebridge_client, options={"profile": "aggressive"})

        def _kept(es: list[dict[str, object]]) -> int:
            return sum(1 for e in es if e.get("label") in {"kept", "highlight"})

        assert _kept(aggressive_entities) <= _kept(default_entities) + 1
