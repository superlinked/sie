"""Integration tests for LoRA adapter support.

Tests validate that LoRA profiles work correctly through the full server stack:
- Profile-based LoRA selection
- Interleaved requests with different LoRAs
- Base model + LoRA coexistence

Mark: integration (run with `mise run test -i packages/sie_server/tests/test_lora_integration.py`)
"""

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

# Skip entire module if CUDA is not available (flash-attn required for BAAI/bge-m3)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LoRA tests require CUDA (BAAI/bge-m3 uses flash-attn)",
)

# Add sie_bench to path for server management
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "packages" / "sie_bench" / "src"))

# LoRA paths matching the profiles in baai-bge-m3.yaml
BANKING_LORA = "saivamshiatukuri/bge-m3-banking77-lora"
MEDICAL_LORA = "doanbao/bge-m3-medical-vn-lora"


def _find_free_port(start: int = 8200, end: int = 8300) -> int:
    """Find an available port in the given range."""
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
def lora_server(device: str) -> Generator[str]:
    """Start a SIE server with BGE-M3 for LoRA testing.

    The server loads model profiles including banking and medical-vn LoRAs.
    Skipped when cuda is not available (BAAI/bge-m3 requires flash-attn which needs CUDA).
    """
    from sie_bench.servers.sie import SIEServer

    models_dir = _project_root / "packages" / "sie_server" / "models"

    # Use dynamic port to avoid conflicts with other test modules
    port = _find_free_port(8200, 8300)

    server = SIEServer(
        port=port,
        models_dir=str(models_dir),
        instrumentation=True,
    )

    model = "BAAI/bge-m3"

    try:
        server.start(model, device)
        server.wait_ready(timeout_s=300.0)  # LoRA downloads may take time
        url = server.get_url()
        logger.info("LoRA test server ready at %s", url)
        yield url
    finally:
        server.stop()
        logger.info("LoRA test server stopped")


@pytest.fixture(scope="module")
def lora_client(lora_server: str) -> SIEClient:
    """Create an SIEClient for LoRA testing."""
    from sie_sdk import SIEClient

    return SIEClient(lora_server, timeout_s=120.0)


@pytest.mark.integration
class TestLoRAProfileIntegration:
    """Integration tests for LoRA profile-based selection."""

    def test_encode_with_default_profile(self, lora_client: SIEClient) -> None:
        """Can encode with default profile (no LoRA)."""
        from sie_sdk.types import Item

        result = lora_client.encode(
            "BAAI/bge-m3",
            Item(text="Transfer money to savings account"),
        )

        assert "dense" in result
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].shape == (1024,)

    def test_encode_with_banking_profile(self, lora_client: SIEClient) -> None:
        """Can encode with banking LoRA profile."""
        from sie_sdk.types import Item

        result = lora_client.encode(
            "BAAI/bge-m3",
            Item(text="Transfer money to savings account"),
            options={"profile": "banking"},
        )

        assert "dense" in result
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].shape == (1024,)

    def test_encode_with_medical_profile(self, lora_client: SIEClient) -> None:
        """Can encode with medical-vn LoRA profile."""
        from sie_sdk.types import Item

        result = lora_client.encode(
            "BAAI/bge-m3",
            Item(text="triệu chứng đau đầu"),  # Vietnamese: headache symptoms
            options={"profile": "medical-vn"},
        )

        assert "dense" in result
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].shape == (1024,)

    def test_lora_produces_different_embeddings(self, lora_client: SIEClient) -> None:
        """LoRA embeddings differ from base model embeddings."""
        from sie_sdk.types import Item

        text = "Transfer money to savings account"

        # Encode with base model (default profile)
        base_result = lora_client.encode("BAAI/bge-m3", Item(text=text))

        # Encode with banking LoRA
        lora_result = lora_client.encode(
            "BAAI/bge-m3",
            Item(text=text),
            options={"profile": "banking"},
        )

        base_vec = base_result["dense"]
        lora_vec = lora_result["dense"]

        # Cosine similarity - should be similar but not identical
        cosine_sim = np.dot(base_vec, lora_vec) / (np.linalg.norm(base_vec) * np.linalg.norm(lora_vec))

        # Should be similar (same semantic meaning) but not identical
        # LoRA fine-tuning typically produces 0.85-0.99 similarity
        assert 0.7 < cosine_sim < 0.999, f"Cosine similarity: {cosine_sim}"


@pytest.mark.integration
class TestLoRAInterleavedRequests:
    """Integration tests for interleaved LoRA requests."""

    def test_interleaved_profile_switching(self, lora_client: SIEClient) -> None:
        """Can switch between profiles (LoRAs) repeatedly."""
        from sie_sdk.types import Item

        # Use same text for consistency checks
        test_text = "Transfer money to savings account"

        results_default = []
        results_banking = []

        # Interleave encoding 3 times with SAME text
        for _ in range(3):
            # Default profile (no LoRA)
            result = lora_client.encode(
                "BAAI/bge-m3",
                Item(text=test_text),
            )
            results_default.append(result["dense"])

            # Banking profile (with LoRA)
            result = lora_client.encode(
                "BAAI/bge-m3",
                Item(text=test_text),
                options={"profile": "banking"},
            )
            results_banking.append(result["dense"])

        # All default results for same text should be identical
        # (same input produces same output - model is deterministic)
        for i in range(1, len(results_default)):
            np.testing.assert_array_almost_equal(
                results_default[0],
                results_default[i],
                decimal=5,
            )

        # All banking results for same text should be identical
        for i in range(1, len(results_banking)):
            np.testing.assert_array_almost_equal(
                results_banking[0],
                results_banking[i],
                decimal=5,
            )

    def test_concurrent_different_profiles(self, lora_client: SIEClient) -> None:
        """Concurrent requests with different profiles return correct embeddings."""
        from concurrent.futures import ThreadPoolExecutor

        from sie_sdk.types import Item

        def encode_with_profile(profile: str | None, idx: int) -> tuple[str | None, np.ndarray]:
            options = {"profile": profile} if profile else None
            result = lora_client.encode(
                "BAAI/bge-m3",
                Item(text=f"Test item {idx}"),
                options=options,
            )
            return (profile, result["dense"])

        # Run concurrent requests with different profiles
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for i in range(3):
                futures.append(executor.submit(encode_with_profile, None, i))
                futures.append(executor.submit(encode_with_profile, "banking", i))

            results = [f.result(timeout=60) for f in futures]

        # Group by profile
        default_results = [r[1] for r in results if r[0] is None]
        banking_results = [r[1] for r in results if r[0] == "banking"]

        assert len(default_results) == 3
        assert len(banking_results) == 3

        # All default results should be similar (same text pattern)
        # All banking results should be similar to each other
        # But default and banking should differ
        default_avg = np.mean(default_results, axis=0)
        banking_avg = np.mean(banking_results, axis=0)

        cosine_sim = np.dot(default_avg, banking_avg) / (np.linalg.norm(default_avg) * np.linalg.norm(banking_avg))
        # Profiles should produce noticeably different embeddings
        assert cosine_sim < 0.999, f"Profiles too similar: {cosine_sim}"


@pytest.mark.integration
class TestLoRABatchIsolation:
    """Tests that per-LoRA batching prevents cross-contamination."""

    def test_batch_isolation_with_ids(self, lora_client: SIEClient) -> None:
        """Batched items maintain correct ID mapping across profiles."""
        from sie_sdk.types import Item

        # Send batch with default profile
        default_items = [
            Item(id="default-1", text="First default item"),
            Item(id="default-2", text="Second default item"),
        ]
        default_results = lora_client.encode("BAAI/bge-m3", default_items)

        # Send batch with banking profile
        banking_items = [
            Item(id="banking-1", text="First banking item"),
            Item(id="banking-2", text="Second banking item"),
        ]
        banking_results = lora_client.encode(
            "BAAI/bge-m3",
            banking_items,
            options={"profile": "banking"},
        )

        # Verify IDs are preserved correctly
        assert len(default_results) == 2
        assert len(banking_results) == 2

        assert default_results[0].get("id") == "default-1"
        assert default_results[1].get("id") == "default-2"
        assert banking_results[0].get("id") == "banking-1"
        assert banking_results[1].get("id") == "banking-2"
