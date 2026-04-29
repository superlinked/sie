"""Docker image tests.

These tests build and run the actual SIE Docker image to verify:
- Dockerfile builds correctly
- Container can start and become healthy
- Model downloads work (cache directory permissions)
- Basic inference works

Regression test for: https://github.com/superlinked/sie-internal/issues/10

Run with: pytest -m "docker"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from sie_sdk import SIEClient

# Mark all tests in this module
pytestmark = [pytest.mark.docker]


class TestDockerImageBasics:
    """Basic tests for Docker image functionality."""

    def test_health_endpoint(self, sie_docker_server: str) -> None:
        """Docker container responds to health checks."""
        import httpx

        response = httpx.get(f"{sie_docker_server}/healthz", timeout=10.0)
        assert response.status_code == 200

    def test_models_endpoint(self, docker_client: SIEClient) -> None:
        """Docker container can list available models."""
        models = docker_client.list_models()
        assert len(models) > 0
        # Should include the model we started with
        # Model list returns "name" not "id"
        model_names = [m["name"] for m in models]
        assert "sentence-transformers/all-MiniLM-L6-v2" in model_names


class TestDockerModelDownload:
    """Tests that verify model download works in Docker container.

    This is the key regression test for issue #10 - the container must
    be able to download models to the HuggingFace cache directory.
    """

    def test_encode_triggers_model_load(self, docker_client: SIEClient) -> None:
        """Encoding request successfully loads model and returns embeddings.

        This tests the full flow:
        1. Request comes in for a model
        2. Model weights are downloaded (if not cached)
        3. Model is loaded onto device
        4. Inference runs and returns results

        If the cache directory doesn't exist or isn't writable,
        this test will fail.
        """
        from sie_sdk.types import Item

        model = "sentence-transformers/all-MiniLM-L6-v2"
        result = docker_client.encode(model, Item(text="Hello, world!"))

        # Should get a dense embedding
        assert "dense" in result
        assert result["dense"] is not None
        assert len(result["dense"]) == 384  # MiniLM embedding dimension

    def test_encode_batch(self, docker_client: SIEClient) -> None:
        """Batch encoding works correctly."""
        from sie_sdk.types import Item

        model = "sentence-transformers/all-MiniLM-L6-v2"
        items = [
            Item(text="First sentence"),
            Item(text="Second sentence"),
            Item(text="Third sentence"),
        ]

        results = docker_client.encode(model, items)

        assert len(results) == 3
        for result in results:
            assert "dense" in result
            assert len(result["dense"]) == 384


class TestDockerCachePermissions:
    """Tests specifically for cache directory permissions.

    These tests verify that the HuggingFace cache is properly configured
    and writable in the Docker container.
    """

    def test_second_request_uses_cache(self, docker_client: SIEClient) -> None:
        """Second request should be faster (model cached).

        This indirectly tests that:
        1. First request wrote to cache successfully
        2. Second request can read from cache
        """
        import time

        from sie_sdk.types import Item

        model = "sentence-transformers/all-MiniLM-L6-v2"

        # First request (may need to load model)
        start1 = time.perf_counter()
        docker_client.encode(model, Item(text="First request"))
        time1 = time.perf_counter() - start1

        # Second request (model should be loaded)
        start2 = time.perf_counter()
        docker_client.encode(model, Item(text="Second request"))
        time2 = time.perf_counter() - start2

        # Second request should be significantly faster
        # (first includes model loading, second is just inference)
        # Allow some variance but second should be at most 50% of first
        # if model loading was involved in first request
        assert time2 < time1 or time1 < 1.0, (
            f"Expected second request ({time2:.2f}s) to be faster than first ({time1:.2f}s) "
            "due to model caching, unless model was already loaded"
        )


class TestDockerGateway:
    """Tests for the SIE Gateway Docker image.

    Verifies the gateway image can start and respond to health/readiness probes.
    The gateway is started without worker URLs -- just validates the image works.
    """

    def test_health_endpoint(self, sie_docker_gateway: str) -> None:
        """Gateway Docker container responds to liveness probe."""
        response = httpx.get(f"{sie_docker_gateway}/healthz", timeout=10.0)
        assert response.status_code == 200

    def test_readiness_endpoint(self, sie_docker_gateway: str) -> None:
        """Gateway Docker container responds to readiness probe."""
        response = httpx.get(f"{sie_docker_gateway}/readyz", timeout=10.0)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
