"""Tests for PostprocessorRegistry."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import MuveraConfig, MuveraPostprocessor
from sie_server.core.postprocessor_registry import PostprocessorRegistry


@pytest.fixture
def cpu_pool() -> ThreadPoolExecutor:
    """Create a test thread pool."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")


@pytest.fixture
def registry(cpu_pool: ThreadPoolExecutor) -> PostprocessorRegistry:
    """Create a test registry."""
    return PostprocessorRegistry(cpu_pool)


@pytest.fixture
def muvera_postprocessor() -> MuveraPostprocessor:
    """Create a test MUVERA postprocessor."""
    config = MuveraConfig(num_repetitions=2, num_simhash_projections=2)  # Small for testing
    return MuveraPostprocessor(token_dim=8, config=config)


class TestPostprocessorRegistry:
    """Tests for PostprocessorRegistry."""

    def test_register_and_get(self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor) -> None:
        """Test registering and retrieving a postprocessor."""
        registry.register("test-model", {"muvera": muvera_postprocessor})

        assert registry.has_postprocessor("test-model", "muvera")
        assert not registry.has_postprocessor("test-model", "quantize")
        assert not registry.has_postprocessor("other-model", "muvera")

        retrieved = registry.get_postprocessor("test-model", "muvera")
        assert retrieved is muvera_postprocessor

    def test_unregister(self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor) -> None:
        """Test unregistering postprocessors."""
        registry.register("test-model", {"muvera": muvera_postprocessor})
        assert registry.has_postprocessor("test-model", "muvera")

        registry.unregister("test-model")
        assert not registry.has_postprocessor("test-model", "muvera")
        assert registry.get_postprocessor("test-model", "muvera") is None

    def test_registered_models(
        self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor
    ) -> None:
        """Test listing registered models."""
        assert registry.registered_models == []

        registry.register("model-a", {"muvera": muvera_postprocessor})
        registry.register("model-b", {"muvera": muvera_postprocessor})

        assert set(registry.registered_models) == {"model-a", "model-b"}

    def test_get_option_keys(self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor) -> None:
        """Test listing option keys for a model."""
        mock_quantize = MagicMock()
        registry.register("test-model", {"muvera": muvera_postprocessor, "quantize": mock_quantize})

        assert set(registry.get_option_keys("test-model")) == {"muvera", "quantize"}
        assert registry.get_option_keys("unknown-model") == []


class TestPostprocessorTransform:
    """Tests for transform functionality."""

    def test_transform_sync_applies_postprocessor(
        self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor
    ) -> None:
        """Test that transform_sync applies the postprocessor when option is set."""
        registry.register("test-model", {"muvera": muvera_postprocessor})

        # Create output with multivector (MUVERA input)
        output = EncodeOutput(
            multivector=[np.random.randn(5, 8).astype(np.float32) for _ in range(2)],
            multivector_token_dim=8,
        )
        assert output.dense is None

        # Transform with muvera option
        elapsed_ms = registry.transform_sync("test-model", output, {"muvera": {"num_repetitions": 2}}, is_query=False)

        # MUVERA should have added dense output
        assert output.dense is not None
        assert output.dense.shape[0] == 2  # Batch size
        assert elapsed_ms > 0

    def test_transform_sync_skips_when_option_null(
        self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor
    ) -> None:
        """Test that transform_sync skips when option is null."""
        registry.register("test-model", {"muvera": muvera_postprocessor})

        output = EncodeOutput(
            multivector=[np.random.randn(5, 8).astype(np.float32)],
            multivector_token_dim=8,
        )

        # Transform with muvera=None
        elapsed_ms = registry.transform_sync("test-model", output, {"muvera": None}, is_query=False)

        # Should not have applied MUVERA
        assert output.dense is None
        # Elapsed time should be negligible (just loop overhead, no actual transform)
        assert elapsed_ms < 1.0  # Less than 1ms

    def test_transform_sync_skips_when_option_missing(
        self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor
    ) -> None:
        """Test that transform_sync skips when option key is missing."""
        registry.register("test-model", {"muvera": muvera_postprocessor})

        output = EncodeOutput(
            multivector=[np.random.randn(5, 8).astype(np.float32)],
            multivector_token_dim=8,
        )

        # Transform without muvera option
        elapsed_ms = registry.transform_sync("test-model", output, {}, is_query=False)

        assert output.dense is None
        # Elapsed time should be negligible (just loop overhead, no actual transform)
        assert elapsed_ms < 1.0  # Less than 1ms

    def test_transform_sync_handles_unregistered_model(self, registry: PostprocessorRegistry) -> None:
        """Test that transform_sync handles unregistered models gracefully."""
        output = EncodeOutput(dense=np.zeros((1, 10)))

        elapsed_ms = registry.transform_sync("unknown-model", output, {"muvera": {}}, is_query=False)

        assert elapsed_ms == 0.0

    @pytest.mark.asyncio
    async def test_transform_async(
        self, registry: PostprocessorRegistry, muvera_postprocessor: MuveraPostprocessor
    ) -> None:
        """Test async transform runs in thread pool."""
        registry.register("test-model", {"muvera": muvera_postprocessor})

        output = EncodeOutput(
            multivector=[np.random.randn(5, 8).astype(np.float32) for _ in range(2)],
            multivector_token_dim=8,
        )

        elapsed_ms = await registry.transform("test-model", output, {"muvera": {"num_repetitions": 2}}, is_query=False)

        assert output.dense is not None
        assert output.dense.shape[0] == 2
        assert elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_transform_async_passes_is_query(self, registry: PostprocessorRegistry) -> None:
        """Test that is_query is passed to postprocessor."""
        mock_postprocessor = MagicMock()
        mock_postprocessor.transform = MagicMock()
        registry.register("test-model", {"muvera": mock_postprocessor})

        output = EncodeOutput(dense=np.zeros((1, 10)))

        await registry.transform("test-model", output, {"muvera": {}}, is_query=True)

        mock_postprocessor.transform.assert_called_once_with(output, is_query=True)


class TestEmptyRegistry:
    """Tests for empty registry behavior."""

    def test_register_empty_dict(self, registry: PostprocessorRegistry) -> None:
        """Test registering empty dict is a no-op."""
        registry.register("test-model", {})
        assert registry.registered_models == []

    def test_unregister_unknown_model(self, registry: PostprocessorRegistry) -> None:
        """Test unregistering unknown model is safe."""
        registry.unregister("unknown-model")  # Should not raise
