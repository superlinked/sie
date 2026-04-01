from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry


def _make_config(
    name: str = "test",
    hf_id: str | None = "org/test",
    dense_dim: int = 768,
    max_sequence_length: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=hf_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=dense_dim))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
        max_sequence_length=max_sequence_length,
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    """Patch ensure_model_cached to avoid actual HF downloads in tests."""
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


class TestMultiModelRouting:
    """Tests for multi-model routing (Project 4.3).

    Verifies:
    - Each loaded model has its own ModelWorker
    - Requests can alternate between multiple loaded models
    - Correct adapter is returned for each model
    """

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters with unique IDs."""
        counter = [0]

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            counter[0] += 1
            mock._test_id = counter[0]  # Unique ID for verification
            return mock

        return make_mock

    @patch("sie_server.core.model_loader.load_adapter")
    def test_each_model_has_own_worker(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """Each loaded model has its own ModelWorker instance."""
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        mock_load_adapter.side_effect = [adapter_a, adapter_b]

        registry = ModelRegistry()

        # Add two model configs
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Get workers for both models
        worker_a = registry.get_worker("model-a")
        worker_b = registry.get_worker("model-b")

        # Each model has its own worker
        assert worker_a is not None
        assert worker_b is not None
        assert worker_a is not worker_b

        # Workers reference correct adapters
        assert worker_a.adapter is adapter_a
        assert worker_b.adapter is adapter_b

    @patch("sie_server.core.model_loader.load_adapter")
    def test_alternating_requests_to_different_models(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Can alternate requests between multiple loaded models."""
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        mock_load_adapter.side_effect = [adapter_a, adapter_b]

        registry = ModelRegistry()

        # Add and load two models
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Alternate requests between models multiple times
        for _ in range(3):
            # Request to model-a
            got_adapter_a = registry.get("model-a")
            assert got_adapter_a is adapter_a
            assert got_adapter_a._test_id == 1

            # Request to model-b
            got_adapter_b = registry.get("model-b")
            assert got_adapter_b is adapter_b
            assert got_adapter_b._test_id == 2

        # Both models still loaded after alternating requests
        assert registry.is_loaded("model-a")
        assert registry.is_loaded("model-b")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_correct_adapter_returned_by_model_name(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Requests are routed to the correct model adapter by name."""
        # Create 3 models with distinct adapters
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        registry = ModelRegistry()

        model_names = ["model-alpha", "model-beta", "model-gamma"]
        for name in model_names:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Request models in random order and verify correct routing
        assert registry.get("model-gamma")._test_id == 3
        assert registry.get("model-alpha")._test_id == 1
        assert registry.get("model-beta")._test_id == 2
        assert registry.get("model-alpha")._test_id == 1  # Same result on repeat
        assert registry.get("model-gamma")._test_id == 3

    @patch("sie_server.core.model_loader.load_adapter")
    def test_worker_not_available_for_unloaded_model(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Worker returns None for models that aren't loaded."""
        mock_load_adapter.return_value = mock_adapter_factory()

        registry = ModelRegistry()

        config = _make_config(name="test-model")
        registry.add_config(config)

        # Worker should be None before loading
        assert registry.get_worker("test-model") is None

        # Load the model
        registry.load("test-model", device="cpu")

        # Worker should be available after loading
        assert registry.get_worker("test-model") is not None

        # Unload
        registry.unload("test-model")

        # Worker should be None again
        assert registry.get_worker("test-model") is None

    @patch("sie_server.core.model_loader.load_adapter")
    def test_lru_updates_on_alternating_access(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """LRU tracking updates correctly when alternating between models."""
        mock_load_adapter.side_effect = [mock_adapter_factory() for _ in range(3)]

        registry = ModelRegistry()

        # Load 3 models in order: A, B, C
        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Initially A is LRU (loaded first)
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Access A -> now B is LRU
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"

        # Access B -> now C is LRU
        registry.get("model-b")
        assert registry.memory_manager.get_lru_model() == "model-c"

        # Access C -> now A is LRU (full rotation)
        registry.get("model-c")
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Access A again -> now B is LRU
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"
