from __future__ import annotations

import pytest
from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.types.inputs import Item


class TestModelCapabilities:
    """Tests for ModelCapabilities."""

    def test_valid_capabilities(self) -> None:
        """Can create capabilities with valid inputs."""
        caps = ModelCapabilities(
            inputs=["text", "image"],
            outputs=["dense", "sparse"],
        )
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense", "sparse"]


class TestModelDims:
    """Tests for ModelDims."""

    def test_valid_dims(self) -> None:
        """Can create dims with valid values."""
        dims = ModelDims(dense=1024, sparse=30522, multivector=128)
        assert dims.dense == 1024
        assert dims.sparse == 30522
        assert dims.multivector == 128

    def test_optional_dims(self) -> None:
        """Dims can be None."""
        dims = ModelDims(dense=768)
        assert dims.dense == 768
        assert dims.sparse is None
        assert dims.multivector is None


class TestModelAdapterABC:
    """Tests for ModelAdapter abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Cannot instantiate ModelAdapter directly."""
        with pytest.raises(TypeError, match="abstract"):
            ModelAdapter()  # type: ignore[abstract]

    def test_default_methods_raise(self) -> None:
        """Default encode/score/extract raise NotImplementedError."""

        class MinimalAdapter(ModelAdapter):
            @property
            def capabilities(self) -> ModelCapabilities:
                return ModelCapabilities(inputs=["text"], outputs=["dense"])

            @property
            def dims(self) -> ModelDims:
                return ModelDims(dense=768)

            def load(self, device: str) -> None:
                pass

            def unload(self) -> None:
                pass

        adapter = MinimalAdapter()

        with pytest.raises(NotImplementedError, match="does not implement encode"):
            adapter.encode([Item(text="test")], output_types=["dense"])

        with pytest.raises(NotImplementedError, match="does not support score"):
            adapter.score(Item(text="query"), [Item(text="doc")])

        with pytest.raises(NotImplementedError, match="does not support extract"):
            adapter.extract([Item(text="test")])
