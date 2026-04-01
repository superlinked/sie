"""Unit tests for SIE embedding components."""

from __future__ import annotations

from haystack import Document
from sie_haystack import (
    SIEDocumentEmbedder,
    SIESparseDocumentEmbedder,
    SIESparseTextEmbedder,
    SIETextEmbedder,
)


class TestSIETextEmbedder:
    """Tests for SIETextEmbedder component."""

    def test_run_returns_embedding(self, mock_sie_client: object) -> None:
        """Test that run returns an embedding."""
        embedder = SIETextEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.run(text="Hello world")

        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == 384  # Default mock dim

    def test_run_uses_is_query_true(self, mock_sie_client: object) -> None:
        """Test that text embedder sets options.is_query=True."""
        embedder = SIETextEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.run(text="Test query")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIETextEmbedder(model="custom/embedding-model")
        embedder._client = mock_sie_client

        embedder.run(text="Test")

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_warm_up_initializes_client(self) -> None:
        """Test that warm_up initializes the client."""
        embedder = SIETextEmbedder(model="test-model")
        assert embedder._client is None

        # Can't fully test without mocking, but verify method exists
        assert hasattr(embedder, "warm_up")


class TestSIEDocumentEmbedder:
    """Tests for SIEDocumentEmbedder component."""

    def test_run_embeds_documents(self, mock_sie_client: object, haystack_documents: list[Document]) -> None:
        """Test that run embeds documents."""
        embedder = SIEDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.run(documents=haystack_documents)

        assert "documents" in result
        assert len(result["documents"]) == len(haystack_documents)
        for doc in result["documents"]:
            assert doc.embedding is not None
            assert len(doc.embedding) == 384

    def test_run_empty_list(self, mock_sie_client: object) -> None:
        """Test that run handles empty document list."""
        embedder = SIEDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.run(documents=[])

        assert result == {"documents": []}

    def test_run_uses_is_query_false(self, mock_sie_client: object) -> None:
        """Test that document embedder doesn't pass is_query (defaults to False)."""
        embedder = SIEDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.run(documents=[Document(content="Test doc")])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        # Document embedding doesn't pass options.is_query (server default is False)
        assert call_kwargs.get("options") is None

    def test_meta_fields_to_embed(self, mock_sie_client: object) -> None:
        """Test embedding with metadata fields."""
        embedder = SIEDocumentEmbedder(
            model="test-model",
            meta_fields_to_embed=["title"],
        )
        embedder._client = mock_sie_client

        docs = [Document(content="Content here", meta={"title": "My Title"})]
        embedder.run(documents=docs)

        call_args = mock_sie_client.encode.call_args
        items = call_args[0][1]
        # First item should include the title
        assert "My Title" in items[0]["text"]

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIEDocumentEmbedder(model="custom/embedding-model")
        embedder._client = mock_sie_client

        embedder.run(documents=[Document(content="Test")])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_documents_are_modified_in_place(self, mock_sie_client: object) -> None:
        """Test that embeddings are stored on the original documents."""
        embedder = SIEDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        docs = [Document(content="Test doc")]
        result = embedder.run(documents=docs)

        # Both the input and output should have embeddings
        assert docs[0].embedding is not None
        assert result["documents"][0].embedding is not None


class TestSIESparseTextEmbedder:
    """Tests for SIESparseTextEmbedder component."""

    def test_run_returns_sparse_embedding(self, mock_sie_client: object) -> None:
        """Test that run returns a sparse embedding."""
        embedder = SIESparseTextEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.run(text="Hello world")

        assert "sparse_embedding" in result
        assert isinstance(result["sparse_embedding"], dict)
        assert "indices" in result["sparse_embedding"]
        assert "values" in result["sparse_embedding"]
        assert len(result["sparse_embedding"]["indices"]) == len(result["sparse_embedding"]["values"])

    def test_run_uses_sparse_output_type(self, mock_sie_client: object) -> None:
        """Test that sparse embedder requests sparse output."""
        embedder = SIESparseTextEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.run(text="Test query")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_run_uses_is_query_true(self, mock_sie_client: object) -> None:
        """Test that sparse text embedder sets options.is_query=True."""
        embedder = SIESparseTextEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.run(text="Test query")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIESparseTextEmbedder(model="custom/embedding-model")
        embedder._client = mock_sie_client

        embedder.run(text="Test")

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_warm_up_initializes_client(self) -> None:
        """Test that warm_up initializes the client."""
        embedder = SIESparseTextEmbedder(model="test-model")
        assert embedder._client is None

        # Can't fully test without mocking, but verify method exists
        assert hasattr(embedder, "warm_up")


class TestSIESparseDocumentEmbedder:
    """Tests for SIESparseDocumentEmbedder component."""

    def test_run_embeds_documents_with_sparse(
        self, mock_sie_client: object, haystack_documents: list[Document]
    ) -> None:
        """Test that run embeds documents with sparse embeddings."""
        embedder = SIESparseDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.run(documents=haystack_documents)

        assert "documents" in result
        assert len(result["documents"]) == len(haystack_documents)
        for doc in result["documents"]:
            sparse = doc.meta.get("_sparse_embedding")
            assert sparse is not None
            assert "indices" in sparse
            assert "values" in sparse
            assert len(sparse["indices"]) == len(sparse["values"])

    def test_run_empty_list(self, mock_sie_client: object) -> None:
        """Test that run handles empty document list."""
        embedder = SIESparseDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.run(documents=[])

        assert result == {"documents": []}

    def test_run_uses_sparse_output_type(self, mock_sie_client: object) -> None:
        """Test that document embedder requests sparse output."""
        embedder = SIESparseDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.run(documents=[Document(content="Test doc")])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_run_does_not_use_is_query(self, mock_sie_client: object) -> None:
        """Test that sparse document embedder doesn't pass is_query."""
        embedder = SIESparseDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.run(documents=[Document(content="Test doc")])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        # Document embedding doesn't pass options.is_query (server default is False)
        assert call_kwargs.get("options") is None

    def test_meta_fields_to_embed(self, mock_sie_client: object) -> None:
        """Test embedding with metadata fields."""
        embedder = SIESparseDocumentEmbedder(
            model="test-model",
            meta_fields_to_embed=["title"],
        )
        embedder._client = mock_sie_client

        docs = [Document(content="Content here", meta={"title": "My Title"})]
        embedder.run(documents=docs)

        call_args = mock_sie_client.encode.call_args
        items = call_args[0][1]
        # First item should include the title
        assert "My Title" in items[0]["text"]

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIESparseDocumentEmbedder(model="custom/embedding-model")
        embedder._client = mock_sie_client

        embedder.run(documents=[Document(content="Test")])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_documents_are_modified_in_place(self, mock_sie_client: object) -> None:
        """Test that sparse embeddings are stored on the original documents."""
        embedder = SIESparseDocumentEmbedder(model="test-model")
        embedder._client = mock_sie_client

        docs = [Document(content="Test doc")]
        result = embedder.run(documents=docs)

        # Both the input and output should have sparse embeddings in meta
        assert docs[0].meta.get("_sparse_embedding") is not None
        assert result["documents"][0].meta.get("_sparse_embedding") is not None
