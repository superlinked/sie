from unittest.mock import MagicMock, patch

import httpx
import msgpack
import numpy as np
import pytest
from sie_sdk import RequestError, ServerError, SIEClient, SIEConnectionError


class TestSIEClientInit:
    """Tests for SIEClient initialization."""

    def test_default_initialization(self) -> None:
        """Client initializes with default settings."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080")
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080"
            assert call_kwargs["timeout"] == 30.0
            assert call_kwargs["headers"]["Content-Type"] == "application/msgpack"
            client.close()

    def test_custom_timeout(self) -> None:
        """Client respects custom timeout."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080", timeout_s=60.0)
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["timeout"] == 60.0
            client.close()

    def test_api_key_sets_auth_header(self) -> None:
        """Client sets Authorization header when api_key provided."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080", api_key="secret")
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer secret"
            client.close()

    def test_trailing_slash_removed(self) -> None:
        """Base URL trailing slash is removed."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            client = SIEClient("http://localhost:8080/")
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8080"
            client.close()

    def test_gpu_stored(self) -> None:
        """Client stores gpu for use in requests."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", gpu="l4")
            assert client._default_gpu == "l4"
            client.close()

    def test_options_stored(self) -> None:
        """Client stores options for use in requests."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", options={"key": "value"})
            assert client._default_options == {"key": "value"}
            client.close()

    def test_resolve_gpu_uses_default(self) -> None:
        """_resolve_gpu returns default when gpu is None."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", gpu="a100")
            assert client._resolve_gpu(None) == "a100"
            assert client._resolve_gpu("l4") == "l4"  # Override
            client.close()

    def test_resolve_options_merges_defaults(self) -> None:
        """_resolve_options merges defaults with per-call options."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080", options={"a": 1, "b": 2})
            # No per-call options returns defaults
            assert client._resolve_options(None) == {"a": 1, "b": 2}
            # Per-call options override defaults
            assert client._resolve_options({"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}
            client.close()

    def test_base_url_property(self) -> None:
        """Client exposes base_url as a public property."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080")
            assert client.base_url == "http://localhost:8080"
            client.close()

    def test_base_url_property_strips_trailing_slash(self) -> None:
        """base_url property returns normalized URL without trailing slash."""
        with patch("sie_sdk.client.sync.httpx.Client"):
            client = SIEClient("http://localhost:8080/")
            assert client.base_url == "http://localhost:8080"
            client.close()


class TestEncode:
    """Tests for encode() method."""

    def test_encode_single_item_returns_single_result(self) -> None:
        """Single item input returns single result (not list)."""
        # Mock response with dense embedding
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "id": "doc-1",
                        "dense": {
                            "dims": 4,
                            "dtype": "float32",
                            "values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                        },
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.encode("bge-m3", {"id": "doc-1", "text": "hello"})

            # Should be single result, not list
            assert isinstance(result, dict)
            assert result["id"] == "doc-1"
            assert isinstance(result["dense"], np.ndarray)
            assert result["dense"].shape == (4,)
            client.close()

    def test_encode_list_returns_list(self) -> None:
        """List of items input returns list of results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([1.0, 2.0, 3.0, 4.0])}},
                    {"dense": {"dims": 4, "dtype": "float32", "values": np.array([5.0, 6.0, 7.0, 8.0])}},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            results = client.encode("bge-m3", [{"text": "hello"}, {"text": "world"}])

            assert isinstance(results, list)
            assert len(results) == 2
            client.close()

    def test_encode_with_output_types(self) -> None:
        """Output types are passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        dense_values = np.array([1, 2, 3, 4], dtype=np.float32)
        mock_response.content = msgpack.packb(
            {"model": "bge-m3", "items": [{"dense": {"dims": 4, "dtype": "float32", "values": dense_values}}]},
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.encode("bge-m3", {"text": "hello"}, output_types=["dense", "sparse"])

            # Check request body
            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["params"]["output_types"] == ["dense", "sparse"]
            client.close()

    def test_encode_with_instruction(self) -> None:
        """Instruction is passed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        dense_values = np.array([1, 2, 3, 4], dtype=np.float32)
        mock_response.content = msgpack.packb(
            {"model": "bge-m3", "items": [{"dense": {"dims": 4, "dtype": "float32", "values": dense_values}}]},
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.encode(
                "gte-qwen2-7b",
                {"text": "What is ML?"},
                instruction="Retrieve passages that answer this question",
                options={"is_query": True},
            )

            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["params"]["instruction"] == "Retrieve passages that answer this question"
            assert request_body["params"]["options"]["is_query"] is True
            client.close()

    def test_encode_parses_sparse_result(self) -> None:
        """Sparse results are parsed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "sparse": {
                            "dims": 30000,
                            "dtype": "float32",
                            "indices": np.array([100, 200, 300], dtype=np.int32),
                            "values": np.array([0.5, 0.3, 0.2], dtype=np.float32),
                        }
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.encode("bge-m3", {"text": "hello"})

            assert "sparse" in result
            assert isinstance(result["sparse"]["indices"], np.ndarray)
            assert isinstance(result["sparse"]["values"], np.ndarray)
            np.testing.assert_array_equal(result["sparse"]["indices"], [100, 200, 300])
            client.close()

    def test_encode_parses_multivector_result(self) -> None:
        """Multivector results are parsed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-m3",
                "items": [
                    {
                        "multivector": {
                            "token_dims": 128,
                            "num_tokens": 3,
                            "dtype": "float32",
                            "values": np.random.default_rng().random((3, 128), dtype=np.float32),
                        }
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.encode("bge-m3", {"text": "hello"})

            assert "multivector" in result
            assert isinstance(result["multivector"], np.ndarray)
            assert result["multivector"].shape == (3, 128)
            client.close()


class TestErrorHandling:
    """Tests for error handling."""

    def test_connection_error(self) -> None:
        """Connection errors are wrapped as SIEConnectionError."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.side_effect = httpx.ConnectError("Connection refused")
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="Failed to connect"):
                client.encode("bge-m3", {"text": "hello"})
            client.close()

    def test_timeout_error(self) -> None:
        """Timeout errors are wrapped as SIEConnectionError."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.side_effect = httpx.TimeoutException("Timeout")
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="timed out"):
                client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=False)
            client.close()

    def test_request_error_400(self) -> None:
        """400 errors are raised as RequestError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"detail": {"code": "INVALID_INPUT", "message": "Invalid text"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 400
            assert exc_info.value.code == "INVALID_INPUT"
            client.close()

    def test_server_error_500(self) -> None:
        """500 errors are raised as ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": {"code": "INFERENCE_ERROR", "message": "Model failed"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 500
            assert exc_info.value.code == "INFERENCE_ERROR"
            client.close()

    def test_server_error_string_format(self) -> None:
        """500 errors with string error format are handled correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "application/json"}
        # Server may return error as a simple string instead of dict
        mock_response.json.return_value = {"error": "Internal server error"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                client.encode("bge-m3", {"text": "hello"})

            assert exc_info.value.status_code == 500
            assert exc_info.value.code is None  # No code in string format
            assert "Internal server error" in str(exc_info.value)
            client.close()

    def test_model_not_found_404(self) -> None:
        """404 errors are raised as RequestError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"detail": {"code": "MODEL_NOT_FOUND", "message": "Model not found"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as exc_info:
                client.encode("unknown-model", {"text": "hello"})

            assert exc_info.value.status_code == 404
            client.close()


class TestListModels:
    """Tests for list_models() method."""

    def test_list_models_returns_list(self) -> None:
        """list_models returns a list of ModelInfo."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Match server's flat ModelInfo structure (no nested capabilities)
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "bge-m3",
                    "loaded": True,
                    "inputs": ["text"],
                    "outputs": ["dense", "sparse", "multivector"],
                    "dims": {"dense": 1024, "sparse": 250002, "multivector": 1024},
                    "max_sequence_length": 8192,
                }
            ]
        }

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            models = client.list_models()

            assert len(models) == 1
            assert models[0]["name"] == "bge-m3"
            assert models[0]["loaded"] is True
            assert "dense" in models[0]["outputs"]
            client.close()

    def test_get_model_returns_info(self) -> None:
        """get_model returns ModelInfo for a specific model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "BAAI/bge-m3",
            "loaded": True,
            "inputs": ["text"],
            "outputs": ["dense", "sparse", "multivector"],
            "dims": {"dense": 1024, "sparse": 250002, "multivector": 1024},
            "max_sequence_length": 8192,
        }

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            info = client.get_model("BAAI/bge-m3")

            assert info["name"] == "BAAI/bge-m3"
            assert info["dims"]["dense"] == 1024
            assert info["dims"]["multivector"] == 1024
            assert info["loaded"] is True
            mock_client.return_value.get.assert_called_once_with(
                "/v1/models/BAAI/bge-m3",
                headers={"Accept": "application/json"},
            )
            client.close()

    def test_get_model_not_found_raises(self) -> None:
        """get_model raises RequestError for unknown model."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "detail": {"code": "MODEL_NOT_FOUND", "message": "Model 'foo' not found"},
        }
        mock_response.text = '{"detail": {"code": "MODEL_NOT_FOUND"}}'

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            with pytest.raises(RequestError):
                client.get_model("foo")
            client.close()


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_closes_client(self) -> None:
        """Context manager calls close() on exit."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            with SIEClient("http://localhost:8080") as client:
                assert client is not None
            mock_client.return_value.close.assert_called_once()


class TestResourceCleanup:
    """Tests for resource cleanup via weakref.finalize safety net."""

    def test_finalizer_closes_transport_on_gc(self) -> None:
        """GC closes httpx transport if close() was never called."""
        import gc

        with patch("sie_sdk.client.sync.httpx.Client") as mock_httpx:
            mock_transport = MagicMock()
            mock_httpx.return_value = mock_transport

            client = SIEClient("http://localhost:8080")
            # Simulate forgetting to close — drop all references
            del client
            gc.collect()

        mock_transport.close.assert_called_once()

    def test_close_detaches_finalizer(self) -> None:
        """Explicit close() prevents double-close from GC finalizer."""
        import gc

        with patch("sie_sdk.client.sync.httpx.Client") as mock_httpx:
            mock_transport = MagicMock()
            mock_httpx.return_value = mock_transport

            client = SIEClient("http://localhost:8080")
            client.close()
            mock_transport.close.assert_called_once()

            # Reset and verify finalizer doesn't fire on GC
            mock_transport.reset_mock()
            del client
            gc.collect()

        mock_transport.close.assert_not_called()

    def test_double_close_is_safe(self) -> None:
        """Calling close() twice does not raise."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_httpx:
            mock_transport = MagicMock()
            mock_httpx.return_value = mock_transport

            client = SIEClient("http://localhost:8080")
            client.close()
            client.close()  # Should not raise


class TestScore:
    """Tests for score() method."""

    def test_score_returns_score_result(self) -> None:
        """score() returns a ScoreResult with sorted scores."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "scores": [
                    {"item_id": "doc-1", "score": 0.95, "rank": 0},
                    {"item_id": "doc-2", "score": 0.72, "rank": 1},
                    {"item_id": "doc-3", "score": 0.31, "rank": 2},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.score(
                "bge-reranker-v2",
                query={"text": "What is machine learning?"},
                items=[
                    {"id": "doc-1", "text": "ML is a subset of AI..."},
                    {"id": "doc-2", "text": "Python is a language..."},
                    {"id": "doc-3", "text": "Cooking recipes..."},
                ],
            )

            assert result["model"] == "bge-reranker-v2"
            assert len(result["scores"]) == 3
            # Scores should be sorted by rank
            assert result["scores"][0]["rank"] == 0
            assert result["scores"][0]["item_id"] == "doc-1"
            assert result["scores"][0]["score"] == 0.95
            client.close()

    def test_score_with_instruction(self) -> None:
        """score() passes instruction correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "scores": [{"item_id": "0", "score": 0.8, "rank": 0}],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.score(
                "bge-reranker-v2",
                query={"text": "What is ML?"},
                items=[{"text": "ML info"}],
                instruction="Rank by relevance to the query",
            )

            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["instruction"] == "Rank by relevance to the query"
            client.close()

    def test_score_with_query_id(self) -> None:
        """score() preserves query_id in result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "bge-reranker-v2",
                "query_id": "q-123",
                "scores": [{"item_id": "0", "score": 0.8, "rank": 0}],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.score(
                "bge-reranker-v2",
                query={"id": "q-123", "text": "What is ML?"},
                items=[{"text": "ML info"}],
            )

            assert result.get("query_id") == "q-123"
            client.close()


class TestExtract:
    """Tests for extract() method."""

    def test_extract_single_item_returns_single_result(self) -> None:
        """Single item input returns single result (not list)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner-multi-v2.1",
                "items": [
                    {
                        "entities": [
                            {"text": "Apple", "label": "organization", "score": 0.98, "start": 0, "end": 5},
                            {"text": "Steve Jobs", "label": "person", "score": 0.97, "start": 22, "end": 32},
                        ]
                    }
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.extract(
                "gliner-multi-v2.1",
                {"text": "Apple was founded by Steve Jobs."},
                labels=["person", "organization"],
            )

            # Should be single result, not list
            assert isinstance(result, dict)
            assert "entities" in result
            assert len(result["entities"]) == 2
            assert result["entities"][0]["text"] == "Apple"
            assert result["entities"][0]["label"] == "organization"
            client.close()

    def test_extract_list_returns_list(self) -> None:
        """List of items input returns list of results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner-multi-v2.1",
                "items": [
                    {"entities": [{"text": "Apple", "label": "org", "score": 0.9, "start": 0, "end": 5}]},
                    {"entities": [{"text": "Tesla", "label": "org", "score": 0.95, "start": 0, "end": 5}]},
                ],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            results = client.extract(
                "gliner-multi-v2.1",
                [{"text": "Apple info"}, {"text": "Tesla info"}],
                labels=["org"],
            )

            assert isinstance(results, list)
            assert len(results) == 2
            client.close()

    def test_extract_with_labels(self) -> None:
        """extract() passes labels correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {"model": "gliner", "items": [{"entities": []}]},
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            client.extract(
                "gliner",
                {"text": "Test"},
                labels=["person", "organization", "location"],
            )

            call_args = mock_client.return_value.post.call_args
            request_body = msgpack.unpackb(call_args.kwargs["content"], raw=False)
            assert request_body["params"]["labels"] == ["person", "organization", "location"]
            client.close()

    def test_extract_preserves_item_id(self) -> None:
        """extract() preserves item IDs in results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = msgpack.packb(
            {
                "model": "gliner",
                "items": [{"id": "doc-123", "entities": []}],
            },
            use_bin_type=True,
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            client = SIEClient("http://localhost:8080")
            result = client.extract("gliner", {"id": "doc-123", "text": "Test"})

            assert result.get("id") == "doc-123"
            client.close()
