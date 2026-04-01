from __future__ import annotations

import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient


@pytest.mark.integration
class TestSyncClient:
    def test_list_models(self, sie_client: SIEClient) -> None:
        models = sie_client.list_models()
        assert len(models) > 0
        model = models[0]
        assert "name" in model
        assert "outputs" in model
        assert "dims" in model

    def test_encode_single_item(self, sie_client: SIEClient) -> None:
        result = sie_client.encode("BAAI/bge-m3:bge_m3_flag", {"text": "Hello world"})

        assert isinstance(result, dict)
        assert "dense" in result
        assert isinstance(result["dense"], np.ndarray)
        assert result["dense"].dtype == np.float32
        assert result["dense"].shape == (1024,)

    def test_encode_batch(self, sie_client: SIEClient) -> None:
        results = sie_client.encode(
            "BAAI/bge-m3:bge_m3_flag",
            [{"text": "Hello"}, {"text": "World"}, {"text": "Test"}],
        )

        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert "dense" in result
            assert result["dense"].shape == (1024,)

    def test_encode_with_id(self, sie_client: SIEClient) -> None:
        result = sie_client.encode("BAAI/bge-m3:bge_m3_flag", {"id": "doc-123", "text": "Test"})
        assert result.get("id") == "doc-123"

    def test_encode_all_output_types(self, sie_client: SIEClient) -> None:
        result = sie_client.encode(
            "BAAI/bge-m3:bge_m3_flag",
            {"text": "Hello world"},
            output_types=["dense", "sparse", "multivector"],
        )

        assert "dense" in result
        assert result["dense"].shape == (1024,)

        assert "sparse" in result
        assert "indices" in result["sparse"]
        assert "values" in result["sparse"]
        assert isinstance(result["sparse"]["indices"], np.ndarray)
        assert isinstance(result["sparse"]["values"], np.ndarray)
        assert len(result["sparse"]["indices"]) > 0
        assert len(result["sparse"]["indices"]) == len(result["sparse"]["values"])

        assert "multivector" in result
        assert isinstance(result["multivector"], np.ndarray)
        assert len(result["multivector"].shape) == 2
        assert result["multivector"].shape[1] == 1024

    def test_encode_sparse_only(self, sie_client: SIEClient) -> None:
        result = sie_client.encode(
            "BAAI/bge-m3:bge_m3_flag",
            {"text": "Hello world"},
            output_types=["sparse"],
        )

        assert "sparse" in result
        assert "dense" not in result
        assert isinstance(result["sparse"]["indices"], np.ndarray)
        assert isinstance(result["sparse"]["values"], np.ndarray)

    def test_encode_multivector_only(self, sie_client: SIEClient) -> None:
        result = sie_client.encode(
            "BAAI/bge-m3:bge_m3_flag",
            {"text": "Hello world"},
            output_types=["multivector"],
        )

        assert "multivector" in result
        assert "dense" not in result
        assert isinstance(result["multivector"], np.ndarray)
        assert len(result["multivector"].shape) == 2

    def test_encode_batch_with_ids(self, sie_client: SIEClient) -> None:
        results = sie_client.encode(
            "BAAI/bge-m3:bge_m3_flag",
            [
                {"id": "doc-1", "text": "First"},
                {"id": "doc-2", "text": "Second"},
            ],
        )

        assert len(results) == 2
        assert results[0].get("id") == "doc-1"
        assert results[1].get("id") == "doc-2"

    def test_context_manager(self, sie_server: str) -> None:
        with SIEClient(sie_server, timeout_s=60.0) as client:
            models = client.list_models()
            assert len(models) > 0


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncClient:
    async def test_list_models(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            models = await async_client.list_models()
            assert len(models) > 0
            model = models[0]
            assert "name" in model
            assert "outputs" in model
            assert "dims" in model

    async def test_encode_single_item(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            result = await async_client.encode("BAAI/bge-m3:bge_m3_flag", {"text": "Hello async world"})

            assert isinstance(result, dict)
            assert "dense" in result
            assert isinstance(result["dense"], np.ndarray)
            assert result["dense"].dtype == np.float32
            assert result["dense"].shape == (1024,)

    async def test_encode_batch(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            results = await async_client.encode(
                "BAAI/bge-m3:bge_m3_flag",
                [{"text": "Async hello"}, {"text": "Async world"}],
            )

            assert isinstance(results, list)
            assert len(results) == 2
            for result in results:
                assert "dense" in result
                assert result["dense"].shape == (1024,)

    async def test_encode_with_id(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            result = await async_client.encode("BAAI/bge-m3:bge_m3_flag", {"id": "async-doc-123", "text": "Test"})
            assert result.get("id") == "async-doc-123"

    async def test_encode_all_output_types(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            result = await async_client.encode(
                "BAAI/bge-m3:bge_m3_flag",
                {"text": "Hello async world"},
                output_types=["dense", "sparse", "multivector"],
            )

            assert "dense" in result
            assert result["dense"].shape == (1024,)

            assert "sparse" in result
            assert isinstance(result["sparse"]["indices"], np.ndarray)
            assert isinstance(result["sparse"]["values"], np.ndarray)

            assert "multivector" in result
            assert isinstance(result["multivector"], np.ndarray)
            assert len(result["multivector"].shape) == 2

    async def test_encode_sparse_only(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            result = await async_client.encode(
                "BAAI/bge-m3:bge_m3_flag",
                {"text": "Hello async world"},
                output_types=["sparse"],
            )

            assert "sparse" in result
            assert "dense" not in result

    async def test_encode_multivector_only(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            result = await async_client.encode(
                "BAAI/bge-m3:bge_m3_flag",
                {"text": "Hello async world"},
                output_types=["multivector"],
            )

            assert "multivector" in result
            assert "dense" not in result

    async def test_encode_batch_with_ids(self, async_client: SIEAsyncClient) -> None:
        async with async_client:
            results = await async_client.encode(
                "BAAI/bge-m3:bge_m3_flag",
                [
                    {"id": "async-doc-1", "text": "First async"},
                    {"id": "async-doc-2", "text": "Second async"},
                ],
            )

            assert len(results) == 2
            assert results[0].get("id") == "async-doc-1"
            assert results[1].get("id") == "async-doc-2"

    async def test_context_manager(self, sie_server: str) -> None:
        async with SIEAsyncClient(sie_server, timeout_s=60.0) as client:
            models = await client.list_models()
            assert len(models) > 0


TestIntegration = TestSyncClient
