"""Tests for the ephemeral payload store."""

import tempfile
import unittest.mock
from unittest.mock import MagicMock

import pytest
from sie_router.payload_store import GCSPayloadStore, LocalPayloadStore, S3PayloadStore, create_payload_store


class TestLocalPayloadStore:
    @pytest.mark.asyncio
    async def test_put_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalPayloadStore(base_dir=tmpdir)
            await store.put("test/key", b"hello world")
            data = await store.get("test/key")
            assert data == b"hello world"

    @pytest.mark.asyncio
    async def test_get_missing_raises_key_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalPayloadStore(base_dir=tmpdir)
            with pytest.raises(KeyError):
                await store.get("nonexistent")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalPayloadStore(base_dir=tmpdir)
            await store.put("test/key", b"data")
            await store.delete("test/key")
            with pytest.raises(KeyError):
                await store.get("test/key")

    @pytest.mark.asyncio
    async def test_delete_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalPayloadStore(base_dir=tmpdir)
            await store.put("payloads/req-1/0", b"item0")
            await store.put("payloads/req-1/1", b"item1")
            await store.put("payloads/req-2/0", b"other")

            await store.delete_prefix("payloads/req-1")

            # req-1 items should be gone
            with pytest.raises(KeyError):
                await store.get("payloads/req-1/0")
            with pytest.raises(KeyError):
                await store.get("payloads/req-1/1")

            # req-2 should still exist
            data = await store.get("payloads/req-2/0")
            assert data == b"other"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalPayloadStore(base_dir=tmpdir)
            # Should not raise
            await store.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_large_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalPayloadStore(base_dir=tmpdir)
            # 2MB payload (simulating an image)
            data = b"x" * (2 * 1024 * 1024)
            await store.put("payloads/req-1/0", data)
            retrieved = await store.get("payloads/req-1/0")
            assert retrieved == data


class TestCreatePayloadStore:
    def test_none_returns_none(self) -> None:
        assert create_payload_store(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert create_payload_store("") is None

    def test_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_payload_store(tmpdir)
            assert isinstance(store, LocalPayloadStore)

    def test_s3_url(self) -> None:
        store = create_payload_store("s3://my-bucket/prefix")
        assert isinstance(store, S3PayloadStore)

    def test_gs_url_returns_gcs_store(self) -> None:
        """gs:// URLs return a GCSPayloadStore."""
        store = create_payload_store("gs://my-bucket/payloads")
        assert isinstance(store, GCSPayloadStore)


class TestS3PayloadStore:
    @pytest.mark.asyncio
    async def test_put(self) -> None:
        store = S3PayloadStore(bucket="my-bucket", prefix="payloads")
        mock_client = MagicMock()
        store._client = mock_client
        await store.put("req-1/0", b"hello")
        mock_client.put_object.assert_called_once_with(Bucket="my-bucket", Key="payloads/req-1/0", Body=b"hello")

    @pytest.mark.asyncio
    async def test_get(self) -> None:
        store = S3PayloadStore(bucket="my-bucket", prefix="payloads")
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"hello"
        mock_client.get_object.return_value = {"Body": mock_body}
        store._client = mock_client
        result = await store.get("req-1/0")
        mock_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="payloads/req-1/0")
        assert result == b"hello"
        mock_body.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_missing_raises_key_error(self) -> None:
        from botocore.exceptions import ClientError

        store = S3PayloadStore(bucket="my-bucket", prefix="payloads")
        mock_client = MagicMock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )
        store._client = mock_client
        with pytest.raises(KeyError):
            await store.get("nonexistent")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = S3PayloadStore(bucket="my-bucket", prefix="payloads")
        mock_client = MagicMock()
        store._client = mock_client
        await store.delete("req-1/0")
        mock_client.delete_object.assert_called_once_with(Bucket="my-bucket", Key="payloads/req-1/0")


class TestGCSPayloadStore:
    def test_gcs_missing_dependency_raises_import_error(self) -> None:
        """GCSPayloadStore raises ImportError with install instructions when google-cloud-storage is missing."""
        store = GCSPayloadStore(bucket="my-bucket", prefix="payloads")
        with unittest.mock.patch.dict("sys.modules", {"google.cloud": None, "google.cloud.storage": None}):
            with pytest.raises(ImportError, match="google-cloud-storage is required"):
                store._get_bucket()

    @pytest.mark.asyncio
    async def test_put_uploads_to_blob(self) -> None:
        """GCSPayloadStore.put uploads data via blob.upload_from_string."""
        store = GCSPayloadStore(bucket="my-bucket", prefix="payloads")
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        store._client = mock_bucket

        await store.put("test-key", b"hello")

        mock_bucket.blob.assert_called_once_with("payloads/test-key")
        mock_blob.upload_from_string.assert_called_once_with(b"hello")

    @pytest.mark.asyncio
    async def test_get_downloads_from_blob(self) -> None:
        """GCSPayloadStore.get returns blob data."""
        store = GCSPayloadStore(bucket="my-bucket", prefix="payloads")
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = b"hello"
        mock_bucket.blob.return_value = mock_blob
        store._client = mock_bucket

        result = await store.get("test-key")

        assert result == b"hello"
        mock_bucket.blob.assert_called_once_with("payloads/test-key")

    @pytest.mark.asyncio
    async def test_get_not_found_raises_key_error(self) -> None:
        """GCSPayloadStore.get raises KeyError for NotFound errors."""
        from google.api_core.exceptions import NotFound

        store = GCSPayloadStore(bucket="my-bucket", prefix="payloads")
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = NotFound("blob not found")
        mock_bucket.blob.return_value = mock_blob
        store._client = mock_bucket

        with pytest.raises(KeyError):
            await store.get("nonexistent")

    @pytest.mark.asyncio
    async def test_get_infra_error_propagates(self) -> None:
        """GCSPayloadStore.get lets non-NotFound errors propagate."""
        store = GCSPayloadStore(bucket="my-bucket", prefix="payloads")
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = PermissionError("access denied")
        mock_bucket.blob.return_value = mock_blob
        store._client = mock_bucket

        with pytest.raises(PermissionError):
            await store.get("some-key")
