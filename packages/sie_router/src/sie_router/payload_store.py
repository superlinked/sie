"""Ephemeral payload store for large work item payloads.

When a work item's serialized size exceeds the NATS inline threshold (1MB),
the router offloads the payload to this store and puts a reference in the
NATS message. Workers fetch the payload by reference before processing.

Backends:
- S3-compatible (production, MinIO for staging)
- Local filesystem (local development)

Objects have a short TTL (default 5 minutes). The router also performs
best-effort cleanup after receiving all results for a request.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# Default payload TTL in seconds (5 minutes)
DEFAULT_TTL_SECONDS = 300


class PayloadStore(ABC):
    """Abstract interface for ephemeral payload storage."""

    @abstractmethod
    async def put(self, key: str, data: bytes) -> None:
        """Store a payload.

        Args:
            key: Object key (e.g., ``payloads/{request_id}/{item_index}``).
            data: Raw bytes to store.
        """

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve a payload.

        Args:
            key: Object key.

        Returns:
            Raw bytes.

        Raises:
            KeyError: If the key does not exist (expired or never stored).
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a payload (best-effort).

        Args:
            key: Object key.
        """

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> None:
        """Delete all objects under a prefix (best-effort cleanup).

        Args:
            prefix: Key prefix (e.g., ``payloads/{request_id}/``).
        """


class LocalPayloadStore(PayloadStore):
    """Local filesystem payload store for development.

    Stores blobs in a directory with background TTL cleanup.
    """

    def __init__(self, base_dir: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds
        self._cleanup_task: asyncio.Task[None] | None = None

    def _safe_path(self, key: str) -> Path:
        """Resolve key to a path inside base_dir, rejecting traversal."""
        target = (self._base_dir / key).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"Path traversal detected: {key}") from None
        return target

    async def put(self, key: str, data: bytes) -> None:
        path = self._safe_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def get(self, key: str) -> bytes:
        path = self._safe_path(key)
        if not path.exists():
            raise KeyError(f"Payload not found: {key}")
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, key: str) -> None:
        path = self._safe_path(key)
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except OSError:
            logger.debug("Failed to delete payload: %s", key)

    async def delete_prefix(self, prefix: str) -> None:
        prefix_dir = self._safe_path(prefix)
        if prefix_dir.is_dir():
            try:
                import shutil

                await asyncio.to_thread(shutil.rmtree, prefix_dir, ignore_errors=True)
            except OSError:
                logger.debug("Failed to delete prefix: %s", prefix)

    async def start_cleanup(self) -> None:
        """Start background TTL cleanup loop."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop background TTL cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Delete files older than TTL."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            try:
                now = time.time()
                paths = await asyncio.to_thread(lambda: list(self._base_dir.rglob("*")))
                for path in paths:
                    is_file = await asyncio.to_thread(path.is_file)
                    if is_file:
                        stat = await asyncio.to_thread(path.stat)
                        age = now - stat.st_mtime
                        if age > self._ttl_seconds:
                            await asyncio.to_thread(path.unlink, True)
            except Exception:  # noqa: BLE001
                logger.debug("Payload cleanup error", exc_info=True)


class S3PayloadStore(PayloadStore):
    """S3-compatible payload store for production.

    Uses boto3 async wrappers. TTL is managed via S3 lifecycle rules
    configured on the bucket (not application-side).
    """

    def __init__(self, bucket: str, prefix: str = "payloads") -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._client: object | None = None

    def _get_client(self) -> object:
        """Get or create a cached boto3 S3 client.

        The client is created on first call and reused for subsequent calls.
        All S3 operations are dispatched via ``asyncio.to_thread`` so the
        client is only accessed from one thread at a time.
        """
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    async def put(self, key: str, data: bytes) -> None:
        client = self._get_client()
        full_key = f"{self._prefix}/{key}" if self._prefix else key
        await asyncio.to_thread(
            client.put_object,  # type: ignore[union-attr]
            Bucket=self._bucket,
            Key=full_key,
            Body=data,
        )

    async def get(self, key: str) -> bytes:
        client = self._get_client()
        full_key = f"{self._prefix}/{key}" if self._prefix else key
        try:
            response = await asyncio.to_thread(
                client.get_object,  # type: ignore[union-attr]
                Bucket=self._bucket,
                Key=full_key,
            )
            body = response["Body"]
            try:
                return await asyncio.to_thread(body.read)
            finally:
                body.close()
        except KeyError:
            raise
        except Exception as e:
            # Only convert "not found" errors to KeyError; let infra errors propagate.
            import botocore.exceptions

            if isinstance(e, botocore.exceptions.ClientError):
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    raise KeyError(f"Payload not found: {key}") from e
            raise

    async def delete(self, key: str) -> None:
        client = self._get_client()
        full_key = f"{self._prefix}/{key}" if self._prefix else key
        try:
            await asyncio.to_thread(
                client.delete_object,  # type: ignore[union-attr]
                Bucket=self._bucket,
                Key=full_key,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to delete payload: %s", key)

    async def delete_prefix(self, prefix: str) -> None:
        client = self._get_client()
        full_prefix = f"{self._prefix}/{prefix}" if self._prefix else prefix
        try:
            continuation_token: str | None = None
            while True:
                kwargs: dict[str, object] = {
                    "Bucket": self._bucket,
                    "Prefix": full_prefix,
                }
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                response = await asyncio.to_thread(
                    client.list_objects_v2,  # type: ignore[union-attr]
                    **kwargs,
                )
                objects = response.get("Contents", [])
                if objects:
                    delete_keys = [{"Key": obj["Key"]} for obj in objects]
                    await asyncio.to_thread(
                        client.delete_objects,  # type: ignore[union-attr]
                        Bucket=self._bucket,
                        Delete={"Objects": delete_keys},
                    )
                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")
        except Exception:  # noqa: BLE001
            logger.debug("Failed to delete prefix: %s", prefix)


class GCSPayloadStore(PayloadStore):
    """Google Cloud Storage payload store for GKE deployments.

    Uses the ``google-cloud-storage`` library. TTL is managed via GCS
    lifecycle rules configured on the bucket (not application-side).
    """

    def __init__(self, bucket: str, prefix: str = "payloads") -> None:
        self._bucket_name = bucket
        self._prefix = prefix
        self._client: object | None = None

    def _get_bucket(self) -> object:
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS payload stores. Install with: pip install sie-router[gcs]"
                ) from None

            client = storage.Client()
            self._client = client.bucket(self._bucket_name)
        return self._client

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    async def put(self, key: str, data: bytes) -> None:
        bucket = self._get_bucket()
        blob = bucket.blob(self._full_key(key))  # type: ignore[union-attr]
        await asyncio.to_thread(blob.upload_from_string, data)

    async def get(self, key: str) -> bytes:
        bucket = self._get_bucket()
        blob = bucket.blob(self._full_key(key))  # type: ignore[union-attr]
        try:
            return await asyncio.to_thread(blob.download_as_bytes)
        except Exception as e:
            # Only convert "not found" errors to KeyError; let infra errors propagate.
            _not_found = None
            with contextlib.suppress(ImportError):
                from google.api_core.exceptions import NotFound

                _not_found = NotFound
            if _not_found is not None and isinstance(e, _not_found):
                raise KeyError(f"Payload not found: {key}") from e
            raise

    async def delete(self, key: str) -> None:
        bucket = self._get_bucket()
        blob = bucket.blob(self._full_key(key))  # type: ignore[union-attr]
        try:
            await asyncio.to_thread(blob.delete)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to delete payload: %s", key)

    async def delete_prefix(self, prefix: str) -> None:
        bucket = self._get_bucket()
        full_prefix = self._full_key(prefix)
        try:
            blobs = await asyncio.to_thread(
                lambda: list(bucket.list_blobs(prefix=full_prefix))  # type: ignore[union-attr]
            )
            for blob in blobs:
                try:
                    await asyncio.to_thread(blob.delete)
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception:  # noqa: BLE001
            logger.debug("Failed to delete prefix: %s", prefix)


def create_payload_store(url: str | None) -> PayloadStore | None:
    """Create a payload store from a URL.

    Args:
        url: Payload store URL. Accepts:
            - ``s3://bucket/prefix`` — S3-compatible store
            - ``gs://bucket/prefix`` — Google Cloud Storage
            - Local path — filesystem store
            - ``None`` — no payload store (all payloads inline)

    Returns:
        PayloadStore instance, or None if URL is not set.
    """
    if not url:
        return None

    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "payloads"
        return S3PayloadStore(bucket=bucket, prefix=prefix)

    if url.startswith("gs://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "payloads"
        return GCSPayloadStore(bucket=bucket, prefix=prefix)

    # Reject URLs with unrecognized schemes to catch typos early
    if "://" in url:
        raise ValueError(
            f"Unsupported payload store URL scheme: {url!r}. "
            "Supported: 's3://bucket/prefix', 'gs://bucket/prefix', or a local filesystem path."
        )
    return LocalPayloadStore(base_dir=url)
