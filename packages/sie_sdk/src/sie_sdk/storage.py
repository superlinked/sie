"""Cloud storage abstraction for S3/GCS/local paths.

Provides a unified interface for:
- Detecting storage type from URL (s3://, gs://, local path)
- Listing objects/files in a location
- Downloading files to local cache
- Checking if a path exists

Used by:
- Config discovery (list configs in models_dir)
- Weight caching (download from cluster cache to local cache)
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def list_dirs(self, path: str) -> Iterator[str]:
        """List immediate subdirectories at the given path.

        Args:
            path: Path to list (bucket path for cloud, directory for local).

        Yields:
            Directory names (not full paths).
        """

    @abstractmethod
    def list_files(self, path: str, pattern: str = "*") -> Iterator[str]:
        """List files at the given path matching pattern.

        Args:
            path: Path to list.
            pattern: Glob pattern to match (e.g., "*.yaml").

        Yields:
            File names (not full paths).
        """

    @abstractmethod
    def download_file(self, src: str, dst: Path) -> None:
        """Download a file to local path.

        Args:
            src: Source path (cloud URL or local path).
            dst: Destination local path.
        """

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists.

        Args:
            path: Path to check.

        Returns:
            True if path exists.
        """

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read text content from a file.

        Args:
            path: Path to read.

        Returns:
            File contents as string.
        """

    @abstractmethod
    def upload_file(self, src: Path, dst: str) -> None:
        """Upload a local file to the storage backend.

        Args:
            src: Source local path.
            dst: Destination path (cloud URL or local path).
        """

    @abstractmethod
    def upload_directory(self, src: Path, dst: str) -> int:
        """Upload a local directory recursively to the storage backend.

        Args:
            src: Source local directory.
            dst: Destination path prefix (cloud URL or local path).

        Returns:
            Number of files uploaded.
        """


class LocalBackend(StorageBackend):
    """Local filesystem backend."""

    def list_dirs(self, path: str) -> Iterator[str]:
        """List immediate subdirectories."""
        p = Path(path)
        if not p.exists():
            return
        for item in p.iterdir():
            if item.is_dir():
                yield item.name

    def list_files(self, path: str, pattern: str = "*") -> Iterator[str]:
        """List files matching pattern."""
        p = Path(path)
        if not p.exists():
            return
        for item in p.glob(pattern):
            if item.is_file():
                yield item.name

    def download_file(self, src: str, dst: Path) -> None:
        """Copy a local file."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def exists(self, path: str) -> bool:
        """Check if local path exists."""
        return Path(path).exists()

    def read_text(self, path: str) -> str:
        """Read text from local file."""
        return Path(path).read_text()

    def upload_file(self, src: Path, dst: str) -> None:
        """Copy a local file to destination."""
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_path)

    def upload_directory(self, src: Path, dst: str) -> int:
        """Copy a local directory recursively."""
        dst_path = Path(dst)
        count = 0
        for file in src.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(src)
                target = dst_path / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
                count += 1
        return count


class S3Backend(StorageBackend):
    """AWS S3 backend using boto3 with parallel uploads."""

    # Parallel upload settings
    MAX_CONCURRENCY = 16  # Max concurrent file uploads
    MULTIPART_THRESHOLD = 8 * 1024 * 1024  # 8MB - use multipart above this
    MULTIPART_CHUNKSIZE = 8 * 1024 * 1024  # 8MB chunks

    def __init__(self) -> None:
        self._client: Any = None
        self._transfer_config: Any = None

    def _get_client(self) -> Any:
        """Lazy-init boto3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("s3")
            except ImportError as e:
                msg = "boto3 is required for S3 storage. Install with: pip install boto3"
                raise ImportError(msg) from e
        return self._client

    def _get_transfer_config(self) -> Any:
        """Get boto3 TransferConfig for parallel multipart uploads."""
        if self._transfer_config is None:
            try:
                from boto3.s3.transfer import TransferConfig

                self._transfer_config = TransferConfig(
                    max_concurrency=self.MAX_CONCURRENCY,
                    multipart_threshold=self.MULTIPART_THRESHOLD,
                    multipart_chunksize=self.MULTIPART_CHUNKSIZE,
                )
            except ImportError:
                self._transfer_config = None
        return self._transfer_config

    def _parse_s3_url(self, url: str) -> tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key)."""
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            msg = f"Not an S3 URL: {url}"
            raise ValueError(msg)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    def list_dirs(self, path: str) -> Iterator[str]:
        """List immediate subdirectories in S3 bucket."""
        client = self._get_client()
        bucket, prefix = self._parse_s3_url(path)

        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                # CommonPrefixes returns full prefix, extract dir name
                dir_path = cp["Prefix"].rstrip("/")
                dir_name = dir_path.split("/")[-1]
                yield dir_name

    def list_files(self, path: str, pattern: str = "*") -> Iterator[str]:
        """List files in S3 bucket matching pattern."""
        client = self._get_client()
        bucket, prefix = self._parse_s3_url(path)

        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Only files directly in this prefix (not in subdirs)
                relative = key[len(prefix) :]
                if "/" not in relative:
                    filename = relative
                    if fnmatch.fnmatch(filename, pattern):
                        yield filename

    def download_file(self, src: str, dst: Path) -> None:
        """Download file from S3."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Downloading s3://%s/%s to %s", bucket, key, dst)
        client.download_file(bucket, key, str(dst))

    def exists(self, path: str) -> bool:
        """Check if S3 object exists."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(path)
        try:
            client.head_object(Bucket=bucket, Key=key)
        except client.exceptions.ClientError:
            return False
        return True

    def read_text(self, path: str) -> str:
        """Read text content from S3."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(path)
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def upload_file(self, src: Path, dst: str) -> None:
        """Upload file to S3 with multipart for large files."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(dst)
        logger.debug("Uploading %s to s3://%s/%s", src, bucket, key)
        config = self._get_transfer_config()
        if config:
            client.upload_file(str(src), bucket, key, Config=config)
        else:
            client.upload_file(str(src), bucket, key)

    def upload_directory(self, src: Path, dst: str) -> int:
        """Upload directory recursively to S3 with parallel file uploads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        client = self._get_client()
        bucket, base_key = self._parse_s3_url(dst)
        config = self._get_transfer_config()

        # Collect all files to upload
        files_to_upload = []
        for file in src.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(src)
                key = f"{base_key}/{rel_path}" if base_key else str(rel_path)
                files_to_upload.append((file, key))

        if not files_to_upload:
            return 0

        def upload_one(item: tuple[Path, str]) -> bool:
            file_path, key = item
            logger.debug("Uploading %s to s3://%s/%s", file_path, bucket, key)
            try:
                if config:
                    client.upload_file(str(file_path), bucket, key, Config=config)
                else:
                    client.upload_file(str(file_path), bucket, key)
                return True
            except OSError as e:
                logger.error("Failed to upload %s: %s", file_path, e)
                return False

        # Upload files in parallel
        count = 0
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENCY) as executor:
            futures = {executor.submit(upload_one, f): f for f in files_to_upload}
            for future in as_completed(futures):
                if future.result():
                    count += 1

        return count


class GCSBackend(StorageBackend):
    """Google Cloud Storage backend with parallel uploads."""

    # Parallel upload settings
    MAX_CONCURRENCY = 16  # Max concurrent file uploads

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage

                self._client = storage.Client()
            except ImportError as e:
                msg = "google-cloud-storage is required for GCS. Install with: pip install google-cloud-storage"
                raise ImportError(msg) from e
        return self._client

    def _parse_gcs_url(self, url: str) -> tuple[str, str]:
        """Parse gs://bucket/path into (bucket, path)."""
        parsed = urlparse(url)
        if parsed.scheme != "gs":
            msg = f"Not a GCS URL: {url}"
            raise ValueError(msg)
        bucket = parsed.netloc
        path = parsed.path.lstrip("/")
        return bucket, path

    def list_dirs(self, path: str) -> Iterator[str]:
        """List immediate subdirectories in GCS bucket."""
        client = self._get_client()
        bucket_name, prefix = self._parse_gcs_url(path)

        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, delimiter="/")

        # Must iterate blobs to populate prefixes
        list(blobs)

        for blob_prefix in blobs.prefixes:
            # prefixes returns full prefix, extract dir name
            dir_path = blob_prefix.rstrip("/")
            dir_name = dir_path.split("/")[-1]
            yield dir_name

    def list_files(self, path: str, pattern: str = "*") -> Iterator[str]:
        """List files in GCS bucket matching pattern."""
        client = self._get_client()
        bucket_name, prefix = self._parse_gcs_url(path)

        # Ensure prefix ends with /
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, delimiter="/")

        for blob in blobs:
            # Only files directly in this prefix
            relative = blob.name[len(prefix) :]
            if "/" not in relative and relative and fnmatch.fnmatch(relative, pattern):
                yield relative

    def download_file(self, src: str, dst: Path) -> None:
        """Download file from GCS."""
        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Downloading gs://%s/%s to %s", bucket_name, blob_path, dst)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(str(dst))

    def exists(self, path: str) -> bool:
        """Check if GCS object exists."""
        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()

    def read_text(self, path: str) -> str:
        """Read text content from GCS."""
        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_text()

    def upload_file(self, src: Path, dst: str) -> None:
        """Upload file to GCS."""
        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(dst)
        logger.debug("Uploading %s to gs://%s/%s", src, bucket_name, blob_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(src))

    def upload_directory(self, src: Path, dst: str) -> int:
        """Upload directory recursively to GCS with parallel file uploads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        client = self._get_client()
        bucket_name, base_path = self._parse_gcs_url(dst)
        bucket = client.bucket(bucket_name)

        # Collect all files to upload
        files_to_upload = []
        for file in src.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(src)
                blob_path = f"{base_path}/{rel_path}" if base_path else str(rel_path)
                files_to_upload.append((file, blob_path))

        if not files_to_upload:
            return 0

        def upload_one(item: tuple[Path, str]) -> bool:
            file_path, blob_path = item
            logger.debug("Uploading %s to gs://%s/%s", file_path, bucket_name, blob_path)
            try:
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))
                return True
            except OSError as e:
                logger.error("Failed to upload %s: %s", file_path, e)
                return False

        # Upload files in parallel
        count = 0
        with ThreadPoolExecutor(max_workers=self.MAX_CONCURRENCY) as executor:
            futures = {executor.submit(upload_one, f): f for f in files_to_upload}
            for future in as_completed(futures):
                if future.result():
                    count += 1

        return count


def get_storage_backend(path: str) -> StorageBackend:
    """Get the appropriate storage backend for a path.

    Args:
        path: A local path, S3 URL (s3://...), or GCS URL (gs://...).

    Returns:
        The appropriate StorageBackend instance.
    """
    if path.startswith("s3://"):
        return S3Backend()
    if path.startswith("gs://"):
        return GCSBackend()
    return LocalBackend()


def is_cloud_path(path: str) -> bool:
    """Check if a path is a cloud URL (S3 or GCS).

    Args:
        path: Path to check.

    Returns:
        True if path is an S3 or GCS URL.
    """
    return path.startswith(("s3://", "gs://"))


def join_path(base: str, *parts: str) -> str:
    """Join path components, handling both local and cloud paths.

    Args:
        base: Base path (local, S3, or GCS).
        *parts: Path components to join.

    Returns:
        Joined path.
    """
    if is_cloud_path(base):
        # For cloud paths, use / separator
        base = base.rstrip("/")
        return "/".join([base, *parts])
    # For local paths, use Path
    return str(Path(base).joinpath(*parts))


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory.

    Returns:
        Path to HF_HOME/hub or default ~/.cache/huggingface/hub.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"
