"""Cloud storage abstraction for S3/GCS/Azure/local paths.

Provides a unified interface for:
- Detecting storage type from URL (s3://, gs://, abfs://, abfss://, local path)
- Listing objects/files in a location
- Downloading files to local cache
- Checking if a path exists

Used by:
- Config discovery (list configs in models_dir)
- Weight caching (download from cluster cache to local cache)
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CLOUD_SCHEMES = ("s3://", "gs://", "abfs://", "abfss://")
AZURE_SCHEMES = ("abfs", "abfss")


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
    def has_children(self, path: str) -> bool:
        """Check if a prefix has any children.

        Required for cache lookups that probe a directory-like prefix
        (e.g. an HF cache ``snapshots/`` folder). Object stores have no
        real directories: a single ``head_object`` on a prefix returns
        404 even when ``list_objects_v2`` shows children clearly present,
        so ``exists`` cannot be used for this check.

        Args:
            path: Prefix to check (cloud URL or local directory path).

        Returns:
            True if the prefix contains at least one child object/file.
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
    def write_text(self, path: str, content: str) -> None:
        """Write text content to a file.

        Args:
            path: Path to write.
            content: Text content to write.
        """

    def delete_file(self, path: str) -> None:
        """Delete one file/object.

        The operation is idempotent: a missing target is already in the
        requested state. Backends that support mutable storage should override
        this method; the default refusal keeps existing third-party/test
        backends source-compatible until they need deletion.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support file deletion")

    def write_text_if_match(self, path: str, content: str, expected_content: str) -> bool:
        """Conditional write: write only if current content matches expected.

        Used for compare-and-swap on epoch files. Subclasses MUST override
        this method with an atomic implementation appropriate for the backend.

        Args:
            path: Path to write.
            content: New content to write.
            expected_content: Expected current content. Empty string means
                the file must not exist (create-only semantics). Cloud
                backends (S3, GCS) use precondition headers that reject
                writes if the object already exists, even if it is empty.
                Local backends treat a missing file as matching empty
                expected content.

        Returns:
            True if write succeeded, False if content didn't match.

        Raises:
            NotImplementedError: Always. Subclasses must provide atomic CAS.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override write_text_if_match with an atomic implementation"
        )

    def try_server_side_copy(self, src: str, dst: str) -> bool:
        """Attempt a provider-native server-side copy between two URLs.

        Backends that support it (same-provider src/dst, compatible
        accounts/credentials) copy the object inside the provider without
        relaying bytes through this host. The base implementation reports
        "not supported" so callers fall back to download + upload.

        Args:
            src: Source URL.
            dst: Destination URL.

        Returns:
            True if the object was copied server-side. False when no
            native copy applies to this src/dst pair; the caller must
            fall back to a download + upload relay.
        """
        return False

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

    def has_children(self, path: str) -> bool:
        """Check if a local directory contains at least one entry."""
        p = Path(path)
        if not p.is_dir():
            return False
        return next(p.iterdir(), None) is not None

    def read_text(self, path: str) -> str:
        """Read text from local file."""
        return Path(path).read_text()

    def write_text(self, path: str, content: str) -> None:
        """Write text to local file atomically.

        Uses a write-to-temp-then-rename pattern so a crash mid-write can
        never leave the destination truncated or empty. `Path.replace` is
        atomic on POSIX and Windows (NTFS) when source and destination
        are on the same filesystem — which they are here because we put
        the temp file next to the destination. Without this, the naive
        `Path.write_text` truncates-then-writes, and a crash in between
        leaves zero bytes on disk. That is particularly bad for the
        epoch file: `ConfigStore.read_epoch` swallows a malformed int
        as 0, which silently collapses the whole replay-detection
        mechanism downstream (poller would see remote==local==0 and
        declare "in sync" forever).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{p.name}.",
            suffix=".tmp",
            dir=str(p.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(p)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    def delete_file(self, path: str) -> None:
        """Delete one local file if it exists."""
        Path(path).unlink(missing_ok=True)

    def write_text_if_match(self, path: str, content: str, expected_content: str) -> bool:
        """Atomic CAS on local filesystem using file locking."""
        import sys

        p = Path(path)

        if sys.platform == "win32":
            # Windows: use msvcrt for file locking (lock entire file, not just 1 byte)
            import msvcrt

            if not p.exists():
                if expected_content != "":
                    return False  # Expected content but file doesn't exist
                # Create-new case: exclusive create with read-write mode
                # so we can lock before writing.
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with p.open("xb+") as f:
                        # Lock immediately after exclusive create
                        content_bytes = content.encode()
                        file_len = max(len(content_bytes), 1)
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, file_len)
                        try:
                            f.write(content_bytes)
                            f.flush()
                        finally:
                            f.seek(0)
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_len)
                        return True
                except FileExistsError:
                    return False
            else:
                with p.open("r+") as f:
                    # Lock the entire file by determining its size first
                    f.seek(0, 2)  # Seek to end
                    file_len = f.tell() or 1
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, file_len)
                    try:
                        current = f.read()
                        if current != expected_content:
                            return False
                        f.seek(0)
                        f.write(content)
                        f.truncate()
                        return True
                    finally:
                        f.seek(0)
                        new_len = max(file_len, f.seek(0, 2)) or 1
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, new_len)
        else:
            import fcntl

            if not p.exists():
                if expected_content != "":
                    return False  # Expected content but file doesn't exist
                # Create-new case: exclusive create to prevent TOCTOU race
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with p.open("x") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            f.write(content)
                            return True
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)
                except FileExistsError:
                    return False  # Another writer created the file first
            else:
                with p.open("r+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        current = f.read()
                        if current != expected_content:
                            return False
                        f.seek(0)
                        f.write(content)
                        f.truncate()
                        return True
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)

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

    def has_children(self, path: str) -> bool:
        """Check if an S3 prefix has at least one object beneath it.

        Uses ``list_objects_v2`` with ``MaxKeys=2`` because S3 has no real
        directories — ``head_object`` on a prefix returns 404 even when
        children are clearly present (see :py:meth:`StorageBackend.has_children`).

        Folder-marker objects whose key equals the normalized prefix exactly
        (a zero-byte placeholder at e.g. ``snapshots/``) are filtered out:
        they exist as objects but represent no real children. ``MaxKeys=2``
        guarantees that if a real child exists alongside such a marker, the
        single non-marker entry is still visible in the response.
        """
        client = self._get_client()
        bucket, prefix = self._parse_s3_url(path)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=2)
        return any(obj.get("Key") != prefix for obj in response.get("Contents", []))

    def read_text(self, path: str) -> str:
        """Read text content from S3."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(path)
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def write_text(self, path: str, content: str) -> None:
        """Write text content to S3."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(path)
        client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))

    def delete_file(self, path: str) -> None:
        """Delete one S3 object (missing objects are successful no-ops)."""
        client = self._get_client()
        bucket, key = self._parse_s3_url(path)
        client.delete_object(Bucket=bucket, Key=key)

    def write_text_if_match(self, path: str, content: str, expected_content: str) -> bool:
        """Conditional write to S3 using ETags for compare-and-swap.

        Uses S3 conditional writes:
        - If expected_content is empty: use IfNoneMatch='*' (create-only, epoch=0)
        - Otherwise: read current ETag, compare content, use IfMatch for write

        .. warning:: Compatibility

            S3 conditional writes (``IfNoneMatch``, ``IfMatch``) are only
            supported on **general purpose buckets** (available since August
            2024).  On S3-compatible stores (MinIO, Ceph, R2) these headers
            may be silently ignored, degrading CAS to an unconditional
            overwrite.  If you use a non-AWS S3-compatible backend, verify
            that conditional writes are enforced, or fall back to an
            external locking mechanism (e.g. DynamoDB).
        """
        client = self._get_client()
        bucket, key = self._parse_s3_url(path)

        if not expected_content.strip():
            # First epoch (0) — object should not exist yet
            try:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=content.encode("utf-8"),
                    IfNoneMatch="*",
                )
                return True
            except client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ("PreconditionFailed", "ConditionalRequestConflict"):
                    return False
                raise
        else:
            # Read current content and ETag
            try:
                response = client.get_object(Bucket=bucket, Key=key)
                current = response["Body"].read().decode("utf-8")
                etag = response["ETag"]
            except client.exceptions.NoSuchKey:
                return False  # Expected content but object doesn't exist

            if current.strip() != expected_content.strip():
                return False

            # Write with ETag condition
            try:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=content.encode("utf-8"),
                    IfMatch=etag,
                )
                return True
            except client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ("PreconditionFailed", "ConditionalRequestConflict"):
                    return False
                raise

    def try_server_side_copy(self, src: str, dst: str) -> bool:
        """Server-side S3 copy via boto3 managed copy.

        ``client.copy`` issues CopyObject (or UploadPartCopy for objects
        above the multipart threshold), so bytes never leave AWS. Requires
        the same s3:GetObject / s3:PutObject permissions as the relay path.
        """
        if not (src.startswith("s3://") and dst.startswith("s3://")):
            return False
        # If boto3/botocore is missing the relay path cannot work either,
        # so let the ImportError propagate rather than masking it.
        from botocore.exceptions import BotoCoreError, ClientError

        client = self._get_client()
        src_bucket, src_key = self._parse_s3_url(src)
        dst_bucket, dst_key = self._parse_s3_url(dst)
        logger.debug("Server-side copy s3://%s/%s to s3://%s/%s", src_bucket, src_key, dst_bucket, dst_key)
        config = self._get_transfer_config()
        copy_source = {"Bucket": src_bucket, "Key": src_key}
        try:
            if config:
                client.copy(copy_source, dst_bucket, dst_key, Config=config)
            else:
                client.copy(copy_source, dst_bucket, dst_key)
        except (BotoCoreError, ClientError) as e:
            logger.warning(
                "S3 server-side copy of %s failed (%s); falling back to download+upload",
                src,
                e,
            )
            return False
        return True

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

    def has_children(self, path: str) -> bool:
        """Check if a GCS prefix has at least one blob beneath it.

        Lists with ``max_results=2`` to avoid materialising the full page
        when only existence is needed.

        Folder-marker blobs whose name equals the normalized prefix exactly
        (a zero-byte placeholder at e.g. ``snapshots/``) are filtered out:
        they exist as blobs but represent no real children. ``max_results=2``
        guarantees that if a real child exists alongside such a marker, the
        single non-marker entry is still visible in the response.
        """
        client = self._get_client()
        bucket_name, prefix = self._parse_gcs_url(path)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, max_results=2)
        return any(blob.name != prefix for blob in blobs)

    def read_text(self, path: str) -> str:
        """Read text content from GCS."""
        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_text()

    def write_text(self, path: str, content: str) -> None:
        """Write text content to GCS."""
        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(content, content_type="text/plain")

    def delete_file(self, path: str) -> None:
        """Delete one GCS object if it exists."""
        from google.api_core.exceptions import NotFound

        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(path)
        bucket = client.bucket(bucket_name)
        try:
            bucket.blob(blob_path).delete()
        except NotFound:
            # Missing objects are the successful no-op case for idempotent deletion.
            pass

    def write_text_if_match(self, path: str, content: str, expected_content: str) -> bool:
        """Conditional write to GCS using generation-based preconditions.

        Uses GCS generation numbers:
        - If expected_content is empty: use if_generation_match=0 (create-only, epoch=0)
        - Otherwise: read current generation, compare content, use if_generation_match for write
        """
        from google.api_core.exceptions import NotFound, PreconditionFailed

        client = self._get_client()
        bucket_name, blob_path = self._parse_gcs_url(path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not expected_content.strip():
            # First epoch (0) — object should not exist yet
            try:
                blob.upload_from_string(
                    content,
                    content_type="text/plain",
                    if_generation_match=0,
                )
                return True
            except PreconditionFailed:
                return False
        else:
            # Read current content and generation atomically.
            # blob.reload() fetches metadata (including generation), then
            # download_as_text() reads the object body.  If the object is
            # mutated between these two calls the generation captured here
            # will be stale and the conditional upload below will correctly
            # fail with PreconditionFailed, preserving CAS semantics.
            try:
                blob.reload()
                generation = blob.generation
                current = blob.download_as_text()
            except NotFound:
                return False

            if current.strip() != expected_content.strip():
                return False

            try:
                blob.upload_from_string(
                    content,
                    content_type="text/plain",
                    if_generation_match=generation,
                )
                return True
            except PreconditionFailed:
                return False

    def try_server_side_copy(self, src: str, dst: str) -> bool:
        """Server-side GCS copy via the rewrite API.

        ``Blob.rewrite`` copies inside GCS (cross-bucket included) and
        chunks arbitrarily large objects through rewrite tokens, so bytes
        never transit this host.
        """
        if not (src.startswith("gs://") and dst.startswith("gs://")):
            return False
        from google.api_core.exceptions import GoogleAPICallError

        client = self._get_client()
        src_bucket_name, src_path = self._parse_gcs_url(src)
        dst_bucket_name, dst_path = self._parse_gcs_url(dst)
        logger.debug(
            "Server-side copy gs://%s/%s to gs://%s/%s",
            src_bucket_name,
            src_path,
            dst_bucket_name,
            dst_path,
        )
        src_blob = client.bucket(src_bucket_name).blob(src_path)
        dst_blob = client.bucket(dst_bucket_name).blob(dst_path)
        token = None
        try:
            while True:
                token, _bytes_rewritten, _total_bytes = dst_blob.rewrite(src_blob, token=token)
                if token is None:
                    return True
        except GoogleAPICallError as e:
            logger.warning(
                "GCS server-side copy of %s failed (%s); falling back to download+upload",
                src,
                e,
            )
            return False

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


class AzureBlobBackend(StorageBackend):
    """Azure Blob / ADLS Gen2 backend with parallel uploads."""

    MAX_CONCURRENCY = 16

    # Same-account server-side copies are usually instantaneous, but the
    # Copy Blob operation is asynchronous by contract: poll, then abort and
    # fall back to the download + upload relay if it never settles.
    SERVER_COPY_POLL_INTERVAL_S = 2.0
    SERVER_COPY_TIMEOUT_S = 900.0

    def __init__(self) -> None:
        self._container_clients: dict[tuple[str, str], Any] = {}
        # account -> ("key" | "sas" | "token", credential) describing how the
        # account was authorized; consulted to authorize the *source* blob of
        # a server-side copy (the destination credential never carries over).
        self._account_auth: dict[str, tuple[str, Any]] = {}

    def _parse_azure_url(self, url: str) -> tuple[str, str, str, str]:
        """Parse abfs(s)://container@account.dfs.core.windows.net/path.

        Returns:
            Tuple of (container, account, blob_path, account_url).
        """
        parsed = urlparse(url)
        if parsed.scheme not in AZURE_SCHEMES:
            msg = f"Not an Azure Blob URL: {url}"
            raise ValueError(msg)

        authority = parsed.netloc
        if "@" not in authority:
            msg = "Azure Blob URL must use <container>@<account>.dfs.core.windows.net"
            raise ValueError(msg)

        container, host = authority.split("@", 1)
        if not container or not host:
            msg = "Azure Blob URL container and account must be non-empty"
            raise ValueError(msg)

        account = host
        if account.endswith(".dfs.core.windows.net"):
            account = account[: -len(".dfs.core.windows.net")]
        elif account.endswith(".blob.core.windows.net"):
            account = account[: -len(".blob.core.windows.net")]

        if not account:
            msg = "Azure Blob URL account must be non-empty"
            raise ValueError(msg)

        account_url = f"https://{account}.blob.core.windows.net"
        blob_path = parsed.path.lstrip("/")
        return container, account, blob_path, account_url

    @staticmethod
    def _connection_string_account_name(connection_string: str) -> str | None:
        """Extract the storage account name from an Azure connection string."""
        parts = dict(part.split("=", 1) for part in connection_string.split(";") if "=" in part)
        account = parts.get("AccountName")
        if account:
            return account

        blob_endpoint = parts.get("BlobEndpoint")
        if not blob_endpoint:
            return None

        host = urlparse(blob_endpoint).netloc
        if host.endswith(".blob.core.windows.net"):
            return host[: -len(".blob.core.windows.net")]
        return host.split(".", 1)[0] if host else None

    def _build_container_client(self, container: str, account: str, account_url: str) -> Any:
        """Build an Azure container client from standard environment credentials."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            msg = (
                "azure-storage-blob is required for Azure Blob storage. "
                "Install with: pip install azure-storage-blob azure-identity"
            )
            raise ImportError(msg) from e

        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            conn_account = self._connection_string_account_name(connection_string)
            if conn_account and conn_account.lower() != account.lower():
                msg = (
                    "AZURE_STORAGE_CONNECTION_STRING AccountName "
                    f"({conn_account}) does not match URL account ({account})"
                )
                raise ValueError(msg)
            service_client = BlobServiceClient.from_connection_string(connection_string)
            self._account_auth[account] = ("key", None)
            return service_client.get_container_client(container)

        credential: Any
        account_key = (
            os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            or os.environ.get("AZURE_STORAGE_ACCESS_KEY")
            or os.environ.get("AZURE_STORAGE_KEY")
        )
        sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
        if account_key:
            credential = account_key
            self._account_auth[account] = ("key", None)
        elif sas_token:
            credential = sas_token.lstrip("?")
            self._account_auth[account] = ("sas", credential)
        else:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as e:
                msg = (
                    "azure-identity is required for Azure Blob storage without "
                    "AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_ACCOUNT_KEY, "
                    "or AZURE_STORAGE_SAS_TOKEN"
                )
                raise ImportError(msg) from e
            credential = DefaultAzureCredential()
            self._account_auth[account] = ("token", credential)

        service_client = BlobServiceClient(account_url=account_url, credential=credential)
        return service_client.get_container_client(container)

    def _get_container_client(self, url: str) -> tuple[Any, str, str, str]:
        """Return (container client, container, account, blob path) for an abfs(s) URL."""
        container, account, blob_path, account_url = self._parse_azure_url(url)
        key = (account, container)
        if key not in self._container_clients:
            self._container_clients[key] = self._build_container_client(container, account, account_url)
        return self._container_clients[key], container, account, blob_path

    @staticmethod
    def _ensure_prefix(path: str) -> str:
        if path and not path.endswith("/"):
            return f"{path}/"
        return path

    def list_dirs(self, path: str) -> Iterator[str]:
        """List immediate virtual directories in an Azure container."""
        container_client, _, _, prefix = self._get_container_client(path)
        prefix = self._ensure_prefix(prefix)

        for item in container_client.walk_blobs(name_starts_with=prefix, delimiter="/"):
            name = getattr(item, "name", "")
            if not name.endswith("/"):
                continue
            relative = name[len(prefix) :].strip("/")
            if relative and "/" not in relative:
                yield relative

    def list_files(self, path: str, pattern: str = "*") -> Iterator[str]:
        """List files in an Azure container matching pattern."""
        container_client, _, _, prefix = self._get_container_client(path)
        prefix = self._ensure_prefix(prefix)

        for blob in container_client.list_blobs(name_starts_with=prefix):
            name = blob.name
            relative = name[len(prefix) :]
            if "/" not in relative and relative and fnmatch.fnmatch(relative, pattern):
                yield relative

    def download_file(self, src: str, dst: Path) -> None:
        """Download a file from Azure Blob storage."""
        container_client, container, account, blob_path = self._get_container_client(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Downloading abfs://%s@%s/%s to %s", container, account, blob_path, dst)
        blob_client = container_client.get_blob_client(blob_path)
        with dst.open("wb") as file:
            blob_client.download_blob().readinto(file)

    def exists(self, path: str) -> bool:
        """Check if an Azure blob exists."""
        container_client, _, _, blob_path = self._get_container_client(path)
        return container_client.get_blob_client(blob_path).exists()

    def has_children(self, path: str) -> bool:
        """Check if an Azure virtual directory has at least one blob beneath it."""
        container_client, _, _, prefix = self._get_container_client(path)
        prefix = self._ensure_prefix(prefix)

        for seen, blob in enumerate(container_client.list_blobs(name_starts_with=prefix, results_per_page=2), start=1):
            if blob.name != prefix:
                return True
            if seen >= 2:
                break
        return False

    def read_text(self, path: str) -> str:
        """Read text content from Azure Blob storage."""
        container_client, _, _, blob_path = self._get_container_client(path)
        return container_client.get_blob_client(blob_path).download_blob().readall().decode("utf-8")

    def write_text(self, path: str, content: str) -> None:
        """Write text content to Azure Blob storage."""
        container_client, _, _, blob_path = self._get_container_client(path)
        container_client.get_blob_client(blob_path).upload_blob(content.encode("utf-8"), overwrite=True)

    def delete_file(self, path: str) -> None:
        """Delete one Azure blob if it exists."""
        try:
            from azure.core.exceptions import ResourceNotFoundError
        except ImportError as e:
            msg = "azure-core is required for Azure Blob deletion"
            raise ImportError(msg) from e

        container_client, _, _, blob_path = self._get_container_client(path)
        try:
            container_client.get_blob_client(blob_path).delete_blob()
        except ResourceNotFoundError:
            # Missing blobs are the successful no-op case for idempotent deletion.
            pass

    def write_text_if_match(self, path: str, content: str, expected_content: str) -> bool:
        """Conditional write to Azure Blob storage using ETags for compare-and-swap."""
        try:
            from azure.core import MatchConditions
            from azure.core.exceptions import ResourceExistsError, ResourceModifiedError, ResourceNotFoundError
        except ImportError as e:
            msg = "azure-core is required for Azure Blob conditional writes"
            raise ImportError(msg) from e

        container_client, _, _, blob_path = self._get_container_client(path)
        blob_client = container_client.get_blob_client(blob_path)

        if not expected_content.strip():
            try:
                blob_client.upload_blob(content.encode("utf-8"), overwrite=False)
                return True
            except ResourceExistsError:
                return False

        try:
            properties = blob_client.get_blob_properties()
            current = blob_client.download_blob().readall().decode("utf-8")
        except ResourceNotFoundError:
            return False

        if current.strip() != expected_content.strip():
            return False

        try:
            blob_client.upload_blob(
                content.encode("utf-8"),
                overwrite=True,
                etag=properties.etag,
                match_condition=MatchConditions.IfNotModified,
            )
            return True
        except ResourceModifiedError:
            return False

    def try_server_side_copy(self, src: str, dst: str) -> bool:
        """Server-side Azure copy within a single storage account.

        Uses the asynchronous Copy Blob From URL operation and polls until
        it settles. Cross-account copies need source SAS provisioning, so
        they report unsupported and take the download + upload relay. Any
        copy-specific failure (e.g. CannotVerifyCopySource) also falls back
        rather than failing the sync.
        """
        azure_url_schemes = ("abfs://", "abfss://")
        if not (src.startswith(azure_url_schemes) and dst.startswith(azure_url_schemes)):
            return False
        _, src_account, _, _ = self._parse_azure_url(src)
        _, dst_account, _, _ = self._parse_azure_url(dst)
        if src_account.lower() != dst_account.lower():
            return False

        from azure.core.exceptions import HttpResponseError

        src_container_client, src_container, _, src_path = self._get_container_client(src)
        dst_container_client, dst_container, _, dst_path = self._get_container_client(dst)
        src_blob = src_container_client.get_blob_client(src_path)
        dst_blob = dst_container_client.get_blob_client(dst_path)

        # Authorize the service-side read of the source. Shared-key /
        # connection-string auth covers same-account sources implicitly;
        # SAS must ride on the source URL; OAuth needs an explicit bearer
        # via x-ms-copy-source-authorization.
        source_url = src_blob.url
        copy_kwargs: dict[str, Any] = {}
        kind, credential = self._account_auth.get(src_account, ("key", None))
        if kind == "sas":
            # BlobClient.url already carries the SAS when the client was
            # built from a SAS credential; only append when it does not.
            if "?" not in source_url:
                source_url = f"{source_url}?{credential}"
        elif kind == "token":
            token = credential.get_token("https://storage.azure.com/.default")
            copy_kwargs["source_authorization"] = f"Bearer {token.token}"

        logger.debug(
            "Server-side copy abfs://%s@%s/%s to abfs://%s@%s/%s",
            src_container,
            src_account,
            src_path,
            dst_container,
            dst_account,
            dst_path,
        )
        copy_started: Any = None
        try:
            copy_started = dst_blob.start_copy_from_url(source_url, **copy_kwargs)
            deadline = time.monotonic() + self.SERVER_COPY_TIMEOUT_S
            while True:
                copy_props = dst_blob.get_blob_properties().copy
                if copy_props.status == "success":
                    return True
                if copy_props.status != "pending":
                    logger.warning(
                        "Azure server-side copy of %s ended with status %r; falling back to download+upload",
                        src,
                        copy_props.status,
                    )
                    return False
                if time.monotonic() >= deadline:
                    dst_blob.abort_copy(copy_props.id)
                    logger.warning(
                        "Azure server-side copy of %s timed out after %.0fs; falling back to download+upload",
                        src,
                        self.SERVER_COPY_TIMEOUT_S,
                    )
                    return False
                time.sleep(self.SERVER_COPY_POLL_INTERVAL_S)
        except HttpResponseError as e:
            # A copy that is still pending blocks the relay's overwrite, so
            # abort it best-effort before falling back.
            copy_id = copy_started.get("copy_id") if copy_started else None
            if copy_id:
                with contextlib.suppress(HttpResponseError):
                    dst_blob.abort_copy(copy_id)
            logger.warning(
                "Azure server-side copy of %s failed (%s); falling back to download+upload",
                src,
                e,
            )
            return False

    def upload_file(self, src: Path, dst: str) -> None:
        """Upload file to Azure Blob storage."""
        container_client, container, account, blob_path = self._get_container_client(dst)
        logger.debug("Uploading %s to abfs://%s@%s/%s", src, container, account, blob_path)
        with src.open("rb") as file:
            container_client.get_blob_client(blob_path).upload_blob(file, overwrite=True)

    def upload_directory(self, src: Path, dst: str) -> int:
        """Upload directory recursively to Azure Blob storage with parallel file uploads."""
        container_client, container, account, base_path = self._get_container_client(dst)

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
            logger.debug("Uploading %s to abfs://%s@%s/%s", file_path, container, account, blob_path)
            try:
                with file_path.open("rb") as file:
                    container_client.get_blob_client(blob_path).upload_blob(file, overwrite=True)
                return True
            except OSError as e:
                logger.error("Failed to upload %s: %s", file_path, e)
                return False

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
        path: A local path, S3 URL (s3://...), GCS URL (gs://...), or Azure URL (abfs(s)://...).

    Returns:
        The appropriate StorageBackend instance.
    """
    if path.startswith("s3://"):
        return S3Backend()
    if path.startswith("gs://"):
        return GCSBackend()
    if path.startswith(("abfs://", "abfss://")):
        return AzureBlobBackend()
    return LocalBackend()


def is_cloud_path(path: str) -> bool:
    """Check if a path is a cloud URL (S3, GCS, or Azure Blob).

    Args:
        path: Path to check.

    Returns:
        True if path is an S3, GCS, or Azure Blob URL.
    """
    return path.startswith(CLOUD_SCHEMES)


def join_path(base: str, *parts: str) -> str:
    """Join path components, handling both local and cloud paths.

    Args:
        base: Base path (local, S3, GCS, or Azure Blob).
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
