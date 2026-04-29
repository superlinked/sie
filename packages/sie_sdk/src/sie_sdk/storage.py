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
import tempfile
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
