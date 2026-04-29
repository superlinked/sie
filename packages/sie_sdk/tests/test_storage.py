"""Tests for storage backend functionality."""

from pathlib import Path
from unittest.mock import MagicMock

from sie_sdk.storage import (
    GCSBackend,
    LocalBackend,
    S3Backend,
    get_storage_backend,
    is_cloud_path,
    join_path,
)


class TestLocalBackend:
    """Tests for LocalBackend."""

    def test_list_dirs(self, tmp_path: Path) -> None:
        """List directories works correctly."""
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir2").mkdir()
        (tmp_path / "file.txt").write_text("content")

        backend = LocalBackend()
        dirs = list(backend.list_dirs(str(tmp_path)))

        assert set(dirs) == {"dir1", "dir2"}

    def test_list_files(self, tmp_path: Path) -> None:
        """List files works correctly."""
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.yaml").write_text("content2")
        (tmp_path / "subdir").mkdir()

        backend = LocalBackend()
        files = list(backend.list_files(str(tmp_path)))
        yaml_files = list(backend.list_files(str(tmp_path), "*.yaml"))

        assert set(files) == {"file1.txt", "file2.yaml"}
        assert yaml_files == ["file2.yaml"]

    def test_download_file(self, tmp_path: Path) -> None:
        """Download (copy) file works correctly."""
        src = tmp_path / "src" / "file.txt"
        src.parent.mkdir()
        src.write_text("test content")
        dst = tmp_path / "dst" / "copied.txt"

        backend = LocalBackend()
        backend.download_file(str(src), dst)

        assert dst.exists()
        assert dst.read_text() == "test content"

    def test_exists(self, tmp_path: Path) -> None:
        """Exists check works correctly."""
        existing = tmp_path / "exists.txt"
        existing.write_text("content")

        backend = LocalBackend()

        assert backend.exists(str(existing))
        assert not backend.exists(str(tmp_path / "nonexistent.txt"))

    def test_read_text(self, tmp_path: Path) -> None:
        """Read text works correctly."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")

        backend = LocalBackend()
        content = backend.read_text(str(file))

        assert content == "hello world"

    def test_upload_file(self, tmp_path: Path) -> None:
        """Upload (copy) file works correctly."""
        src = tmp_path / "source.txt"
        src.write_text("upload content")
        dst_path = str(tmp_path / "dest" / "uploaded.txt")

        backend = LocalBackend()
        backend.upload_file(src, dst_path)

        assert Path(dst_path).exists()
        assert Path(dst_path).read_text() == "upload content"

    def test_upload_directory(self, tmp_path: Path) -> None:
        """Upload directory recursively works correctly."""
        # Create source directory structure
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "subdir").mkdir()
        (src_dir / "subdir" / "file2.txt").write_text("content2")
        (src_dir / "subdir" / "nested").mkdir()
        (src_dir / "subdir" / "nested" / "file3.txt").write_text("content3")

        dst_dir = str(tmp_path / "destination")

        backend = LocalBackend()
        count = backend.upload_directory(src_dir, dst_dir)

        assert count == 3
        assert Path(dst_dir, "file1.txt").exists()
        assert Path(dst_dir, "file1.txt").read_text() == "content1"
        assert Path(dst_dir, "subdir", "file2.txt").exists()
        assert Path(dst_dir, "subdir", "nested", "file3.txt").exists()


class TestLocalBackendAtomicity:
    """Durability/atomicity properties of `LocalBackend.write_text`."""

    def test_write_text_overwrite_does_not_leave_empty_file_on_simulated_crash(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Atomicity regression: if the write fails mid-flight the original
        content must remain intact, not be truncated to zero bytes.

        Before the tempfile+rename refactor, `Path.write_text` would
        truncate the destination first and then write. A crash in
        between left an empty file on disk; for the epoch file
        specifically that collapsed to `read_epoch() -> 0` and silently
        wedged gateway drift detection (remote==local==0 = "in sync").
        """
        import os

        target = tmp_path / "epoch"
        target.write_text("7")
        assert target.read_text() == "7"

        backend = LocalBackend()
        real_replace = os.replace

        def failing_replace(_src: str, _dst: str) -> None:
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(os, "replace", failing_replace)
        try:
            backend.write_text(str(target), "8")
        except OSError:
            pass
        monkeypatch.setattr(os, "replace", real_replace)

        # Original content is still there — the failed write did not
        # truncate the destination, which is the whole point of the
        # tempfile+rename pattern.
        assert target.read_text() == "7"
        # No leftover temp files should be lying around either.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".epoch.")]
        assert leftovers == [], f"tempfiles leaked: {leftovers}"

    def test_write_text_is_durable(self, tmp_path: Path) -> None:
        """A successful write must be visible with its full content."""
        backend = LocalBackend()
        target = tmp_path / "nested" / "subdir" / "f.txt"
        backend.write_text(str(target), "hello world")
        assert target.exists()
        assert target.read_text() == "hello world"


class TestLocalBackendHasChildren:
    """Tests for ``LocalBackend.has_children`` — the prefix-aware probe."""

    def test_directory_with_files_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "snapshots").mkdir()
        (tmp_path / "snapshots" / "abc123").mkdir()
        (tmp_path / "snapshots" / "abc123" / "config.json").write_text("{}")

        backend = LocalBackend()
        assert backend.has_children(str(tmp_path / "snapshots")) is True

    def test_empty_directory_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "snapshots").mkdir()

        backend = LocalBackend()
        assert backend.has_children(str(tmp_path / "snapshots")) is False

    def test_missing_path_returns_false(self, tmp_path: Path) -> None:
        backend = LocalBackend()
        assert backend.has_children(str(tmp_path / "does-not-exist")) is False

    def test_regular_file_is_not_a_prefix(self, tmp_path: Path) -> None:
        """A leaf file is not a children-bearing prefix."""
        target = tmp_path / "config.json"
        target.write_text("{}")

        backend = LocalBackend()
        assert backend.has_children(str(target)) is False


class TestS3BackendHasChildren:
    """Tests for ``S3Backend.has_children`` — exercises the prefix-vs-key
    divergence that `has_children` exists to fix.

    Why a unit-level test is necessary: ``exists`` calls ``head_object``,
    which on a real S3 bucket returns 404 for a prefix even when
    ``list_objects_v2`` shows children clearly present (S3 has no real
    directories). The cluster-cache reader probed the ``snapshots`` prefix
    with ``exists`` and always failed on real S3, MinIO, Ceph, and R2.
    These tests pin the contract:
    ``has_children`` calls ``list_objects_v2(MaxKeys=2)`` and returns
    ``True`` iff at least one returned object's key differs from the
    normalized prefix — folder-marker objects whose key equals the prefix
    exactly are filtered out.
    """

    def _backend_with_mock_client(self, list_response: dict) -> tuple[S3Backend, MagicMock]:
        backend = S3Backend()
        client = MagicMock()
        client.list_objects_v2.return_value = list_response
        backend._client = client  # bypass lazy init
        return backend, client

    def test_returns_true_when_prefix_has_children(self) -> None:
        backend, client = self._backend_with_mock_client(
            {
                "KeyCount": 1,
                "Contents": [{"Key": "sie-cache-validate/models--BAAI--bge-m3/snapshots/abc/config.json"}],
            }
        )

        assert backend.has_children("s3://sie-cache/sie-cache-validate/models--BAAI--bge-m3/snapshots") is True

        client.list_objects_v2.assert_called_once_with(
            Bucket="sie-cache",
            Prefix="sie-cache-validate/models--BAAI--bge-m3/snapshots/",
            MaxKeys=2,
        )

    def test_returns_false_when_prefix_is_empty(self) -> None:
        backend, _ = self._backend_with_mock_client({"KeyCount": 0})

        assert backend.has_children("s3://sie-cache/empty-prefix") is False

    def test_existing_trailing_slash_is_not_doubled(self) -> None:
        """``has_children`` must accept paths both with and without a trailing slash."""
        backend, client = self._backend_with_mock_client(
            {"KeyCount": 1, "Contents": [{"Key": "seeded/snapshots/abc/config.json"}]}
        )

        assert backend.has_children("s3://sie-cache/seeded/snapshots/") is True
        client.list_objects_v2.assert_called_once_with(
            Bucket="sie-cache",
            Prefix="seeded/snapshots/",  # not 'snapshots//'
            MaxKeys=2,
        )

    def test_does_not_call_head_object(self) -> None:
        """Regression guard: head_object on a prefix returns 404 on real S3
        even when children exist; ``has_children`` must avoid it entirely.
        """
        backend, client = self._backend_with_mock_client(
            {"KeyCount": 1, "Contents": [{"Key": "seeded/snapshots/abc/config.json"}]}
        )

        backend.has_children("s3://sie-cache/seeded/snapshots")

        assert not client.head_object.called

    def test_returns_false_when_only_folder_marker_exists(self) -> None:
        """Regression: a zero-byte object whose key equals the normalized
        prefix is a folder marker (e.g. created by an S3 console "Create
        folder" action), not a real child. ``has_children`` must filter
        these out — otherwise the cluster-cache reader would see an empty
        ``snapshots/`` prefix as populated and try to download nothing.
        """
        backend, client = self._backend_with_mock_client(
            {"KeyCount": 1, "Contents": [{"Key": "seeded/snapshots/", "Size": 0}]}
        )

        assert backend.has_children("s3://sie-cache/seeded/snapshots") is False

        client.list_objects_v2.assert_called_once_with(
            Bucket="sie-cache",
            Prefix="seeded/snapshots/",
            MaxKeys=2,
        )

    def test_returns_true_when_marker_coexists_with_real_child(self) -> None:
        """``MaxKeys=2`` ensures we still see a real child when a folder
        marker is also present. Without this margin (``MaxKeys=1``) the
        single returned key could be the marker, producing a false negative.
        """
        backend, _ = self._backend_with_mock_client(
            {
                "KeyCount": 2,
                "Contents": [
                    {"Key": "seeded/snapshots/", "Size": 0},
                    {"Key": "seeded/snapshots/abc/config.json"},
                ],
            }
        )

        assert backend.has_children("s3://sie-cache/seeded/snapshots") is True


class TestGCSBackendHasChildren:
    """Tests for ``GCSBackend.has_children`` — same contract as S3.

    GCS has the same prefix-vs-blob divergence (``blob.exists()`` on a
    prefix returns False); this test uses a mocked client to assert that
    we list with ``max_results=2`` and return ``True`` iff at least one
    returned blob's name differs from the normalized prefix — folder-marker
    blobs whose name equals the prefix exactly are filtered out.
    """

    def _backend_with_mock_client(self, blobs: list) -> tuple[GCSBackend, MagicMock]:
        backend = GCSBackend()
        client = MagicMock()
        bucket = MagicMock()
        bucket.list_blobs.return_value = iter(blobs)
        client.bucket.return_value = bucket
        backend._client = client  # bypass lazy init
        return backend, client

    def test_returns_true_when_prefix_has_children(self) -> None:
        blob = MagicMock()
        blob.name = "seeded/snapshots/abc/config.json"
        backend, client = self._backend_with_mock_client([blob])

        assert backend.has_children("gs://sie-cache/seeded/snapshots") is True

        client.bucket.assert_called_once_with("sie-cache")
        client.bucket.return_value.list_blobs.assert_called_once_with(
            prefix="seeded/snapshots/",
            max_results=2,
        )

    def test_returns_false_when_prefix_is_empty(self) -> None:
        backend, _ = self._backend_with_mock_client([])

        assert backend.has_children("gs://sie-cache/empty-prefix") is False

    def test_returns_false_when_only_folder_marker_exists(self) -> None:
        """Regression: a zero-byte blob whose name equals the normalized
        prefix is a folder marker (e.g. created by ``gsutil`` placeholder
        semantics or a GCS console folder action), not a real child.
        ``has_children`` must filter these out — otherwise the cluster-cache
        reader would see an empty ``snapshots/`` prefix as populated.
        """
        marker = MagicMock()
        marker.name = "seeded/snapshots/"
        backend, client = self._backend_with_mock_client([marker])

        assert backend.has_children("gs://sie-cache/seeded/snapshots") is False

        client.bucket.return_value.list_blobs.assert_called_once_with(
            prefix="seeded/snapshots/",
            max_results=2,
        )

    def test_returns_true_when_marker_coexists_with_real_child(self) -> None:
        """``max_results=2`` ensures we still see a real child when a folder
        marker is also present. Without this margin (``max_results=1``) the
        iterator could yield only the marker, producing a false negative.
        """
        marker = MagicMock()
        marker.name = "seeded/snapshots/"
        child = MagicMock()
        child.name = "seeded/snapshots/abc/config.json"
        backend, _ = self._backend_with_mock_client([marker, child])

        assert backend.has_children("gs://sie-cache/seeded/snapshots") is True


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_cloud_path(self) -> None:
        """Cloud path detection works correctly."""
        assert is_cloud_path("s3://bucket/key")
        assert is_cloud_path("gs://bucket/key")
        assert not is_cloud_path("/local/path")
        assert not is_cloud_path("./relative/path")

    def test_join_path_local(self) -> None:
        """Path joining works for local paths."""
        result = join_path("/base", "dir", "file.txt")
        assert result == "/base/dir/file.txt"

    def test_join_path_s3(self) -> None:
        """Path joining works for S3 paths."""
        result = join_path("s3://bucket/prefix", "dir", "file.txt")
        assert result == "s3://bucket/prefix/dir/file.txt"

    def test_join_path_gcs(self) -> None:
        """Path joining works for GCS paths."""
        result = join_path("gs://bucket/prefix/", "dir", "file.txt")
        assert result == "gs://bucket/prefix/dir/file.txt"

    def test_get_storage_backend(self) -> None:
        """Backend selection works correctly."""
        assert isinstance(get_storage_backend("/local/path"), LocalBackend)
        # S3 and GCS backends are lazily initialized,
        # so we just check we get the right type
        from sie_sdk.storage import GCSBackend, S3Backend

        assert isinstance(get_storage_backend("s3://bucket"), S3Backend)
        assert isinstance(get_storage_backend("gs://bucket"), GCSBackend)
