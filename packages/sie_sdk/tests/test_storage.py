"""Tests for storage backend functionality."""

from pathlib import Path

from sie_sdk.storage import (
    LocalBackend,
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
