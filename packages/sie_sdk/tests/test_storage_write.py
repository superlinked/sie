"""Tests for write_text and write_text_if_match in storage backends."""

from pathlib import Path

import pytest
from sie_sdk.storage import LocalBackend, StorageBackend


class TestLocalBackendWriteText:
    """Tests for LocalBackend.write_text()."""

    def test_write_text_creates_new_file(self, tmp_path: Path) -> None:
        """write_text creates a file that didn't exist before."""
        backend = LocalBackend()
        target = tmp_path / "new_file.txt"

        backend.write_text(str(target), "hello world")

        assert target.exists()
        assert target.read_text() == "hello world"

    def test_write_text_overwrites_existing_content(self, tmp_path: Path) -> None:
        """write_text replaces previous file content entirely."""
        backend = LocalBackend()
        target = tmp_path / "overwrite.txt"
        target.write_text("old content")

        backend.write_text(str(target), "new content")

        assert target.read_text() == "new content"

    def test_write_text_creates_parent_directories(self, tmp_path: Path) -> None:
        """write_text creates intermediate parent dirs if they don't exist."""
        backend = LocalBackend()
        target = tmp_path / "deeply" / "nested" / "dir" / "file.txt"

        backend.write_text(str(target), "nested content")

        assert target.exists()
        assert target.read_text() == "nested content"


class TestLocalBackendWriteTextIfMatch:
    """Tests for LocalBackend.write_text_if_match() (atomic CAS)."""

    def test_succeeds_when_content_matches(self, tmp_path: Path) -> None:
        """CAS write succeeds when file content matches expected."""
        backend = LocalBackend()
        target = tmp_path / "cas.txt"
        target.write_text("epoch=1")

        result = backend.write_text_if_match(str(target), "epoch=2", "epoch=1")

        assert result is True
        assert target.read_text() == "epoch=2"

    def test_fails_when_content_does_not_match(self, tmp_path: Path) -> None:
        """CAS write fails when file content differs from expected."""
        backend = LocalBackend()
        target = tmp_path / "cas_fail.txt"
        target.write_text("epoch=5")

        result = backend.write_text_if_match(str(target), "epoch=6", "epoch=3")

        assert result is False
        # Content unchanged
        assert target.read_text() == "epoch=5"

    def test_first_write_empty_expected_file_does_not_exist(self, tmp_path: Path) -> None:
        """CAS with empty expected content succeeds when file doesn't exist.

        write_text_if_match creates the file via touch() then checks that
        the content is empty (matching the empty expected_content).
        """
        backend = LocalBackend()
        target = tmp_path / "first_write.txt"

        result = backend.write_text_if_match(str(target), "epoch=1", "")

        assert result is True
        assert target.read_text() == "epoch=1"

    def test_first_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """CAS write creates parent directories for new files."""
        backend = LocalBackend()
        target = tmp_path / "nested" / "dir" / "epoch.txt"

        result = backend.write_text_if_match(str(target), "epoch=1", "")

        assert result is True
        assert target.exists()

    def test_empty_expected_fails_when_file_has_content(self, tmp_path: Path) -> None:
        """CAS with empty expected fails when file already has content."""
        backend = LocalBackend()
        target = tmp_path / "nonempty.txt"
        target.write_text("existing")

        result = backend.write_text_if_match(str(target), "new", "")

        assert result is False
        assert target.read_text() == "existing"  # Unchanged


class TestStorageBackendWriteTextIfMatch:
    """Tests for base StorageBackend.write_text_if_match()."""

    def test_base_class_raises_not_implemented(self) -> None:
        """Base StorageBackend.write_text_if_match raises NotImplementedError."""

        # Create a minimal concrete subclass to test the base method
        class StubBackend(StorageBackend):
            def list_dirs(self, path):
                return iter([])

            def list_files(self, path, pattern="*"):
                return iter([])

            def download_file(self, src, dst):
                pass

            def exists(self, path):
                return False

            def has_children(self, path):
                return False

            def read_text(self, path):
                return ""

            def write_text(self, path, content):
                pass

            def upload_file(self, src, dst):
                pass

            def upload_directory(self, src, dst):
                return 0

        stub = StubBackend()
        with pytest.raises(NotImplementedError, match="StubBackend must override write_text_if_match"):
            stub.write_text_if_match("/some/path", "new", "old")
