"""Tests for storage backend functionality."""

import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sie_sdk.storage import (
    AzureBlobBackend,
    GCSBackend,
    LocalBackend,
    OSSBackend,
    S3Backend,
    _AlibabaCredentialsProvider,
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


class TestAzureBlobBackend:
    """Tests for ``AzureBlobBackend`` URL parsing and prefix semantics."""

    def _install_fake_blob_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        blob_service_client: type[object],
    ) -> None:
        azure_module = types.ModuleType("azure")
        storage_module = types.ModuleType("azure.storage")
        blob_module = types.ModuleType("azure.storage.blob")
        blob_module.BlobServiceClient = blob_service_client
        monkeypatch.setitem(sys.modules, "azure", azure_module)
        monkeypatch.setitem(sys.modules, "azure.storage", storage_module)
        monkeypatch.setitem(sys.modules, "azure.storage.blob", blob_module)

    def _backend_with_mock_container(self, container_client: MagicMock) -> AzureBlobBackend:
        backend = AzureBlobBackend()
        backend._container_clients[("sieacct", "sie-cache")] = container_client
        return backend

    def test_parse_abfs_url(self) -> None:
        backend = AzureBlobBackend()

        container, account, path, account_url = backend._parse_azure_url(
            "abfs://sie-cache@sieacct.dfs.core.windows.net/models/org/model"
        )

        assert container == "sie-cache"
        assert account == "sieacct"
        assert path == "models/org/model"
        assert account_url == "https://sieacct.blob.core.windows.net"

    def test_parse_abfss_blob_endpoint_url(self) -> None:
        backend = AzureBlobBackend()

        container, account, path, account_url = backend._parse_azure_url(
            "abfss://sie-cache@sieacct.blob.core.windows.net/models"
        )

        assert container == "sie-cache"
        assert account == "sieacct"
        assert path == "models"
        assert account_url == "https://sieacct.blob.core.windows.net"

    def test_parse_requires_container_account_separator(self) -> None:
        backend = AzureBlobBackend()

        with pytest.raises(ValueError, match="must use <container>@<account>"):
            backend._parse_azure_url("abfs://sie-cache/models")

    def test_has_children_returns_true_when_prefix_has_children(self) -> None:
        container_client = MagicMock()
        child = MagicMock()
        child.name = "models/models--BAAI--bge-m3/snapshots/abc/config.json"
        container_client.list_blobs.return_value = iter([child])
        backend = self._backend_with_mock_container(container_client)

        assert (
            backend.has_children("abfs://sie-cache@sieacct.dfs.core.windows.net/models/models--BAAI--bge-m3/snapshots")
            is True
        )

        container_client.list_blobs.assert_called_once_with(
            name_starts_with="models/models--BAAI--bge-m3/snapshots/",
            results_per_page=2,
        )

    def test_has_children_returns_false_when_only_folder_marker_exists(self) -> None:
        container_client = MagicMock()
        marker = MagicMock()
        marker.name = "models/seeded/snapshots/"
        container_client.list_blobs.return_value = iter([marker])
        backend = self._backend_with_mock_container(container_client)

        assert backend.has_children("abfs://sie-cache@sieacct.dfs.core.windows.net/models/seeded/snapshots") is False

    def test_list_dirs_uses_hierarchical_prefixes(self) -> None:
        container_client = MagicMock()
        directory = MagicMock()
        directory.name = "models/models--BAAI--bge-m3/"
        file_blob = MagicMock()
        file_blob.name = "models/config.json"
        container_client.walk_blobs.return_value = iter([directory, file_blob])
        backend = self._backend_with_mock_container(container_client)

        assert list(backend.list_dirs("abfs://sie-cache@sieacct.dfs.core.windows.net/models")) == [
            "models--BAAI--bge-m3"
        ]
        container_client.walk_blobs.assert_called_once_with(name_starts_with="models/", delimiter="/")

    def test_list_files_filters_to_immediate_files(self) -> None:
        container_client = MagicMock()
        direct = MagicMock()
        direct.name = "models/config.yaml"
        nested = MagicMock()
        nested.name = "models/subdir/config.yaml"
        other = MagicMock()
        other.name = "models/readme.md"
        container_client.list_blobs.return_value = iter([direct, nested, other])
        backend = self._backend_with_mock_container(container_client)

        assert list(backend.list_files("abfss://sie-cache@sieacct.dfs.core.windows.net/models", "*.yaml")) == [
            "config.yaml"
        ]

    def test_connection_string_account_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeBlobServiceClient:
            @classmethod
            def from_connection_string(cls, connection_string: str) -> "FakeBlobServiceClient":
                raise AssertionError("mismatched connection string should be rejected before client construction")

        backend = AzureBlobBackend()
        self._install_fake_blob_module(monkeypatch, FakeBlobServiceClient)
        monkeypatch.setenv(
            "AZURE_STORAGE_CONNECTION_STRING",
            "DefaultEndpointsProtocol=https;AccountName=otheracct;AccountKey=fake;EndpointSuffix=core.windows.net",
        )

        with pytest.raises(ValueError, match="does not match URL account"):
            backend._build_container_client(
                "sie-cache",
                "sieacct",
                "https://sieacct.blob.core.windows.net",
            )

    def test_connection_string_matching_account_returns_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        container_client = MagicMock()

        class FakeBlobServiceClient:
            @classmethod
            def from_connection_string(cls, connection_string: str) -> "FakeBlobServiceClient":
                assert "AccountName=sieacct" in connection_string
                return cls()

            def get_container_client(self, container: str) -> MagicMock:
                assert container == "sie-cache"
                return container_client

        backend = AzureBlobBackend()
        self._install_fake_blob_module(monkeypatch, FakeBlobServiceClient)
        monkeypatch.setenv(
            "AZURE_STORAGE_CONNECTION_STRING",
            "DefaultEndpointsProtocol=https;AccountName=sieacct;AccountKey=fake;EndpointSuffix=core.windows.net",
        )

        assert (
            backend._build_container_client(
                "sie-cache",
                "sieacct",
                "https://sieacct.blob.core.windows.net",
            )
            is container_client
        )


class TestOSSBackend:
    """Alibaba OSS uses strict URLs, V4 signing, and provider credentials."""

    @staticmethod
    def _backend_with_bucket(bucket: MagicMock) -> OSSBackend:
        backend = OSSBackend()
        backend._buckets["sie-cache"] = bucket
        return backend

    @pytest.mark.parametrize(
        "url",
        [
            "oss://",
            "oss://ABCD/models",
            "oss://ab/models",
            "oss://user:password@sie-cache/models",
            "oss://sie-cache/models?token=sensitive",
            "oss://sie-cache/models#fragment",
            "oss://sie-cache//models",
            "oss://sie-cache/models/../payloads",
            "oss://sie-cache/models/%2e%2e/payloads",
            "oss://sie-cache/models\\payloads",
            "oss://sie-cache/models with spaces",
            f"oss://sie-cache/{'a' * 1024}",
        ],
    )
    def test_parse_rejects_ambiguous_or_credential_bearing_urls(self, url: str) -> None:
        with pytest.raises(ValueError, match="OSS"):
            OSSBackend._parse_oss_url(url)

    def test_parse_accepts_bucket_root_and_safe_prefix(self) -> None:
        assert OSSBackend._parse_oss_url("oss://sie-cache") == ("sie-cache", "")
        assert OSSBackend._parse_oss_url("oss://sie-cache/models/snapshots/") == (
            "sie-cache",
            "models/snapshots/",
        )

    def test_shared_payload_wire_fixture_matches_python_path_contract(self) -> None:
        fixture_path = Path(__file__).parents[2] / "wire-fixtures" / "oss_payload_store.json"
        fixture = json.loads(fixture_path.read_text())

        bucket, object_key = OSSBackend._parse_oss_url(fixture["full_reference"])
        assert fixture["version"] == 1
        assert fixture["scheme"] == "oss"
        assert bucket == fixture["bucket"]
        assert object_key == fixture["object_key"]
        assert object_key == f"{fixture['prefix']}/{fixture['plain_key']}"
        assert join_path(f"oss://{bucket}", fixture["prefix"], fixture["plain_key"]) == fixture["full_reference"]

    def test_paginated_listing_filters_markers_and_nested_files(self) -> None:
        bucket = MagicMock()
        page_marker = "page-2"
        bucket.list_objects_v2.side_effect = [
            SimpleNamespace(
                prefix_list=["models/first/"],
                object_list=[],
                is_truncated=True,
                next_continuation_token=page_marker,
            ),
            SimpleNamespace(
                prefix_list=["models/second/"],
                object_list=[],
                is_truncated=False,
                next_continuation_token="",
            ),
            SimpleNamespace(
                prefix_list=[],
                object_list=[
                    SimpleNamespace(key="models/"),
                    SimpleNamespace(key="models/config.yaml"),
                    SimpleNamespace(key="models/readme.md"),
                    SimpleNamespace(key="models/nested/ignored.yaml"),
                ],
                is_truncated=False,
                next_continuation_token="",
            ),
        ]
        backend = self._backend_with_bucket(bucket)

        assert list(backend.list_dirs("oss://sie-cache/models")) == ["first", "second"]
        assert list(backend.list_files("oss://sie-cache/models", "*.yaml")) == ["config.yaml"]
        assert bucket.list_objects_v2.call_args_list[0].kwargs["continuation_token"] == ""
        assert bucket.list_objects_v2.call_args_list[1].kwargs["continuation_token"] == page_marker

    def test_has_children_ignores_folder_marker(self) -> None:
        bucket = MagicMock()
        bucket.list_objects_v2.side_effect = [
            SimpleNamespace(object_list=[SimpleNamespace(key="models/snapshots/")]),
            SimpleNamespace(
                object_list=[
                    SimpleNamespace(key="models/snapshots/"),
                    SimpleNamespace(key="models/snapshots/abc/config.json"),
                ]
            ),
        ]
        backend = self._backend_with_bucket(bucket)

        assert backend.has_children("oss://sie-cache/models/snapshots") is False
        assert backend.has_children("oss://sie-cache/models/snapshots") is True
        assert bucket.list_objects_v2.call_args.kwargs == {
            "prefix": "models/snapshots/",
            "delimiter": "",
            "continuation_token": "",
            "max_keys": 2,
        }

    def test_download_exists_and_read_text_use_exact_key(self, tmp_path: Path) -> None:
        bucket = MagicMock()
        bucket.object_exists.return_value = True
        result = MagicMock()
        result.read.return_value = b"model: clip\n"
        bucket.get_object.return_value = result
        backend = self._backend_with_bucket(bucket)
        destination = tmp_path / "nested" / "config.yaml"

        backend.download_file("oss://sie-cache/models/config.yaml", destination)
        assert backend.exists("oss://sie-cache/models/config.yaml") is True
        assert backend.read_text("oss://sie-cache/models/config.yaml") == "model: clip\n"

        bucket.get_object_to_file.assert_called_once_with("models/config.yaml", str(destination))
        bucket.object_exists.assert_called_once_with("models/config.yaml")
        bucket.get_object.assert_called_once_with("models/config.yaml")
        result.close.assert_called_once_with()

    def test_provider_failure_is_sanitized(self) -> None:
        class SecretError(Exception):
            status = 403
            code = "AccessDenied"

            def __str__(self) -> str:
                return "token=projected-secret request-id=raw-id"

        bucket = MagicMock()
        bucket.get_object.side_effect = SecretError()
        backend = self._backend_with_bucket(bucket)

        with pytest.raises(RuntimeError) as exc_info:
            backend.read_text("oss://sie-cache/models/config.yaml")

        message = str(exc_info.value)
        assert message == "Alibaba OSS read failed (status=403, code=AccessDenied)"
        assert "projected-secret" not in message
        assert "raw-id" not in message

    def test_bucket_construction_uses_v4_and_derived_internal_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, object] = {}
        oss2_module = types.ModuleType("oss2")

        def provider_auth_v4(provider: object) -> object:
            calls["provider"] = provider
            return "v4-auth"

        def bucket_factory(auth: object, endpoint: str, bucket_name: str, **kwargs: object) -> object:
            calls["bucket"] = (auth, endpoint, bucket_name, kwargs)
            return object()

        oss2_module.ProviderAuthV4 = provider_auth_v4
        oss2_module.Bucket = bucket_factory
        monkeypatch.setitem(sys.modules, "oss2", oss2_module)
        monkeypatch.setenv("SIE_OSS_REGION", "eu-central-1")
        monkeypatch.setenv("SIE_OSS_USE_INTERNAL_ENDPOINT", "true")

        backend = OSSBackend()
        backend._credential_client = MagicMock()
        backend._get_bucket("sie-cache")

        assert calls["bucket"] == (
            "v4-auth",
            "https://oss-eu-central-1-internal.aliyuncs.com",
            "sie-cache",
            {"app_name": "sie-sdk", "region": "eu-central-1"},
        )
        assert calls["provider"] is backend._credentials_provider
        assert not hasattr(oss2_module, "Auth")
        assert logging.getLogger("oss2.http").getEffectiveLevel() >= logging.WARNING

    def test_endpoint_config_rejects_invalid_region_and_boolean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OSS_REGION", "https://attacker.invalid")
        with pytest.raises(ValueError, match="SIE_OSS_REGION"):
            OSSBackend._resolve_runtime_endpoint()

        monkeypatch.setenv("SIE_OSS_REGION", "eu-central-1")
        monkeypatch.setenv("SIE_OSS_USE_INTERNAL_ENDPOINT", "sometimes")
        with pytest.raises(ValueError, match="SIE_OSS_USE_INTERNAL_ENDPOINT"):
            OSSBackend._resolve_runtime_endpoint()

    def test_partial_rrsa_configuration_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALIBABA_CLOUD_ROLE_ARN", "sensitive-role-value")
        monkeypatch.delenv("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", raising=False)
        monkeypatch.delenv("ALIBABA_CLOUD_OIDC_TOKEN_FILE", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            OSSBackend._build_credential_client()

        message = str(exc_info.value)
        assert "incomplete" in message
        assert "sensitive-role-value" not in message

    def test_complete_rrsa_configuration_selects_explicit_oidc_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        package = types.ModuleType("alibabacloud_credentials")
        client_module = types.ModuleType("alibabacloud_credentials.client")
        models_module = types.ModuleType("alibabacloud_credentials.models")

        class FakeConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeClient:
            def __init__(self, config: FakeConfig | None = None) -> None:
                self.config = config

        client_module.Client = FakeClient
        models_module.Config = FakeConfig
        monkeypatch.setitem(sys.modules, "alibabacloud_credentials", package)
        monkeypatch.setitem(sys.modules, "alibabacloud_credentials.client", client_module)
        monkeypatch.setitem(sys.modules, "alibabacloud_credentials.models", models_module)
        monkeypatch.setenv("ALIBABA_CLOUD_ROLE_ARN", "acs:ram::example:role/workload")
        monkeypatch.setenv("ALIBABA_CLOUD_OIDC_PROVIDER_ARN", "acs:ram::example:oidc-provider/ack")
        monkeypatch.setenv("ALIBABA_CLOUD_OIDC_TOKEN_FILE", "/var/run/secrets/ack/token")
        monkeypatch.setenv("ALIBABA_CLOUD_ROLE_SESSION_NAME", "sie-worker")

        client = OSSBackend._build_credential_client()

        assert client.config is not None
        assert client.config.kwargs == {
            "type": "oidc_role_arn",
            "role_arn": "acs:ram::example:role/workload",
            "oidc_provider_arn": "acs:ram::example:oidc-provider/ack",
            "oidc_token_file_path": "/var/run/secrets/ack/token",
            "role_session_name": "sie-worker",
        }

    def test_credentials_adapter_refreshes_and_sanitizes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        credentials_module = types.ModuleType("oss2.credentials")

        class FakeCredentials:
            def __init__(self, access_key_id: str, access_key_secret: str, security_token: str) -> None:
                self.values = (access_key_id, access_key_secret, security_token)

        credentials_module.Credentials = FakeCredentials
        monkeypatch.setitem(sys.modules, "oss2.credentials", credentials_module)
        client = MagicMock()
        first_values = ("id-1", "secret-1", "token-1")
        second_values = ("id-2", "secret-2", "token-2")
        client.get_credential.side_effect = [
            SimpleNamespace(
                access_key_id=first_values[0],
                access_key_secret=first_values[1],
                security_token=first_values[2],
            ),
            SimpleNamespace(
                access_key_id=second_values[0],
                access_key_secret=second_values[1],
                security_token=second_values[2],
            ),
        ]
        provider = _AlibabaCredentialsProvider(client)

        assert provider.get_credentials().values == first_values
        assert provider.get_credentials().values == second_values
        assert client.get_credential.call_count == 2

        client.get_credential.side_effect = RuntimeError("OIDC token=projected-secret")
        with pytest.raises(RuntimeError, match="could not be resolved") as exc_info:
            provider.get_credentials()
        assert "projected-secret" not in str(exc_info.value)


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_cloud_path(self) -> None:
        """Cloud path detection works correctly."""
        assert is_cloud_path("s3://bucket/key")
        assert is_cloud_path("gs://bucket/key")
        assert is_cloud_path("abfs://container@account.dfs.core.windows.net/key")
        assert is_cloud_path("abfss://container@account.dfs.core.windows.net/key")
        assert is_cloud_path("oss://bucket/key")
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

    def test_join_path_azure(self) -> None:
        """Path joining works for Azure Blob paths."""
        result = join_path("abfs://container@account.dfs.core.windows.net/models/", "dir", "file.txt")
        assert result == "abfs://container@account.dfs.core.windows.net/models/dir/file.txt"

    def test_join_path_oss(self) -> None:
        result = join_path("oss://bucket/models/", "dir", "file.txt")
        assert result == "oss://bucket/models/dir/file.txt"

    def test_get_storage_backend(self) -> None:
        """Backend selection works correctly."""
        assert isinstance(get_storage_backend("/local/path"), LocalBackend)
        # Cloud backends are lazily initialized,
        # so we just check we get the right type
        assert isinstance(get_storage_backend("s3://bucket"), S3Backend)
        assert isinstance(get_storage_backend("gs://bucket"), GCSBackend)
        assert isinstance(get_storage_backend("oss://bucket"), OSSBackend)
        assert isinstance(get_storage_backend("abfs://container@account.dfs.core.windows.net/models"), AzureBlobBackend)
        assert isinstance(
            get_storage_backend("abfss://container@account.dfs.core.windows.net/models"), AzureBlobBackend
        )
