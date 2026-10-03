"""Tests for StorageBackend.try_server_side_copy implementations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from azure.core.exceptions import HttpResponseError
from botocore.exceptions import ClientError
from google.api_core.exceptions import Forbidden
from sie_sdk.storage import AzureBlobBackend, GCSBackend, LocalBackend, OSSBackend, S3Backend


class FakeS3Client:
    def __init__(self) -> None:
        self.copies: list[tuple[dict[str, str], str, str]] = []

    def copy(self, copy_source: dict[str, str], bucket: str, key: str, Config: Any = None) -> None:  # noqa: N803
        self.copies.append((copy_source, bucket, key))


class TestS3ServerSideCopy:
    def test_cross_provider_is_unsupported(self) -> None:
        backend = S3Backend()
        assert backend.try_server_side_copy("s3://bucket/a.bin", "gs://bucket/a.bin") is False
        assert backend.try_server_side_copy("gs://bucket/a.bin", "s3://bucket/a.bin") is False
        # Unsupported pairs must not touch (or lazily build) the client.
        assert backend._client is None

    def test_same_provider_issues_copy_object(self) -> None:
        backend = S3Backend()
        client = FakeS3Client()
        backend._client = client

        result = backend.try_server_side_copy("s3://src-bucket/path/a.bin", "s3://dst-bucket/other/a.bin")

        assert result is True
        assert client.copies == [({"Bucket": "src-bucket", "Key": "path/a.bin"}, "dst-bucket", "other/a.bin")]

    def test_client_error_falls_back(self) -> None:
        backend = S3Backend()

        class FailingS3Client:
            def copy(self, copy_source: dict[str, str], bucket: str, key: str, Config: Any = None) -> None:  # noqa: N803
                raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "CopyObject")

        backend._client = FailingS3Client()

        assert backend.try_server_side_copy("s3://src-bucket/a.bin", "s3://dst-bucket/a.bin") is False


class FakeGcsBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.rewrite_calls: list[tuple[str, str | None]] = []

    def rewrite(self, source: FakeGcsBlob, token: str | None = None) -> tuple[str | None, int, int]:
        self.rewrite_calls.append((source.name, token))
        # First call returns a continuation token, second completes — covers
        # the multi-step rewrite loop for large objects.
        next_token = "tok-1" if token is None else None
        return (next_token, 100, 200)


class FakeGcsBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, FakeGcsBlob] = {}

    def blob(self, path: str) -> FakeGcsBlob:
        return self.blobs.setdefault(path, FakeGcsBlob(path))


class FakeGcsClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeGcsBucket] = {}

    def bucket(self, name: str) -> FakeGcsBucket:
        return self.buckets.setdefault(name, FakeGcsBucket(name))


class TestGcsServerSideCopy:
    def test_cross_provider_is_unsupported(self) -> None:
        backend = GCSBackend()
        assert backend.try_server_side_copy("gs://bucket/a.bin", "s3://bucket/a.bin") is False
        assert backend._client is None

    def test_same_provider_loops_rewrite_until_done(self) -> None:
        backend = GCSBackend()
        client = FakeGcsClient()
        backend._client = client

        result = backend.try_server_side_copy("gs://src-bucket/path/a.bin", "gs://dst-bucket/other/a.bin")

        assert result is True
        dst_blob = client.buckets["dst-bucket"].blobs["other/a.bin"]
        assert dst_blob.rewrite_calls == [("path/a.bin", None), ("path/a.bin", "tok-1")]

    def test_api_error_falls_back(self) -> None:
        backend = GCSBackend()
        client = FakeGcsClient()
        backend._client = client

        def raise_forbidden(source: FakeGcsBlob, token: str | None = None) -> tuple[str | None, int, int]:
            raise Forbidden("denied")

        client.bucket("dst-bucket").blob("other/a.bin").rewrite = raise_forbidden  # type: ignore[method-assign]

        assert backend.try_server_side_copy("gs://src-bucket/path/a.bin", "gs://dst-bucket/other/a.bin") is False


class FakeAzureBlobClient:
    def __init__(self, url: str, statuses: list[str]) -> None:
        self.url = url
        self._statuses = statuses
        self.copy_calls: list[tuple[str, dict[str, Any]]] = []
        self.aborted: list[str] = []
        self.properties_error: Exception | None = None

    def start_copy_from_url(self, source_url: str, **kwargs: Any) -> dict[str, Any]:
        self.copy_calls.append((source_url, kwargs))
        return {"copy_id": "copy-1", "copy_status": "pending"}

    def get_blob_properties(self) -> Any:
        if self.properties_error is not None:
            raise self.properties_error
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return SimpleNamespace(copy=SimpleNamespace(status=status, id="copy-1"))

    def abort_copy(self, copy_id: str) -> None:
        self.aborted.append(copy_id)


class FakeAzureContainerClient:
    def __init__(self, account: str, container: str, statuses: list[str] | None = None) -> None:
        self.account = account
        self.container = container
        self.statuses = statuses if statuses is not None else ["success"]
        self.blob_clients: dict[str, FakeAzureBlobClient] = {}

    def get_blob_client(self, path: str) -> FakeAzureBlobClient:
        url = f"https://{self.account}.blob.core.windows.net/{self.container}/{path}"
        return self.blob_clients.setdefault(path, FakeAzureBlobClient(url, self.statuses))


class FakeTokenCredential:
    def get_token(self, scope: str) -> Any:
        return SimpleNamespace(token="tok-123")  # noqa: S106 - test fixture, not a secret


def _azure_backend_with_fakes(
    statuses: list[str] | None = None,
) -> tuple[AzureBlobBackend, FakeAzureContainerClient, FakeAzureContainerClient]:
    backend = AzureBlobBackend()
    backend.SERVER_COPY_POLL_INTERVAL_S = 0.0  # type: ignore[misc]
    src_container = FakeAzureContainerClient("sieacct", "source")
    dst_container = FakeAzureContainerClient("sieacct", "dest", statuses=statuses)
    backend._container_clients[("sieacct", "source")] = src_container
    backend._container_clients[("sieacct", "dest")] = dst_container
    return backend, src_container, dst_container


class TestAzureServerSideCopy:
    SRC = "abfs://source@sieacct.dfs.core.windows.net/models/a.bin"
    DST = "abfss://dest@sieacct.dfs.core.windows.net/models/a.bin"

    def test_cross_provider_is_unsupported(self) -> None:
        backend = AzureBlobBackend()
        assert backend.try_server_side_copy("abfs://c@a.dfs.core.windows.net/x", "s3://bucket/x") is False
        assert backend._container_clients == {}

    def test_cross_account_is_unsupported(self) -> None:
        backend = AzureBlobBackend()
        result = backend.try_server_side_copy(
            "abfs://c@account-one.dfs.core.windows.net/x",
            "abfs://c@account-two.dfs.core.windows.net/x",
        )
        assert result is False
        # Decided from URL parsing alone — no clients built.
        assert backend._container_clients == {}

    def test_same_account_copy_succeeds(self) -> None:
        backend, src_container, dst_container = _azure_backend_with_fakes()

        result = backend.try_server_side_copy(self.SRC, self.DST)

        assert result is True
        dst_blob = dst_container.blob_clients["models/a.bin"]
        ((source_url, kwargs),) = dst_blob.copy_calls
        assert source_url == src_container.blob_clients["models/a.bin"].url
        assert kwargs == {}

    def test_pending_copy_polls_until_success(self) -> None:
        backend, _, _ = _azure_backend_with_fakes(statuses=["pending", "success"])

        assert backend.try_server_side_copy(self.SRC, self.DST) is True

    def test_failed_copy_falls_back(self) -> None:
        backend, _, _ = _azure_backend_with_fakes(statuses=["failed"])

        assert backend.try_server_side_copy(self.SRC, self.DST) is False

    def test_http_error_falls_back(self) -> None:
        backend, _, dst_container = _azure_backend_with_fakes()

        def raise_http_error(source_url: str, **kwargs: Any) -> None:
            raise HttpResponseError(message="CannotVerifyCopySource")

        dst_blob = dst_container.get_blob_client("models/a.bin")
        dst_blob.start_copy_from_url = raise_http_error  # type: ignore[method-assign]

        assert backend.try_server_side_copy(self.SRC, self.DST) is False
        # No copy ever started — nothing to abort.
        assert dst_blob.aborted == []

    def test_http_error_after_copy_started_aborts_pending_copy(self) -> None:
        backend, _, dst_container = _azure_backend_with_fakes()
        dst_blob = dst_container.get_blob_client("models/a.bin")
        dst_blob.properties_error = HttpResponseError(message="transient poll failure")

        assert backend.try_server_side_copy(self.SRC, self.DST) is False
        # The in-flight copy must be aborted so the relay's overwrite
        # is not rejected with a pending-copy conflict.
        assert dst_blob.aborted == ["copy-1"]

    def test_token_credential_sets_source_authorization(self) -> None:
        backend, _, dst_container = _azure_backend_with_fakes()
        backend._account_auth["sieacct"] = ("token", FakeTokenCredential())

        assert backend.try_server_side_copy(self.SRC, self.DST) is True
        ((_, kwargs),) = dst_container.blob_clients["models/a.bin"].copy_calls
        assert kwargs == {"source_authorization": "Bearer tok-123"}

    def test_sas_credential_rides_on_source_url(self) -> None:
        backend, src_container, dst_container = _azure_backend_with_fakes()
        backend._account_auth["sieacct"] = ("sas", "sig=abc")

        assert backend.try_server_side_copy(self.SRC, self.DST) is True
        ((source_url, kwargs),) = dst_container.blob_clients["models/a.bin"].copy_calls
        assert source_url == f"{src_container.blob_clients['models/a.bin'].url}?sig=abc"
        assert kwargs == {}

    def test_sas_already_on_source_url_is_not_double_appended(self) -> None:
        backend, src_container, dst_container = _azure_backend_with_fakes()
        backend._account_auth["sieacct"] = ("sas", "sig=abc")
        # BlobClient.url carries the SAS when the client was built from a
        # SAS credential — the copy must not append it a second time.
        src_blob = src_container.get_blob_client("models/a.bin")
        src_blob.url += "?sig=abc"

        assert backend.try_server_side_copy(self.SRC, self.DST) is True
        ((source_url, _),) = dst_container.blob_clients["models/a.bin"].copy_calls
        assert source_url == src_blob.url
        assert source_url.count("?") == 1


class TestOSSServerSideCopy:
    def test_cross_provider_is_unsupported(self) -> None:
        backend = OSSBackend()

        assert backend.try_server_side_copy("oss://source/models/a.bin", "s3://dest/models/a.bin") is False
        assert backend._buckets == {}

    def test_same_region_copy_uses_destination_bucket(self) -> None:
        backend = OSSBackend()
        destination = FakeOSSBucket()
        backend._buckets["dest-bucket"] = destination

        assert (
            backend.try_server_side_copy(
                "oss://source-bucket/models/a.bin",
                "oss://dest-bucket/other/a.bin",
            )
            is True
        )
        assert destination.copies == [("source-bucket", "models/a.bin", "other/a.bin")]

    def test_provider_error_falls_back_without_exposing_secrets(self, caplog: Any) -> None:
        class SecretProviderError(Exception):
            status = 403
            code = "AccessDenied"

            def __str__(self) -> str:
                return "Authorization=secret-token request-id=private"

        backend = OSSBackend()
        destination = FakeOSSBucket()
        destination.error = SecretProviderError()
        backend._buckets["dest-bucket"] = destination

        assert (
            backend.try_server_side_copy(
                "oss://source-bucket/models/a.bin",
                "oss://dest-bucket/other/a.bin",
            )
            is False
        )
        assert "secret-token" not in caplog.text
        assert "request-id" not in caplog.text


class FakeOSSBucket:
    def __init__(self) -> None:
        self.copies: list[tuple[str, str, str]] = []
        self.error: Exception | None = None

    def copy_object(self, source_bucket: str, source_key: str, destination_key: str) -> None:
        if self.error is not None:
            raise self.error
        self.copies.append((source_bucket, source_key, destination_key))


class TestBaseBackendDefault:
    def test_local_backend_reports_unsupported(self, tmp_path: Any) -> None:
        backend = LocalBackend()
        assert backend.try_server_side_copy(str(tmp_path / "a"), str(tmp_path / "b")) is False
