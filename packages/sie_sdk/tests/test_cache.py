"""Tests for model weight caching hierarchy."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from sie_sdk.cache import (
    CacheConfig,
    ensure_model_cached,
    get_cache_config,
    is_model_cached,
    populate_cluster_cache,
)
from sie_sdk.exceptions import GatedModelError
from sie_sdk.storage import OSSBackend


@pytest.fixture(autouse=True)
def offline_hf_metadata():
    with patch("huggingface_hub.file_exists", return_value=False) as metadata:
        yield metadata


class FakeOSSCacheBucket:
    """Small hierarchical OSS fake backed by realistic HF cache objects."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def list_objects_v2(
        self,
        *,
        prefix: str,
        delimiter: str,
        continuation_token: str,
        max_keys: int,
    ) -> SimpleNamespace:
        assert continuation_token == ""
        matching = [key for key in sorted(self.objects) if key.startswith(prefix)]
        object_keys: list[str] = []
        prefix_keys: set[str] = set()
        for key in matching:
            relative = key[len(prefix) :]
            if delimiter and delimiter in relative:
                prefix_keys.add(f"{prefix}{relative.split(delimiter, maxsplit=1)[0]}{delimiter}")
            else:
                object_keys.append(key)
        return SimpleNamespace(
            object_list=[SimpleNamespace(key=key) for key in object_keys[:max_keys]],
            prefix_list=sorted(prefix_keys)[:max_keys],
            is_truncated=False,
            next_continuation_token="",
        )

    def get_object_to_file(self, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])


class TestGetCacheConfig:
    """Tests for get_cache_config()."""

    def test_default_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default config uses HF_HOME and enables fallback."""
        monkeypatch.delenv("SIE_LOCAL_CACHE", raising=False)
        monkeypatch.delenv("SIE_CLUSTER_CACHE", raising=False)
        monkeypatch.delenv("SIE_HF_FALLBACK", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        config = get_cache_config()

        assert config.local_cache == Path.home() / ".cache" / "huggingface" / "hub"
        assert config.cluster_cache is None
        assert config.hf_fallback is True

    def test_explicit_local_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_LOCAL_CACHE takes precedence."""
        monkeypatch.setenv("SIE_LOCAL_CACHE", "/custom/cache")
        monkeypatch.delenv("HF_HOME", raising=False)

        config = get_cache_config()

        assert config.local_cache == Path("/custom/cache")

    def test_hf_home_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF_HOME used if SIE_LOCAL_CACHE not set."""
        monkeypatch.delenv("SIE_LOCAL_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", "/hf/home")

        config = get_cache_config()

        assert config.local_cache == Path("/hf/home/hub")

    def test_cluster_cache_s3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """S3 cluster cache URL parsed correctly."""
        monkeypatch.setenv("SIE_CLUSTER_CACHE", "s3://my-bucket/models")

        config = get_cache_config()

        assert config.cluster_cache == "s3://my-bucket/models"

    def test_cluster_cache_gcs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GCS cluster cache URL parsed correctly."""
        monkeypatch.setenv("SIE_CLUSTER_CACHE", "gs://my-bucket/models")

        config = get_cache_config()

        assert config.cluster_cache == "gs://my-bucket/models"

    def test_cluster_cache_azure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Azure cluster cache URL parsed correctly."""
        monkeypatch.setenv("SIE_CLUSTER_CACHE", "abfs://models@sieacct.dfs.core.windows.net/models")

        config = get_cache_config()

        assert config.cluster_cache == "abfs://models@sieacct.dfs.core.windows.net/models"

    def test_cluster_cache_oss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_CLUSTER_CACHE", "oss://my-bucket/models")

        config = get_cache_config()

        assert config.cluster_cache == "oss://my-bucket/models"

    def test_hf_fallback_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_HF_FALLBACK=false disables HF fallback."""
        monkeypatch.setenv("SIE_HF_FALLBACK", "false")

        config = get_cache_config()

        assert config.hf_fallback is False

    def test_hf_fallback_disabled_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Various false values work."""
        for value in ["false", "0", "no", "FALSE", "No"]:
            monkeypatch.setenv("SIE_HF_FALLBACK", value)
            config = get_cache_config()
            assert config.hf_fallback is False, f"Failed for value: {value}"


class TestIsModelCached:
    """Tests for is_model_cached()."""

    def test_model_not_cached(self, tmp_path: Path) -> None:
        """Returns False for non-existent model."""
        config = CacheConfig(local_cache=tmp_path)

        assert is_model_cached("BAAI/bge-m3", config) is False

    def test_model_cached_with_snapshot(self, tmp_path: Path) -> None:
        """Returns True for model with snapshot directory."""
        # Create HF-style cache structure
        model_dir = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        config = CacheConfig(local_cache=tmp_path)

        assert is_model_cached("BAAI/bge-m3", config) is True

    def test_model_cached_requires_exact_revision_when_pinned(self, tmp_path: Path) -> None:
        revision = "a" * 40
        model_dir = tmp_path / "models--BAAI--bge-m3" / "snapshots" / revision
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        config = CacheConfig(local_cache=tmp_path)

        assert is_model_cached("BAAI/bge-m3", config, revision=revision) is True
        assert is_model_cached("BAAI/bge-m3", config, revision="b" * 40) is False
        assert is_model_cached("BAAI/bge-m3", config, revision="../outside") is False
        assert is_model_cached("BAAI/bge-m3", config, revision="") is False
        assert is_model_cached("BAAI/bge-m3", config, revision=".") is False

    def test_model_cached_empty_snapshot(self, tmp_path: Path) -> None:
        """Returns False for model with empty snapshot directory."""
        model_dir = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)
        # Empty snapshot directory

        config = CacheConfig(local_cache=tmp_path)

        assert is_model_cached("BAAI/bge-m3", config) is False

    def test_model_dir_no_snapshots(self, tmp_path: Path) -> None:
        """Returns False for model dir without snapshots."""
        model_dir = tmp_path / "models--BAAI--bge-m3"
        model_dir.mkdir(parents=True)
        # No snapshots subdirectory

        config = CacheConfig(local_cache=tmp_path)

        assert is_model_cached("BAAI/bge-m3", config) is False


class TestEnsureModelCached:
    """Tests for ensure_model_cached()."""

    def test_model_in_local_cache(self, tmp_path: Path) -> None:
        """Returns path if model in local cache."""
        # Create cached model
        model_dir = tmp_path / "models--BAAI--bge-m3" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_text("weights")

        config = CacheConfig(local_cache=tmp_path)

        result = ensure_model_cached("BAAI/bge-m3", config)

        assert result is not None
        assert result == tmp_path / "models--BAAI--bge-m3"

    def test_model_not_cached_no_cluster(self, tmp_path: Path) -> None:
        """Raises error if not cached, no cluster cache, and HF fallback disabled."""
        config = CacheConfig(local_cache=tmp_path, cluster_cache=None, hf_fallback=False)

        with pytest.raises(RuntimeError, match="not found in local or cluster cache"):
            ensure_model_cached("BAAI/bge-m3", config)

    def test_model_from_cluster_cache(self, tmp_path: Path) -> None:
        """Downloads from cluster cache if configured."""
        local_cache = tmp_path / "local"
        local_cache.mkdir()

        config = CacheConfig(
            local_cache=local_cache,
            cluster_cache="s3://my-bucket/cache",
            hf_fallback=False,
        )

        # Mock the storage backend
        with patch("sie_sdk.cache.get_storage_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend

            # Simulate model exists in cluster cache (snapshots prefix non-empty)
            mock_backend.has_children.return_value = True

            cluster_root = "s3://my-bucket/cache/models--BAAI--bge-m3"
            directories = {cluster_root: ["snapshots"], f"{cluster_root}/snapshots": ["abc123"]}
            mock_backend.list_dirs.side_effect = lambda path: directories.get(path, [])
            mock_backend.list_files.side_effect = lambda path: ["model.bin"] if path.endswith("/abc123") else []
            mock_backend.download_file.side_effect = lambda _source, destination: destination.write_bytes(b"weights")

            result = ensure_model_cached("BAAI/bge-m3", config)

            assert result == local_cache / "models--BAAI--bge-m3"
            assert is_model_cached("BAAI/bge-m3", config)
            mock_backend.has_children.assert_called()

    def test_model_not_in_cluster_cache(self, tmp_path: Path) -> None:
        """Raises error if model not in cluster cache and HF fallback disabled."""
        config = CacheConfig(
            local_cache=tmp_path,
            cluster_cache="s3://my-bucket/cache",
            hf_fallback=False,
        )

        with patch("sie_sdk.cache.get_storage_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend
            mock_backend.has_children.return_value = False

            with pytest.raises(RuntimeError, match="not found in local or cluster cache"):
                ensure_model_cached("BAAI/bge-m3", config)

    def test_realistic_hf_snapshot_tree_downloads_from_oss(self, tmp_path: Path) -> None:
        model_root = "models/models--openai--clip-vit-base-patch32"
        snapshot = "0123456789abcdef"
        objects = {
            f"{model_root}/refs/main": snapshot.encode(),
            f"{model_root}/blobs/config-hash": b"{}",
            f"{model_root}/snapshots/{snapshot}/config.json": b'{"model_type":"clip"}',
            f"{model_root}/snapshots/{snapshot}/model.safetensors": b"weights",
        }
        backend = OSSBackend()
        backend._buckets["model-cache"] = FakeOSSCacheBucket(objects)
        config = CacheConfig(
            local_cache=tmp_path,
            cluster_cache="oss://model-cache/models",
            hf_fallback=False,
        )

        with patch("sie_sdk.cache.get_storage_backend", return_value=backend):
            result = ensure_model_cached("openai/clip-vit-base-patch32", config)

        assert result == tmp_path / "models--openai--clip-vit-base-patch32"
        assert (result / "refs" / "main").read_text() == snapshot
        assert (result / "snapshots" / snapshot / "config.json").read_text() == '{"model_type":"clip"}'
        assert (result / "snapshots" / snapshot / "model.safetensors").read_bytes() == b"weights"


class TestCacheHierarchy:
    """Integration tests for the full caching hierarchy."""

    def test_local_cache_first(self, tmp_path: Path) -> None:
        """Local cache is checked before cluster cache."""
        # Create local cached model
        model_dir = tmp_path / "models--org--model" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.bin").write_text("local")

        config = CacheConfig(
            local_cache=tmp_path,
            cluster_cache="s3://bucket/cache",  # Should not be accessed
        )

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            result = ensure_model_cached("org/model", config)

            # Should not even try to access cluster cache
            mock_get.assert_not_called()
            assert result is not None

    def test_cluster_cache_fallback(self, tmp_path: Path) -> None:
        """Cluster cache used if local cache empty, raises if not found and no HF fallback."""
        config = CacheConfig(
            local_cache=tmp_path,
            cluster_cache="s3://bucket/cache",
            hf_fallback=False,
        )

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            mock_backend = MagicMock()
            mock_get.return_value = mock_backend
            mock_backend.has_children.return_value = False  # Not in cluster either

            with pytest.raises(RuntimeError, match="not found in local or cluster cache"):
                ensure_model_cached("org/model", config)

            # Should have tried cluster cache
            mock_get.assert_called_once()


class TestPopulateClusterCache:
    """Tests for populate_cluster_cache()."""

    @staticmethod
    def _cached_model(local_cache: Path, model_id: str = "BAAI/bge-m3") -> Path:
        """Create an HF-style cached model directory and return its root."""
        model_dir = local_cache / f"models--{model_id.replace('/', '--')}"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_text("weights")
        (snapshot / "config.json").write_text("{}")
        return model_dir

    def test_no_cluster_cache_configured(self, tmp_path: Path) -> None:
        """Returns False without touching storage when cluster cache is unset."""
        self._cached_model(tmp_path)
        config = CacheConfig(local_cache=tmp_path, cluster_cache=None)

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            assert populate_cluster_cache("BAAI/bge-m3", config) is False
            mock_get.assert_not_called()

    def test_model_not_in_local_cache(self, tmp_path: Path) -> None:
        """Returns False without touching storage when the model is not cached."""
        config = CacheConfig(local_cache=tmp_path, cluster_cache="s3://bucket/cache")

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            assert populate_cluster_cache("BAAI/bge-m3", config) is False
            mock_get.assert_not_called()

    def test_uploads_model_directory(self, tmp_path: Path) -> None:
        """Happy path: uploads the model dir to the mirrored cluster path."""
        model_dir = self._cached_model(tmp_path)
        config = CacheConfig(local_cache=tmp_path, cluster_cache="s3://bucket/cache")

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.upload_directory.return_value = 2
            mock_get.return_value = mock_backend

            assert populate_cluster_cache("BAAI/bge-m3", config) is True

            mock_get.assert_called_once_with("s3://bucket/cache")
            mock_backend.upload_directory.assert_called_once_with(
                model_dir,
                "s3://bucket/cache/models--BAAI--bge-m3",
            )

    def test_zero_files_uploaded_returns_false(self, tmp_path: Path) -> None:
        """Backend reporting zero uploaded files is surfaced as failure."""
        self._cached_model(tmp_path)
        config = CacheConfig(local_cache=tmp_path, cluster_cache="s3://bucket/cache")

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.upload_directory.return_value = 0
            mock_get.return_value = mock_backend

            assert populate_cluster_cache("BAAI/bge-m3", config) is False

    def test_reads_config_from_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """config=None resolves via get_cache_config() (env variables)."""
        self._cached_model(tmp_path)
        monkeypatch.setenv("SIE_LOCAL_CACHE", str(tmp_path))
        monkeypatch.setenv("SIE_CLUSTER_CACHE", "s3://env-bucket/cache")

        with patch("sie_sdk.cache.get_storage_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.upload_directory.return_value = 1
            mock_get.return_value = mock_backend

            assert populate_cluster_cache("BAAI/bge-m3") is True
            mock_get.assert_called_once_with("s3://env-bucket/cache")

    def test_roundtrip_with_local_backend(self, tmp_path: Path) -> None:
        """Uploading via the real LocalBackend mirrors the HF cache layout."""
        local_cache = tmp_path / "local"
        local_cache.mkdir()
        self._cached_model(local_cache)
        cluster = tmp_path / "cluster"
        cluster.mkdir()

        config = CacheConfig(local_cache=local_cache, cluster_cache=str(cluster))

        assert populate_cluster_cache("BAAI/bge-m3", config) is True
        uploaded = cluster / "models--BAAI--bge-m3" / "snapshots" / "abc123"
        assert (uploaded / "model.safetensors").read_text() == "weights"
        assert (uploaded / "config.json").read_text() == "{}"


class TestGatedModelErrorHandling:
    """Tests for gated model error handling in HuggingFace downloads."""

    def test_gated_repo_error_raises_gated_model_error(self, tmp_path: Path) -> None:
        """GatedRepoError from HF Hub is converted to GatedModelError."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = GatedRepoError("Access denied")

            with pytest.raises(GatedModelError) as exc_info:
                ensure_model_cached("meta-llama/Llama-2-7b", config)

            assert exc_info.value.model_id == "meta-llama/Llama-2-7b"
            assert "Access denied" in str(exc_info.value.original_error)

    def test_http_401_raises_gated_model_error(self, tmp_path: Path) -> None:
        """HTTP 401 errors are converted to GatedModelError."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        # Mock HfHubHTTPError with 401 status code
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_error = HfHubHTTPError("Unauthorized")
        mock_error.response = mock_response

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = mock_error

            with pytest.raises(GatedModelError) as exc_info:
                ensure_model_cached("org/gated-model", config)

            assert exc_info.value.model_id == "org/gated-model"

    def test_http_403_raises_gated_model_error(self, tmp_path: Path) -> None:
        """HTTP 403 errors are converted to GatedModelError."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        # Mock HfHubHTTPError with 403 status code
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_error = HfHubHTTPError("Forbidden")
        mock_error.response = mock_response

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = mock_error

            with pytest.raises(GatedModelError) as exc_info:
                ensure_model_cached("org/gated-model", config)

            assert exc_info.value.model_id == "org/gated-model"

    def test_repo_not_found_without_token_suggests_gated(self, tmp_path: Path) -> None:
        """RepositoryNotFoundError without token suggests it might be gated."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = RepositoryNotFoundError("Not found")

            with patch("os.environ.get", return_value=None):
                with pytest.raises(RuntimeError) as exc_info:
                    ensure_model_cached("org/maybe-gated", config)

                error_msg = str(exc_info.value)
                assert "org/maybe-gated" in error_msg
                assert "gated" in error_msg.lower()
                assert "HF_TOKEN" in error_msg

    def test_repo_not_found_with_token_reraises(self, tmp_path: Path) -> None:
        """RepositoryNotFoundError with token is re-raised (actually not found)."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        with (
            patch("huggingface_hub.snapshot_download") as mock_download,
            patch("huggingface_hub.file_exists", return_value=False),
        ):
            mock_download.side_effect = RepositoryNotFoundError("Not found")

            with patch("os.environ.get", return_value="hf_token123"):
                with pytest.raises(RepositoryNotFoundError):
                    ensure_model_cached("org/nonexistent", config)

    def test_hf_token_passed_to_snapshot_download(self, tmp_path: Path) -> None:
        """HF_TOKEN from environment is passed to snapshot_download."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        # Create a minimal model structure to satisfy is_model_cached check
        model_dir = tmp_path / "models--org--model" / "snapshots" / "abc"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_text("weights")

        with (
            patch("huggingface_hub.snapshot_download") as mock_download,
            patch("huggingface_hub.file_exists", return_value=False),
        ):
            mock_download.return_value = str(tmp_path / "models--org--model")

            with patch("os.environ.get", return_value="hf_mytoken"):
                # First call will find model cached, so create scenario where it's not
                (tmp_path / "models--org--newmodel").mkdir(parents=True, exist_ok=True)
                ensure_model_cached("org/newmodel", config)

                # Verify token was passed
                mock_download.assert_called_once()
                call_kwargs = mock_download.call_args[1]
                assert call_kwargs["token"] == "hf_mytoken"  # noqa: S105 — test fixture token assertion

    def test_cache_dir_uses_local_cache_path(self, tmp_path: Path) -> None:
        """snapshot_download uses correct cache_dir (HF_HOME/hub)."""
        config = CacheConfig(local_cache=tmp_path, hf_fallback=True)

        with (
            patch("huggingface_hub.snapshot_download") as mock_download,
            patch("huggingface_hub.file_exists", return_value=False),
        ):
            mock_download.return_value = str(tmp_path / "models--org--model")

            # Ensure model is not cached to trigger download
            ensure_model_cached("org/model", config)

            # Verify cache_dir points to HF_HOME/hub (which is tmp_path in test)
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["cache_dir"] == str(tmp_path)
