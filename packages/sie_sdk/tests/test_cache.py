"""Tests for model weight caching hierarchy."""

from pathlib import Path
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
)
from sie_sdk.exceptions import GatedModelError


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
        )

        # Mock the storage backend
        with patch("sie_sdk.cache.get_storage_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_get_backend.return_value = mock_backend

            # Simulate model exists in cluster cache
            mock_backend.exists.return_value = True

            # list_dirs returns subdirectories - we need to simulate the HF cache structure
            # First call lists snapshots dir contents, subsequent calls return empty
            list_dirs_calls = [["abc123"], []]  # abc123 is a snapshot, then no more subdirs
            mock_backend.list_dirs.side_effect = lambda _: list_dirs_calls.pop(0) if list_dirs_calls else []
            mock_backend.list_files.return_value = iter(["model.bin"])

            ensure_model_cached("BAAI/bge-m3", config)

            # Should call backend methods
            mock_backend.exists.assert_called()
            # Result depends on whether files were actually downloaded
            # In mocked scenario, we just verify the logic flow

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
            mock_backend.exists.return_value = False

            with pytest.raises(RuntimeError, match="not found in local or cluster cache"):
                ensure_model_cached("BAAI/bge-m3", config)


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
            mock_backend.exists.return_value = False  # Not in cluster either

            with pytest.raises(RuntimeError, match="not found in local or cluster cache"):
                ensure_model_cached("org/model", config)

            # Should have tried cluster cache
            mock_get.assert_called_once()


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
