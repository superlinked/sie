"""Integration tests for Router with ModelRegistry.

These tests verify the full request routing flow with ModelRegistry-based
bundle resolution, including 404/409 fast-fail behavior.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sie_router.app.app_factory import AppFactory
from sie_router.app.app_state_config import AppStateConfig


@pytest.fixture
def config_dirs():
    """Create temporary directories with test bundle and model configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        bundles_dir = tmppath / "bundles"
        models_dir = tmppath / "models"
        bundles_dir.mkdir()
        models_dir.mkdir()

        # Create bundle configs (YAML with adapters)
        default_bundle = bundles_dir / "default.yaml"
        default_bundle.write_text(
            "name: default\n"
            "priority: 10\n"
            "default: true\n"
            "adapters:\n"
            "  - sie_server.adapters.bge_m3\n"
            "  - sie_server.adapters.sentence_transformer\n"
        )

        sglang_bundle = bundles_dir / "sglang.yaml"
        sglang_bundle.write_text(
            "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.bge_m3\n  - sie_server.adapters.sglang\n"
        )

        # Create model configs (flat YAML files with profiles)
        (models_dir / "baai-bge-m3.yaml").write_text(
            "name: BAAI/bge-m3\n"
            "hf_id: BAAI/bge-m3\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bge_m3:BGEM3Adapter\n"
        )

        (models_dir / "intfloat-e5-small-v2.yaml").write_text(
            "name: intfloat/e5-small-v2\n"
            "hf_id: intfloat/e5-small-v2\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.sentence_transformer:SentenceTransformerAdapter\n"
        )

        (models_dir / "qwen-qwen3-embedding-8b.yaml").write_text(
            "name: Qwen/Qwen3-Embedding-8B\n"
            "hf_id: Qwen/Qwen3-Embedding-8B\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.sglang:SGLangAdapter\n"
        )

        yield bundles_dir, models_dir


@pytest.fixture
def client(config_dirs):
    """Create test client for router app with ModelRegistry configured via env vars."""
    bundles_dir, models_dir = config_dirs

    # Configure paths via environment variables (read by lifespan)
    old_bundles = os.environ.get("SIE_BUNDLES_DIR")
    old_models = os.environ.get("SIE_MODELS_DIR")
    os.environ["SIE_BUNDLES_DIR"] = str(bundles_dir)
    os.environ["SIE_MODELS_DIR"] = str(models_dir)

    try:
        # Create app - lifespan will initialize ModelRegistry with env var paths
        config = AppStateConfig(worker_urls=[])
        app = AppFactory.create_app(config)
        with TestClient(app) as client:
            yield client
    finally:
        # Restore environment
        if old_bundles is None:
            os.environ.pop("SIE_BUNDLES_DIR", None)
        else:
            os.environ["SIE_BUNDLES_DIR"] = old_bundles
        if old_models is None:
            os.environ.pop("SIE_MODELS_DIR", None)
        else:
            os.environ["SIE_MODELS_DIR"] = old_models


class TestModelsEndpointWithRegistry:
    """Test /v1/models endpoint with ModelRegistry."""

    def test_models_returns_complete_catalog(self, client: TestClient) -> None:
        """Models endpoint returns complete catalog from ModelRegistry."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data

        # All models should be present
        model_names = [m["name"] for m in data["models"]]
        assert "BAAI/bge-m3" in model_names
        assert "intfloat/e5-small-v2" in model_names
        assert "Qwen/Qwen3-Embedding-8B" in model_names

    def test_models_includes_bundle_info(self, client: TestClient) -> None:
        """Models endpoint includes bundle information."""
        response = client.get("/v1/models")
        data = response.json()

        # Find bge-m3 which is in both default and sglang
        bge_m3 = next(m for m in data["models"] if m["name"] == "BAAI/bge-m3")
        assert "bundles" in bge_m3
        assert "default" in bge_m3["bundles"]
        assert "sglang" in bge_m3["bundles"]
        # Default should be first (lower priority)
        assert bge_m3["bundles"][0] == "default"

    def test_models_shows_loaded_status(self, client: TestClient) -> None:
        """Models endpoint shows loaded status (false when no workers)."""
        response = client.get("/v1/models")
        data = response.json()

        # All models should show loaded=False (no workers)
        for model in data["models"]:
            assert model["loaded"] is False
            assert model["worker_count"] == 0


class TestBundleResolution:
    """Test bundle resolution in proxy requests."""

    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        """Unknown model returns 404 without triggering autoscaling."""
        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                response = client.post(
                    "/v1/encode/unknown/nonexistent-model",
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "model_not_found"
        assert data["model"] == "unknown/nonexistent-model"

        # Pending demand should NOT be recorded for 404
        mock_demand.assert_not_called()

    def test_incompatible_bundle_override_returns_409(self, client: TestClient) -> None:
        """Incompatible bundle override returns 409."""
        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                # e5-small-v2 is only in default, not sglang
                response = client.post(
                    "/v1/encode/sglang:/intfloat/e5-small-v2",
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 409
        data = response.json()
        assert data["error"] == "bundle_conflict"
        assert data["model"] == "intfloat/e5-small-v2"
        assert data["requested_bundle"] == "sglang"
        assert "default" in data["compatible_bundles"]

        # Pending demand should NOT be recorded for 409
        mock_demand.assert_not_called()

    def test_valid_model_without_workers_returns_202(self, client: TestClient) -> None:
        """Valid model without workers returns 202 with autoscaling."""
        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                response = client.post(
                    "/v1/encode/BAAI/bge-m3",
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "provisioning"

        # Pending demand SHOULD be recorded for valid model
        mock_demand.assert_called_once()

    def test_valid_bundle_override_returns_202(self, client: TestClient) -> None:
        """Valid bundle override without workers returns 202."""
        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                # bge-m3 is in both default and sglang
                response = client.post(
                    "/v1/encode/sglang:/BAAI/bge-m3",
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 202

        # Demand should be recorded with sglang bundle
        # record_pending_demand signature: (machine_profile, bundle)
        mock_demand.assert_called_once()
        call_args = mock_demand.call_args[0]
        assert call_args[1] == "sglang"  # bundle is at index 1

    def test_auto_bundle_selection_uses_priority(self, client: TestClient) -> None:
        """Auto bundle selection uses lowest priority bundle."""
        with patch("sie_router.proxy.record_pending_demand") as mock_demand:
            with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
                # bge-m3 is in default (10) and sglang (20), should use default
                response = client.post(
                    "/v1/encode/BAAI/bge-m3",
                    headers={"X-SIE-MACHINE-PROFILE": "l4"},
                    json={"text": "hello"},
                )

        assert response.status_code == 202

        # Demand should be recorded with default bundle (lower priority)
        # record_pending_demand signature: (machine_profile, bundle)
        mock_demand.assert_called_once()
        call_args = mock_demand.call_args[0]
        assert call_args[1] == "default"  # bundle is at index 1


class TestAllEndpointsWithRegistry:
    """Test all proxy endpoints work with ModelRegistry."""

    @pytest.mark.parametrize("endpoint", ["/v1/encode", "/v1/score", "/v1/extract"])
    def test_404_for_unknown_model(self, client: TestClient, endpoint: str) -> None:
        """All endpoints return 404 for unknown models."""
        with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
            response = client.post(
                f"{endpoint}/unknown/model",
                headers={"X-SIE-MACHINE-PROFILE": "l4"},
                json={"text": "hello"},
            )

        assert response.status_code == 404

    @pytest.mark.parametrize("endpoint", ["/v1/encode", "/v1/score", "/v1/extract"])
    def test_409_for_incompatible_bundle(self, client: TestClient, endpoint: str) -> None:
        """All endpoints return 409 for incompatible bundle overrides."""
        with patch("sie_router.proxy.CONFIGURED_GPU_TYPES", ["l4"]):
            response = client.post(
                f"{endpoint}/sglang:/intfloat/e5-small-v2",
                headers={"X-SIE-MACHINE-PROFILE": "l4"},
                json={"text": "hello"},
            )

        assert response.status_code == 409
