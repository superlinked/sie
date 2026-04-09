import tempfile
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_router.config_api import router as config_router
from sie_router.config_store import ConfigStore
from sie_router.model_registry import ModelRegistry
from sie_router.registry import WorkerRegistry


def _create_test_app(
    bundles_dir: Path,
    models_dir: Path,
    config_store_dir: str | None = None,
) -> FastAPI:
    """Create a minimal test app with config API."""
    app = FastAPI()
    app.include_router(config_router)

    model_registry = ModelRegistry(bundles_dir, models_dir)
    app.state.model_registry = model_registry
    app.state.registry = WorkerRegistry()
    app.state.nats_manager = None
    app.state.config_store = ConfigStore(config_store_dir) if config_store_dir else None

    return app


def _write_bundle(bundles_dir: Path, name: str, adapters: list[str], priority: int = 10) -> None:
    bundle = {"name": name, "priority": priority, "adapters": adapters}
    (bundles_dir / f"{name}.yaml").write_text(yaml.dump(bundle))


def _write_model(models_dir: Path, sie_id: str, adapter_path: str) -> None:
    config = {
        "sie_id": sie_id,
        "profiles": {
            "default": {
                "adapter_path": adapter_path,
                "max_batch_tokens": 8192,
            }
        },
    }
    filename = sie_id.replace("/", "__") + ".yaml"
    (models_dir / filename).write_text(yaml.dump(config))


class TestConfigAPIModels:
    """Tests for /v1/configs/models endpoints."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._store = self._root / "store"
        self._bundles.mkdir()
        self._models.mkdir()

        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])
        _write_model(self._models, "test/model", "sie_server.adapters.bert_flash:BertFlashAdapter")

        self.app = _create_test_app(self._bundles, self._models, str(self._store))
        self.client = TestClient(self.app)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_list_models(self) -> None:
        resp = self.client.get("/v1/configs/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "test/model"
        assert data["models"][0]["source"] == "filesystem"

    def test_get_model_not_found(self) -> None:
        resp = self.client.get("/v1/configs/models/nonexistent/model")
        assert resp.status_code == 404

    def test_get_model_filesystem_returns_200(self) -> None:
        """GET /v1/configs/models/{model_id} returns 200 with YAML for filesystem models."""
        resp = self.client.get("/v1/configs/models/test/model")
        assert resp.status_code == 200
        assert "application/x-yaml" in resp.headers["content-type"]

        # Parse the YAML response body
        data = yaml.safe_load(resp.text)
        assert data["source"] == "filesystem"
        assert data["sie_id"] == "test/model"
        assert "bundles" in data

    def test_add_model_success(self) -> None:
        yaml_body = """
sie_id: new/model
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
    max_batch_tokens: 8192
"""
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_id"] == "new/model"
        assert data["created_profiles"] == ["default"]
        assert data["worker_ack_pending"] is True  # no workers connected
        assert "no_eligible_workers_connected" in data["warnings"]

    def test_add_model_unroutable_adapter(self) -> None:
        yaml_body = """
sie_id: bad/model
profiles:
  default:
    adapter_path: sie_server.adapters.unknown:UnknownAdapter
    max_batch_tokens: 8192
"""
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 422
        assert "validation_error" in resp.json()["detail"]["error"]

    def test_add_model_invalid_yaml(self) -> None:
        resp = self.client.post(
            "/v1/configs/models",
            content="{{invalid yaml",
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 400

    def test_add_model_missing_sie_id(self) -> None:
        yaml_body = """
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
"""
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 422

    def test_add_model_persisted_to_store(self) -> None:
        yaml_body = "sie_id: stored/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        self.client.post("/v1/configs/models", content=yaml_body)

        store = self.app.state.config_store
        assert store.read_model("stored/model") is not None
        assert store.read_epoch() == 1

    def test_add_model_idempotent_profiles(self) -> None:
        """Adding same model twice skips existing profiles."""
        yaml_body = "sie_id: new/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        resp1 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp1.status_code == 201

        resp2 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["created_profiles"] == []
        assert data["existing_profiles_skipped"] == ["default"]

    def test_list_models_includes_api_added(self) -> None:
        yaml_body = "sie_id: api/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        self.client.post("/v1/configs/models", content=yaml_body)

        resp = self.client.get("/v1/configs/models")
        models = resp.json()["models"]
        api_model = next(m for m in models if m["model_id"] == "api/model")
        assert api_model["source"] == "api"


class TestConfigAPIBundles:
    """Tests for /v1/configs/bundles endpoints."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._bundles.mkdir()
        self._models.mkdir()

        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"], priority=10)
        _write_bundle(self._bundles, "sglang", ["sie_server.adapters.sglang"], priority=20)

        self.app = _create_test_app(self._bundles, self._models)
        self.client = TestClient(self.app)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_list_bundles(self) -> None:
        resp = self.client.get("/v1/configs/bundles")
        assert resp.status_code == 200
        bundles = resp.json()["bundles"]
        assert len(bundles) == 2
        assert bundles[0]["bundle_id"] == "default"
        assert bundles[0]["priority"] == 10
        assert bundles[0]["connected_workers"] == 0

    def test_get_bundle(self) -> None:
        resp = self.client.get("/v1/configs/bundles/default")
        assert resp.status_code == 200
        assert "application/x-yaml" in resp.headers["content-type"]

    def test_get_bundle_not_found(self) -> None:
        resp = self.client.get("/v1/configs/bundles/nonexistent")
        assert resp.status_code == 404


class TestConfigAPIEdgeCases:
    """Tests for Config API edge cases."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._bundles.mkdir()
        self._models.mkdir()
        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_post_without_config_store_works_in_memory(self) -> None:
        """POST works when no config store is configured (in-memory only)."""
        app = _create_test_app(self._bundles, self._models, config_store_dir=None)
        client = TestClient(app)

        yaml_body = "sie_id: mem/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201

        # Model exists in registry but not persisted
        resp2 = client.get("/v1/configs/models")
        model_ids = [m["model_id"] for m in resp2.json()["models"]]
        assert "mem/model" in model_ids

    def test_post_with_nats_disconnected_returns_503(self) -> None:
        """POST returns 503 when NATS is configured but disconnected."""
        from unittest.mock import MagicMock

        from sie_router.nats_manager import NatsManager

        app = _create_test_app(self._bundles, self._models)
        # Set up a mock NatsManager that reports disconnected
        mock_nats = MagicMock(spec=NatsManager)
        mock_nats.connected = False
        app.state.nats_manager = mock_nats

        client = TestClient(app)
        yaml_body = "sie_id: test/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 503
        assert "nats_unavailable" in resp.json()["detail"]["error"]

    def test_post_model_with_multiple_profiles(self) -> None:
        """POST with multiple profiles in a single request."""
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)

        yaml_body = "sie_id: multi/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n  custom:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 2\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201
        data = resp.json()
        assert sorted(data["created_profiles"]) == ["custom", "default"]

    def test_post_empty_body_returns_400(self) -> None:
        """POST with empty body returns 400."""
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)
        resp = client.post("/v1/configs/models", content="")
        # Empty YAML parses as None, which is not a dict
        assert resp.status_code == 400

    def test_serving_readiness_with_no_workers(self) -> None:
        """POST with no workers returns worker_ack_pending=true + warning."""
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)

        yaml_body = "sie_id: lonely/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["worker_ack_pending"] is True
        assert "no_eligible_workers_connected" in data["warnings"]
        assert data["total_eligible"] == 0

    def test_auth_write_rejected_with_inference_token(self, monkeypatch) -> None:
        """POST with inference token (not admin) is rejected when admin token is set."""
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)

        monkeypatch.setenv("SIE_ADMIN_TOKEN", "admin-secret")
        monkeypatch.setenv("SIE_AUTH_TOKEN", "read-only")
        yaml_body = "sie_id: test/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Authorization": "Bearer read-only"},
        )
        assert resp.status_code == 403

    def test_auth_read_allowed_with_inference_token(self, monkeypatch) -> None:
        """GET with inference token is allowed."""
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)

        monkeypatch.setenv("SIE_ADMIN_TOKEN", "admin-secret")
        monkeypatch.setenv("SIE_AUTH_TOKEN", "read-only")
        resp = client.get(
            "/v1/configs/models",
            headers={"Authorization": "Bearer read-only"},
        )
        assert resp.status_code == 200


class TestConfigAPICASRetry:
    """Tests for CAS retry loop in add_model."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._store = self._root / "store"
        self._bundles.mkdir()
        self._models.mkdir()
        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_post_model_cas_retry_succeeds(self) -> None:
        """CAS retry loop should succeed after transient epoch conflict."""
        from unittest.mock import patch

        from sie_router.config_store import EpochCASError

        app = _create_test_app(self._bundles, self._models, str(self._store))
        client = TestClient(app)

        store = app.state.config_store
        original_cas = store.cas_epoch
        call_count = 0

        def cas_fail_once(expected: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EpochCASError(expected, expected + 1)
            return original_cas(expected)

        with patch.object(store, "cas_epoch", side_effect=cas_fail_once):
            yaml_body = "sie_id: cas/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
            resp = client.post(
                "/v1/configs/models",
                content=yaml_body,
                headers={"Content-Type": "application/x-yaml"},
            )

        assert resp.status_code == 201
        assert resp.json()["model_id"] == "cas/model"
        assert call_count == 2

    def test_post_model_cas_exhausted_returns_503(self) -> None:
        """CAS retry loop should return 503 after all retries exhausted."""
        from unittest.mock import patch

        from sie_router.config_store import EpochCASError

        app = _create_test_app(self._bundles, self._models, str(self._store))
        client = TestClient(app)

        store = app.state.config_store

        def cas_always_fail(expected: int) -> int:
            raise EpochCASError(expected, expected + 1)

        with patch.object(store, "cas_epoch", side_effect=cas_always_fail):
            yaml_body = "sie_id: cas/fail\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
            resp = client.post(
                "/v1/configs/models",
                content=yaml_body,
                headers={"Content-Type": "application/x-yaml"},
            )

        assert resp.status_code == 503
        assert "cas_conflict" in resp.json()["detail"]["error"]


class TestConfigAPIIdempotency:
    """Tests for Idempotency-Key header behavior in add_model."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._store = self._root / "store"
        self._bundles.mkdir()
        self._models.mkdir()
        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])

        self.app = _create_test_app(self._bundles, self._models, str(self._store))
        self.client = TestClient(self.app)

        # Clear module-level idempotency cache and in-flight tracker between tests
        from sie_router.config_api import _idempotency_cache, _idempotency_in_flight

        _idempotency_cache.clear()
        _idempotency_in_flight.clear()

    def teardown_method(self) -> None:
        from sie_router.config_api import _idempotency_cache, _idempotency_in_flight

        _idempotency_cache.clear()
        _idempotency_in_flight.clear()
        self._tmpdir.cleanup()

    def test_duplicate_key_returns_cached_response(self) -> None:
        """Second POST with same Idempotency-Key and body returns cached response."""
        yaml_body = "sie_id: idem/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        headers = {
            "Content-Type": "application/x-yaml",
            "Idempotency-Key": "test-key-1",
        }

        resp1 = self.client.post("/v1/configs/models", content=yaml_body, headers=headers)
        assert resp1.status_code == 201

        # Second request with same key and same body → cached 201
        resp2 = self.client.post("/v1/configs/models", content=yaml_body, headers=headers)
        assert resp2.status_code == 201
        assert resp2.json() == resp1.json()

    def test_duplicate_key_different_body_returns_422(self) -> None:
        """Reusing Idempotency-Key with a different payload returns 422."""
        yaml_body_1 = "sie_id: idem/model-a\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        yaml_body_2 = "sie_id: idem/model-b\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 2\n"
        key = "test-key-mismatch"

        resp1 = self.client.post(
            "/v1/configs/models",
            content=yaml_body_1,
            headers={"Content-Type": "application/x-yaml", "Idempotency-Key": key},
        )
        assert resp1.status_code == 201

        # Different body with same key → 422
        resp2 = self.client.post(
            "/v1/configs/models",
            content=yaml_body_2,
            headers={"Content-Type": "application/x-yaml", "Idempotency-Key": key},
        )
        assert resp2.status_code == 422
        assert resp2.json()["detail"]["error"] == "idempotency_mismatch"

    def test_no_idempotency_key_skips_cache(self) -> None:
        """Requests without Idempotency-Key are never cached."""
        yaml_body = "sie_id: nocache/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"

        resp1 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp1.status_code == 201

        # Same body, no key — should be treated as a new request (200 = profiles skipped)
        resp2 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp2.status_code == 200
        assert resp2.json()["existing_profiles_skipped"] == ["default"]


class TestConfigAPINATSPublishFailure:
    """Tests for NATS publish failure path in add_model."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._store = self._root / "store"
        self._bundles.mkdir()
        self._models.mkdir()
        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_nats_publish_failure_still_persists_model(self) -> None:
        """Model is persisted and response contains warning when NATS publish fails."""
        from unittest.mock import AsyncMock, MagicMock

        from sie_router.nats_manager import NatsManager

        app = _create_test_app(self._bundles, self._models, str(self._store))

        mock_nats = MagicMock(spec=NatsManager)
        mock_nats.connected = True
        mock_nats.router_id = "test-router"
        mock_nats.publish_config_notification = AsyncMock(side_effect=RuntimeError("NATS down"))
        app.state.nats_manager = mock_nats

        client = TestClient(app)
        yaml_body = "sie_id: nats/fail\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["model_id"] == "nats/fail"
        assert any("nats_publish_failed" in w for w in data["warnings"])

        # Verify model was persisted to config store
        store = app.state.config_store
        assert store.read_model("nats/fail") is not None
