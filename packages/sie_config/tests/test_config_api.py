import asyncio
import tempfile
import threading
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_config.config_api import router as config_router
from sie_config.config_store import ConfigStore
from sie_config.model_registry import ModelRegistry


def _create_test_app(
    bundles_dir: Path,
    models_dir: Path,
    config_store_dir: str | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(config_router)

    model_registry = ModelRegistry(bundles_dir, models_dir)
    app.state.model_registry = model_registry
    app.state.nats_publisher = None
    app.state.config_store = ConfigStore(config_store_dir) if config_store_dir else None

    return app


def _write_bundle(
    bundles_dir: Path,
    name: str,
    adapters: list[str],
    priority: int = 10,
    *,
    engine: str | None = None,
) -> None:
    bundle = {"name": name, "priority": priority, "adapters": adapters}
    if engine is not None:
        bundle["engine"] = engine
    (bundles_dir / f"{name}.yaml").write_text(yaml.dump(bundle))


def _write_model(models_dir: Path, sie_id: str, adapter_path: str, *, pool: str | None = None) -> None:
    config = {
        "sie_id": sie_id,
        "profiles": {
            "default": {
                "adapter_path": adapter_path,
                "max_batch_tokens": 8192,
            }
        },
    }
    if pool is not None:
        config["pool"] = pool
    filename = sie_id.replace("/", "__") + ".yaml"
    (models_dir / filename).write_text(yaml.dump(config))


class TestConfigAPIModels:
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

    def test_list_models_uses_profile_only_candle_route_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundles = root / "bundles"
            models = root / "models"
            bundles.mkdir()
            models.mkdir()
            _write_bundle(bundles, "candle", ["sie_server_rust.adapters.candle"], priority=30, engine="candle")
            (models / "candle-only.yaml").write_text(
                "sie_id: org/candle-only\n"
                "profiles:\n"
                "  candle:\n"
                "    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter\n"
            )
            app = _create_test_app(bundles, models)
            client = TestClient(app)

            resp = client.get("/v1/configs/models")

        assert resp.status_code == 200
        models_body = resp.json()["models"]
        assert [model["model_id"] for model in models_body] == ["org/candle-only:candle"]
        assert models_body[0]["profiles"] == ["candle"]
        assert models_body[0]["source"] == "filesystem"

    def test_get_model_not_found(self) -> None:
        resp = self.client.get("/v1/configs/models/nonexistent/model")
        assert resp.status_code == 404

    def test_get_model_filesystem_returns_200(self) -> None:
        resp = self.client.get("/v1/configs/models/test/model")
        assert resp.status_code == 200
        assert "application/x-yaml" in resp.headers["content-type"]

    def test_add_model_success(self) -> None:
        yaml_body = "sie_id: new/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n    max_batch_tokens: 8192\n"
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_id"] == "new/model"
        assert data["created_profiles"] == ["default"]

    def test_add_model_normalizes_pool_in_export(self) -> None:
        yaml_body = "sie_id: pool/model\npool: Customer-A\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n    max_batch_tokens: 8192\n"
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 201

        export = self.client.get("/v1/configs/export")
        assert export.status_code == 200
        models = {m["model_id"]: m for m in export.json()["models"]}
        assert models["pool/model"]["model_config"]["pool"] == "customer-a"
        assert models["pool/model"]["pool"] == "customer-a"
        assert "pool: customer-a" in models["pool/model"]["raw_yaml"]

    def test_append_model_profile_preserves_pool_in_export(self) -> None:
        first = (
            "sie_id: pool/model\n"
            "pool: Customer-A\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
            "    max_batch_tokens: 1024\n"
        )
        assert self.client.post("/v1/configs/models", content=first).status_code == 201

        appended = (
            "sie_id: pool/model\n"
            "profiles:\n"
            "  fast:\n"
            "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
            "    max_batch_tokens: 8192\n"
        )
        assert self.client.post("/v1/configs/models", content=appended).status_code == 201

        export = self.client.get("/v1/configs/export")
        assert export.status_code == 200
        models = {m["model_id"]: m for m in export.json()["models"]}
        entry = models["pool/model"]
        assert entry["model_config"]["pool"] == "customer-a"
        assert entry["pool"] == "customer-a"
        assert set(entry["model_config"]["profiles"]) == {"default", "fast"}
        assert "pool: customer-a" in entry["raw_yaml"]

    def test_add_model_invalid_pool_returns_422(self) -> None:
        yaml_body = "sie_id: bad/model\npool: customer.a\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n    max_batch_tokens: 8192\n"
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 422
        assert "pool" in str(resp.json()["detail"])

    def test_add_model_unroutable_adapter(self) -> None:
        yaml_body = "sie_id: bad/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.unknown:UnknownAdapter\n    max_batch_tokens: 8192\n"
        resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert resp.status_code == 422
        assert "validation_error" in resp.json()["detail"]["error"]

    def test_add_model_invalid_yaml(self) -> None:
        resp = self.client.post(
            "/v1/configs/models", content="{{invalid yaml", headers={"Content-Type": "application/x-yaml"}
        )
        assert resp.status_code == 400

    def test_add_model_missing_sie_id(self) -> None:
        yaml_body = "profiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
        resp = self.client.post("/v1/configs/models", content=yaml_body, headers={"Content-Type": "application/x-yaml"})
        assert resp.status_code == 422

    def test_add_model_persisted_to_store(self) -> None:
        yaml_body = "sie_id: stored/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        self.client.post("/v1/configs/models", content=yaml_body)
        store = self.app.state.config_store
        assert store.read_model("stored/model") is not None
        assert store.read_epoch() == 1

    def test_add_model_idempotent_profiles(self) -> None:
        yaml_body = "sie_id: new/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        resp1 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp1.status_code == 201
        resp2 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["created_profiles"] == []
        assert data["existing_profiles_skipped"] == ["default"]

    def test_add_model_conflicting_existing_profile_returns_409_without_store(self) -> None:
        client = TestClient(_create_test_app(self._bundles, self._models))
        yaml_body = (
            "sie_id: new/conflict\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:Bert\n"
            "    max_batch_tokens: 8192\n"
            "    max_sequence_length: 256\n"
        )
        resp1 = client.post("/v1/configs/models", content=yaml_body)
        assert resp1.status_code == 201

        conflict_body = yaml_body.replace("max_sequence_length: 256", "max_sequence_length: 128")
        resp2 = client.post("/v1/configs/models", content=conflict_body)
        assert resp2.status_code == 409
        data = resp2.json()
        assert data["detail"]["error"] == "content_conflict"
        assert data["detail"]["conflicting_profiles"] == ["default"]

    def test_add_model_conflicting_top_level_field_returns_409_without_store(self) -> None:
        client = TestClient(_create_test_app(self._bundles, self._models))
        yaml_body = (
            "sie_id: new/top-level-conflict\n"
            "max_sequence_length: 256\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:Bert\n"
            "    max_batch_tokens: 8192\n"
        )
        resp1 = client.post("/v1/configs/models", content=yaml_body)
        assert resp1.status_code == 201

        conflict_body = yaml_body.replace("max_sequence_length: 256", "max_sequence_length: 128")
        resp2 = client.post("/v1/configs/models", content=conflict_body)
        assert resp2.status_code == 409
        data = resp2.json()
        assert data["detail"]["error"] == "content_conflict"
        assert data["detail"]["conflicting_fields"] == ["max_sequence_length"]

    def test_list_models_includes_api_added(self) -> None:
        yaml_body = "sie_id: api/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        self.client.post("/v1/configs/models", content=yaml_body)
        resp = self.client.get("/v1/configs/models")
        models = resp.json()["models"]
        api_model = next(m for m in models if m["model_id"] == "api/model")
        assert api_model["source"] == "api"


class TestConfigAPIBundles:
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
        assert bundles[0]["engine"] == "pytorch"
        assert bundles[0]["priority"] == 10

    def test_get_bundle(self) -> None:
        resp = self.client.get("/v1/configs/bundles/default")
        assert resp.status_code == 200
        assert "application/x-yaml" in resp.headers["content-type"]
        assert "engine: pytorch" in resp.text

    def test_get_bundle_not_found(self) -> None:
        resp = self.client.get("/v1/configs/bundles/nonexistent")
        assert resp.status_code == 404


class TestConfigAPIEdgeCases:
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
        app = _create_test_app(self._bundles, self._models, config_store_dir=None)
        client = TestClient(app)
        yaml_body = "sie_id: mem/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201
        resp2 = client.get("/v1/configs/models")
        model_ids = [m["model_id"] for m in resp2.json()["models"]]
        assert "mem/model" in model_ids

    def test_post_with_nats_disconnected_returns_503(self) -> None:
        from unittest.mock import MagicMock

        from sie_config.nats_publisher import NatsPublisher

        app = _create_test_app(self._bundles, self._models)
        mock_nats = MagicMock(spec=NatsPublisher)
        mock_nats.connected = False
        app.state.nats_publisher = mock_nats
        client = TestClient(app)
        yaml_body = "sie_id: test/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 503
        assert "nats_unavailable" in resp.json()["detail"]["error"]

    def test_post_model_with_multiple_profiles(self) -> None:
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)
        yaml_body = "sie_id: multi/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n  custom:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 2\n"
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201
        data = resp.json()
        assert sorted(data["created_profiles"]) == ["custom", "default"]

    def test_post_empty_body_returns_400(self) -> None:
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)
        resp = client.post("/v1/configs/models", content="")
        assert resp.status_code == 400

    def test_auth_write_rejected_with_inference_token(self, monkeypatch) -> None:
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
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "admin-secret")
        monkeypatch.setenv("SIE_AUTH_TOKEN", "read-only")
        resp = client.get(
            "/v1/configs/models",
            headers={"Authorization": "Bearer read-only"},
        )
        assert resp.status_code == 200

    @pytest.mark.parametrize("env_var", ["SIE_ENV", "SIE_DEPLOYMENT_ENV"])
    def test_prod_without_any_token_refuses_reads_and_writes(self, monkeypatch, env_var: str) -> None:
        # A production deploy with NO auth token must fail closed, not serve open —
        # armed by EITHER env signal (SIE_ENV managed / SIE_DEPLOYMENT_ENV Helm).
        monkeypatch.delenv("SIE_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("SIE_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("SIE_ENV", raising=False)
        monkeypatch.delenv("SIE_DEPLOYMENT_ENV", raising=False)
        monkeypatch.setenv(env_var, "production")
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)
        yaml_body = "sie_id: test/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        assert client.get("/v1/configs/models").status_code == 403
        assert client.post("/v1/configs/models", content=yaml_body).status_code == 403

    def test_dev_without_any_token_stays_open(self, monkeypatch) -> None:
        # Self-host / dev (no prod env signal) keeps the open-localhost posture.
        monkeypatch.delenv("SIE_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("SIE_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("SIE_ENV", raising=False)
        monkeypatch.delenv("SIE_DEPLOYMENT_ENV", raising=False)
        app = _create_test_app(self._bundles, self._models)
        client = TestClient(app)
        assert client.get("/v1/configs/models").status_code == 200


class TestConfigAPIIdempotency:
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
        # Idempotency state now lives on `app.state` (see
        # `_get_idempotency_state`), so a fresh app per test gives us a
        # fresh cache/in-flight map automatically.

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_duplicate_key_returns_cached_response(self) -> None:
        yaml_body = "sie_id: idem/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        headers = {"Content-Type": "application/x-yaml", "Idempotency-Key": "test-key-1"}
        resp1 = self.client.post("/v1/configs/models", content=yaml_body, headers=headers)
        assert resp1.status_code == 201
        resp2 = self.client.post("/v1/configs/models", content=yaml_body, headers=headers)
        assert resp2.status_code == 201
        assert resp2.json() == resp1.json()

    def test_duplicate_key_different_body_returns_422(self) -> None:
        yaml_body_1 = "sie_id: idem/model-a\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        yaml_body_2 = "sie_id: idem/model-b\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 2\n"
        key = "test-key-mismatch"
        resp1 = self.client.post(
            "/v1/configs/models",
            content=yaml_body_1,
            headers={"Content-Type": "application/x-yaml", "Idempotency-Key": key},
        )
        assert resp1.status_code == 201
        resp2 = self.client.post(
            "/v1/configs/models",
            content=yaml_body_2,
            headers={"Content-Type": "application/x-yaml", "Idempotency-Key": key},
        )
        assert resp2.status_code == 422
        assert resp2.json()["detail"]["error"] == "idempotency_mismatch"

    def test_no_idempotency_key_skips_cache(self) -> None:
        yaml_body = "sie_id: nocache/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp1 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp1.status_code == 201
        resp2 = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp2.status_code == 200
        assert resp2.json()["existing_profiles_skipped"] == ["default"]


class TestConfigAPINATSPublishFailure:
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
        from unittest.mock import AsyncMock, MagicMock

        from sie_config.nats_publisher import NatsPublisher

        app = _create_test_app(self._bundles, self._models, str(self._store))
        mock_nats = MagicMock(spec=NatsPublisher)
        mock_nats.connected = True
        mock_nats.router_id = "test-router"
        mock_nats.publish_config_notification = AsyncMock(side_effect=RuntimeError("NATS down"))
        app.state.nats_publisher = mock_nats
        client = TestClient(app)
        yaml_body = "sie_id: nats/fail\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"
        resp = client.post("/v1/configs/models", content=yaml_body, headers={"Content-Type": "application/x-yaml"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_id"] == "nats/fail"
        assert any("nats_publish_failed" in w for w in data["warnings"])
        store = app.state.config_store
        assert store.read_model("nats/fail") is not None


class TestConfigAPIExport:
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

    def test_export_returns_snapshot(self) -> None:
        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot_version"] == 1
        assert data["epoch"] == 0
        assert "generated_at" in data
        assert data["bundle_config_hashes"]["default"]
        assert data["bundle_config_hashes"]["default"] == self.app.state.model_registry.compute_bundle_config_hash(
            "default"
        )
        assert data["bundle_pool_config_hashes"]["default"]["default"] == data["bundle_config_hashes"]["default"]
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "test/model"
        assert data["models"][0]["affected_bundles"] == ["default"]
        assert data["models"][0]["pool"] == "default"

    def test_export_includes_api_added_models(self) -> None:
        yaml_body = "sie_id: api/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        self.client.post("/v1/configs/models", content=yaml_body)
        resp = self.client.get("/v1/configs/export")
        data = resp.json()
        assert data["epoch"] == 1
        model_ids = [m["model_id"] for m in data["models"]]
        assert "api/model" in model_ids
        assert "test/model" in model_ids

    def test_export_empty_registry(self) -> None:
        (self._root / "empty_models").mkdir()
        app = _create_test_app(self._bundles, self._root / "empty_models")
        client = TestClient(app)
        resp = client.get("/v1/configs/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []
        assert data["bundle_config_hashes"] == {"default": ""}
        assert data["bundle_pool_config_hashes"] == {"default": {}}
        assert data["epoch"] == 0

    def test_export_bundle_pool_hashes_are_scoped_by_model_pool(self) -> None:
        _write_model(
            self._models,
            "tenant/model",
            "sie_server.adapters.bert_flash:BertFlashAdapter",
            pool="customer-a",
        )
        registry: ModelRegistry = self.app.state.model_registry
        registry.reload()

        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        data = resp.json()

        global_hash = data["bundle_config_hashes"]["default"]
        default_pool_hash = data["bundle_pool_config_hashes"]["default"]["default"]
        tenant_pool_hash = data["bundle_pool_config_hashes"]["default"]["customer-a"]
        assert default_pool_hash == registry.compute_bundle_config_hash_for_pool("default", "default")
        assert tenant_pool_hash == registry.compute_bundle_config_hash_for_pool("default", "customer-a")
        assert global_hash != default_pool_hash
        assert global_hash != tenant_pool_hash

    def test_export_generated_at_is_utc_iso8601(self) -> None:
        from datetime import datetime

        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        data = resp.json()
        parsed = datetime.fromisoformat(data["generated_at"])
        assert parsed.utcoffset() is not None, "generated_at must carry a timezone (expected UTC)"
        assert parsed.utcoffset().total_seconds() == 0, "generated_at must be UTC"

    def test_export_returns_raw_yaml_for_api_added_model(self) -> None:
        yaml_body = (
            "sie_id: api/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:Bert\n"
            "    max_batch_tokens: 8192\n"
        )
        post_resp = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Content-Type": "application/x-yaml"},
        )
        assert post_resp.status_code == 201

        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        models = {m["model_id"]: m for m in resp.json()["models"]}
        api_entry = models["api/model"]
        assert api_entry["raw_yaml"] is not None, (
            "API-added model must round-trip raw_yaml so the gateway bootstrap can replay it"
        )
        assert "api/model" in api_entry["raw_yaml"]
        assert api_entry["model_config"]["sie_id"] == "api/model"
        assert "default" in api_entry["model_config"]["profiles"]

    def test_export_epoch_monotonically_increases(self) -> None:
        yaml_a = "sie_id: a/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:A\n    max_batch_tokens: 1\n"
        yaml_b = "sie_id: b/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:B\n    max_batch_tokens: 1\n"

        self.client.post("/v1/configs/models", content=yaml_a)
        first = self.client.get("/v1/configs/export").json()["epoch"]

        self.client.post("/v1/configs/models", content=yaml_b)
        second = self.client.get("/v1/configs/export").json()["epoch"]

        assert second > first, f"epoch should increase after a write; got {first} -> {second}"


class TestConfigAPIExportNoConfigStore:
    """Export/propagation contract when `config.configStore.enabled=false` (default).

    Regression guards for the path where the control plane has no persistent
    ConfigStore: we still must serve a complete merged model YAML on
    `/v1/configs/export` and publish the full merged YAML on NATS deltas,
    so a fresh gateway that bootstraps after a write (or that only hears
    the delta) rebuilds the same registry state as one with a store.
    """

    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._bundles = self._root / "bundles"
        self._models = self._root / "models"
        self._bundles.mkdir()
        self._models.mkdir()
        _write_bundle(self._bundles, "default", ["sie_server.adapters.bert_flash"])
        # No config_store_dir → app.state.config_store is None.
        self.app = _create_test_app(self._bundles, self._models, config_store_dir=None)
        self.client = TestClient(self.app)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_export_reconstructs_full_yaml_for_api_added_model_without_store(self) -> None:
        yaml_body = (
            "sie_id: mem/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 8192\n"
        )
        post_resp = self.client.post("/v1/configs/models", content=yaml_body)
        assert post_resp.status_code == 201

        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        models = {m["model_id"]: m for m in resp.json()["models"]}
        entry = models["mem/model"]

        # Before the fix this collapsed to `{"sie_id": "mem/model"}` and
        # raw_yaml was None, so a fresh gateway bootstrapping from this
        # snapshot silently dropped the profile.
        assert entry["model_config"].get("profiles"), "export must include merged profiles even without a ConfigStore"
        assert "default" in entry["model_config"]["profiles"]
        assert entry["model_config"]["profiles"]["default"]["adapter_path"] == "sie_server.adapters.bert_flash:B"
        assert entry["raw_yaml"] is not None, "raw_yaml must round-trip so gateway replay on bootstrap is authoritative"

    def test_export_merges_appended_profile_without_store(self) -> None:
        first = (
            "sie_id: mem/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 1024\n"
        )
        assert self.client.post("/v1/configs/models", content=first).status_code == 201

        appended = (
            "sie_id: mem/model\n"
            "profiles:\n"
            "  fast:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 8192\n"
        )
        assert self.client.post("/v1/configs/models", content=appended).status_code == 201

        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        models = {m["model_id"]: m for m in resp.json()["models"]}
        entry = models["mem/model"]

        # Both profiles must survive in the exported snapshot — this is
        # the core contract the earlier body.decode()-only path broke.
        profiles = entry["model_config"]["profiles"]
        assert set(profiles.keys()) == {"default", "fast"}, (
            f"both profiles must be present after append; got {list(profiles.keys())}"
        )
        assert profiles["default"]["max_batch_tokens"] == 1024
        assert profiles["fast"]["max_batch_tokens"] == 8192

    def test_export_preserves_pool_on_append_without_store(self) -> None:
        first = (
            "sie_id: mem/model\n"
            "pool: Customer-A\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 1024\n"
        )
        assert self.client.post("/v1/configs/models", content=first).status_code == 201

        appended = (
            "sie_id: mem/model\n"
            "profiles:\n"
            "  fast:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 8192\n"
        )
        assert self.client.post("/v1/configs/models", content=appended).status_code == 201

        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 200
        models = {m["model_id"]: m for m in resp.json()["models"]}
        entry = models["mem/model"]
        assert entry["model_config"]["pool"] == "customer-a"
        assert entry["pool"] == "customer-a"
        assert set(entry["model_config"]["profiles"]) == {"default", "fast"}
        assert "pool: customer-a" in entry["raw_yaml"]

    def test_export_targets_variant_bundle_for_filesystem_model_without_store(self) -> None:
        _write_bundle(
            self._bundles,
            "default",
            ["sie_server.adapters.bert_flash", "sie_server.adapters.bge_m3_flash"],
        )
        _write_bundle(self._bundles, "candle", ["sie_server_rust.adapters.candle"], priority=30, engine="candle")
        (self._models / "BAAI__bge-m3.yaml").write_text(
            "sie_id: BAAI/bge-m3\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter\n"
            "    max_batch_tokens: 16384\n"
            "  candle:\n"
            "    extends: default\n"
            "    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter\n"
        )
        self.app = _create_test_app(self._bundles, self._models, config_store_dir=None)
        self.client = TestClient(self.app)

        resp = self.client.get("/v1/configs/export")

        assert resp.status_code == 200
        data = resp.json()
        models = {m["model_id"]: m for m in data["models"]}
        entry = models["BAAI/bge-m3"]
        assert entry["affected_bundles"] == ["default", "candle"]
        assert entry["pool"] == "default"
        assert set(entry["model_config"]["profiles"]) == {"default", "candle"}
        registry: ModelRegistry = self.app.state.model_registry
        assert data["bundle_pool_config_hashes"]["candle"]["default"] == registry.compute_bundle_config_hash_for_pool(
            "candle", "default"
        )
        candle_applied = [m for m in data["models"] if "candle" in m["affected_bundles"]]
        assert [m["model_id"] for m in candle_applied] == ["BAAI/bge-m3"]

    def test_export_prefers_registry_merge_for_filesystem_append_without_store(self) -> None:
        _write_bundle(
            self._bundles,
            "default",
            ["sie_server.adapters.bert_flash", "sie_server.adapters.bge_m3_flash"],
        )
        _write_bundle(self._bundles, "candle", ["sie_server_rust.adapters.candle"], priority=30, engine="candle")
        (self._models / "BAAI__bge-m3.yaml").write_text(
            "sie_id: BAAI/bge-m3\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter\n"
            "    max_batch_tokens: 16384\n"
        )
        self.app = _create_test_app(self._bundles, self._models, config_store_dir=None)
        self.client = TestClient(self.app)

        appended = (
            "sie_id: BAAI/bge-m3\n"
            "profiles:\n"
            "  candle:\n"
            "    extends: default\n"
            "    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter\n"
        )
        post = self.client.post("/v1/configs/models", content=appended)
        assert post.status_code == 201
        assert post.json()["routable_bundles_by_profile"] == {"candle": ["candle"]}

        resp = self.client.get("/v1/configs/export")

        assert resp.status_code == 200
        models = {m["model_id"]: m for m in resp.json()["models"]}
        entry = models["BAAI/bge-m3"]
        assert entry["affected_bundles"] == ["default", "candle"]
        assert set(entry["model_config"]["profiles"]) == {"default", "candle"}
        assert "candle:" in entry["raw_yaml"]

    def test_export_replaces_invalid_raw_yaml_with_registry_config(self) -> None:
        _write_model(self._models, "test/model", "sie_server.adapters.bert_flash:BertFlashAdapter")
        app = _create_test_app(self._bundles, self._models, str(self._root / "invalid_store"))
        client = TestClient(app)
        store: ConfigStore = app.state.config_store
        store.write_model("test/model", "profiles: [")

        resp = client.get("/v1/configs/export")

        assert resp.status_code == 200
        models = {m["model_id"]: m for m in resp.json()["models"]}
        entry = models["test/model"]
        assert entry["raw_yaml"] != "profiles: ["
        assert "profiles:" in entry["raw_yaml"]
        assert entry["model_config"]["profiles"]["default"]["adapter_path"] == (
            "sie_server.adapters.bert_flash:BertFlashAdapter"
        )

    def test_nats_delta_carries_merged_yaml_without_store(self) -> None:
        """On the NATS publish path, the delta payload must be the full
        merged model YAML (not the incremental request body) so a fresh
        gateway that only sees the delta rebuilds the complete model.
        """
        from unittest.mock import AsyncMock, MagicMock

        from sie_config.nats_publisher import NatsPublisher

        mock_nats = MagicMock(spec=NatsPublisher)
        mock_nats.connected = True
        mock_nats.router_id = "test-publisher"
        mock_nats.publish_config_notification = AsyncMock()
        self.app.state.nats_publisher = mock_nats

        first = (
            "sie_id: mem/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 1024\n"
        )
        assert self.client.post("/v1/configs/models", content=first).status_code == 201

        appended = (
            "sie_id: mem/model\n"
            "profiles:\n"
            "  fast:\n"
            "    adapter_path: sie_server.adapters.bert_flash:B\n"
            "    max_batch_tokens: 8192\n"
        )
        assert self.client.post("/v1/configs/models", content=appended).status_code == 201

        # Second publish (append) must carry both profiles in the YAML.
        last_call = mock_nats.publish_config_notification.await_args_list[-1]
        published_yaml = last_call.kwargs["model_config_yaml"]
        parsed = yaml.safe_load(published_yaml)
        assert parsed["sie_id"] == "mem/model"
        assert set(parsed.get("profiles", {}).keys()) == {"default", "fast"}, (
            f"NATS delta must carry merged profiles; got {list(parsed.get('profiles', {}).keys())}"
        )


class TestConfigAPIExportAuth:
    """Export is admin-gated. These tests protect the internal-only contract."""

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

    def test_export_unauth_when_admin_token_configured(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "the-real-admin")
        resp = self.client.get("/v1/configs/export")
        assert resp.status_code == 401
        assert "Missing Authorization header" in resp.text

    def test_export_forbidden_with_wrong_admin_token(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "the-real-admin")
        resp = self.client.get(
            "/v1/configs/export",
            headers={"Authorization": "Bearer WRONG-TOKEN"},
        )
        assert resp.status_code == 403

    def test_export_forbidden_with_inference_only_token(self, monkeypatch) -> None:
        # SIE_AUTH_TOKEN alone is not an admin credential, even for reads that
        # go through the write-auth gate (export is admin-only).
        monkeypatch.delenv("SIE_ADMIN_TOKEN", raising=False)
        monkeypatch.setenv("SIE_AUTH_TOKEN", "inference-token")
        resp = self.client.get(
            "/v1/configs/export",
            headers={"Authorization": "Bearer inference-token"},
        )
        assert resp.status_code == 403
        assert "SIE_ADMIN_TOKEN" in resp.text

    def test_export_allowed_with_correct_admin_token(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "the-real-admin")
        resp = self.client.get(
            "/v1/configs/export",
            headers={"Authorization": "Bearer the-real-admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot_version"] == 1
        assert any(m["model_id"] == "test/model" for m in data["models"])


class TestConfigAPIEpoch:
    """Lightweight epoch endpoint consumed by the Rust gateway's config poller."""

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

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_epoch_returns_zero_on_empty_store(self) -> None:
        resp = self.client.get("/v1/configs/epoch")
        assert resp.status_code == 200
        body = resp.json()
        assert body["epoch"] == 0
        # Bundle hash is non-empty whenever the registry has loaded any
        # bundles (the test fixture seeds one). The exact value is opaque to
        # the gateway and only compared as a string.
        assert isinstance(body["bundles_hash"], str)
        assert len(body["bundles_hash"]) == 64  # sha256 hex
        assert isinstance(body["bundle_config_hashes_hash"], str)
        assert len(body["bundle_config_hashes_hash"]) == 64  # sha256 hex

    def test_epoch_bundles_hash_changes_when_bundles_reload(self) -> None:
        # The hash is the gateway's only signal that bundles need re-fetching
        # from /v1/configs/bundles after a sie-config redeploy. If the value
        # is stable across a real bundle change, the gateway will silently
        # serve a stale adapter set until a model write happens to bump the
        # epoch (or until the gateway pod restarts) — the original bug this
        # whole change is closing.
        before = self.client.get("/v1/configs/epoch").json()["bundles_hash"]
        _write_bundle(self._bundles, "extra", ["sie_server.adapters.cross_encoder"])
        registry: ModelRegistry = self.app.state.model_registry
        registry.reload()
        after = self.client.get("/v1/configs/epoch").json()["bundles_hash"]
        assert before != after
        assert len(after) == 64

    def test_epoch_bundle_config_hashes_hash_changes_when_model_hash_changes(self) -> None:
        # This is the compact signal the gateway uses to detect stale
        # expected bundle_config_hash values without polling the full export
        # snapshot on every interval.
        before = self.client.get("/v1/configs/epoch").json()["bundle_config_hashes_hash"]
        yaml_body = (
            "sie_id: hash/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        self.client.post("/v1/configs/models", content=yaml_body)
        after = self.client.get("/v1/configs/epoch").json()["bundle_config_hashes_hash"]
        assert before != after
        assert len(after) == 64

    def test_epoch_bundle_config_hashes_hash_changes_when_only_model_pool_changes(self) -> None:
        # A top-level pool assignment is routing/readiness state, not part of
        # the worker-parity bundle_config_hash. The compact /epoch fingerprint
        # must still move so no-store/same-epoch redeploys trigger export
        # recovery when only model ownership changes.
        _write_model(self._models, "pool/hash-model", "sie_server.adapters.bert_flash:A")
        registry: ModelRegistry = self.app.state.model_registry
        registry.reload()

        before_epoch = self.client.get("/v1/configs/epoch").json()
        before_export = self.client.get("/v1/configs/export").json()
        before_bundle_hash = before_export["bundle_config_hashes"]["default"]

        _write_model(
            self._models,
            "pool/hash-model",
            "sie_server.adapters.bert_flash:A",
            pool="customer-a",
        )
        registry.reload()

        after_epoch = self.client.get("/v1/configs/epoch").json()
        after_export = self.client.get("/v1/configs/export").json()
        after_bundle_hash = after_export["bundle_config_hashes"]["default"]

        assert before_epoch["epoch"] == after_epoch["epoch"] == 0
        assert before_bundle_hash == after_bundle_hash
        assert before_epoch["bundle_config_hashes_hash"] != after_epoch["bundle_config_hashes_hash"]
        models = {m["model_id"]: m for m in after_export["models"]}
        assert models["pool/hash-model"]["model_config"]["pool"] == "customer-a"

    def test_epoch_advances_after_write(self) -> None:
        before = self.client.get("/v1/configs/epoch").json()["epoch"]
        yaml_body = (
            "sie_id: a/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        self.client.post("/v1/configs/models", content=yaml_body)
        after = self.client.get("/v1/configs/epoch").json()["epoch"]
        assert after > before, f"epoch should advance after a write; got {before} -> {after}"

    def test_epoch_accepts_read_auth_token(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", "inference-token")
        monkeypatch.delenv("SIE_ADMIN_TOKEN", raising=False)
        resp = self.client.get(
            "/v1/configs/epoch",
            headers={"Authorization": "Bearer inference-token"},
        )
        assert resp.status_code == 200
        assert "epoch" in resp.json()

    def test_epoch_accepts_admin_token(self, monkeypatch) -> None:
        # Read auth accepts the admin token too — the gateway passes its
        # admin token here rather than maintaining two credentials.
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "real-admin")
        resp = self.client.get(
            "/v1/configs/epoch",
            headers={"Authorization": "Bearer real-admin"},
        )
        assert resp.status_code == 200

    def test_epoch_rejects_missing_token_when_auth_configured(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "real-admin")
        resp = self.client.get("/v1/configs/epoch")
        assert resp.status_code == 401

    def test_epoch_rejects_wrong_token(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", "right")
        resp = self.client.get(
            "/v1/configs/epoch",
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 403


class TestWritePathOrdering:
    """Regression tests for fixes #5/#6/#10: persist before mutate, atomic
    epoch increment, idempotent re-execution safety, and partial-publish
    reporting.
    """

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

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_status_suffix_model_id_rejected(self) -> None:
        # Fix #9 complement on sie-config side: model IDs ending in
        # /status collide with the gateway's status endpoint route and
        # must be rejected at ingest.
        yaml_body = (
            "sie_id: foo/status\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        resp = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 400
        assert "/status" in resp.text

    def test_invalid_config_does_not_create_disk_artifact(self) -> None:
        # Fix #10 regression: validation failures must abort BEFORE the
        # on-disk model file is written. Otherwise a 422 would still
        # leave a stale YAML that the next reload picks up.
        yaml_body = (
            "sie_id: bad/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: nonexistent.adapter.module:Adapter\n"
            "    max_batch_tokens: 1\n"
        )
        resp = self.client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 422
        # No model file should have been written.
        expected_path = self._store / "models" / "bad__model.yaml"
        assert not expected_path.exists(), (
            "422 validation failure left an on-disk artifact; write-before-validate reintroduced"
        )
        # Epoch must not have advanced either (fix #5 contract).
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 0

    def test_sequential_writes_produce_monotonic_epochs(self) -> None:
        # Fix #5 contract (sequential baseline): every successful write
        # bumps the epoch by exactly one. Concurrent asyncio-level
        # contention is covered by the Rust gateway tests that exercise
        # the consumer side — the Python fix relies on `_WRITE_LOCK`
        # serialization and is best verified structurally (the lock is
        # module-level and wraps the full critical section).
        for i in range(5):
            yaml_body = (
                f"sie_id: seq/model-{i}\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.bert_flash:A\n"
                "    max_batch_tokens: 1\n"
            )
            resp = self.client.post("/v1/configs/models", content=yaml_body)
            assert resp.status_code == 201, resp.text
        final_epoch = self.client.get("/v1/configs/epoch").json()["epoch"]
        assert final_epoch == 5, f"expected 5 epoch bumps, got {final_epoch}"


class TestIdempotencyEvictionSafety:
    """Regression for fix #11: a waiter that wakes after the in-flight
    request's cache entry was evicted must NOT re-execute the write.
    """

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

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_idempotency_key_replay_hits_cache(self) -> None:
        # Baseline sanity check for the idempotency path: a repeat call
        # with the same Idempotency-Key and body must return the cached
        # response (not re-execute the write and not double-bump the
        # epoch). Fix #11 tightens the failure mode when the cache was
        # evicted between the in-flight wait and the re-read; that path
        # is covered by code review and the `already_waited` flag in
        # `config_api.add_model` (see the module docstring there).
        yaml_body = (
            "sie_id: idem/model\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        resp1 = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Idempotency-Key": "K1"},
        )
        assert resp1.status_code == 201, resp1.text
        epoch_after_first = self.client.get("/v1/configs/epoch").json()["epoch"]

        resp2 = self.client.post(
            "/v1/configs/models",
            content=yaml_body,
            headers={"Idempotency-Key": "K1"},
        )
        assert resp2.status_code == 201
        assert resp2.json() == resp1.json()
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == epoch_after_first, (
            "idempotent replay must NOT bump the epoch"
        )

    def test_concurrent_same_key_waiter_gets_cached_success(self, monkeypatch) -> None:
        body = (
            "sie_id: idem/concurrent-success\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        original_write_model = self.app.state.config_store.write_model
        release_write = threading.Event()
        calls: list[str] = []

        async def run() -> None:
            loop = asyncio.get_running_loop()
            write_started = asyncio.Event()

            def blocking_write_model(model_id: str, config_yaml: str) -> None:
                calls.append(model_id)
                loop.call_soon_threadsafe(write_started.set)
                if not release_write.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release write")
                original_write_model(model_id, config_yaml)

            monkeypatch.setattr(self.app.state.config_store, "write_model", blocking_write_model)

            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                owner = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-success"})
                )
                await asyncio.wait_for(write_started.wait(), timeout=1)

                waiter = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-success"})
                )
                await asyncio.sleep(0.05)
                assert not waiter.done(), "same-key request did not wait on the in-flight owner"

                release_write.set()
                owner_resp, waiter_resp = await asyncio.gather(owner, waiter)

                assert owner_resp.status_code == 201, owner_resp.text
                assert waiter_resp.status_code == 201, waiter_resp.text
                assert waiter_resp.json() == owner_resp.json()
                assert calls == ["idem/concurrent-success"]

        asyncio.run(run())
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 1

    def test_concurrent_same_key_different_body_returns_422_while_owner_in_flight(self, monkeypatch) -> None:
        owner_body = (
            "sie_id: idem/concurrent-mismatch-a\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        mismatch_body = (
            "sie_id: idem/concurrent-mismatch-b\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        original_write_model = self.app.state.config_store.write_model
        release_write = threading.Event()
        calls: list[str] = []

        async def run() -> None:
            loop = asyncio.get_running_loop()
            write_started = asyncio.Event()

            def blocking_write_model(model_id: str, config_yaml: str) -> None:
                calls.append(model_id)
                loop.call_soon_threadsafe(write_started.set)
                if not release_write.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release write")
                original_write_model(model_id, config_yaml)

            monkeypatch.setattr(self.app.state.config_store, "write_model", blocking_write_model)

            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                owner = asyncio.create_task(
                    ac.post("/v1/configs/models", content=owner_body, headers={"Idempotency-Key": "K-mismatch"})
                )
                await asyncio.wait_for(write_started.wait(), timeout=1)

                try:
                    mismatch = await asyncio.wait_for(
                        ac.post(
                            "/v1/configs/models",
                            content=mismatch_body,
                            headers={"Idempotency-Key": "K-mismatch"},
                        ),
                        timeout=1,
                    )
                    assert mismatch.status_code == 422, mismatch.text
                    assert mismatch.json()["detail"]["error"] == "idempotency_mismatch"
                finally:
                    release_write.set()

                owner_resp = await asyncio.wait_for(owner, timeout=1)
                assert owner_resp.status_code == 201, owner_resp.text
                assert calls == ["idem/concurrent-mismatch-a"]

        asyncio.run(run())
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 1

    def test_concurrent_same_key_waiter_gets_owner_failure(self, monkeypatch) -> None:
        body = (
            "sie_id: idem/concurrent-failure\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        release_write = threading.Event()
        calls: list[str] = []

        async def run() -> None:
            loop = asyncio.get_running_loop()
            write_started = asyncio.Event()

            def failing_write_model(model_id: str, _config_yaml: str) -> None:
                calls.append(model_id)
                loop.call_soon_threadsafe(write_started.set)
                if not release_write.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release write")
                raise RuntimeError("injected store write failure")

            monkeypatch.setattr(self.app.state.config_store, "write_model", failing_write_model)

            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                owner = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-failure"})
                )
                await asyncio.wait_for(write_started.wait(), timeout=1)

                waiter = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-failure"})
                )
                await asyncio.sleep(0.05)
                assert not waiter.done(), "same-key request did not wait on the in-flight owner"

                release_write.set()
                owner_resp, waiter_resp = await asyncio.gather(owner, waiter)

                assert owner_resp.status_code == 500, owner_resp.text
                assert waiter_resp.status_code == 500, waiter_resp.text
                assert "idempotent_replay_evicted" not in waiter_resp.text
                assert calls == ["idem/concurrent-failure"]

        asyncio.run(run())
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 0

    def test_waiter_reports_failed_owner_even_if_later_retry_starts_first(self, monkeypatch) -> None:
        body = (
            "sie_id: idem/concurrent-failure-retry\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        original_write_model = self.app.state.config_store.write_model
        release_first_write = threading.Event()
        calls: list[str] = []
        call_lock = threading.Lock()

        async def run() -> None:
            loop = asyncio.get_running_loop()
            write_started = asyncio.Event()
            write_failed = asyncio.Event()

            def first_write_fails(model_id: str, config_yaml: str) -> None:
                with call_lock:
                    calls.append(model_id)
                    call_index = len(calls)
                if call_index == 1:
                    loop.call_soon_threadsafe(write_started.set)
                    if not release_first_write.wait(timeout=5):
                        raise RuntimeError("timed out waiting to release failed write")
                    loop.call_soon_threadsafe(write_failed.set)
                    raise RuntimeError("injected first store write failure")
                original_write_model(model_id, config_yaml)

            monkeypatch.setattr(self.app.state.config_store, "write_model", first_write_fails)

            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                owner = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-fail-retry"})
                )
                await asyncio.wait_for(write_started.wait(), timeout=1)

                waiter = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-fail-retry"})
                )
                await asyncio.sleep(0.05)
                assert not waiter.done(), "same-key request did not wait on the in-flight owner"

                idem = self.app.state._config_idempotency_state
                await idem.lock.acquire()
                try:
                    release_first_write.set()
                    await asyncio.wait_for(write_failed.wait(), timeout=1)
                    await asyncio.sleep(0.05)
                    retry = asyncio.create_task(
                        ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-fail-retry"})
                    )
                    await asyncio.sleep(0.05)
                finally:
                    idem.lock.release()

                owner_resp, waiter_resp, retry_resp = await asyncio.gather(owner, waiter, retry)

                assert owner_resp.status_code == 500, owner_resp.text
                assert waiter_resp.status_code == 500, waiter_resp.text
                assert "idempotent_replay_evicted" not in waiter_resp.text
                assert retry_resp.status_code == 201, retry_resp.text
                assert calls == ["idem/concurrent-failure-retry", "idem/concurrent-failure-retry"]

        asyncio.run(run())
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 1

    def test_concurrent_same_key_cancellation_waits_for_durable_outcome(self, monkeypatch) -> None:
        body = (
            "sie_id: idem/concurrent-cancel\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        original_write_model = self.app.state.config_store.write_model
        release_write = threading.Event()
        calls: list[str] = []
        call_lock = threading.Lock()

        async def run() -> None:
            loop = asyncio.get_running_loop()
            write_started = asyncio.Event()

            def slow_write_model(model_id: str, config_yaml: str) -> None:
                with call_lock:
                    calls.append(model_id)
                    call_index = len(calls)
                if call_index == 1:
                    loop.call_soon_threadsafe(write_started.set)
                    if not release_write.wait(timeout=5):
                        raise RuntimeError("timed out waiting to release cancelled write")
                original_write_model(model_id, config_yaml)

            monkeypatch.setattr(self.app.state.config_store, "write_model", slow_write_model)

            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                owner = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-cancel"})
                )
                await asyncio.wait_for(write_started.wait(), timeout=1)

                waiter = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-cancel"})
                )
                await asyncio.sleep(0.05)
                assert not waiter.done(), "same-key request did not wait on the in-flight owner"

                owner.cancel()
                await asyncio.sleep(0.05)
                assert not owner.done(), "cancelled owner returned before the store write finished"
                assert not waiter.done(), "waiter returned before the owner published its durable outcome"
                release_write.set()

                try:
                    owner_result = await asyncio.wait_for(owner, timeout=1)
                except asyncio.CancelledError:
                    pass
                else:
                    pytest.fail(f"cancelled owner unexpectedly returned {owner_result!r}")

                waiter_resp = await asyncio.wait_for(waiter, timeout=1)
                assert waiter_resp.status_code == 201, waiter_resp.text

                retry_resp = await ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-cancel"})
                assert retry_resp.status_code == 201, retry_resp.text
                assert retry_resp.json() == waiter_resp.json()

        asyncio.run(run())
        assert calls == ["idem/concurrent-cancel"]
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 1

    def test_pre_commit_cancellation_cleanup_survives_second_cancel(self, monkeypatch) -> None:
        body = (
            "sie_id: idem/pre-commit-cancel\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        original_read_model = self.app.state.config_store.read_model
        release_read = threading.Event()
        calls: list[str] = []

        async def run() -> None:
            loop = asyncio.get_running_loop()
            read_started = asyncio.Event()

            def blocking_read_model(model_id: str) -> str | None:
                calls.append(model_id)
                loop.call_soon_threadsafe(read_started.set)
                if not release_read.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release read")
                return original_read_model(model_id)

            monkeypatch.setattr(self.app.state.config_store, "read_model", blocking_read_model)

            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                owner = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-pre-cancel"})
                )
                await asyncio.wait_for(read_started.wait(), timeout=1)

                waiter = asyncio.create_task(
                    ac.post("/v1/configs/models", content=body, headers={"Idempotency-Key": "K-pre-cancel"})
                )
                await asyncio.sleep(0.05)
                assert not waiter.done(), "same-key request did not wait on the in-flight owner"

                idem = self.app.state._config_idempotency_state
                await idem.lock.acquire()
                try:
                    owner.cancel()
                    await asyncio.sleep(0.05)
                    assert not owner.done(), "owner cleanup finished while idempotency lock was held"
                    assert not waiter.done(), "waiter returned before cancellation cleanup completed"

                    owner.cancel()
                    await asyncio.sleep(0.05)
                    assert not owner.done(), "second cancellation interrupted idempotency cleanup"
                    assert not waiter.done(), "second cancellation left waiter with a partial outcome"
                finally:
                    idem.lock.release()
                    release_read.set()

                try:
                    owner_result = await asyncio.wait_for(owner, timeout=1)
                except asyncio.CancelledError:
                    pass
                else:
                    pytest.fail(f"cancelled owner unexpectedly returned {owner_result!r}")

                waiter_resp = await asyncio.wait_for(waiter, timeout=1)
                assert waiter_resp.status_code == 503, waiter_resp.text
                assert waiter_resp.json()["detail"]["error"] == "idempotent_inflight_cancelled"

        asyncio.run(run())
        assert calls == ["idem/pre-commit-cancel"]
        assert self.client.get("/v1/configs/epoch").json()["epoch"] == 0

    def test_concurrent_writes_serialize_and_bump_epoch_once_each(self) -> None:
        # Real concurrent-write regression: fire off N POSTs from the same
        # async event loop and assert (a) every write succeeded, (b) the
        # final epoch equals exactly N (so no lost bump), (c) every write
        # observed a distinct epoch value in its response chain. Starlette's
        # TestClient is sync, so we drive this through an AsyncClient +
        # asyncio.gather to actually exercise the `_WRITE_LOCK` contention
        # — without the lock, two concurrent `increment_epoch` calls both
        # read N, both write N+1, and the final epoch would be N+1 instead
        # of N+M.
        async def run() -> None:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                bodies = [
                    (
                        f"sie_id: race/model-{i}\n"
                        "profiles:\n"
                        "  default:\n"
                        "    adapter_path: sie_server.adapters.bert_flash:A\n"
                        "    max_batch_tokens: 1\n"
                    )
                    for i in range(10)
                ]
                results = await asyncio.gather(*(ac.post("/v1/configs/models", content=b) for b in bodies))
                for r in results:
                    assert r.status_code == 201, r.text
                resp = await ac.get("/v1/configs/epoch")
                assert resp.json()["epoch"] == 10, (
                    "concurrent writes must each bump the epoch exactly once; "
                    "if this fails, `_WRITE_LOCK` is not serializing the "
                    "read-modify-write on `config_store.increment_epoch`."
                )

        asyncio.run(run())

    def test_export_snapshot_is_consistent_with_concurrent_writes(self) -> None:
        # TOCTOU regression for `GET /export`: the returned `(epoch, models)`
        # pair MUST be a real serialization point. Specifically, the dangerous
        # escape is `epoch_returned > state_reflected_by_models` — i.e. a
        # snapshot that says "we're at epoch N+1" but whose `models` list
        # predates the write that bumped the epoch. A gateway bootstrapping
        # on such a snapshot would set `ConfigEpoch = N+1` with model M
        # missing; the poller would see `remote == local`, log "in sync",
        # and silently wedge.
        #
        # We verify the consistency invariant holds under interleaved
        # writes + exports. For every export we capture, the set of models
        # in the snapshot must be ≥ the set that had been persisted by the
        # time the returned `epoch` was assigned. We check that the
        # snapshot's `epoch` never exceeds `len(models_in_snapshot) -
        # preseed_count` (each successful write bumps the epoch once and
        # adds exactly one model).
        async def run() -> None:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:

                async def writer(i: int) -> None:
                    body = (
                        f"sie_id: export_race/model-{i}\n"
                        "profiles:\n"
                        "  default:\n"
                        "    adapter_path: sie_server.adapters.bert_flash:A\n"
                        "    max_batch_tokens: 1\n"
                    )
                    r = await ac.post("/v1/configs/models", content=body)
                    assert r.status_code == 201, r.text

                async def exporter() -> dict:
                    r = await ac.get("/v1/configs/export")
                    assert r.status_code == 200, r.text
                    return r.json()

                # Seed state so `test/model` from setup_method is present.
                preseed = await exporter()
                preseed_count = len(preseed["models"])
                preseed_epoch = preseed["epoch"]

                # Interleave 10 writes with 20 exports. Every export we
                # capture must satisfy the invariant below.
                export_tasks = [exporter() for _ in range(20)]
                write_tasks = [writer(i) for i in range(10)]
                results = await asyncio.gather(*(write_tasks + export_tasks))
                exports = [r for r in results if isinstance(r, dict)]

                for snap in exports:
                    snap_epoch = snap["epoch"]
                    snap_models = {m["model_id"] for m in snap["models"]}
                    race_models = {m for m in snap_models if m.startswith("export_race/")}
                    committed_writes = snap_epoch - preseed_epoch
                    assert len(race_models) >= committed_writes, (
                        f"/export invariant violated: epoch={snap_epoch} "
                        f"(committed writes={committed_writes}) but only "
                        f"{len(race_models)} race models in snapshot. "
                        "This is the `epoch > state` TOCTOU bug that would "
                        "make the gateway poller silently wedge."
                    )
                    assert len(snap_models) - preseed_count >= committed_writes, (
                        f"/export invariant violated: total non-preseed models="
                        f"{len(snap_models) - preseed_count} < committed writes="
                        f"{committed_writes}."
                    )

                # After everything drains, final export must reflect the
                # full set of 10 writes and epoch == preseed_epoch + 10.
                final = await exporter()
                final_race = {m["model_id"] for m in final["models"] if m["model_id"].startswith("export_race/")}
                assert len(final_race) == 10
                assert final["epoch"] == preseed_epoch + 10

        asyncio.run(run())

    def test_idempotency_key_mismatched_body_returns_422(self) -> None:
        yaml_body_a = (
            "sie_id: idem/model-a\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        yaml_body_b = (
            "sie_id: idem/model-b\n"
            "profiles:\n"
            "  default:\n"
            "    adapter_path: sie_server.adapters.bert_flash:A\n"
            "    max_batch_tokens: 1\n"
        )
        resp1 = self.client.post(
            "/v1/configs/models",
            content=yaml_body_a,
            headers={"Idempotency-Key": "K2"},
        )
        assert resp1.status_code == 201
        resp2 = self.client.post(
            "/v1/configs/models",
            content=yaml_body_b,
            headers={"Idempotency-Key": "K2"},
        )
        assert resp2.status_code == 422
        assert "idempotency_mismatch" in resp2.text


class TestMergePreservesTopLevelFields:
    """Appending a profile via `POST /v1/configs/models` must merge on
    top of the stored document, not replace it. A minimal append body
    cannot erase previously-written top-level fields (`description`,
    `default_bundle`, ...); conflicting values raise 409 because the
    config API is append-only for model metadata.
    """

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

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_append_preserves_existing_top_level_fields(self) -> None:
        # First write: model with extra top-level metadata.
        resp1 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: acme/bert\n"
                "description: keep-me-around\n"
                "default_bundle: premium\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 4096\n"
            ),
        )
        assert resp1.status_code == 201

        # Append a second profile with a minimal body (no description /
        # default_bundle in the incoming payload).
        resp2 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: acme/bert\n"
                "profiles:\n"
                "  fast:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 8192\n"
            ),
        )
        assert resp2.status_code == 201

        stored_path = self._store / "models" / "acme__bert.yaml"
        stored = yaml.safe_load(stored_path.read_text())
        assert stored["description"] == "keep-me-around"
        assert stored["default_bundle"] == "premium"
        assert set(stored["profiles"].keys()) == {"default", "fast"}

    def test_conflicting_top_level_field_returns_409(self) -> None:
        resp1 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: acme/bert\n"
                "description: initial\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 4096\n"
            ),
        )
        assert resp1.status_code == 201

        # Reusing the same sie_id but mutating `description` must fail 409
        # — config API is append-only for top-level metadata.
        resp2 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: acme/bert\n"
                "description: mutated!\n"
                "profiles:\n"
                "  fast:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 8192\n"
            ),
        )
        assert resp2.status_code == 409
        body = resp2.json()
        assert body["detail"]["error"] == "content_conflict"
        assert "description" in body["detail"]["conflicting_fields"]

    def test_append_can_introduce_new_top_level_field(self) -> None:
        resp1 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: acme/bert\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 4096\n"
            ),
        )
        assert resp1.status_code == 201

        resp2 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: acme/bert\n"
                "description: added-later\n"
                "profiles:\n"
                "  fast:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 8192\n"
            ),
        )
        assert resp2.status_code == 201

        stored = yaml.safe_load((self._store / "models" / "acme__bert.yaml").read_text())
        assert stored["description"] == "added-later"


class TestRejectUnroutableModels:
    """A model whose new profiles do not contribute any `adapter_path`
    that a known bundle owns cannot be routed and must be rejected.
    This covers new models whose profiles are all `extends`-only (the
    resolved `affected_bundles` is empty) as well as the equivalent
    append case.
    """

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

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_new_model_with_only_extends_profiles_rejected(self) -> None:
        resp = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: extends-only/model\nprofiles:\n  derived:\n    extends: base\n    max_batch_tokens: 4096\n"
            ),
        )
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["detail"]["error"] == "validation_error"

    def test_new_model_with_adapter_path_still_accepted(self) -> None:
        # Control: an otherwise-identical model that DOES resolve to the
        # default bundle must still succeed, so the fix doesn't regress
        # normal writes.
        resp = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: ok/model\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 4096\n"
            ),
        )
        assert resp.status_code == 201, resp.text

    def test_append_extends_profile_to_routable_model_accepted(self) -> None:
        # Appending an `extends`-only profile is fine IF the model
        # already has a routable profile (the existing adapter module
        # still maps to a bundle). The fix must not regress this.
        resp1 = self.client.post(
            "/v1/configs/models",
            content=(
                "sie_id: composite/model\n"
                "profiles:\n"
                "  base:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
                "    max_batch_tokens: 4096\n"
            ),
        )
        assert resp1.status_code == 201, resp1.text

        resp2 = self.client.post(
            "/v1/configs/models",
            content=("sie_id: composite/model\nprofiles:\n  fast:\n    extends: base\n    max_batch_tokens: 8192\n"),
        )
        assert resp2.status_code == 201, resp2.text


class TestMissingRegistryReturns503:
    """When `app.state.model_registry` is `None` (registry init failed;
    see `app_factory._model_registry`), config routes must return a
    structured 503 via `_require_model_registry`. This keeps them on
    the same contract as `/readyz` instead of surfacing AttributeError
    as an unhandled HTTP 500.
    """

    def _app_without_registry(self) -> FastAPI:
        app = FastAPI()
        app.include_router(config_router)
        app.state.model_registry = None
        app.state.nats_publisher = None
        app.state.config_store = None
        return app

    def test_list_models_returns_503_when_registry_none(self) -> None:
        client = TestClient(self._app_without_registry())
        resp = client.get("/v1/configs/models")
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "registry_unavailable"

    def test_list_bundles_returns_503_when_registry_none(self) -> None:
        client = TestClient(self._app_without_registry())
        resp = client.get("/v1/configs/bundles")
        assert resp.status_code == 503

    def test_resolve_returns_503_when_registry_none(self) -> None:
        client = TestClient(self._app_without_registry())
        resp = client.post("/v1/configs/resolve", content="sie_id: foo/bar\n")
        assert resp.status_code == 503

    def test_get_model_returns_503_when_registry_none(self) -> None:
        client = TestClient(self._app_without_registry())
        resp = client.get("/v1/configs/models/foo/bar")
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "registry_unavailable"

    def test_get_bundle_returns_503_when_registry_none(self) -> None:
        client = TestClient(self._app_without_registry())
        resp = client.get("/v1/configs/bundles/default")
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "registry_unavailable"

    def test_add_model_returns_503_when_registry_none(self) -> None:
        client = TestClient(self._app_without_registry())
        resp = client.post(
            "/v1/configs/models",
            content=(
                "sie_id: foo/bar\n"
                "profiles:\n"
                "  default:\n"
                "    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
            ),
        )
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "registry_unavailable"

    def test_epoch_works_without_registry(self) -> None:
        # /epoch is the gateway's liveness signal for sie-config and must
        # keep answering even during a registry init failure. The bundles
        # hash degrades to the empty string so the gateway treats the
        # registry-absent state as "nothing to sync" rather than as a real
        # change worth re-fetching against.
        app = self._app_without_registry()
        app.state.config_store = None
        client = TestClient(app)
        resp = client.get("/v1/configs/epoch")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "epoch": 0,
            "bundles_hash": "",
            "bundle_config_hashes_hash": "",
        }

    def test_export_returns_503_when_registry_none(self) -> None:
        # /export reads from the registry so it must 503, matching the
        # rest of the config surface.
        client = TestClient(self._app_without_registry())
        resp = client.get("/v1/configs/export")
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "registry_unavailable"


class TestConfigAPIReplace:
    """PUT /v1/configs/models/{id}: the catalog-convergence write (#1771).

    Unlike append-only POST, PUT replaces a model's stored config wholesale so
    a catalog entry whose content has drifted from the source YAML converges
    (the stale-bge-m3 heal). Unchanged content is a genuine no-op (no epoch
    churn). Same admin write-auth + Idempotency-Key semantics as POST.
    """

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

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    @staticmethod
    def _yaml(sie_id: str, *, tokens: int = 8192, profiles: str = "default") -> str:
        blocks = "".join(
            f"  {p}:\n    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n    max_batch_tokens: {tokens}\n"
            for p in profiles.split(",")
        )
        return f"sie_id: {sie_id}\nprofiles:\n{blocks}"

    def test_put_creates_new_model(self) -> None:
        resp = self.client.put("/v1/configs/models/new/model", content=self._yaml("new/model"))
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["model_id"] == "new/model"
        assert data["replaced"] is True
        assert data["unchanged"] is False
        assert self.app.state.config_store.read_model("new/model") is not None
        assert self.app.state.config_store.read_epoch() == 1

    def test_put_heals_a_differing_entry(self) -> None:
        # Seed a "stale" entry (append-only POST would 409 on a changed
        # profile) then converge it with a full replace.
        assert self.client.post("/v1/configs/models", content=self._yaml("stale/model", tokens=1024)).status_code == 201
        resp = self.client.put("/v1/configs/models/stale/model", content=self._yaml("stale/model", tokens=4096))
        assert resp.status_code == 200, resp.text
        assert resp.json()["replaced"] is True

        export = self.client.get("/v1/configs/export")
        entry = {m["model_id"]: m for m in export.json()["models"]}["stale/model"]
        assert entry["model_config"]["profiles"]["default"]["max_batch_tokens"] == 4096

    def test_put_unchanged_content_is_a_noop_no_epoch_bump(self) -> None:
        body = self._yaml("noop/model")
        assert self.client.put("/v1/configs/models/noop/model", content=body).status_code == 200
        epoch_after_first = self.app.state.config_store.read_epoch()
        assert epoch_after_first == 1

        resp = self.client.put("/v1/configs/models/noop/model", content=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["unchanged"] is True
        assert resp.json()["replaced"] is False
        # No churn: the epoch must not advance on an unchanged re-sync.
        assert self.app.state.config_store.read_epoch() == epoch_after_first

    def test_put_drops_stale_profiles(self) -> None:
        # Seed two profiles, then replace with only one — the dropped profile
        # must not survive (replace is not a merge).
        assert (
            self.client.put(
                "/v1/configs/models/multi/model", content=self._yaml("multi/model", profiles="default,fast")
            ).status_code
            == 200
        )
        resp = self.client.put("/v1/configs/models/multi/model", content=self._yaml("multi/model", profiles="default"))
        assert resp.status_code == 200, resp.text

        export = self.client.get("/v1/configs/export")
        entry = {m["model_id"]: m for m in export.json()["models"]}["multi/model"]
        assert set(entry["model_config"]["profiles"]) == {"default"}

    def test_put_body_sie_id_mismatch_is_400(self) -> None:
        resp = self.client.put("/v1/configs/models/path/id", content=self._yaml("other/id"))
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "model_id_mismatch"

    def test_put_defaults_sie_id_from_path(self) -> None:
        body = (
            "profiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:Bert\n    max_batch_tokens: 8192\n"
        )
        resp = self.client.put("/v1/configs/models/derived/id", content=body)
        assert resp.status_code == 200, resp.text
        assert resp.json()["model_id"] == "derived/id"

    def test_put_unroutable_adapter_is_422(self) -> None:
        body = "profiles:\n  default:\n    adapter_path: sie_server.adapters.unknown:X\n    max_batch_tokens: 8192\n"
        resp = self.client.put("/v1/configs/models/bad/model", content=body)
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "validation_error"

    def test_put_invalid_yaml_is_400(self) -> None:
        resp = self.client.put("/v1/configs/models/x/y", content="{{bad")
        assert resp.status_code == 400

    def test_put_requires_admin_token(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_ADMIN_TOKEN", "admin-secret")
        resp = self.client.put("/v1/configs/models/auth/model", content=self._yaml("auth/model"))
        assert resp.status_code == 401  # missing Authorization header
        resp = self.client.put(
            "/v1/configs/models/auth/model",
            content=self._yaml("auth/model"),
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 403
        resp = self.client.put(
            "/v1/configs/models/auth/model",
            content=self._yaml("auth/model"),
            headers={"Authorization": "Bearer admin-secret"},
        )
        assert resp.status_code == 200, resp.text

    def test_put_idempotency_key_replay(self) -> None:
        body = self._yaml("idem/model")
        headers = {"Idempotency-Key": "put-k1"}
        r1 = self.client.put("/v1/configs/models/idem/model", content=body, headers=headers)
        assert r1.status_code == 200, r1.text
        r2 = self.client.put("/v1/configs/models/idem/model", content=body, headers=headers)
        assert r2.status_code == 200
        assert r2.json() == r1.json()
        # Exactly one write applied.
        assert self.app.state.config_store.read_epoch() == 1

    def test_put_idempotency_key_reuse_different_body_is_422(self) -> None:
        headers = {"Idempotency-Key": "put-k2"}
        assert (
            self.client.put(
                "/v1/configs/models/idem2/model", content=self._yaml("idem2/model", tokens=1024), headers=headers
            ).status_code
            == 200
        )
        resp = self.client.put(
            "/v1/configs/models/idem2/model", content=self._yaml("idem2/model", tokens=2048), headers=headers
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "idempotency_mismatch"

    def test_put_idempotency_same_key_body_different_path_does_not_false_replay(self) -> None:
        # The idempotency cache is application-wide, so the fingerprint folds in
        # the request target (model path). A body that omits sie_id (filled from
        # the path) is byte-identical across two paths; WITHOUT the target in the
        # fingerprint the second PUT would replay the first (returning 200 with
        # model_id="idemx/first") and never write idemx/second. WITH it the second
        # request has a distinct fingerprint, so it is caught as a key reused with a
        # different payload (422) rather than silently mis-written as the first.
        body = "profiles:\n  default:\n    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter\n"
        headers = {"Idempotency-Key": "put-shared"}
        r1 = self.client.put("/v1/configs/models/idemx/first", content=body, headers=headers)
        assert r1.status_code == 200, r1.text
        assert r1.json()["model_id"] == "idemx/first"
        r2 = self.client.put("/v1/configs/models/idemx/second", content=body, headers=headers)
        # NOT a false 200 replay writing the wrong model — the target is in the hash.
        assert r2.status_code == 422
        assert r2.json()["detail"]["error"] == "idempotency_mismatch"
        # The first model still exists; the second was never (mis)written under the key.
        assert self.client.get("/v1/configs/models/idemx/first").status_code == 200
        assert self.client.get("/v1/configs/models/idemx/second").status_code == 404
