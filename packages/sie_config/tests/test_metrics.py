"""Config telemetry-facade integration tests.

The application emits OTel-only through one facade. These tests observe that
semantic seam directly; exact SDK instruments and OTLP transport are covered
in ``test_managed_metrics.py``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from sie_config import metrics as sie_metrics
from sie_config.app_factory import AppFactory


@dataclass
class CapturedTelemetry:
    requests: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    models: list[tuple[str, int]] = field(default_factory=list)
    publishes: list[tuple[str, str]] = field(default_factory=list)
    store_writes: list[tuple[str, str]] = field(default_factory=list)
    messaging_ready: list[bool] = field(default_factory=list)

    def record_request(self, **observation: Any) -> None:
        self.requests.append(observation)

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def set_models(self, *, source: str, count: int) -> None:
        self.models.append((source, count))

    def record_publish(self, *, operation: str, outcome: str) -> None:
        self.publishes.append((operation, outcome))

    def record_store_write(self, *, operation: str, outcome: str) -> None:
        self.store_writes.append((operation, outcome))

    def set_messaging_ready(self, ready: bool) -> None:
        self.messaging_ready.append(ready)


def _write_fixtures(root: Path) -> tuple[Path, Path, Path]:
    bundles = root / "bundles"
    models = root / "models"
    store = root / "store"
    bundles.mkdir()
    models.mkdir()
    (bundles / "default.yaml").write_text(
        yaml.dump({"name": "default", "priority": 10, "adapters": ["sie_server.adapters.bert_flash"]})
    )
    (models / "test__model.yaml").write_text(
        yaml.dump(
            {
                "sie_id": "test/model",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertFlashAdapter",
                        "max_batch_tokens": 8192,
                    }
                },
            }
        )
    )
    return bundles, models, store


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    bundles, models, store = _write_fixtures(root)
    telemetry = CapturedTelemetry()

    monkeypatch.setenv("SIE_BUNDLES_DIR", str(bundles))
    monkeypatch.setenv("SIE_MODELS_DIR", str(models))
    monkeypatch.setenv("SIE_CONFIG_STORE_DIR", str(store))
    monkeypatch.delenv("SIE_NATS_URL", raising=False)
    monkeypatch.delenv("SIE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SIE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SIE_METRICS_ENABLED", raising=False)
    monkeypatch.setattr(sie_metrics, "managed_metrics", lambda: telemetry)
    monkeypatch.setattr(sie_metrics, "telemetry_enabled", lambda: True)

    with TestClient(AppFactory.create_app()) as client:
        yield client, root, telemetry
    tmp.cleanup()


def _last_request(telemetry: CapturedTelemetry) -> dict[str, Any]:
    assert telemetry.requests
    return telemetry.requests[-1]


class TestHTTPMiddleware:
    def test_disabled_app_omits_request_instrumentation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sie_config.app_factory import _TelemetryHTTPMiddleware

        monkeypatch.setattr("sie_config.app_factory.setup_managed_metrics", lambda: None)
        monkeypatch.setattr(sie_metrics, "telemetry_enabled", lambda: False)
        app = AppFactory.create_app()

        assert all(middleware.cls is not _TelemetryHTTPMiddleware for middleware in app.user_middleware)

    def test_records_success_once(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        before = len(telemetry.requests)
        response = client.get("/v1/configs/models")
        assert response.status_code == 200
        assert len(telemetry.requests) == before + 1
        observation = _last_request(telemetry)
        assert observation["method"] == "GET"
        assert observation["route"] == "/v1/configs/models"
        assert observation["status_code"] == 200
        assert observation["duration_s"] >= 0

    def test_uses_route_template_not_raw_model_url(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        response = client.get("/v1/configs/models/test/model")
        assert response.status_code == 200
        observation = _last_request(telemetry)
        assert observation["route"] == "/v1/configs/models/{model_id:path}"
        assert "test/model" not in observation["route"]

    def test_records_error_response(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        response = client.get("/v1/configs/models/does/not/exist")
        assert response.status_code == 404
        observation = _last_request(telemetry)
        assert observation["route"] == "/v1/configs/models/{model_id:path}"
        assert observation["status_code"] == 404

    def test_unmatched_path_and_method_collapse_to_contract_other(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        response = client.request("ATTACKER-METHOD-1", "/arbitrary/customer/path")
        assert response.status_code == 404
        observation = _last_request(telemetry)
        assert observation["method"] == "other"
        assert observation["route"] == "other"
        assert "/arbitrary/customer/path" not in str(observation)

    def test_application_prometheus_endpoint_is_removed(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        response = client.get("/metrics")
        assert response.status_code == 404
        assert _last_request(telemetry)["route"] == "other"

    def test_uncaught_exception_is_recorded_as_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI
        from sie_config.app_factory import _TelemetryHTTPMiddleware

        telemetry = CapturedTelemetry()
        monkeypatch.setattr(sie_metrics, "managed_metrics", lambda: telemetry)
        app = FastAPI()
        app.add_middleware(_TelemetryHTTPMiddleware)  # type: ignore[arg-type]

        @app.get("/boom")
        def boom() -> None:
            raise RuntimeError("synthetic")

        response = TestClient(app, raise_server_exceptions=False).get("/boom")
        assert response.status_code == 500
        observation = _last_request(telemetry)
        assert observation["route"] == "/boom"
        assert observation["status_code"] == 500
        assert observation["duration_s"] >= 0


class TestStateAndMutationEvents:
    def test_startup_seeds_epoch_and_filesystem_model_count(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        assert telemetry.epochs[-1] == 0
        assert telemetry.messaging_ready[0] is False
        assert ("filesystem", 1) in telemetry.models
        assert client.get("/v1/configs/epoch").json()["epoch"] == 0

    @pytest.mark.asyncio
    async def test_no_store_mode_seeds_authoritative_zero_epoch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import FastAPI

        telemetry = CapturedTelemetry()
        monkeypatch.delenv("SIE_CONFIG_STORE_DIR", raising=False)
        monkeypatch.setattr(sie_metrics, "managed_metrics", lambda: telemetry)
        app = FastAPI()

        async with AppFactory._config_store(app):
            assert app.state.config_store is None

        assert telemetry.epochs == [0]

    def test_add_model_records_one_store_event_per_operation(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        before_writes = len(telemetry.store_writes)
        body = yaml.dump(
            {
                "sie_id": "metrics/store-test",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertFlashAdapter",
                        "max_batch_tokens": 8192,
                    }
                },
            }
        )
        response = client.post("/v1/configs/models", content=body)
        assert response.status_code == 201
        assert telemetry.store_writes[before_writes:] == [
            (sie_metrics.STORE_OP_WRITE_MODEL, sie_metrics.STORE_RESULT_SUCCESS),
            (sie_metrics.STORE_OP_INCREMENT_EPOCH, sie_metrics.STORE_RESULT_SUCCESS),
        ]
        assert telemetry.epochs[-1] == 1
        assert ("api", 1) in telemetry.models

    def test_store_failure_records_bounded_failure(self, app_client: Any) -> None:
        client, _, telemetry = app_client
        store = client.app.state.config_store
        original_write = store._backend.write_text  # type: ignore[attr-defined]

        def boom(path: str, content: str) -> None:
            del path, content
            raise OSError("simulated disk failure")

        store._backend.write_text = boom  # type: ignore[attr-defined]
        try:
            with pytest.raises(OSError, match="simulated disk failure"):
                store.write_model("broken/model", "sie_id: broken/model\n")
            assert telemetry.store_writes[-1] == (
                sie_metrics.STORE_OP_WRITE_MODEL,
                sie_metrics.STORE_RESULT_FAILURE,
            )
        finally:
            store._backend.write_text = original_write  # type: ignore[attr-defined]
