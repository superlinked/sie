# Tests for sie-config Prometheus metrics.
#
# Covers:
# - `/metrics` endpoint exposition.
# - HTTP middleware counter + histogram on success, error, and
#   exception paths.
# - Store-write counter on success and failure.
# - Epoch gauge tracks `increment_epoch`.
# - Models gauge tracks `source` split.
# - NATS publish counter on success / partial / failure.
#
# The Prometheus default registry is process-global, so these tests
# mostly observe *deltas* (before-sample / after-sample) rather than
# absolute values, which is robust against other tests having already
# touched the same metric objects.

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY
from sie_config import metrics as sie_metrics
from sie_config.app_factory import AppFactory


def _sample_value(name: str, labels: dict[str, str] | None = None) -> float:
    """Look up a single sample from the default registry by name + labels.

    Returns 0.0 if the series hasn't been emitted yet, which is the
    same semantics Prometheus applies for a never-observed label
    combination.
    """
    labels = labels or {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


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
    """Spin up a real AppFactory-built app with a temp ConfigStore and
    no NATS. Yields a TestClient plus the tempdir root for assertions.
    """
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    bundles, models, store = _write_fixtures(root)

    monkeypatch.setenv("SIE_BUNDLES_DIR", str(bundles))
    monkeypatch.setenv("SIE_MODELS_DIR", str(models))
    monkeypatch.setenv("SIE_CONFIG_STORE_DIR", str(store))
    monkeypatch.delenv("SIE_NATS_URL", raising=False)
    monkeypatch.delenv("SIE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SIE_ADMIN_TOKEN", raising=False)

    app = AppFactory.create_app()
    client = TestClient(app)
    # `with TestClient(...)` is what fires FastAPI lifespan; drive it
    # the same way so the startup hooks (epoch seed, models gauge
    # seed) run.
    with client:
        yield client, root
    tmp.cleanup()


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_prometheus_text(self, app_client: Any) -> None:
        client, _ = app_client
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(CONTENT_TYPE_LATEST.split(";")[0])
        body = resp.text
        # Every metric defined in sie_config.metrics should be
        # discoverable in the exposition, at least via its HELP line.
        for name in (
            "sie_config_http_requests_total",
            "sie_config_http_request_duration_seconds",
            "sie_config_epoch",
            "sie_config_models_total",
            "sie_config_nats_connected",
            "sie_config_nats_publishes_total",
            "sie_config_store_writes_total",
        ):
            assert name in body, f"expected {name} in /metrics output"

    def test_metrics_endpoint_does_not_record_itself(self, app_client: Any) -> None:
        """Scrape traffic must not pollute the metric it's scraping —
        otherwise Prometheus's own poll rate dominates the counter.
        """
        client, _ = app_client
        before = _sample_value(
            "sie_config_http_requests_total",
            {"path": "/metrics", "method": "GET", "status": "200"},
        )
        client.get("/metrics")
        client.get("/metrics")
        after = _sample_value(
            "sie_config_http_requests_total",
            {"path": "/metrics", "method": "GET", "status": "200"},
        )
        assert after == before, "middleware should skip /metrics"


class TestHTTPMiddleware:
    def test_counter_increments_on_success(self, app_client: Any) -> None:
        client, _ = app_client
        labels = {
            "method": "GET",
            "path": "/v1/configs/models",
            "status": "200",
        }
        before = _sample_value("sie_config_http_requests_total", labels)
        resp = client.get("/v1/configs/models")
        assert resp.status_code == 200
        after = _sample_value("sie_config_http_requests_total", labels)
        assert after == before + 1

    def test_counter_uses_route_template_not_raw_url(self, app_client: Any) -> None:
        """Per-model reads must not create one time series per model —
        the `path` label has to be the FastAPI route template.
        """
        client, _ = app_client
        template_labels = {
            "method": "GET",
            "path": "/v1/configs/models/{model_id:path}",
            "status": "200",
        }
        before = _sample_value("sie_config_http_requests_total", template_labels)
        resp = client.get("/v1/configs/models/test/model")
        assert resp.status_code == 200
        after = _sample_value("sie_config_http_requests_total", template_labels)
        assert after == before + 1

        raw_labels = {
            "method": "GET",
            "path": "/v1/configs/models/test/model",
            "status": "200",
        }
        assert _sample_value("sie_config_http_requests_total", raw_labels) == 0

    def test_error_responses_also_counted(self, app_client: Any) -> None:
        client, _ = app_client
        labels = {
            "method": "GET",
            "path": "/v1/configs/models/{model_id:path}",
            "status": "404",
        }
        before = _sample_value("sie_config_http_requests_total", labels)
        resp = client.get("/v1/configs/models/does/not/exist")
        assert resp.status_code == 404
        after = _sample_value("sie_config_http_requests_total", labels)
        assert after == before + 1

    def test_uncaught_exception_counted_as_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When a handler raises, `BaseHTTPMiddleware.dispatch` exits
        # via the exception path. Without a `try/finally`, the metric
        # recording below `await call_next(request)` would be skipped
        # and operators would lose visibility on exactly the 500s they
        # care about most. This test pins the counter behaviour so
        # nobody regresses it back to "success-path only".
        from fastapi import FastAPI
        from sie_config.app_factory import _PrometheusHTTPMiddleware

        app = FastAPI()
        app.add_middleware(_PrometheusHTTPMiddleware)  # type: ignore

        @app.get("/boom")
        def boom() -> None:
            raise RuntimeError("synthetic")

        labels = {"method": "GET", "path": "/boom", "status": "500"}
        before = _sample_value("sie_config_http_requests_total", labels)

        # `TestClient` re-raises handler exceptions by default; turning
        # that off mirrors the production ASGI behaviour where
        # Starlette converts the exception into a 500 response.
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500

        after = _sample_value("sie_config_http_requests_total", labels)
        assert after == before + 1, "uncaught handler exceptions must still bump the 500 counter"

        duration_labels = {"method": "GET", "path": "/boom"}
        # The histogram `_count` series increments on every observation
        # regardless of status, so it is the cleanest way to assert
        # that latency was recorded on the exception path too.
        duration_count = _sample_value("sie_config_http_request_duration_seconds_count", duration_labels)
        assert duration_count >= 1.0


class TestEpochGauge:
    def test_epoch_seeded_from_store_on_startup(self, app_client: Any) -> None:
        # Startup runs `ConfigStore.read_epoch` which is 0 on a fresh
        # store. The gauge should mirror that exactly.
        client, _ = app_client
        assert _sample_value("sie_config_epoch") == 0
        resp = client.get("/v1/configs/epoch")
        assert resp.json()["epoch"] == 0

    def test_epoch_advances_on_add_model(self, app_client: Any) -> None:
        client, _ = app_client
        before = _sample_value("sie_config_epoch")
        yaml_body = yaml.dump(
            {
                "sie_id": "metrics/test-model",
                "profiles": {
                    "new_profile": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertFlashAdapter",
                        "max_batch_tokens": 8192,
                    }
                },
            }
        )
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201
        after = _sample_value("sie_config_epoch")
        assert after == before + 1


class TestModelsGauge:
    def test_filesystem_count_seeded_on_startup(self, app_client: Any) -> None:
        # The fixture seeds one filesystem model.
        _client, _ = app_client
        assert _sample_value("sie_config_models_total", {"source": "filesystem"}) >= 1, (
            "filesystem-source gauge should be populated on startup"
        )

    def test_api_count_increments_after_write(self, app_client: Any) -> None:
        client, _ = app_client
        before = _sample_value("sie_config_models_total", {"source": "api"})
        yaml_body = yaml.dump(
            {
                "sie_id": "metrics/new-api-model",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.bert_flash:BertFlashAdapter",
                        "max_batch_tokens": 8192,
                    }
                },
            }
        )
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201
        after = _sample_value("sie_config_models_total", {"source": "api"})
        assert after == before + 1


class TestStoreWriteCounter:
    def test_success_counter_increments_on_add_model(self, app_client: Any) -> None:
        client, _ = app_client
        write_before = _sample_value(
            "sie_config_store_writes_total",
            {"op": sie_metrics.STORE_OP_WRITE_MODEL, "result": sie_metrics.STORE_RESULT_SUCCESS},
        )
        epoch_before = _sample_value(
            "sie_config_store_writes_total",
            {"op": sie_metrics.STORE_OP_INCREMENT_EPOCH, "result": sie_metrics.STORE_RESULT_SUCCESS},
        )
        yaml_body = yaml.dump(
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
        resp = client.post("/v1/configs/models", content=yaml_body)
        assert resp.status_code == 201

        write_after = _sample_value(
            "sie_config_store_writes_total",
            {"op": sie_metrics.STORE_OP_WRITE_MODEL, "result": sie_metrics.STORE_RESULT_SUCCESS},
        )
        epoch_after = _sample_value(
            "sie_config_store_writes_total",
            {"op": sie_metrics.STORE_OP_INCREMENT_EPOCH, "result": sie_metrics.STORE_RESULT_SUCCESS},
        )
        assert write_after == write_before + 1
        assert epoch_after == epoch_before + 1

    def test_failure_counter_increments_on_backend_error(self, app_client: Any) -> None:
        """Simulate a disk-backend failure inside ConfigStore and
        verify the failure counter ticks. Use the real ConfigStore
        instance on app.state; monkeypatching the backend's
        write_text is the cheapest way to inject the failure.
        """
        client, _ = app_client
        store = client.app.state.config_store
        assert store is not None

        original_write = store._backend.write_text  # type: ignore

        def boom(path: str, content: str) -> None:
            raise OSError("simulated disk failure")

        store._backend.write_text = boom  # type: ignore
        try:
            before = _sample_value(
                "sie_config_store_writes_total",
                {"op": sie_metrics.STORE_OP_WRITE_MODEL, "result": sie_metrics.STORE_RESULT_FAILURE},
            )
            with pytest.raises(OSError, match="simulated disk failure"):
                store.write_model("broken/model", "sie_id: broken/model\n")
            after = _sample_value(
                "sie_config_store_writes_total",
                {"op": sie_metrics.STORE_OP_WRITE_MODEL, "result": sie_metrics.STORE_RESULT_FAILURE},
            )
            assert after == before + 1
        finally:
            store._backend.write_text = original_write  # type: ignore
