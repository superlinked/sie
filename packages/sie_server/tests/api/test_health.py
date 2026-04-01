"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.readiness import mark_not_ready, mark_ready


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the SIE server."""
    app = AppFactory.create_app(AppStateConfig())
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for /healthz and /readyz endpoints."""

    def test_healthz_returns_ok(self, client: TestClient) -> None:
        """Liveness probe returns 200 OK."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_readyz_returns_ok_when_ready(self, client: TestClient) -> None:
        """Readiness probe returns 200 OK when ready."""
        # Ensure ready state (TestClient invokes lifespan which calls mark_ready(),
        # but other tests may have modified the global state)
        mark_ready()
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.text == "ok"
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_readyz_returns_503_when_not_ready(self, client: TestClient) -> None:
        """Readiness probe returns 503 when not ready."""
        # Force not-ready state
        mark_not_ready()
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.text == "not ready"
        # Restore ready state for other tests
        mark_ready()
