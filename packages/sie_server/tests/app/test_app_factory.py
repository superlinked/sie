"""Tests for the FastAPI app factory."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the SIE server."""
    app = AppFactory.create_app(AppStateConfig())
    return TestClient(app)


class TestAppFactory:
    """Tests for the FastAPI app factory."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """App factory returns a FastAPI application."""
        app = AppFactory.create_app(AppStateConfig())
        assert isinstance(app, FastAPI)

    def test_app_has_correct_metadata(self) -> None:
        """App has correct title and version."""
        app = AppFactory.create_app(AppStateConfig())
        assert app.title == "SIE Server"
        assert app.version == "0.1.0"

    def test_health_routes_registered(self, client: TestClient) -> None:
        """Health routes are registered in the app."""
        # Get OpenAPI schema to verify routes exist
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/healthz" in paths
        assert "/readyz" in paths
